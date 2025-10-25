import streamlit as st
import pandas as pd
import time
from pathlib import Path
from insrenew import MLContext

# ------------------------------
# INITIALIZE CONTEXT
# ------------------------------
st.set_page_config(page_title="Insurance Renewal Prediction", layout="wide")

st.title("💡 Insurance Renewal Prediction App")
st.write("""
This app allows you to:
1. Upload and train on a CSV dataset  
2. Test the model on unseen data  
3. Predict renewal for individual customers  
""")

# Initialize MLContext (core engine)
BASE_DIR = Path(".")
ctx = MLContext(base_dir=str(BASE_DIR))

# Session state variables
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
if "test_done" not in st.session_state:
    st.session_state.test_done = False
if "best_model" not in st.session_state:
    st.session_state.best_model = None
if "test_df" not in st.session_state:
    st.session_state.test_df = None


# ------------------------------
# SECTION 1 — TRAINING
# ------------------------------
st.header(" Train Model on Uploaded Dataset")

train_file = st.file_uploader("Upload Training CSV", type=["csv"], key="train")

if st.button("Train Model"):
    if not train_file:
        st.error("Please upload a training dataset first.")
    else:
        with st.spinner("Running training pipeline..."):
            df_train = pd.read_csv(train_file)
            ctx.df = df_train

            progress = st.progress(0, text="Starting EDA...")
            time.sleep(0.5)
            ctx.run_eda()
            progress.progress(33, text="EDA completed. Running feature engineering...")

            ctx.run_fe()
            progress.progress(40, text="Feature engineering completed. Preprocessing...")

            # Feature config & preprocessor
            ctx.prepare_feature_config()
            ctx.build_preprocessor()
            ctx.split_train_test(test_size=0.2)
            progress.progress(45, text="Preprocessor built. Training models...")

            ctx.train_all_models()
            progress.progress(60, text="Model training complete.")

            ctx.adasyn_experiment()
            progress.progress(75, text="ADASYN Imbalance.")

            ctx.tune_with_optuna('rf')
            ctx.tune_with_optuna('xgb')
            progress.progress(80, text="Optuna tuning.")

            ctx.export_shap()
            progress.progress(85, text="SHAP exports.")

            ctx.prepare_artifacts()
            saved = ctx.save_models()
            progress.progress(90, text="Prepare artifacts.")

            # Finally, generate all the plots
            ctx.generate_all_plots(shap_model="xgb")
            ctx.generate_standard_outputs()
            progress.progress(100, text="Generate Plots.")

            st.session_state.model_trained = True
            best_summary = (ctx.results_dir / "final_model_selection_summary.json")
            if best_summary.exists():
                import json
                with open(best_summary, "r") as f:
                    summary = json.load(f)
                    st.session_state.best_model = summary.get("recommendation", {}).get("recommended_model", "Unknown")
                    st.success(f"✅ Training complete! Best model: **{st.session_state.best_model}**")
            else:
                st.warning("Training completed, but summary file not found.")


# ------------------------------
# SECTION 2 — TESTING
# ------------------------------
st.header("2️⃣ Test Model on Uploaded Test Data")

test_file = st.file_uploader("Upload Test CSV", type=["csv"], key="test")

if st.button("Run Test Predictions"):
    if not st.session_state.model_trained:
        st.error("Please train the model first.")
    elif not test_file:
        st.error("Please upload a test dataset.")
    else:
        with st.spinner("Running test predictions..."):
            test_path = BASE_DIR / "test_uploaded.csv"
            with open(test_path, "wb") as f:
                f.write(test_file.getvalue())

            results = ctx.predict_external_test(str(test_path))
            if isinstance(results, dict) and "preview" in results:
                df_preview = pd.DataFrame(results["preview"])
                st.dataframe(pd.DataFrame(df_preview),height=600)
            else:
                st.write("Test predictions completed.")
            st.session_state.test_done = True

            # Save test data for later prediction
            st.session_state.test_df = pd.read_csv(test_path)
            st.session_state.df_preview = df_preview
            st.success("✅ Test predictions completed successfully!")


# ------------------------------
# SECTION 3 — INDIVIDUAL PREDICTION
# ------------------------------
st.header("3️⃣ Predict for an Individual Customer")

if not st.session_state.test_done:
    st.info("Please upload test data and run test predictions first.")
else:
    test_df = st.session_state.test_df
    if "id" not in test_df.columns:
        st.warning("The test file must contain an 'id' column for lookup.")
    else:
        selected_id = st.selectbox("Select a Customer ID", options=test_df["id"].astype(str).tolist())

        if st.button("Predict Renewal for Selected Customer"):
            customer_row = test_df[test_df["id"].astype(str) == selected_id].iloc[0].to_dict()
            st.json(customer_row)

            with st.spinner("Running prediction..."):
                df_preview = st.session_state.df_preview
                target_id = selected_id

                # Check if the ID exists
                if target_id in df_preview["Id"].values:
                    # Filter the row
                    row = df_preview[df_preview["Id"] == target_id].iloc[0]

                    # Extract values
                    y_pred_value = row["y_pred"]
                    prob_value = row["prob"]
                    #st.write(f"Id: {target_id}, y_pred: {y_pred_value}, prob: {prob_value}")
                    if y_pred_value == 1:
                        st.info(f'Customer id {target_id} **likely to renew** with a proability of {prob_value}')
                    else:
                        st.error(f'Customer id {target_id} **less likely to renew** with a proability of {prob_value}')
                else:
                    st.write("ID not found in df_preview.")

# ------------------------------
# FOOTER
# ------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, FastAPI-style architecture, and MLCore engine.")