# app.py
"""
Streamlit front-end for the ML pipeline.

Three tabs:
- Train: upload training CSV, press Train -> calls backend /train
- Test: upload test CSV, pick model (or use chosen), press Test -> calls backend /test and shows top-10
- Predict: pick an ID from uploaded test set or enter fields manually, press Predict -> calls backend /predict
"""

import streamlit as st
import requests
import pandas as pd
import time
import json
from pathlib import Path

BACKEND_URL = "http://localhost:8000"
st.set_page_config(page_title="Insurance Renewal ML App", layout="wide")

st.title("Insurance Renewal — Training / Testing / Single Prediction")

if "train_result" not in st.session_state:
    st.session_state["train_result"] = None
if "test_df" not in st.session_state:
    st.session_state["test_df"] = None
if "chosen_model" not in st.session_state:
    st.session_state["chosen_model"] = None

tab = st.tabs(["Train", "Test", "Predict"])

# --------------------------
# TRAIN TAB
# --------------------------
with tab[0]:
    st.header("1) Upload Training CSV and Train Models")
    train_file = st.file_uploader("Upload training CSV", type=["csv"], key="train_file")

    if st.button("Start Training"):
        if train_file is None:
            st.error("Please upload a training CSV first.")
        else:
            with st.spinner("Uploading and training... this runs on the backend (may take minutes)"):
                # send file to backend
                files = {"train_file": ("train.csv", train_file.getvalue())}
                try:
                    resp = requests.post(f"{BACKEND_URL}/train", files=files, timeout=60*20)  # long timeout
                except Exception as e:
                    st.error(f"Failed to reach backend: {e}")
                    st.stop()

                if resp.status_code != 200:
                    st.error(f"Training failed: {resp.text}")
                else:
                    data = resp.json()
                    st.session_state["train_result"] = data
                    chosen = data.get("chosen_model")
                    if chosen:
                        st.session_state["chosen_model"] = chosen
                    st.success("Training completed")
                    st.write("Status log:")
                    st.json(data.get("status_log"))

    # Show last train result
    if st.session_state["train_result"]:
        st.subheader("Last training result")
        st.write("Chosen model:", st.session_state.get("chosen_model"))
        st.json(st.session_state["train_result"])

# --------------------------
# TEST TAB
# --------------------------
with tab[1]:
    st.header("2) Upload Test CSV and Run Predictions")
    st.write("Chosen model from training:", st.session_state.get("chosen_model"))
    test_file = st.file_uploader("Upload test CSV", type=["csv"], key="test_file")
    model_override = st.text_input("Model override (optional, e.g., xgb, lr, nn, brf, eec, tab)", value="", key="model_override")

    if st.button("Run Test Predictions"):
        if test_file is None:
            st.error("Please upload a test CSV first.")
        else:
            # send file to backend test endpoint
            #files = {"test_file": ("test.csv", test_file.getvalue())}
            files = {"file": ("test.csv", test_file.getvalue())}
            data = {}
            if model_override:
                data['model_name'] = model_override
            try:
                # Use multipart/form-data; include model override as form field if present
                resp = requests.post(f"{BACKEND_URL}/test", files=files, data=data, timeout=60*5)
            except Exception as e:
                st.error(f"Test endpoint error: {e}")
                st.stop()

            if resp.status_code != 200:
                st.error(f"Test failed: {resp.text}")
            else:
                result = resp.json()
                st.success("Test predictions created")
                st.write("Model used:", result.get("model"))
                st.write("Predictions file:", result.get("predictions_path"))
                st.write("Confidence file:", result.get("predictions_confidence_path"))
                preview = result.get("preview", [])
                df_preview = pd.DataFrame(preview)
                st.dataframe(df_preview)

                # store uploaded test_df in session for Predict tab convenience
                test_df = pd.read_csv(pd.io.common.BytesIO(test_file.getvalue()))
                st.session_state["test_df"] = test_df
                st.info("Uploaded test file stored in session state (for single-ID predictions).")

# --------------------------
# PREDICT TAB
# --------------------------
with tab[2]:
    st.header("3) Single Customer Prediction")
    st.write("You can either select an ID from uploaded test CSV (if you uploaded one in Test tab) or manually enter feature values.")

    test_df = st.session_state.get("test_df")
    if test_df is not None:
        st.subheader("Select ID from uploaded test set")
        if 'id' in test_df.columns:
            selected_id = st.selectbox("Choose ID", test_df['id'].astype(str).tolist())
            use_row = st.button("Use selected ID for prediction")
            if use_row:
                row = test_df[test_df['id'].astype(str) == selected_id].iloc[0].to_dict()
                st.write("Selected row data:")
                st.json(row)
                # call backend predict
                #payload = {"customer": row}
                payload = row
                try:
                    resp = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=30)
                except Exception as e:
                    st.error(f"Predict endpoint error: {e}")
                    st.stop()
                if resp.status_code == 200:
                    out = resp.json()
                    st.success(f"Prediction: {out.get('prediction')} (prob={out.get('prob')}) using model {out.get('model')}")
                else:
                    st.error(f"Prediction failed: {resp.text}")
        else:
            st.info("Uploaded test CSV doesn't have an 'id' column.")

    st.subheader("Or: Manually enter a customer's features")
    st.write("Enter JSON-like dictionary of feature_name: value (e.g., {\"age_years\": 45, \"Income\": 42000, ...})")
    cust_text = st.text_area("Customer feature dict", height=200, value='{"age_years": 45, "Income": 42000}')
    if st.button("Predict for manual customer"):
        try:
            cust = json.loads(cust_text)
        except Exception as e:
            st.error(f"Invalid JSON: {e}")
            st.stop()
        payload = {"customer": cust}
        try:
            resp = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=30)
        except Exception as e:
            st.error(f"Predict endpoint error: {e}")
            st.stop()

        if resp.status_code == 200:
            out = resp.json()
            st.success(f"Prediction: {out.get('prediction')} (prob={out.get('prob')}) using model {out.get('model')}")
        else:
            st.error(f"Prediction failed: {resp.text}")
