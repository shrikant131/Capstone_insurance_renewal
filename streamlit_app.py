# streamlit_app.py
# Finalized Streamlit UI for Insurance Renewal Prediction
# Drop this file into your repo (replace existing streamlit_app.py).
# Make sure models/model_metadata.json and models/*.joblib are present if you want automatic model selection.

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import plotly.express as px
import plotly.graph_objects as go
import shap
import base64
from datetime import datetime
import json
from pathlib import Path
import hashlib
import matplotlib.pyplot as plt

# -----------------------
# Helper functions (must be defined before use)
# -----------------------
@st.cache_data(ttl=3600)
def load_model(path="models/final_model.joblib"):
    try:
        model = joblib.load(path)
    except Exception as e:
        st.warning(f"Could not load model at {path}: {e}")
        model = None
    return model

@st.cache_data
def load_sample_data(path="data/sample.csv"):
    try:
        return pd.read_csv(path)
    except Exception:
        # tiny sample fallback
        return pd.DataFrame({
            "id": [1,2,3],
            "age_years": [35,45,60],
            "income": [40000,70000,30000],
            "premium": [2000,5000,1500],
            "Count_3-6_months_late":[0,1,0],
            "Count_6-12_months_late":[0,0,1],
            "Count_more_than_12_months_late":[0,0,0],
            "sourcing_channel":["Agent","Online","Agent"],
            "residence_area_type":["Urban","Semiurban","Rural"],
            "renewal":[1,0,1]
        })

@st.cache_data
def fe_transform(df):
    df = df.copy()
    # minimal feature engineering for UI demo: create premium_to_income if missing
    if "premium_to_income" not in df.columns and "premium" in df.columns and "income" in df.columns:
        df["premium_to_income"] = df["premium"] / (df["income"].replace(0, np.nan))
    # fill na for demo (in production use the real pipeline)
    df = df.fillna(-999)
    return df

def to_download_link(df: pd.DataFrame, filename="predictions.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f"data:file/csv;base64,{b64}", filename

def dataset_fingerprint(df):
    """Same logic used in Streamlit app"""
    cols = ",".join(sorted(df.columns.astype(str)))
    meta = f"{cols}|rows={len(df)}"
    return hashlib.sha1(meta.encode("utf-8")).hexdigest()[:12]

def save_model_metadata(
    model_key: str,
    model_path: str,
    model_name: str,
    version: str,
    auc: float,
    accuracy: float,
    description: str,
    feature_names: list,
    df_eval=None,
    per_dataset_metrics: dict=None,
    metadata_path="models/model_metadata.json"
):
    """
    Create or update model metadata JSON automatically after training.
    """
    path = Path(metadata_path)
    if path.exists():
        with open(path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    # Build per-dataset fingerprint entry if df_eval given
    per_dataset = {}
    if df_eval is not None:
        fp = dataset_fingerprint(df_eval)
        per_dataset[fp] = {
            "auc": round(auc, 3),
            "accuracy": round(accuracy, 3),
            "notes": "auto-logged from training dataset"
        }

    # Compose metadata record
    metadata[model_key] = {
        "model_name": model_name,
        "model_version": version,
        "trained_on": str(date.today()),
        "auc": round(auc, 3),
        "accuracy": round(accuracy, 3),
        "description": description,
        "path": model_path,
        "feature_names": feature_names,
        "per_dataset": per_dataset or per_dataset_metrics or {}
    }

    # Save back to file
    Path(metadata_path).parent.mkdir(exist_ok=True, parents=True)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Updated {metadata_path} with entry: {model_key}")


save_model_metadata(
    model_key="xgb_v1.3",
    model_path=best_model_path,
    model_name="XGBoost Classifier",
    version="v1.3",
    auc=best_auc,
    accuracy=best_acc,
    description="Best model selected from Optuna tuning across 5 candidates.",
    feature_names=feature_names,
    df_eval=X_valid  # optional; used to record dataset fingerprint
)

# -----------------------
# Model metadata utilities
# -----------------------
def dataset_fingerprint(df):
    # simple fingerprint: sorted column names + row count
    cols = ",".join(sorted(df.columns.astype(str)))
    meta = f"{cols}|rows={len(df)}"
    return hashlib.sha1(meta.encode("utf-8")).hexdigest()[:12]

@st.cache_data(ttl=3600)
def load_all_model_metadata(path="models/model_metadata.json"):
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Failed to load model metadata: {e}")
        return {}

def choose_best_model_for_df(df, metadata):
    if not metadata:
        return None, None
    fp = dataset_fingerprint(df)
    # look for per_dataset matches
    candidates = []
    for key, meta in metadata.items():
        per = meta.get("per_dataset", {})
        if fp in per:
            candidates.append((key, per[fp].get("auc", meta.get("auc", 0.0)), "per_dataset"))
    if candidates:
        best_key, best_auc, _ = max(candidates, key=lambda x: x[1])
        return best_key, "per_dataset"
    # fallback: global best by 'auc'
    candidates = [(key, meta.get("auc", 0.0)) for key, meta in metadata.items()]
    best_key, _ = max(candidates, key=lambda x: x[1])
    return best_key, "global"

def render_model_card(meta):
    if not meta:
        st.info("No model metadata available.")
        return
    model_name = meta.get("model_name", "Unknown model")
    version = meta.get("model_version", "-")
    trained_on = meta.get("trained_on", "-")
    auc = meta.get("auc", None)
    acc = meta.get("accuracy", None)
    metric_str = []
    if auc is not None:
        metric_str.append(f"AUC = {auc:.3f}")
    if acc is not None:
        metric_str.append(f"Accuracy = {acc*100:.1f}%")
    metric_line = ", ".join(metric_str) if metric_str else meta.get("metric", "-")

    phrasing = (
        f"🧠 **Active Model:** {model_name} ({version})  \n"
        f"Selected as best performer ({metric_line}) from automated Optuna tuning across 5 candidate models on the provided dataset."
    )

    st.markdown(
        f"""
        <div style="background:#f2f8fc; padding:12px; border-radius:10px;">
          <div style="font-weight:700; font-size:15px;">{phrasing}</div>
          <div style="margin-top:6px; color:#345; font-size:13px;"><i>{meta.get('description','')}</i></div>
          <div style="margin-top:8px; font-size:12px; color:#555;">Trained: {trained_on}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------
# Page config & styling
# -----------------------
st.set_page_config(page_title="Insurance Renewal Predictor", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #f7fbff 0%, #ffffff 100%); }
    .big-title { font-size:28px; font-weight:700; color:#0b4c6a; }
    .subtitle { color:#2b6777; margin-bottom:12px; }
    .card { background: white; padding: 16px; border-radius: 10px; box-shadow: 0 2px 8px rgba(14,30,37,0.06); }
    </style>
    """, unsafe_allow_html=True
)

# -----------------------
# Top header
# -----------------------
col1, col2 = st.columns([0.15, 0.85])
with col1:
    st.image("assets/logo.png" if st.secrets.get("has_logo", False) else "https://placehold.co/80x80?text=Logo", width=80)
with col2:
    st.markdown('<div class="big-title">Insurance Renewal Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Predict customer renewals & surface who to target for retention campaigns</div>', unsafe_allow_html=True)
    st.markdown(f"**Model:** `{st.secrets.get('model_version', 'v1.0')}` &nbsp;&middot;&nbsp; **Last updated:** {st.secrets.get('model_updated', datetime.utcnow().isoformat())}")

# -----------------------
# Sidebar controls
# -----------------------
st.sidebar.header("Controls")
upload = st.sidebar.file_uploader("Upload policy CSV (optional)", type=["csv"])
sample_mode = st.sidebar.selectbox("Load dataset", ["Sample data", "Upload data"], index=0)
manual_model_path = st.sidebar.text_input("Manual model path (optional)", value="models/final_model.joblib")
predict_button = st.sidebar.button("Run Predictions")
threshold = st.sidebar.slider("Renewal probability threshold", 0.0, 1.0, 0.5, 0.01)
show_shap = st.sidebar.checkbox("Show SHAP explainability", value=True)

# -----------------------
# Load data + basic FE
# -----------------------
if sample_mode == "Sample data":
    df_raw = load_sample_data()
elif upload:
    try:
        df_raw = pd.read_csv(upload)
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")
        df_raw = load_sample_data()
else:
    df_raw = load_sample_data()

df = fe_transform(df_raw)

# -----------------------
# Model metadata selection (run AFTER df exists)
# -----------------------
all_meta = load_all_model_metadata("models/model_metadata.json")
model_keys = list(all_meta.keys())

best_key = None
source = None
if df is not None and model_keys:
    try:
        best_key, source = choose_best_model_for_df(df, all_meta)
    except Exception as e:
        st.warning(f"Model auto-selection failed: {e}")

# Sidebar selectbox: let user override. If no metadata, allow manual path.
if model_keys:
    default_index = model_keys.index(best_key) if (best_key in model_keys) else 0
    selected_key = st.sidebar.selectbox("Active model (auto-selected or choose)", options=model_keys, index=default_index)
else:
    st.sidebar.info("No model metadata found. You can provide a manual model path.")
    selected_key = None

# determine which model to load: metadata selected or manual path
selected_meta = all_meta.get(selected_key) if selected_key else None
render_model_card(selected_meta)

model = None
if selected_meta and selected_meta.get("path"):
    model = load_model(selected_meta.get("path"))
    if model:
        st.sidebar.success(f"Loaded model: {selected_meta.get('model_name')} ({selected_meta.get('model_version')})")
    else:
        st.sidebar.error(f"Failed to load model at {selected_meta.get('path')}.")
else:
    # if metadata not present or user wants manual model, try manual path
    if manual_model_path:
        try:
            model = load_model(manual_model_path)
            if model:
                st.sidebar.success(f"Loaded model from manual path")
        except Exception:
            model = None

# -----------------------
# KPI cards
# -----------------------
pred_col1, pred_col2, pred_col3, pred_col4 = st.columns(4)
with pred_col1:
    st.metric("Data rows", f"{len(df):,}")
with pred_col2:
    churn_guess = round(df.get("renewal", pd.Series([0]*len(df))).mean() * 100, 2)
    st.metric("Existing renewal %", f"{churn_guess}%")
with pred_col3:
    st.metric("Model loaded", "Yes" if model else "No")
with pred_col4:
    st.metric("Threshold", f"{threshold:.2f}")

st.write("")

# -----------------------
# Tabs: Data / EDA / Predict / Explain / Monitoring
# -----------------------
tabs = st.tabs(["Data", "EDA", "Predictions", "Explainability", "Monitoring"])

# ----- Data tab -----
with tabs[0]:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(200), use_container_width=True)
    st.download_button("Download preview CSV", data=df.head(200).to_csv(index=False).encode('utf-8'), file_name="preview.csv", mime="text/csv")

# ----- EDA tab -----
with tabs[1]:
    st.subheader("Exploratory Data Analysis")
    left, right = st.columns([2,1])
    with left:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if num_cols:
            col_sel = st.selectbox("Feature for distribution", num_cols, index=0)
            fig = px.histogram(df, x=col_sel, nbins=40, marginal="box", title=f"Distribution of {col_sel}")
            st.plotly_chart(fig, use_container_width=True)
        if len(num_cols) > 1:
            corr = df[num_cols].corr()
            fig2 = px.imshow(corr, text_auto=True, title="Correlation matrix")
            st.plotly_chart(fig2, use_container_width=True)
    with right:
        st.markdown("### Cohort & KPIs")
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if not cat_cols:
            cat_cols = ["sourcing_channel"] if "sourcing_channel" in df.columns else []
        if cat_cols:
            group_col = st.selectbox("Group by (categorical)", cat_cols, index=0)
            cohort = df.groupby(group_col).agg(total=("id","count"), mean_premium=("premium","mean"))
            st.dataframe(cohort.reset_index())
        else:
            st.info("No categorical columns detected for cohorting.")

# ----- Predictions tab -----
with tabs[2]:
    st.subheader("Batch & Single Predictions")
    if not model:
        st.warning("Model not loaded. Upload model file or set correct path in sidebar.")
    # Batch predict button
    if predict_button and model:
        with st.spinner("Running model predictions..."):
            X_cols = [c for c in df.columns if c not in ("id", "renewal")]
            X = df[X_cols].copy()

            # Validate expected features if provided in metadata
            expected_features = None
            if selected_meta:
                expected_features = selected_meta.get("feature_names")
            if expected_features:
                missing = [c for c in expected_features if c not in X.columns]
                if missing:
                    st.error(f"Model expects features not present in data: {missing}")
                    st.stop()
                # reorder columns
                X = X[expected_features]
            else:
                st.warning("No 'feature_names' in metadata — ensure features match model training set.")

            try:
                proba = model.predict_proba(X)[:, 1]
            except Exception:
                proba = model.predict(X)
                proba = np.clip(proba, 0, 1)

            df_out = df.copy()
            df_out["renewal_prob"] = proba
            df_out["renewal_pred"] = (df_out["renewal_prob"] >= threshold).astype(int)

            st.success("Predictions finished")
            st.dataframe(df_out.sort_values("renewal_prob", ascending=False).head(50), use_container_width=True)

            to_target = df_out[(df_out["renewal_pred"]==0)].sort_values("premium", ascending=False).head(200)
            st.markdown(f"**Top {len(to_target)} customers to target** (predicted non-renewals with highest premium)")
            st.dataframe(to_target[["id","premium","renewal_prob"]].head(50))

            csv_bytes = df_out.to_csv(index=False).encode("utf-8")
            st.download_button("Download full predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")

    # Single-prediction UI
    st.markdown("---")
    st.markdown("### Predict for a single customer")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    first_cols = numeric_cols[:6]
    if first_cols:
        cols = st.columns(len(first_cols))
        single = {}
        for i, c in enumerate(first_cols):
            single[c] = cols[i].number_input(c, value=float(df[c].median() if c in df.columns else 0.0))
        for c in numeric_cols[6:12]:
            single[c] = st.number_input(c, value=float(df[c].median() if c in df.columns else 0.0))
        if st.button("Predict single"):
            x_single = pd.DataFrame([single])
            x_single = fe_transform(x_single)
            if model:
                try:
                    p = model.predict_proba(x_single)[:,1][0]
                except Exception:
                    p = float(model.predict(x_single)[0])
                st.metric("Renewal probability", f"{p:.3f}")
                st.info("Action: Add to campaign list" if p < threshold else "Likely to renew — low priority")
            else:
                st.warning("Model not loaded. Can't predict.")
    else:
        st.info("No numeric columns available for single-prediction UI.")

# ----- Explainability tab -----
with tabs[3]:
    st.subheader("Explainability (SHAP)")
    if not model:
        st.warning("Model not loaded — SHAP disabled")
    else:
        try:
            sample_for_shap = df.sample(min(200, len(df)), random_state=42)
            Xshap = sample_for_shap[[c for c in sample_for_shap.columns if c not in ("id","renewal")]]
            explainer = shap.Explainer(model.predict_proba if hasattr(model, "predict_proba") else model.predict, Xshap, feature_names=Xshap.columns)
            shap_values = explainer(Xshap)
            st.markdown("### SHAP Summary (beeswarm)")
            plt.figure(figsize=(8,4))
            shap.plots.beeswarm(shap_values, max_display=12)
            st.pyplot(plt.gcf())
            plt.clf()
        except Exception as e:
            st.warning(f"SHAP plotting failed: {e}")

# ----- Monitoring tab -----
with tabs[4]:
    st.subheader("Monitoring & Logs")
    st.markdown("Show recent EDA / FE / Training logs, metrics, and dataset snapshots.")
    if st.button("Show last 200 lines of app.log (if available)"):
        try:
            with open("app.log") as f:
                lines = f.read().splitlines()[-200:]
                st.text("\n".join(lines))
        except Exception:
            st.info("No app.log found in working directory.")

# -----------------------
# Footer
# -----------------------
st.markdown("---")
st.markdown("Built with ❤️  •  Tips: Use `st.cache_data` for fe/model loads and avoid heavy model training in the UI.")
