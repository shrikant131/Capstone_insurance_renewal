# streamlit_app.py
# Streamlit UI for Insurance Renewal Prediction
# Safe to run when Plotly is missing (falls back to matplotlib)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
import json
from pathlib import Path
import hashlib
from datetime import datetime
import matplotlib.pyplot as plt
import asyncio
import requests

# optional imports
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# -----------------------
# Utility / Helper funcs
# -----------------------
@st.cache_data(ttl=3600)
def load_model(path: str = "models/final_model.joblib"):
    try:
        return joblib.load(path)
    except Exception as e:
        st.warning(f"Could not load model at {path}: {e}")
        return None

@st.cache_data
def load_sample_data(path: str = "data/sample.csv") -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        # fallback tiny sample
        return pd.DataFrame({
            "id": [1, 2, 3],
            "age_years": [35, 45, 60],
            "income": [40000, 70000, 30000],
            "premium": [2000, 5000, 1500],
            "Count_3-6_months_late": [0, 1, 0],
            "Count_6-12_months_late": [0, 0, 1],
            "Count_more_than_12_months_late": [0, 0, 0],
            "sourcing_channel": ["Agent", "Online", "Agent"],
            "residence_area_type": ["Urban", "Semiurban", "Rural"],
            "renewal": [1, 0, 1]
        })

@st.cache_data
def fe_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal feature engineering for demo / UI. Replace with real pipeline in production."""
    df = df.copy()
    if "premium_to_income" not in df.columns and "premium" in df.columns and "income" in df.columns:
        df["premium_to_income"] = df["premium"] / (df["income"].replace(0, np.nan))
    # simple derived features if present
    if "Count_3-6_months_late" in df.columns:
        df["total_late_counts"] = df.get("Count_3-6_months_late", 0) + df.get("Count_6-12_months_late", 0) + df.get("Count_more_than_12_months_late", 0)
    # fill missing for safe display / predict
    retu
