import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncio
# Make uvicorn optional — prevents Streamlit import failure if not installed
try:
    import uvicorn
except ImportError:
    uvicorn = None
    print("⚠️ uvicorn not found (safe to ignore if running Streamlit only)")
from typing import Optional, Dict, Any
import joblib
import time
import logging
import sys
from fastapi.responses import JSONResponse
import threading
from insrenew import MLContext

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("app")

# Initialize FastAPI app
api = FastAPI(title="Insurance Renewal ML Backend")
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables and paths
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
ARTIFACTS_FILE = MODELS_DIR / "artifacts.json"
FINAL_SUMMARY = RESULTS_DIR / "final_model_selection_summary.json"

# Create necessary directories
for d in (DATA_DIR, MODELS_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Pydantic models for request/response
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class StatusLog(BaseModel):
    step: str
    msg: str
    ts: float

class TrainResponse(BaseModel):
    status_log: List[StatusLog]
    trained_models: Dict[str, bool]
    chosen_model: Optional[str]

class TestResponse(BaseModel):
    message: str
    metrics: Dict[str, Any]
    preview: List[Dict[str, Any]]
    predictions_path: str

class PredictResponse(BaseModel):
    prediction: int
    prob: float
    model: str

# FastAPI endpoints
@api.post("/train", response_model=TrainResponse)
async def train_endpoint(train_data: Dict[str, Any]):
    """Train models using uploaded data"""
    status_log = []
    start_ts = time.time()
    
    try:
        # Convert dict to DataFrame
        df = pd.DataFrame(train_data)
        
        # Save training data
        train_path = DATA_DIR / "train_uploaded.csv"
        df.to_csv(train_path, index=False)
        status_log.append(StatusLog(step="upload", msg=f"Saved training file", ts=time.time()))
        
        # Initialize ML Context and run pipeline
        ctx = MLContext(base_dir=str(BASE_DIR))
        ctx.set_df(df)
        
        # Run EDA
        try:
            ctx.run_eda()
            status_log.append({"step": "eda", "msg": "EDA completed", "ts": time.time()})
        except Exception as e:
            status_log.append({"step": "eda", "msg": f"EDA failed: {e}", "ts": time.time()})
        
        # Feature engineering
        try:
            ctx.run_fe()
            ctx.prepare_feature_config()
            ctx.build_preprocessor()
            status_log.append({"step": "fe", "msg": "Feature engineering complete", "ts": time.time()})
        except Exception as e:
            status_log.append({"step": "fe", "msg": f"Feature engineering failed: {e}", "ts": time.time()})
        
        # Train/test split
        ctx.split_train_test(test_size=0.2)
        
        # Train models
        trained = {}
        models_to_train = [
            ('lr', ctx.train_lr),
            ('xgb', ctx.train_xgboost),
            ('nn', ctx.train_neural_net),
            ('brf', ctx.train_balanced_random_forest),
            ('eec', ctx.train_easy_ensemble),
            ('lgb', ctx.train_lightgbm),
            ('tab', ctx.train_tabnet)
        ]
        
        for model_name, train_func in models_to_train:
            try:
                train_func()
                trained[model_name] = True
                status_log.append({"step": f"train_{model_name}", "msg": f"{model_name} trained successfully", "ts": time.time()})
            except Exception as e:
                status_log.append({"step": f"train_{model_name}", "msg": f"{model_name} training failed: {e}", "ts": time.time()})
        
        # Save models and artifacts
        ctx.prepare_artifacts()
        saved = ctx.save_models(prefix="trained")
        artifacts = {
            'artifacts': {k: str(v) for k, v in saved.items()},
            'feature_names': ctx.artifacts.get('feature_names', []),
        }
        with open(ARTIFACTS_FILE, "w") as f:
            json.dump(artifacts, f, indent=2, default=str)
        
        # Aggregate metrics and save final selection
        ctx.aggregate_model_comparison_metrics()
        summary_path = ctx.prepare_final_model_selection_summary()
        
        # Get chosen model
        chosen = None
        try:
            with open(FINAL_SUMMARY, "r") as f:
                final = json.load(f)
                chosen = final.get("recommendation", {}).get("recommended_model")
        except Exception:
            pass
        
        return {"status_log": status_log, "trained_models": trained, "chosen_model": chosen}
    
    except Exception as e:
        status_log.append({"step": "error", "msg": str(e)})
        return {"status_log": status_log, "error": str(e)}

@api.post("/test", response_model=TestResponse)
async def test_endpoint(test_data: Dict[str, Any]):
    """Run predictions on test data"""
    try:
        # Convert dict to DataFrame
        df = pd.DataFrame(test_data)
        
        # Save test data
        test_path = DATA_DIR / "test_uploaded.csv"
        df.to_csv(test_path, index=False)
        
        ctx = MLContext(base_dir=str(BASE_DIR))
        test_results = ctx.predict_external_test(str(test_path))
        
        # Store test data for later use
        ctx.test_df = df
        
        return TestResponse(
            message=test_results.get("message", ""),
            metrics=test_results.get("metrics", {}),
            preview=test_results.get("preview", []),
            predictions_path=str(test_results.get("output_path", ""))
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@api.post("/predict", response_model=PredictResponse)
async def predict_endpoint(customer_data: Dict[str, Any]):
    """Predict for a single customer"""
    try:
        ctx = MLContext(base_dir=str(BASE_DIR))
        
        # Align features with training schema
        feature_path = ctx.results_dir / "feature_names.npy"
        if feature_path.exists():
            expected_features = list(np.load(feature_path, allow_pickle=True))
            for col in expected_features:
                if col not in customer_data:
                    customer_data[col] = 0
            customer_data = {k: customer_data[k] for k in expected_features if k in customer_data}
        
        result = ctx.predict_single(customer_dict=customer_data)
        return PredictResponse(
            prediction=result.get("prediction", 0),
            prob=result.get("prob", 0.0),
            model=result.get("model", "unknown")
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# Streamlit UI
def main():
    st.set_page_config(page_title="Insurance Renewal ML App", layout="wide")
    st.title("Insurance Renewal — Training / Testing / Single Prediction")

    # Initialize session state
    if "train_result" not in st.session_state:
        st.session_state["train_result"] = None
    if "test_df" not in st.session_state:
        st.session_state["test_df"] = None
    if "chosen_model" not in st.session_state:
        st.session_state["chosen_model"] = None

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Train", "Test", "Predict"])

    # Train Tab
    with tab1:
        st.header("1) Upload Training CSV and Train Models")
        train_file = st.file_uploader("Upload training CSV", type=["csv"], key="train_file")

        if st.button("Start Training"):
            if train_file is None:
                st.error("Please upload a training CSV first.")
            else:
                with st.spinner("Training models... this may take several minutes"):
                    # Read CSV and call training endpoint
                    train_data = pd.read_csv(train_file)
                    # Convert DataFrame to dict for FastAPI
                    train_dict = train_data.to_dict(orient='list')
                    result = asyncio.run(train_endpoint(train_dict))
                    
                    if "error" in result:
                        st.error(f"Training failed: {result['error']}")
                    else:
                        st.session_state["train_result"] = result
                        if result.chosen_model:
                            st.session_state["chosen_model"] = result.chosen_model
                        st.success("Training completed")
                        st.write("Status log:")
                        st.json([log.dict() for log in result.status_log])

        if st.session_state["train_result"]:
            st.subheader("Last training result")
            st.write("Chosen model:", st.session_state.get("chosen_model"))
            st.json(st.session_state["train_result"])

    # Test Tab
    with tab2:
        st.header("2) Upload Test CSV and Run Predictions")
        st.write("Chosen model from training:", st.session_state.get("chosen_model"))
        test_file = st.file_uploader("Upload test CSV", type=["csv"], key="test_file")
        model_override = st.text_input("Model override (optional)", value="", key="model_override")

        if st.button("Run Test Predictions"):
            if test_file is None:
                st.error("Please upload a test CSV first.")
            else:
                with st.spinner("Running predictions..."):
                    test_data = pd.read_csv(test_file)
                    # Convert DataFrame to dict for FastAPI
                    test_dict = test_data.to_dict(orient='list')
                    result = asyncio.run(test_endpoint(test_dict))
                    
                    st.success("Test predictions created")
                    st.write("Predictions file:", result.predictions_path)
                    
                    if result.preview:
                        st.dataframe(pd.DataFrame(result.preview))
                    
                    st.session_state["test_df"] = test_data
                    st.info("Test data stored for single predictions")

                    # Display metrics
                    st.write("Model evaluation metrics:")
                    st.json(result.metrics)

    # Predict Tab
    with tab3:
        st.header("3) Single Customer Prediction")
        st.write("Select an ID from uploaded test data or enter feature values manually")

        test_df = st.session_state.get("test_df")
        if test_df is not None:
            st.subheader("Select ID from uploaded test set")
            if 'id' in test_df.columns:
                selected_id = st.selectbox("Choose ID", test_df['id'].astype(str).tolist())
                if st.button("Use selected ID for prediction"):
                    row = test_df[test_df['id'].astype(str) == selected_id].iloc[0].to_dict()
                    st.write("Selected row data:")
                    st.json(row)
                    
                    result = asyncio.run(predict_endpoint(row))
                    st.success(f"Prediction: {result.prediction} (prob={result.prob:.3f}) using model {result.model}")

        st.subheader("Or: Manually enter customer features")
        st.write("Enter feature values as JSON")
        cust_text = st.text_area("Customer feature dict", height=200, value='{"age_years": 45, "Income": 42000}')
        
        if st.button("Predict for manual customer"):
            try:
                cust = json.loads(cust_text)
                result = asyncio.run(predict_endpoint(cust))
                st.success(f"Prediction: {result.prediction} (prob={result.prob:.3f}) using model {result.model}")
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

if __name__ == "__main__":
    # If you are running FastAPI directly (not Streamlit)
    if uvicorn:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        print("⚠️ uvicorn not installed — skipping FastAPI launch.")
