# backend.py
"""
FastAPI backend for ML pipeline using ml_core.MLContext.

Endpoints:
- POST /train : multipart/form-data, file field 'train_file'
    Runs the full pipeline synchronously, saves models & artifacts, returns status log + best model.
- POST /test  : multipart/form-data, file field 'test_file'
    Run predictions using the selected best model (discovered from final_model_selection_summary.json),
    saves test predictions CSV and returns first 10 rows and path.
- POST /predict : application/json
    Body: {"customer": {...}}  OR {"customer_id": "12345"}
    When using customer_id, the frontend should have uploaded the test file earlier and can send the
    selected row instead; we accept either a full dict of features or a single-row dict.
"""

import os
import json
import shutil
import sys
import joblib
import numpy as np
import pandas as pd
import traceback
import time
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware # Correct import
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
from typing import Optional, Dict, Any

# Import your ML core
from insrenew import MLContext

# Configure global logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("backend")

app = FastAPI(title="Insurance Renewal ML Backend")

# Allow Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ctx = None

BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
ARTIFACTS_FILE = MODELS_DIR / "artifacts.json"
FINAL_SUMMARY = RESULTS_DIR / "final_model_selection_summary.json"

for d in (DATA_DIR, MODELS_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

import math

def sanitize_for_json(obj):
    """Recursively convert NaN, Inf, -Inf to None for JSON serialization."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    else:
        return obj

class PredictRequest(BaseModel):
    customer: Optional[Dict[str, Any]] = None
    # If customer_id is provided, frontend is expected to provide the test dataframe separately.
    customer_id: Optional[str] = None
    model_name: Optional[str] = None  # optional override of chosen model


def _write_uploaded_file(upload: UploadFile, dest: Path) -> None:
    with open(dest, "wb") as out:
        shutil.copyfileobj(upload.file, out)


def _read_final_selection():
    """Return parsed final model selection JSON or None."""
    if FINAL_SUMMARY.exists():
        try:
            with open(FINAL_SUMMARY, "r") as fh:
                return json.load(fh)
        except Exception:
            return None
    return None


@app.post("/train")
async def train_endpoint(train_file: UploadFile = File(...)):
    """
    Receive a train CSV file and run the pipeline fully (synchronous).
    Returns a status log (list of steps + messages) and the chosen best model.
    """
    status_log = []
    start_ts = time.time()

    # Save uploaded train file to data/train_uploaded.csv
    train_path = DATA_DIR / "train_uploaded.csv"
    try:
        _write_uploaded_file(train_file, train_path)
        status_log.append({"step": "upload", "msg": f"Saved training file to {train_path}", "ts": time.time()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded training file: {e}")

    # Run pipeline
    try:
        ctx = MLContext(base_dir=str(BASE_DIR))
        ctx.set_df(pd.read_csv(train_path))

        status_log.append({"step": "set_df", "msg": "Loaded training dataframe", "shape": ctx.df.shape, "ts": time.time()})

        # EDA
        try:
            ctx.run_eda()
            status_log.append({"step": "eda", "msg": "EDA completed and plots saved", "ts": time.time()})
        except Exception as e:
            status_log.append({"step": "eda", "msg": f"EDA failed: {e}", "ts": time.time()})

        # Feature engineering
        try:
            ctx.run_fe()
            status_log.append({"step": "fe", "msg": "Feature engineering complete", "ts": time.time()})
        except Exception as e:
            status_log.append({"step": "fe", "msg": f"Feature engineering failed: {e}", "ts": time.time()})

        # Feature config & preprocessor
        try:
            ctx.prepare_feature_config()
            ctx.build_preprocessor()
            status_log.append({"step": "preproc", "msg": "Preprocessor built", "ts": time.time()})
        except Exception as e:
            status_log.append({"step": "preproc", "msg": f"Preprocessor build failed: {e}", "ts": time.time()})

        # Train/test split
        try:
            ctx.split_train_test(test_size=0.2)
            status_log.append({"step": "split", "msg": "Train/test split completed", "X_train": str(ctx.X_train.shape), "ts": time.time()})
        except Exception as e:
            status_log.append({"step": "split", "msg": f"Train/test split failed: {e}", "ts": time.time()})
            raise

        # Train core models (LR, XGB, NN)
        trained = {}
        try:
            # train LR
            try:
                lr_model = ctx.train_lr()
                trained['lr'] = True
                status_log.append({"step": "train_lr", "msg": "Logistic Regression trained", "ts": time.time()})
            except Exception as e:
                status_log.append({"step": "train_lr", "msg": f"LR train failed: {e}", "ts": time.time()})

            # train XGB / HGB fallback
            try:
                xgb_model = ctx.train_xgboost()
                trained['xgb'] = True
                status_log.append({"step": "train_xgb", "msg": "XGBoost/HGB trained", "ts": time.time()})
            except Exception as e:
                status_log.append({"step": "train_xgb", "msg": f"XGB train failed: {e}", "ts": time.time()})

            # train NN
            try:
                nn_model = ctx.train_neural_net(epochs=5)
                trained['nn'] = True
                status_log.append({"step": "train_nn", "msg": "Neural Net trained", "ts": time.time()})
            except Exception as e:
                status_log.append({"step": "train_nn", "msg": f"NN train failed: {e}", "ts": time.time()})



        except Exception as e:
            status_log.append({"step": "train_all", "msg": f"Model training failed: {e}", "ts": time.time()})
            raise

        # Optional: train extra models if available (balanced rf, easy ensemble, tabnet)
        try:
            try:
                ctx.train_balanced_random_forest()
                status_log.append({"step": "train_brf", "msg": "Balanced RF trained", "ts": time.time()})
            except Exception as e:
                status_log.append({"step": "train_brf", "msg": f"BRF train failed or skipped: {e}", "ts": time.time()})

            try:
                ctx.train_easy_ensemble()
                status_log.append({"step": "train_eec", "msg": "Easy Ensemble trained", "ts": time.time()})
            except Exception as e:
                status_log.append({"step": "train_eec", "msg": f"EEC train failed or skipped: {e}", "ts": time.time()})

            # train LGBM
            try:
                lbm_model = ctx.train_lightgbm()
                trained['lbm'] = True
                status_log.append({"step": "train_lbm", "msg": "Light GBM trained", "ts": time.time()})
            except Exception as e:
                status_log.append({"step": "train_lbm", "msg": f"Light GBM failed: {e}", "ts": time.time()})

            try:
                ctx.train_tabnet(max_epochs=10)
                status_log.append({"step": "train_tabnet", "msg": "TabNet trained (or fallback)", "ts": time.time()})
            except Exception as e:
                status_log.append({"step": "train_tabnet", "msg": f"TabNet train failed or skipped: {e}", "ts": time.time()})
        except Exception:
            # swallow, continue
            pass

        # Prepare artifacts & save models
        try:
            ctx.prepare_artifacts()
            saved = ctx.save_models(prefix="trained")
            # Also save a small artifacts.json mapping used by frontend/backends
            artifacts = {
                'artifacts': {k: str(v) for k, v in saved.items()},
                'feature_names': ctx.artifacts.get('feature_names', []),
            }
            with open(ARTIFACTS_FILE, "w") as fh:
                json.dump(artifacts, fh, indent=2, default=str)
            status_log.append({"step": "save_models", "msg": f"Saved model artifacts: {list(saved.keys())}", "ts": time.time()})
        except Exception as e:
            status_log.append({"step": "save_models", "msg": f"Save models failed: {e}", "ts": time.time()})

        # Aggregate metrics and final selection (notebook-style)
        try:
            ctx.aggregate_model_comparison_metrics()
            summary_path = ctx.prepare_final_model_selection_summary()  # writes final_model_selection_summary.json
            status_log.append({"step": "aggregate_metrics", "msg": "Aggregated model metrics and created final selection summary", "path": summary_path, "ts": time.time()})
        except Exception as e:
            status_log.append({"step": "aggregate_metrics", "msg": f"Aggregation/selection failed: {e}", "ts": time.time()})

        # Save shap values (guarded)
        try:
            ctx.save_shap_values_csv(model_name='xgb')
            status_log.append({"step": "shap", "msg": "Saved SHAP values (if possible)", "ts": time.time()})
        except Exception as e:
            status_log.append({"step": "shap", "msg": f"SHAP save failed: {e}", "ts": time.time()})

        # Finalize
        elapsed = time.time() - start_ts
        status_log.append({"step": "done", "msg": "Training pipeline finished", "elapsed_sec": elapsed, "ts": time.time()})

        # Read final selection to return chosen model
        final = _read_final_selection()
        chosen = None
        if final:
            try:
                chosen = final.get("recommendation", {}).get("recommended_model") or final.get("selected_model")
            except Exception:
                chosen = None

        return JSONResponse({"status_log": status_log, "trained_models": trained, "chosen_model": chosen})
    except Exception as e:
        tb = traceback.format_exc()
        status_log.append({"step": "error", "msg": str(e), "traceback": tb})
        return JSONResponse({"status_log": status_log}, status_code=500)

@app.post("/test")
async def test_endpoint(file: UploadFile = File(...)):
    """Upload test CSV and run predictions using the best model."""
    try:
        file_path = DATA_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Uploaded test file saved to: {file_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save test file: {e}")

    try:
        ctx = MLContext(base_dir=str(BASE_DIR))
        test_results = ctx.predict_external_test(str(file_path))

        # Save the test dataset in context for later single predictions
        ctx.test_df = pd.read_csv(file_path)
        message = test_results.get("message", "")
        metrics = test_results.get("metrics", {})
        preview = test_results.get("preview", [])
        output_path = test_results.get("output_path", "")

        response = {
            "message": message,
            "metrics": metrics,
            "preview": preview,
            "predictions_path": str(output_path)
        }
        return JSONResponse(sanitize_for_json(response))
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Test prediction failed: {e}\n{tb}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")



@app.post("/predict")
async def predict_endpoint(request: Request):
    """
    Predict renewal for a single record.
    Uses the tested dataset’s schema (from /test) and best trained model.
    """
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("Input must be a JSON object with customer features.")
        logger.info(f"Received predict payload: {list(payload.keys())}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input JSON: {e}")

    try:
        ctx = MLContext(base_dir=str(BASE_DIR))

        # Align to training schema if feature_names.npy exists
        feature_path = ctx.results_dir / "feature_names.npy"
        if feature_path.exists():
            expected_features = list(np.load(feature_path, allow_pickle=True))
            logger.info(f"Aligning input to training schema: {len(expected_features)} features")
            for col in expected_features:
                if col not in payload:
                    payload[col] = 0
            # remove extra keys not seen during training
            payload = {k: payload[k] for k in expected_features if k in payload}
        else:
            logger.warning("No feature schema found; using raw payload as-is.")

        # Call predict_single() from ml_core.py
        result = ctx.predict_single(customer_dict=payload)
        response = sanitize_for_json(result)
        return JSONResponse(response)

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Prediction failed: {e}\n{tb}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")