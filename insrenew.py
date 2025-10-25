# ml_core.py
"""
ml_core.py
Modular ML core for Insurance Renewal Prediction (core models only):
 - Logistic Regression
 - XGBoost (with sklearn HGB fallback)
 - Neural Net (TensorFlow if available, else sklearn MLP)

Design:
 - MLContext holds shared state (df, paths, preprocessor, trained models, etc.)
 - Methods are instance methods on MLContext for clarity and easy backend usage.
 - EDA plots are saved to disk (plots/ under base_dir).
 - Models/artifacts are saved using joblib.
"""

import os
import logging
from pathlib import Path
import sys
from typing import Optional, Dict, Any, List, Tuple
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score

# Wrap with a simple lambda for predict/predict_proba compatibility
class TFWrapper:
    def __init__(self, model):
        self.model = model
    def predict(self, X):
        p = self.model.predict(X)
        return (p.ravel() >= 0.5).astype(int)
    def predict_proba(self, X):
        p = self.model.predict(X)
        return np.vstack([1-p.ravel(), p.ravel()]).T

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)

# Try optional heavy libraries
try:
    import xgboost as xgb  # type: ignore
    XGBOOST_AVAILABLE = True
    #self.logger.info("xgboost available.")
except Exception:
    XGBOOST_AVAILABLE = False
    #self.logger.info("xgboost not available; will fallback to sklearn HGB where needed.")

try:
    import tensorflow as tf  # type: ignore
    from tensorflow import keras  # type: ignore
    TENSORFLOW_AVAILABLE = True
    #self.logger,info("TensorFlow available.")
except Exception:
    TENSORFLOW_AVAILABLE = False
    #self.logger,info("TensorFlow not available; will fallback to sklearn MLP for neural net.")


class MLContext:
    """
    Shared context for ML pipeline. Instantiate once per logical pipeline run (per request/session).
    """

    def __init__(self, base_dir: Optional[str] = "."):
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "data"
        self.model_dir = self.base_dir / "models"
        self.plot_dir = self.base_dir / "plots"
        self.results_dir = self.base_dir / "results"

        # ensure directories
        for d in [self.data_dir, self.model_dir, self.plot_dir, self.results_dir]:
            os.makedirs(d, exist_ok=True)


        # Data / split
        self.df: Optional[pd.DataFrame] = None
        self.df_train: Optional[pd.DataFrame] = None
        self.df_test: Optional[pd.DataFrame] = None

        # Train/test split outputs
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None

        # Preprocessor & feature lists
        self.preprocessor: Optional[ColumnTransformer] = None
        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = []
        self.feature_config: Dict[str, Any] = {}

        # Trained models
        self.lr_model = None
        self.xgb_model = None
        self.nn_model = None

        # artifacts & metadata
        self.artifacts: Dict[str, Any] = {}
        self.random_state = 42
        self.target_col = "renewal"
        def setup_logger(name="insrenew"):
            logger = logging.getLogger(name)
            if not logger.handlers:  #  prevents duplicate handlers
                logger.setLevel(logging.INFO)
                handler = logging.StreamHandler(sys.stdout)
                formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
                handler.setFormatter(formatter)
                logger.addHandler(handler)
            logger.propagate = False  #  avoid double propagation to root
            return logger
        
        self.logger = setup_logger("insrenew")

    # ----------------------------
    # Data loading / setters
    # ----------------------------
    def load_data_from_csv(self, path: str) -> pd.DataFrame:
        path = Path(path)
        self.logger.info(f"Loading CSV from {path}")
        df = pd.read_csv(path)
        self.set_df(df)
        return df

    def set_df(self, df: pd.DataFrame):
        """Set the main dataframe for this context (raw)."""
        self.df = df.copy()
        self.logger.info(f"Dataframe set with shape {self.df.shape}")

    # ----------------------------
    # EDA (saves plots)
    # ----------------------------
    def run_eda(self, sample_n: Optional[int] = 5000):
        """Run lightweight EDA and save essential plots to plot_dir."""
        if self.df is None:
            raise ValueError("Dataframe (self.df) is not set.")

        df = self.df.copy()
        self.logger.info("Starting EDA...")

        # Head & missing counts saved as JSON
        head_path = self.results_dir / "df_head.json"
        df.head(10).to_json(head_path, orient="records")
        self.logger.info(f"Saved df head to {head_path}")

        # Missing values heatmap
        try:
            plt.figure(figsize=(10, 6))
            sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
            plt.title("Missing Value Heatmap")
            p = self.plot_dir / "missing_values_heatmap.png"
            plt.tight_layout()
            plt.savefig(p, dpi=200)
            plt.close()
            self.logger.info(f"Saved missing values heatmap: {p}")
        except Exception as e:
            self.logger.warning("Failed to create missing values heatmap: %s", e)

        # Target distribution
        if self.target_col in df.columns:
            try:
                plt.figure(figsize=(6, 4))
                sns.countplot(x=self.target_col, data=df)
                plt.title(f"Target Distribution ({self.target_col})")
                p = self.plot_dir / "target_distribution.png"
                plt.tight_layout()
                plt.savefig(p, dpi=200)
                plt.close()
                self.logger.info(f"Saved target distribution plot: {p}")

                # save normalized counts
                vc = df[self.target_col].value_counts(normalize=True).round(4).to_dict()
                with open(self.results_dir / "target_distribution.json", "w") as fh:
                    json.dump(vc, fh, indent=2)
                self.logger.info("Saved target distribution JSON.")
            except Exception as e:
                self.logger.warning("Target distribution plot failed: %s", e)

        # Numeric distributions for top numeric cols
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            candidate = numeric_cols[:min(6, len(numeric_cols))]
            try:
                df[candidate].hist(bins=20, figsize=(12, 6))
                plt.suptitle("Distribution of key numeric features")
                p = self.plot_dir / "numeric_distributions.png"
                plt.tight_layout()
                plt.savefig(p, dpi=200)
                plt.close()
                self.logger.info(f"Saved numeric distributions: {p}")
            except Exception as e:
                self.logger.warning("Numeric distributions plotting failed: %s", e)

        # Correlation heatmap (numeric)
        try:
            corr_cols = numeric_cols[:20]
            if len(corr_cols) > 1:
                plt.figure(figsize=(10, 8))
                sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
                plt.title("Correlation Heatmap (selected numeric features)")
                p = self.plot_dir / "correlation_heatmap.png"
                plt.tight_layout()
                plt.savefig(p, dpi=200)
                plt.close()
                self.logger.info(f"Saved correlation heatmap: {p}")
        except Exception as e:
            self.logger.warning("Correlation heatmap failed: %s", e)

        self.logger.info("EDA complete.")

    # ----------------------------
    # Feature engineering (canonical)
    # ----------------------------
    def run_fe(self):
        """Canonical cleaning and feature engineering. Updates self.df in-place."""
        if self.df is None:
            raise ValueError("Dataframe (self.df) is not set.")
        self.logger.info("Starting feature engineering...")

        df = self.df.copy()

        # Example canonical steps — adapt to your dataset
        # 1) late payment counts to numeric
        late_cols = ['Count_3-6_months_late', 'Count_6-12_months_late', 'Count_more_than_12_months_late']
        for c in late_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)

        # 2) Drop rows with missing underwriting score (if present)
        if 'application_underwriting_score' in df.columns:
            df = df.dropna(subset=['application_underwriting_score']).reset_index(drop=True)

        # 3) Derived features
        if 'age_in_days' in df.columns and 'age_years' not in df.columns:
            df['age_years'] = df['age_in_days'] / 365.25
        if 'Income' in df.columns and 'income_log' not in df.columns:
            df['income_log'] = np.log1p(df['Income'])
        if all(c in df.columns for c in late_cols):
            df['total_late_counts'] = df[late_cols].sum(axis=1)
        if 'no_of_premiums_paid' in df.columns:
            df['no_of_premiums_paid'] = pd.to_numeric(df['no_of_premiums_paid'], errors='coerce').fillna(0)
            df['late_rate'] = (df['total_late_counts'] / df['no_of_premiums_paid'].replace(0, np.nan)).fillna(0)
        if 'premium' in df.columns and 'Income' in df.columns:
            df['premium_to_income'] = df['premium'] / (df['Income'] + 1)

        if 'application_underwriting_score' in df.columns:
            thr = df['application_underwriting_score'].quantile(0.99)
            df['high_underwriting_flag'] = (df['application_underwriting_score'] >= thr).astype(int)

        # Enhanced interactions
        if 'age_years' in df.columns and 'income_log' in df.columns:
            df['age_income_interaction'] = df['age_years'] * df['income_log']
        if 'total_late_counts' in df.columns and 'premium' in df.columns:
            df['late_premium_interaction'] = df['total_late_counts'] * df['premium']
        if 'Income' in df.columns and 'income_bucket' not in df.columns:
            try:
                df['income_bucket'] = pd.qcut(df['Income'].fillna(0) + 1, q=5, labels=False, duplicates='drop')
            except Exception:
                pass

        self.logger.info(f"In run_fe DF columns: {df.columns.tolist()}")
        # persist engineered df
        self.df = df
        self.logger.info("Feature engineering complete. New shape: %s", df.shape)

    # ----------------------------
    # Feature config detection
    # ----------------------------
    def prepare_feature_config(self):
        """Detect numeric and categorical features and save dtypes."""
        if self.df is None:
            raise ValueError("Dataframe (self.df) is not set.")

        dtypes = {col: str(self.df[col].dtype) for col in self.df.columns}
        num_feats = self.df.select_dtypes(include=[np.number]).columns.tolist()
        cat_feats = self.df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Remove target if present
        if self.target_col in num_feats:
            num_feats = [c for c in num_feats if c != self.target_col]
        if self.target_col in cat_feats:
            cat_feats = [c for c in cat_feats if c != self.target_col]

        self.numeric_features = num_feats
        self.categorical_features = cat_feats
        self.feature_config = {
            'dtypes': dtypes,
            'numeric_features': num_feats,
            'categorical_features': cat_feats
        }

        # Save feature config
        with open(self.results_dir / "feature_config.json", "w") as fh:
            json.dump(self.feature_config, fh, indent=2)
        self.logger.info("Prepared feature_config with %d numeric and %d categorical features.", len(num_feats), len(cat_feats))
        return self.feature_config

    # ----------------------------
    # Preprocessor construction
    # ----------------------------
    def build_preprocessor(self, scaler: bool = True, onehot_sparse: bool = False):
        """Build ColumnTransformer preprocessor based on detected features."""
        if not self.numeric_features and not self.categorical_features:
            self.prepare_feature_config()

        num_feats = self.numeric_features
        cat_feats = self.categorical_features

        num_transformer = Pipeline([('scaler', StandardScaler())]) if scaler and num_feats else 'passthrough'
        # OneHotEncoder config: use sparse_output param depending on sklearn version
        try:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=not onehot_sparse)
        except TypeError:
            # older sklearn
            ohe = OneHotEncoder(handle_unknown='ignore', sparse=not onehot_sparse)

        cat_transformer = Pipeline([('onehot', ohe)]) if cat_feats else 'passthrough'

        transformers = []
        if num_feats:
            transformers.append(('num', num_transformer, num_feats))
        if cat_feats:
            transformers.append(('cat', cat_transformer, cat_feats))

        self.preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
        self.logger.info("Built preprocessor. Num feats: %d, Cat feats: %d", len(num_feats), len(cat_feats))
        return self.preprocessor

    # ----------------------------
    # Split train/test
    # ----------------------------
    def split_train_test(self, test_size: float = 0.2):
        """Split self.df into train/test with stratify on target if present."""
        if self.df is None:
            raise ValueError("Dataframe (self.df) is not set.")

        if self.target_col not in self.df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in dataframe.")

        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.random_state
        )
        self.logger.info("Train/test split complete: X_train=%s, X_test=%s", self.X_train.shape, self.X_test.shape)
        self.feature_names_ = list(self.X_train.columns)
        (np.save(self.results_dir / "feature_names.npy", np.array(self.feature_names_)))

        return (self.X_train, self.X_test, self.y_train, self.y_test)

    # ----------------------------
    # Model training: Logistic Regression
    # ----------------------------
    def train_lr(self):
        """Train a Logistic Regression pipeline."""
        if self.X_train is None or self.y_train is None:
            raise ValueError("Train/test split not done. Call split_train_test() first.")

        self.logger.info("Training Logistic Regression...")
        if self.preprocessor is None:
            self.build_preprocessor()

        pipe_lr = Pipeline([('preproc', self.preprocessor), ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=self.random_state))])
        pipe_lr.fit(self.X_train, self.y_train)
        self.lr_model = pipe_lr
        self.logger.info("Logistic Regression trained.")
        # Save predictions
        try:
            probs = pipe_lr.predict_proba(self.X_test)[:, 1]
            preds = pipe_lr.predict(self.X_test)
        except Exception:
            preds = pipe_lr.predict(self.X_test)
            probs = np.array(preds, dtype=float)
        self._save_predictions("lr", self.X_test, self.y_test, preds, probs)
        return pipe_lr

    # ----------------------------
    # Model training: XGBoost (or HGB fallback)
    # ----------------------------
    def train_xgboost(self):
        """Train XGBoost if available; otherwise fallback to HistGradientBoostingClassifier."""
        if self.X_train is None or self.y_train is None:
            raise ValueError("Train/test split not done. Call split_train_test() first.")

        if self.preprocessor is None:
            self.build_preprocessor()

        if XGBOOST_AVAILABLE:
            self.logger.info("Training XGBoost (skipping heavy hyperparam search for speed).")
            xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=-1, random_state=self.random_state)
            pipe_xgb = Pipeline([('preproc', self.preprocessor), ('clf', xgb_clf)])
            # lightweight fit
            pipe_xgb.fit(self.X_train, self.y_train)
            self.xgb_model = pipe_xgb
            self.logger.info("XGBoost trained.")
        else:
            self.logger.info("XGBoost not available — using HistGradientBoostingClassifier fallback.")
            hgb = HistGradientBoostingClassifier(random_state=self.random_state)
            pipe_hgb = Pipeline([('preproc', self.preprocessor), ('clf', hgb)])
            pipe_hgb.fit(self.X_train, self.y_train)
            self.xgb_model = pipe_hgb
            self.logger.info("HGB fallback trained.")

        # Save predictions
        try:
            preds = self.xgb_model.predict(self.X_test)
            #probs = self.xgb_model.predict_proba(self.X_test)[:, 1]
            probs = pipe_xgb.predict_proba(self.X_test)[:, 1] if hasattr(pipe_xgb, "predict_proba") else None

        except Exception:
            preds = self.xgb_model.predict(self.X_test)
            probs = np.array(preds, dtype=float)
        self._save_predictions("xgb", self.X_test, self.y_test, preds, probs)
        return self.xgb_model

    # ----------------------------
    # Model training: Neural Net
    # ----------------------------
    def train_neural_net(self, epochs: int = 10):
        """Train a neural network. Use TensorFlow if available, else sklearn MLPClassifier."""
        if self.X_train is None or self.y_train is None:
            raise ValueError("Train/test split not done. Call split_train_test() first.")
        if self.preprocessor is None:
            self.build_preprocessor()

        self.logger.info("Training Neural Network...")
        # Fit preprocessor and transform to numpy arrays
        try:
            self.preprocessor.fit(self.X_train)
            X_tr = self.preprocessor.transform(self.X_train)
            X_te = self.preprocessor.transform(self.X_test)
        except Exception:
            # Fallback: use raw fillna
            X_tr = self.X_train.fillna(0).values
            X_te = self.X_test.fillna(0).values

        self.logger.info(f"In train_neural_network X_tr columns: {X_tr.shape()}")   
        self.logger.info(f"In train_neural_network X_te columns: {X_te.shape()}")   

        if TENSORFLOW_AVAILABLE:
            try:
                self.logger.info("Using TensorFlow Keras model.")
                input_dim = X_tr.shape[1]
                nn = keras.Sequential([
                    keras.layers.Input(shape=(input_dim,)),
                    keras.layers.Dense(64, activation='relu'),
                    keras.layers.Dropout(0.2),
                    keras.layers.Dense(32, activation='relu'),
                    keras.layers.Dense(1, activation='sigmoid')
                ])
                nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                nn.fit(X_tr, self.y_train, validation_split=0.1, epochs=epochs, batch_size=64, verbose=0)
                self.nn_model = TFWrapper(nn)
                self.logger.info("Neural Net (TF) trained.")
            except Exception as e:
                self.logger.warning("TensorFlow training failed: %s. Falling back to sklearn MLP.", e)
                TENSORFLOW_MODEL_FAILED = True
            else:
                TENSORFLOW_MODEL_FAILED = False
        else:
            TENSORFLOW_MODEL_FAILED = True

        if TENSORFLOW_MODEL_FAILED:
            from sklearn.neural_network import MLPClassifier
            mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=self.random_state)
            mlp.fit(X_tr, self.y_train)
            # Create pipeline: note preprocessor already included separately in context
            # We'll store a wrapper pipeline for consistency
            pipe_mlp = Pipeline([('preproc', self.preprocessor), ('clf', mlp)])
            self.nn_model = pipe_mlp
            self.logger.info("Neural Net (sklearn MLP) trained.")

        # Save predictions
        try:
            preds = self.nn_model.predict(self.X_test if hasattr(self.nn_model, 'predict') else X_te)
            probs = self.nn_model.predict_proba(self.X_test if hasattr(self.nn_model, 'predict_proba') else X_te)[:, 1]
        except Exception:
            # If TFWrapper used earlier, it expects transformed arrays
            try:
                preds = self.nn_model.predict(X_te)
                probs = self.nn_model.predict_proba(X_te)[:, 1]
            except Exception:
                preds = np.zeros(len(self.X_test), dtype=int)
                probs = np.zeros(len(self.X_test), dtype=float)
        self._save_predictions("nn", self.X_test, self.y_test, preds, probs)
        return self.nn_model
    
        # ----------------------------
    # Imbalanced ensembles: BalancedRandomForest & EasyEnsemble
    # ----------------------------
    def train_balanced_random_forest(self, n_estimators: int = 100):
        """Train BalancedRandomForestClassifier (imblearn) or RandomForest fallback."""
        if self.X_train is None or self.y_train is None:
            raise ValueError("Run split_train_test() first.")

        self.logger.info("Training Balanced Random Forest...")
        try:
            from imblearn.ensemble import BalancedRandomForestClassifier  # type: ignore
            brf = BalancedRandomForestClassifier(n_estimators=n_estimators, random_state=self.random_state)
            brf.fit(self.preprocessor.transform(self.X_train) if self.preprocessor is not None else self.X_train.fillna(0), self.y_train)
            self.brf_model = brf
            self.logger.info("BalancedRandomForest trained via imblearn.")
        except Exception as e:
            self.logger.warning("imblearn not available or BRF failed: %s — falling back to sklearn RandomForest", e)
            from sklearn.ensemble import RandomForestClassifier
            rf = RandomForestClassifier(n_estimators=max(100, n_estimators), class_weight='balanced', random_state=self.random_state, n_jobs=-1)
            rf.fit(self.preprocessor.transform(self.X_train) if self.preprocessor is not None else self.X_train.fillna(0), self.y_train)
            self.brf_model = rf
            self.logger.info("Fallback RandomForest trained.")

        # Save preds
        try:
            Xtest_proc = self.preprocessor.transform(self.X_test) if self.preprocessor is not None else self.X_test.fillna(0)
            #preds = self.brf_model.predict(Xtest_proc)
            #probs = self.brf_model.predict_proba(Xtest_proc)[:, 1] if hasattr(self.brf_model, 'predict_proba') else np.array(preds, dtype=float)
            probs = self.brf_model.predict_proba(Xtest_proc)[:, 1] if hasattr(self.brf_model, "predict_proba") else self.brf_model.decision_function(Xtest_proc)
            preds = (probs >= 0.5).astype(int)
        except Exception:
            preds = self.brf_model.predict(self.X_test)
            probs = np.array(preds, dtype=float)
        self._save_predictions("brf", self.X_test, self.y_test, preds, probs)
        return self.brf_model

    def train_easy_ensemble(self, n_estimators: int = 10):
        """Train EasyEnsemble (imblearn) or bagging fallback."""
        if self.X_train is None or self.y_train is None:
            raise ValueError("Run split_train_test() first.")

        self.logger.info("Training Easy Ensemble / bagging fallback...")
        try:
            from imblearn.ensemble import EasyEnsembleClassifier  # type: ignore
            eec = EasyEnsembleClassifier(n_estimators=n_estimators, random_state=self.random_state)
            eec.fit(self.preprocessor.transform(self.X_train) if self.preprocessor is not None else self.X_train.fillna(0), self.y_train)
            self.eec_model = eec
            self.logger.info("EasyEnsemble trained (imblearn).")
        except Exception as e:
            self.logger.warning("imblearn EasyEnsemble not available: %s — using Bagging + RF fallback", e)
            from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
            try:
                bag = BaggingClassifier(estimator=RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=self.random_state), n_estimators=5, random_state=self.random_state, n_jobs=-1)
            except TypeError:
                # older sklearn
                bag = BaggingClassifier(base_estimator=RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=self.random_state), n_estimators=5, random_state=self.random_state, n_jobs=-1)
            bag.fit(self.preprocessor.transform(self.X_train) if self.preprocessor is not None else self.X_train.fillna(0), self.y_train)
            self.eec_model = bag
            self.logger.info("Bagging fallback trained for EasyEnsemble.")

        # Save preds
        try:
            Xtest_proc = self.preprocessor.transform(self.X_test) if self.preprocessor is not None else self.X_test.fillna(0)
            #preds = self.eec_model.predict(Xtest_proc)
            #probs = self.eec_model.predict_proba(Xtest_proc)[:, 1] if hasattr(self.eec_model, 'predict_proba') else np.array(preds, dtype=float)
            probs = self.eec_model.predict_proba(Xtest_proc)[:, 1] if hasattr(self.eec_model, "predict_proba") else self.eec_model.decision_function(Xtest_proc)
            preds = (probs >= 0.5).astype(int)
        except Exception:
            preds = self.eec_model.predict(self.X_test)
            probs = np.array(preds, dtype=float)
        self._save_predictions("eec", self.X_test, self.y_test, preds, probs)
        return self.eec_model

    # ----------------------------
    # LightGBM trainer (fast fallback)
    # ----------------------------
    def train_lightgbm(self):
        """Train LightGBM if available; else raise."""
        if self.X_train is None or self.y_train is None:
            raise ValueError("Run split_train_test() first.")
        self.logger.info("Training LightGBM (guarded)...")
        try:
            import lightgbm as lgb  # type: ignore
            lgb_clf = lgb.LGBMClassifier(random_state=self.random_state, n_jobs=-1)
            pipe_lgb = Pipeline([('preproc', self.preprocessor), ('clf', lgb_clf)]) if self.preprocessor is not None else Pipeline([('clf', lgb_clf)])
            pipe_lgb.fit(self.X_train, self.y_train)
            self.lgb_model = pipe_lgb
            self.logger.info("LightGBM trained successfully.")
            preds = pipe_lgb.predict(self.X_test)
            probs = pipe_lgb.predict_proba(self.X_test)[:, 1]
            self._save_predictions("lgb", self.X_test, self.y_test, preds, probs)
            return pipe_lgb
        except Exception as e:
            self.logger.warning("LightGBM training failed / not available: %s", e)
            raise

    # ----------------------------
    # TabNet training (guarded) with LGB/HGB fallback
    # ----------------------------
    def train_tabnet(self, max_epochs: int = 50):
        """Train TabNet (pytorch-tabnet) if available; fallback to LightGBM or HGB."""
        if self.X_train is None or self.y_train is None:
            raise ValueError("Run split_train_test() first.")
        if self.preprocessor is None:
            self.build_preprocessor()

        self.logger.info("Attempting TabNet training (guarded).")
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier  # type: ignore
            # prepare numeric arrays
            self.preprocessor.fit(self.X_train)
            Xtr = self.preprocessor.transform(self.X_train)
            Xte = self.preprocessor.transform(self.X_test)
            Xtr = np.asarray(Xtr)
            Xte = np.asarray(Xte)
            tabnet = TabNetClassifier(n_d=8, n_a=8, n_steps=3, verbose=0, seed=self.random_state)
            tabnet.fit(Xtr, self.y_train.values, eval_set=[(Xte, self.y_test.values)], max_epochs=max_epochs, patience=10, batch_size=1024, virtual_batch_size=64)
            self.tabnet_model = tabnet
            self.logger.info("TabNet trained successfully.")
            preds = tabnet.predict(Xte)
            probs = tabnet.predict_proba(Xte)[:, 1]
            # TabNet preds: note DataFrame Id mapping may differ
            self._save_predictions("tabnet", self.X_test, self.y_test, preds, probs)
            return tabnet
        except Exception as e:
            self.logger.warning("TabNet not available or failed: %s — attempting LightGBM fallback.", e)
            try:
                return self.train_lightgbm()
            except Exception:
                self.logger.info("LightGBM fallback failed; trying HGB fallback.")
                from sklearn.ensemble import HistGradientBoostingClassifier
                hgb = HistGradientBoostingClassifier(random_state=self.random_state)
                pipe_hgb = Pipeline([('preproc', self.preprocessor), ('clf', hgb)])
                pipe_hgb.fit(self.X_train, self.y_train)
                self.tabnet_model = pipe_hgb
                preds = pipe_hgb.predict(self.X_test)
                try:
                    probs = pipe_hgb.predict_proba(self.X_test)[:, 1]
                except Exception:
                    probs = np.array(preds, dtype=float)
                self._save_predictions("tabnet_hgb", self.X_test, self.y_test, preds, probs)
                return pipe_hgb

    # ----------------------------
    # ADASYN resampling experiment
    # ----------------------------
    def adasyn_experiment(self, sample_limit: Optional[int] = 20000):
        """Run ADASYN oversampling experiment and train a RandomForest on resampled data."""
        self.logger.info("Starting ADASYN experiment (guarded).")
        try:
            from imblearn.over_sampling import ADASYN  # type: ignore
            from sklearn.ensemble import RandomForestClassifier
        except Exception as e:
            self.logger.warning("imblearn not available; skipping ADASYN experiment: %s", e)
            return None

        # Prepare dataset
        if self.X_train is None or self.y_train is None:
            self.logger.warning("No train/test split; using df if available.")
            if self.df is None or self.target_col not in self.df.columns:
                raise ValueError("No data to run ADASYN.")
            Xs = self.df.drop(columns=[self.target_col]).fillna(0)
            ys = self.df[self.target_col]
        else:
            Xs = self.X_train.copy()
            ys = self.y_train.copy()

        # Ensure numeric encoding
        try:
            if self.preprocessor is not None:
                X_proc = self.preprocessor.fit_transform(Xs)
            else:
                X_proc = pd.get_dummies(Xs, drop_first=True).fillna(0)
        except Exception as e:
            self.logger.warning("Encoding failed, fallback to numeric-only: %s", e)
            X_proc = Xs.select_dtypes(include=[np.number]).fillna(0)

        if hasattr(X_proc, 'shape') and X_proc.shape[0] > sample_limit:
            idx = np.random.choice(X_proc.shape[0], sample_limit, replace=False)
            X_proc = X_proc[idx]
            ys = ys.iloc[idx]

        ad = ADASYN(random_state=self.random_state)
        X_res, y_res = ad.fit_resample(X_proc, ys)
        self.logger.info("ADASYN resampled shapes: %s, %s", getattr(X_res, 'shape', None), getattr(y_res, 'shape', None))

        clf = RandomForestClassifier(n_estimators=200, random_state=self.random_state, n_jobs=-1)
        clf.fit(X_res, y_res)
        self.adasyn_rf = clf

        # Evaluate on X_test if available
        if self.X_test is not None and self.y_test is not None:
            try:
                Xte_proc = self.preprocessor.transform(self.X_test) if self.preprocessor is not None else self.X_test.fillna(0)
                probs = clf.predict_proba(Xte_proc)[:, 1]
                preds = clf.predict(Xte_proc)
                self._save_predictions("adasyn_rf", self.X_test, self.y_test, preds, probs)
                self.logger.info("ADASYN experiment evaluation saved.")
            except Exception as e:
                self.logger.warning("Failed ADASYN evaluation: %s", e)
        return clf

    # ----------------------------
    # Optuna hyperparameter tuning (guarded)
    # ----------------------------
    def tune_with_optuna(self, estimator: str = "rf", n_trials: int = 10):
        """
        Small example using optuna to tune hyperparams for RandomForest or XGBoost.
        estimator: 'rf' or 'xgb'
        """
        self.logger.info("Starting Optuna tuning for %s (guarded)", estimator)
        try:
            import optuna  # type: ignore
            from sklearn.model_selection import cross_val_score, StratifiedKFold
            from sklearn.ensemble import RandomForestClassifier
        except Exception as e:
            self.logger.warning("Optuna or required libs not available: %s", e)
            return None

        # Prepare lightweight dataset
        if self.X_train is None or self.y_train is None:
            if self.df is None or self.target_col not in self.df.columns:
                raise ValueError("No data for tuning")
            X_raw = self.df.drop(columns=[self.target_col])
            y_t = self.df[self.target_col]
        else:
            X_raw = self.X_train
            y_t = self.y_train.copy()
        
        # If a preprocessor exists, use it to encode X
        if self.preprocessor is not None:
            try:
                X_t = self.preprocessor.fit_transform(X_raw)
            except Exception as e:
                self.logger.warning("Preprocessor transform failed: %s", e)
                X_t = pd.get_dummies(X_raw, drop_first=True).fillna(0)
        else:
            X_t = pd.get_dummies(X_raw, drop_first=True).fillna(0)

        def rf_objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4)
            }
            clf = RandomForestClassifier(random_state=self.random_state, n_jobs=-1, **params)
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
            return float(cross_val_score(clf, X_t, y_t, cv=cv, scoring='roc_auc', error_score='raise').mean())


        def xgb_objective(trial):
            if not XGBOOST_AVAILABLE:
                raise RuntimeError("XGBoost not available for tuning")
            import xgboost as xgb  # type: ignore
            params = {
                'n_estimators': trial.suggest_categorical('n_estimators', [50, 100, 200]),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2)
            }
            clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=self.random_state, n_jobs=1, **params)
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
            return float(cross_val_score(clf, X_t, y_t, cv=cv, scoring='roc_auc', error_score='raise').mean())

        try:
            study = optuna.create_study(direction='maximize')
            if estimator == 'rf':
                study.optimize(rf_objective, n_trials=n_trials)
            elif estimator == 'xgb':
                if not XGBOOST_AVAILABLE:
                    self.logger.warning("XGBoost not available for tuning.")
                    return None
                study.optimize(xgb_objective, n_trials=n_trials)
            else:
                self.logger.warning("Unsupported estimator for tuning: %s", estimator)
                return None
            outp = self.results_dir / f"optuna_best_{estimator}.json"
            with open(outp, 'w') as fh:
                json.dump({'best_params': study.best_params, 'best_value': study.best_value}, fh, indent=2)
            self.logger.info("Optuna tuning complete. Results saved to %s", outp)
            return study.best_params
        except Exception as e:
            self.logger.warning("Optuna tuning failed: %s", e)
            return None

    # ----------------------------
    # SHAP export for tree models (guarded)
    # ----------------------------
    def export_shap(self, model_name: str = "xgb", sample_size: int = 2000):
        """
        Compute and save SHAP values for tree-based models (XGBoost/LGB/HGB).
        Saves CSV of shap values (sample) and summary PNGs if possible.
        """
        try:
            import shap  # type: ignore
        except Exception as e:
            self.logger.warning("SHAP not available: %s", e)
            return None

        # pick model object
        model_attr = None
        if model_name == "xgb":
            model_attr = getattr(self, 'xgb_model', None)
        elif model_name == "lgb":
            model_attr = getattr(self, 'lgb_model', None)
        elif model_name == "brf":
            model_attr = getattr(self, 'brf_model', None)
        else:
            model_attr = getattr(self, f"{model_name}_model", None)

        if model_attr is None:
            self.logger.warning("No such model available for SHAP: %s", model_name)
            return None

        # Need transformed test set
        if self.X_test is None:
            self.logger.warning("No X_test available for SHAP export.")
            return None

        try:
            # Transform X_test using preprocessor if present
            if self.preprocessor is not None:
                X_shap = self.preprocessor.transform(self.X_test)
            else:
                X_shap = self.X_test.fillna(0).values
            X_shap = np.asarray(X_shap)
            nsave = min(sample_size, X_shap.shape[0])
            idx = np.random.choice(X_shap.shape[0], nsave, replace=False)
            Xs = X_shap[idx]

            # Extract raw model for tree explainer if inside a pipeline
            raw_model = model_attr.named_steps['clf'] if hasattr(model_attr, 'named_steps') and 'clf' in model_attr.named_steps else model_attr
            expl = shap.TreeExplainer(raw_model)
            shap_vals = expl.shap_values(Xs) if hasattr(expl, 'shap_values') else expl(Xs)
            # Save a CSV sample of shap values
            feat_names = getattr(self, 'numeric_features', []) + getattr(self, 'categorical_features', [])
            if not feat_names:
                # fallback to columns from X_test
                feat_names = list(self.X_test.columns)
            shap_df = pd.DataFrame(shap_vals if not isinstance(shap_vals, list) else shap_vals[0], columns=feat_names)
            out_csv = self.results_dir / f"shap_values_{model_name}.csv"
            shap_df.to_csv(out_csv, index=False)
            self.logger.info("Saved SHAP values sample to %s", out_csv)
            # summary plot
            try:
                plt.figure(figsize=(8,6))
                shap.summary_plot(shap_vals, features=Xs, feature_names=feat_names, show=False)
                out_png = self.plot_dir / f"shap_summary_{model_name}.png"
                plt.tight_layout()
                plt.savefig(out_png, dpi=200)
                plt.close()
                self.logger.info("Saved SHAP summary plot to %s", out_png)
            except Exception as eplot:
                self.logger.warning("SHAP plotting failed: %s", eplot)
            return out_csv
        except Exception as e:
            self.logger.warning("SHAP computation failed: %s", e)
            return None

    # ----------------------------
    # Plots recreation: confusion matrices, missing heatmap, numeric dists, ROC/PR, SHAP plots
    # ----------------------------
    def _load_preds_df(self, tag: str):
        """Helper to load predictions CSV created by _save_predictions()."""
        p = self.results_dir / f"{tag}_predictions.csv"
        if not p.exists():
            self.logger.warning("Predictions file not found for tag %s at %s", tag, p)
            return None
        try:
            return pd.read_csv(p)
        except Exception as e:
            self.logger.warning("Failed to read predictions for %s: %s", tag, e)
            return None

    def _plot_confusion_matrix_from_preds(self, tag: str, fname: str):
        """Plot and save confusion matrix PNG for model tag."""
        dfp = self._load_preds_df(tag)
        if dfp is None:
            return None
        try:
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
            y_true = dfp['y_true'].values
            y_pred = dfp['y_pred'].values
            cm = confusion_matrix(y_true, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            fig, ax = plt.subplots(figsize=(6, 5))
            disp.plot(ax=ax, cmap='Blues', values_format='d')
            ax.set_title(f"Confusion Matrix: {tag.upper()}")
            out = self.plot_dir / fname
            plt.tight_layout()
            plt.savefig(out, dpi=200)
            plt.close(fig)
            self.logger.info("Saved confusion matrix for %s -> %s", tag, out)
            return out
        except Exception as e:
            self.logger.warning("Failed to plot confusion matrix for %s: %s", tag, e)
            return None

    def generate_confusion_matrices(self):
        """Create confusion matrices for known tags and save PNGs."""
        tag_map = {
            'lr': 'confusion_matrix_lr.png',
            'xgb': 'confusion_matrix_xgb.png',
            'nn': 'confusion_matrix_nn.png',
            'brf': 'confusion_matrix_brf.png',
            'eec': 'confusion_matrix_eec.png',
            'tabnet': 'confusion_matrix_tab.png'
        }
        results = {}
        for tag, fname in tag_map.items():
            out = self._plot_confusion_matrix_from_preds(tag, fname)
            results[tag] = str(out) if out is not None else None
        return results

    def plot_missing_values_heatmap(self):
        """Recreate missing_values_heatmap.png from self.df."""
        if self.df is None:
            self.logger.warning("No df available for missing values heatmap.")
            return None
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(self.df.isnull(), cbar=False, cmap="viridis", ax=ax)
            ax.set_title("Missing Value Heatmap")
            out = self.plot_dir / "missing_values_heatmap.png"
            plt.tight_layout()
            plt.savefig(out, dpi=200)
            plt.close(fig)
            self.logger.info("Saved missing values heatmap -> %s", out)
            return out
        except Exception as e:
            self.logger.warning("Failed to create missing values heatmap: %s", e)
            return None

    def plot_numeric_distributions(self, top_n: int = 6):
        """Recreate numeric_distributions.png using top_n numeric features."""
        if self.df is None:
            self.logger.warning("No df available for numeric distributions.")
            return None
        try:
            num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if not num_cols:
                self.logger.warning("No numeric columns found for distributions.")
                return None
            cols = num_cols[:min(top_n, len(num_cols))]
            fig = plt.figure(figsize=(12, 6))
            self.df[cols].hist(bins=20, figsize=(12, 6))
            plt.suptitle("Distribution of key numeric features")
            out = self.plot_dir / "numeric_distributions.png"
            plt.tight_layout()
            plt.savefig(out, dpi=200)
            plt.close(fig)
            self.logger.info("Saved numeric distributions -> %s", out)
            return out
        except Exception as e:
            self.logger.warning("Numeric distributions plotting failed: %s", e)
            return None

    def plot_roc_pr_comparison(self):
        """
        Create a combined ROC & Precision-Recall comparison plot for available models.
        Looks for *_predictions.csv files written by _save_predictions().
        """
        model_tags = ['lr', 'xgb', 'nn', 'brf', 'eec', 'tabnet']
        plt.figure(figsize=(12, 6))
        has_any = False

        # ROC subplot
        ax_roc = plt.subplot(1, 2, 1)
        ax_pr = plt.subplot(1, 2, 2)
        from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

        for tag in model_tags:
            dfp = self._load_preds_df(tag)
            if dfp is None:
                continue
            if 'prob' not in dfp.columns:
                continue
            y_true = dfp['y_true'].values
            y_prob = dfp['prob'].values
            # ROC
            try:
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                roc_auc = auc(fpr, tpr)
                ax_roc.plot(fpr, tpr, label=f"{tag.upper()} (AUC={roc_auc:.3f})")
            except Exception as e:
                self.logger.debug("ROC failed for %s: %s", tag, e)
            # PR
            try:
                prec, rec, _ = precision_recall_curve(y_true, y_prob)
                ap = average_precision_score(y_true, y_prob)
                ax_pr.plot(rec, prec, label=f"{tag.upper()} (AP={ap:.3f})")
            except Exception as e:
                self.logger.debug("PR failed for %s: %s", tag, e)
            has_any = True

        if not has_any:
            self.logger.warning("No model predictions with probabilities found for ROC/PR plot.")
            plt.close()
            return None

        ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=0.6)
        ax_roc.set_title("ROC Comparison")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend(loc='lower right', fontsize='small')

        ax_pr.set_title("Precision-Recall Comparison")
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.legend(loc='lower left', fontsize='small')

        out = self.plot_dir / "roc_pr_comparison.png"
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close()
        self.logger.info("Saved ROC/PR comparison -> %s", out)
        return out

    def plot_shap_bar_and_summary(self, model_name: str = "xgb"):
        """
        Attempt to create SHAP bar plot, interaction heatmap and summary plots.
        Requires shap and that SHAP values CSV or model exists.
        Looks for self.results_dir/shap_values_{model_name}.csv or uses model + X_test.
        Saves:
         - shap_bar_plot.png
         - shap_interaction_heatmap.png
         - shap_summary_plot.png
         - shap_summary_plot_defensive.png
        """
        try:
            import shap  # type: ignore
        except Exception as e:
            self.logger.warning("SHAP not available: %s", e)
            return None

        shap_csv = self.results_dir / f"shap_values_{model_name}.csv"
        # If CSV exists, use it; else try computing shap from model & X_test
        shap_values = None
        feature_names = None
        try:
            if shap_csv.exists():
                shap_df = pd.read_csv(shap_csv)
                shap_values = shap_df.values
                feature_names = list(shap_df.columns)
            else:
                model_obj = getattr(self, f"{model_name}_model", None)
                if model_obj is None:
                    self.logger.warning("No model found for shap plotting: %s", model_name)
                    return None
                if self.X_test is None:
                    self.logger.warning("No X_test available to compute SHAP.")
                    return None
                # transform test set
                if self.preprocessor is not None:
                    X_input = self.preprocessor.transform(self.X_test)
                else:
                    X_input = self.X_test.fillna(0).values
                raw_model = model_obj.named_steps['clf'] if hasattr(model_obj, 'named_steps') and 'clf' in model_obj.named_steps else model_obj
                expl = shap.TreeExplainer(raw_model)

                shap_vals = expl.shap_values(X_input)

                self.logger.debug(f"SHAP structure: type={type(shap_vals)}, len={len(shap_vals) if isinstance(shap_vals, list) else 'n/a'}, shape={np.shape(shap_vals)}")

                # Handle different SHAP output shapes
                if isinstance(shap_vals, list):
                    if len(shap_vals) == 0:
                        self.logger.warning("SHAP returned empty list — skipping plot.")
                        return None
                    elif len(shap_vals) == 1:
                        shap_values = shap_vals[0]
                    else:
                        # For multi-class, pick the largest class-wise variance as representative
                        #shap_values = shap_vals[np.argmax([np.abs(v).mean() for v in shap_vals])]
                        class_means = [np.abs(v).mean() for v in shap_vals if isinstance(v, np.ndarray)]
                        shap_values = shap_vals[np.argmax(class_means)]
                else:
                    shap_values = shap_vals
                
                if shap_values is None or len(np.shape(shap_values)) != 2:
                    self.logger.warning("Invalid SHAP value shape; skipping SHAP plots.")
                    return None

                feature_names = getattr(self, 'numeric_features', []) + getattr(self, 'categorical_features', [])
                if not feature_names:
                    feature_names = list(self.X_test.columns)

            # Convert to numpy array
            shap_arr = np.asarray(shap_values)
            # Bar plot: mean absolute SHAP per feature
            mean_abs = np.abs(shap_arr).mean(axis=0)
            idx = np.argsort(mean_abs)[::-1][:30]
            top_feats = [feature_names[i] for i in idx]
            top_vals = mean_abs[idx]

            fig, ax = plt.subplots(figsize=(8, 6))
            y_pos = np.arange(len(top_feats))
            ax.barh(y_pos, top_vals[::-1])
            ax.set_yticks(y_pos)
            ax.set_yticklabels([f for f in top_feats[::-1]])
            ax.set_title(f"SHAP mean(|value|) - top features ({model_name})")
            out_bar = self.plot_dir / "shap_bar_plot.png"
            plt.tight_layout()
            plt.savefig(out_bar, dpi=200)
            plt.close(fig)
            self.logger.info("Saved SHAP bar plot -> %s", out_bar)

            # Interaction heatmap: compute mean abs interaction (if shap supports)
            try:
                # if shap has interaction_values
                if shap_csv.exists():
                    # fallback: compute correlation of absolute shap values as proxy for interaction
                    abs_shap = np.abs(shap_arr)
                    corr = np.corrcoef(abs_shap.T)
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(corr, cmap='coolwarm', xticklabels=feature_names, yticklabels=feature_names)
                    ax.set_title("SHAP interaction heatmap (proxy via abs-corr)")
                    out_heat = self.plot_dir / "shap_interaction_heatmap.png"
                    plt.tight_layout()
                    plt.savefig(out_heat, dpi=200)
                    plt.close(fig)
                    self.logger.info("Saved SHAP interaction heatmap -> %s", out_heat)
                else:
                    # Try shap interaction values (may be costly)
                    if hasattr(expl, "shap_interaction_values"):
                        inter = expl.shap_interaction_values(X_input)
                        # inter may be list per class; pick first
                        inter_mat = np.mean(np.abs(inter), axis=0)
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(inter_mat, cmap='coolwarm', xticklabels=feature_names, yticklabels=feature_names)
                        ax.set_title("SHAP interaction heatmap")
                        out_heat = self.plot_dir / "shap_interaction_heatmap.png"
                        plt.tight_layout()
                        plt.savefig(out_heat, dpi=200)
                        plt.close(fig)
                        self.logger.info("Saved SHAP interaction heatmap -> %s", out_heat)
            except Exception as e:
                self.logger.warning("SHAP interaction heatmap creation failed: %s", e)

            # Summary plot
            try:
                fig = plt.figure(figsize=(8, 6))
                shap.summary_plot(shap_arr, features=(self.X_test if self.preprocessor is None else None), feature_names=feature_names, show=False)
                out_sum = self.plot_dir / "shap_summary_plot.png"
                plt.tight_layout()
                plt.savefig(out_sum, dpi=200)
                plt.close()
                self.logger.info("Saved SHAP summary plot -> %s", out_sum)
            except Exception as e:
                self.logger.warning("SHAP summary plot failed (defensive). Attempting defensive summary.")
                # Defensive summary: make a violin-style summary using mean abs values
                try:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.boxplot(data=pd.DataFrame(shap_arr[:, idx], columns=[feature_names[i] for i in idx]))
                    out_def = self.plot_dir / "shap_summary_plot_defensive.png"
                    plt.tight_layout()
                    plt.savefig(out_def, dpi=200)
                    plt.close(fig)
                    self.logger.info("Saved defensive SHAP summary -> %s", out_def)
                except Exception as e2:
                    self.logger.warning("Defensive SHAP summary also failed: %s", e2)

            return {
                'bar': str(out_bar),
                'interaction': str(self.plot_dir / "shap_interaction_heatmap.png") if (self.plot_dir / "shap_interaction_heatmap.png").exists() else None,
                'summary': str(self.plot_dir / "shap_summary_plot.png") if (self.plot_dir / "shap_summary_plot.png").exists() else None,
                'summary_defensive': str(self.plot_dir / "shap_summary_plot_defensive.png") if (self.plot_dir / "shap_summary_plot_defensive.png").exists() else None
            }

        except Exception as e:
            self.logger.warning("SHAP plotting failed overall: %s", e)
            return None

    def generate_all_plots(self, shap_model: str = "xgb"):
        """
        Generate all requested plots (confusion matrices, missing heatmap, numeric dists,
        ROC/PR comparison, SHAP plots, and target distribution).
        """
        outputs = {}
        outputs['confusion_matrices'] = self.generate_confusion_matrices()
        outputs['missing_values_heatmap'] = str(self.plot_missing_values_heatmap()) if self.plot_missing_values_heatmap() else None
        outputs['numeric_distributions'] = str(self.plot_numeric_distributions()) if self.plot_numeric_distributions() else None
        outputs['roc_pr_comparison'] = str(self.plot_roc_pr_comparison()) if self.plot_roc_pr_comparison() else None

        # Target distribution (from df)
        try:
            if self.df is not None and self.target_col in self.df.columns:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.countplot(x=self.target_col, data=self.df, ax=ax)
                ax.set_title(f"Target Distribution ({self.target_col})")
                out = self.plot_dir / "target_distribution.png"
                plt.tight_layout()
                plt.savefig(out, dpi=200)
                plt.close(fig)
                outputs['target_distribution'] = str(out)
            else:
                outputs['target_distribution'] = None
        except Exception as e:
            self.logger.warning("Target distribution plotting failed: %s", e)
            outputs['target_distribution'] = None

        # SHAP plots
        outputs['shap'] = self.plot_shap_bar_and_summary(model_name=shap_model)
        return outputs

    # ----------------------------
    # Train all (convenience)
    # ----------------------------
    def train_all_models(self, train_lr: bool = True, train_xgb: bool = True, train_nn: bool = True, 
             train_brf: bool = True, train_ee: bool = True, train_lbm: bool = True, train_tab: bool = True):
        """Train selected models sequentially. Returns dict of trained model objects."""
        trained = {}
        if train_lr:
            trained['lr'] = self.train_lr()
        if train_xgb:
            trained['xgb'] = self.train_xgboost()
        if train_nn:
            trained['nn'] = self.train_neural_net()
        if train_brf:
            trained['brf'] = self.train_balanced_random_forest()
        if train_ee:
            trained['eec'] = self.train_easy_ensemble()
        if train_lbm:
            trained['lbm'] = self.train_lightgbm()
        if train_tab:
            trained['tab'] = self.train_tabnet()

        self.logger.info("Completed training selected models.")
        return trained

    # ----------------------------
    # Predictions and evaluation saver
    # ----------------------------
    def _save_predictions(self, tag: str, X_test: pd.DataFrame, y_test: pd.Series, preds, probs):
        """Save a small CSV with predictions for the given model tag."""
        outp = self.results_dir / f"{tag}_predictions.csv"
        try:
            ids = X_test['id'] if 'id' in X_test.columns else range(len(X_test))
        except Exception:
            ids = range(len(X_test))
        df_out = pd.DataFrame({
            'Id': list(ids),
            'y_true': list(y_test),
            'y_pred': list(preds),
            'prob': list(probs)
        })
        df_out.to_csv(outp, index=False)
        self.logger.info("Saved predictions for %s to %s", tag, outp)

        # Basic metrics
        try:
            auc = roc_auc_score(y_test, probs) if probs is not None else None
        except Exception:
            auc = None
        try:
            f1 = f1_score(y_test, preds)
        except Exception:
            f1 = None
        metrics = {'AUC': auc, 'F1': f1}
        with open(self.results_dir / f"{tag}_metrics.json", "w") as fh:
            json.dump(metrics, fh, default=str, indent=2)
        self.logger.info("Saved metrics for %s", tag)

    # ----------------------------
    # Artifacts & save/load
    # ----------------------------
    def prepare_artifacts(self):
        """Collect artifacts (models, preprocessor, feature names) for saving or serving."""
        artifacts = {}
        # prepare feature names from training X if available
        feature_names = list(self.X_train.columns) if self.X_train is not None else (list(self.df.columns.drop(self.target_col)) if self.df is not None and self.target_col in self.df.columns else [])
        artifacts['feature_names'] = feature_names
        artifacts['preprocessor'] = self.preprocessor
        artifacts['lr_model'] = self.lr_model
        artifacts['xgb_model'] = self.xgb_model
        artifacts['nn_model'] = self.nn_model
        self.artifacts = artifacts
        self.logger.info("Prepared artifacts dictionary.")
        return artifacts

    def save_models(self, prefix: Optional[str] = None) -> Dict[str, str]:
        """Save artifacts to self.model_dir. Returns dict mapping artifact_name->filepath."""
        if not self.artifacts:
            self.prepare_artifacts()

        saved = {}
        for name, obj in self.artifacts.items():
            if obj is None:
                continue
            # create filename
            fname = f"{prefix + '_' if prefix else ''}{name}.pkl"
            path = self.model_dir / fname
            try:
                joblib.dump(obj, path)
                saved[name] = str(path)
                self.logger.info("Saved artifact %s -> %s", name, path)
            except Exception as e:
                self.logger.warning("Failed to save artifact %s: %s", name, e)
        return saved

    # ----------------------------
    # Helpers to save predictions + confidence CSVs and other summary files
    # ----------------------------
    def save_prediction_with_confidence(self, tag: str, X: pd.DataFrame, y_true, preds, probs):
        """
        Notebook-accurate implementation of 'Prediction-confidence analysis':
        - Reads probability-style outputs (prob, prob_1, probability)
        - Computes confidence = max(prob, 1 - prob)
        - Saves both predictions.csv and predictions_confidence.csv
        """
        try:
            import os
            # 1️⃣ Build base DataFrame
            ids = X['id'].astype(str).tolist() if 'id' in X.columns else [str(i) for i in range(len(preds))]

            # 2️⃣ Normalize probability column naming
            if isinstance(probs, pd.Series):
                probs = probs.values
            elif isinstance(probs, (list, tuple)):
                probs = np.array(probs)

            if probs is None or len(probs) == 0:
                # fallback: treat y_pred as probability if nothing else
                probs = np.array(preds, dtype=float)

            # 3️⃣ Compute confidence (same as original notebook)
            conf = np.max(np.vstack([probs, 1 - probs]).T, axis=1)

            # 4️⃣ Construct dataframe
            df_out1 = pd.DataFrame({
                'Id': ids,
                'y_true': list(y_true),
                'y_pred': list(preds),
                'prob': probs
            })

            df_out2 = pd.DataFrame({
                'Id': ids,
                'y_true': list(y_true),
                'y_pred': list(preds),
                'prob': probs,
                'confidence': conf
            })
            # 5️⃣ Write two CSVs: one main, one with confidence suffix
            base_path = self.results_dir / f"{tag}_predictions.csv"
            conf_path = self.results_dir / f"{tag}_predictions_confidence.csv"

            df_out1.to_csv(base_path, index=False)
            df_out2.to_csv(conf_path, index=False)
            self.logger.info("Saved predictions and confidence CSVs for %s -> %s", tag, conf_path)
            return str(conf_path)
        except Exception as e:
            self.logger.warning("Failed to save prediction confidence for %s: %s", tag, e)
            return None

    def _save_predictions(self, tag: str, X_test: pd.DataFrame, y_test: pd.Series, preds, probs):
        """
        Backwards-compatible wrapper used elsewhere. It will call save_prediction_with_confidence.
        Also writes {tag}_metrics.json as earlier.
        """
        paths = self.save_prediction_with_confidence(tag, X_test, y_test, preds, probs)
        # compute metrics and write metrics file (if not already)
        try:
            auc = roc_auc_score(y_test, probs) if probs is not None else None
        except Exception:
            auc = None
        try:
            f1 = f1_score(y_test, preds)
        except Exception:
            f1 = None
        try:
            acc = accuracy_score(y_test, preds)
        except Exception:
            acc = None
        try:
            prec = precision_score(y_test, preds, zero_division=0)
        except Exception:
            prec = None
        try:
            rec = recall_score(y_test, preds, zero_division=0)
        except Exception:
            rec = None

        metrics = {'AUC': auc, 'F1': f1, 'Accuracy': acc, 'Precision': prec, 'Recall': rec}
        with open(self.results_dir / f"{tag}_metrics.json", "w") as fh:
            json.dump(metrics, fh, default=str, indent=2)
        self.logger.info("Saved metrics for %s", tag)

    # ----------------------------
    # External test evaluation (external_test_eval.csv)
    # ----------------------------
    def create_external_test_eval(self, preds: List, probs: List, y_true: Optional[pd.Series] = None, ids: Optional[List] = None):
        """
        Create external_test_eval.csv: Id, y_true (if given), y_pred, prob
        """
        try:
            if ids is None:
                ids = [str(i) for i in range(len(preds))]
            df_out = pd.DataFrame({
                'Id': ids,
                'y_pred': list(preds),
                'prob': list(probs)
            })
            if y_true is not None:
                df_out['y_true'] = list(y_true)
            path = self.results_dir / "external_test_eval.csv"
            df_out.to_csv(path, index=False)
            self.logger.info("Saved external_test_eval -> %s", path)
            return str(path)
        except Exception as e:
            self.logger.warning("Failed to create external_test_eval.csv: %s", e)
            return None

    # ----------------------------
    # Feature engineering CV comparison placeholder (feature_engineering_cv_comparison.csv)
    # ----------------------------
    def save_feature_engineering_cv_comparison(self, comparison_df: pd.DataFrame):
        """
        Save a DataFrame summarizing cross-validated results for different FE variants.
        Expects columns like: 'fe_variant','model','mean_auc','std_auc' etc.
        """
        try:
            outp = self.results_dir / "feature_engineering_cv_comparison.csv"
            comparison_df.to_csv(outp, index=False)
            self.logger.info("Saved feature_engineering_cv_comparison -> %s", outp)
            return str(outp)
        except Exception as e:
            self.logger.warning("Failed to save feature_engineering_cv_comparison.csv: %s", e)
            return None

    # ----------------------------
    # Model comparison metrics aggregation (model_comparison_metrics.csv)
    # ----------------------------
    def aggregate_model_comparison_metrics(self):
        """
        Reproduce notebook logic:
        Compute Accuracy, F1, Precision, Recall, ROC AUC, PR AUC
        directly from prediction CSVs and save as model_comparison_metrics.csv
        """
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

        pred_files = {
            'LR': self.results_dir / 'lr_predictions.csv',
            'XGB': self.results_dir / 'xgb_predictions.csv',
            'NN': self.results_dir / 'nn_predictions.csv',
            'BRF': self.results_dir / 'brf_predictions.csv',
            'EEC': self.results_dir / 'eec_predictions.csv',
            'TAB': self.results_dir / 'tabnet_predictions.csv'
        }

        models = {}
        for name, path in pred_files.items():
            if not path.exists():
                continue
            try:
                dfp = pd.read_csv(path)
                y_true = dfp['y_true'].values
                prob = dfp['prob'].values if 'prob' in dfp.columns else None
                pred = dfp['y_pred'].values if 'y_pred' in dfp.columns else (
                    (prob >= 0.5).astype(int) if prob is not None else None
                )
                if pred is None:
                    continue

                metrics = {
                    'Accuracy': float(accuracy_score(y_true, pred)),
                    'F1': float(f1_score(y_true, pred)),
                    'Precision': float(precision_score(y_true, pred)),
                    'Recall': float(recall_score(y_true, pred)),
                    'ROC AUC': float(roc_auc_score(y_true, prob)) if prob is not None else None,
                    'PR AUC': float(average_precision_score(y_true, prob)) if prob is not None else None
                }
                models[name] = metrics
                self.logger.info("Computed metrics for %s", name)
            except Exception as e:
                self.logger.warning("Failed to compute metrics for %s: %s", name, e)

        if not models:
            self.logger.warning("No prediction CSVs found to build model_comparison_metrics.csv.")
            return None

        df = pd.DataFrame(models).T
        df.index.name = "Model"
        outp = self.results_dir / "model_comparison_metrics.csv"
        df.to_csv(outp)
        self.logger.info("Saved model_comparison_metrics.csv -> %s", outp)
        return str(outp)

    # ----------------------------
    # Mutual information scores (mutual_info_scores.csv)
    # ----------------------------
    def compute_and_save_mutual_info(self, X: pd.DataFrame = None, y: pd.Series = None, discrete_features: Optional[List[str]] = None):
        """
        Compute mutual_info_classif for features and save mutual_info_scores.csv
        If X/y not provided, uses self.df / self.target_col
        """
        try:
            from sklearn.feature_selection import mutual_info_classif
        except Exception as e:
            self.logger.warning("sklearn.feature_selection.mutual_info_classif not available: %s", e)
            return None

        if X is None or y is None:
            if self.df is None or self.target_col not in self.df.columns:
                self.logger.warning("No data for mutual information.")
                return None
            X = self.df.drop(columns=[self.target_col]).copy()
            y = self.df[self.target_col]

        # encode categoricals temporarily
        X_enc = pd.get_dummies(X, drop_first=True).fillna(0)
        try:
            mi = mutual_info_classif(X_enc, y, discrete_features='auto', random_state=self.random_state)
            cols = X_enc.columns.tolist()
            df_mi = pd.DataFrame({'feature': cols, 'mutual_info': mi})
            df_mi = df_mi.sort_values('mutual_info', ascending=False)
            outp = self.results_dir / "mutual_info_scores.csv"
            df_mi.to_csv(outp, index=False)
            self.logger.info("Saved mutual_info_scores -> %s", outp)
            return str(outp)
        except Exception as e:
            self.logger.warning("Failed to compute mutual info: %s", e)
            return None

    # ----------------------------
    # Final model selection summary (final_model_selection_summary.json)
    # ----------------------------
    def prepare_final_model_selection_summary(self):
        """
        Final model selection with notebook logic.
        Ensures all six models (LR, XGB, NN, BRF, EEC, TAB) appear in JSON,
        computes weighted rubric score (ROC/PR/F1 normalized),
        penalizes low recall, and includes missing models as placeholders.
        """
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score, recall_score,
            roc_auc_score, average_precision_score
        )

        pred_files = {
            'LR': self.results_dir / 'lr_predictions.csv',
            'XGB': self.results_dir / 'xgb_predictions.csv',
            'NN': self.results_dir / 'nn_predictions.csv',
            'BRF': self.results_dir / 'brf_predictions.csv',
            'EEC': self.results_dir / 'eec_predictions.csv',
            'TAB': self.results_dir / 'tabnet_predictions.csv',
        }

        models = {}
        for name, path in pred_files.items():
            if not path.exists():
                self.logger.warning("%s prediction file missing (%s)", name, path)
                models[name] = {
                    'Accuracy': None, 'F1': None, 'Precision': None,
                    'Recall': None, 'ROC AUC': None, 'PR AUC': None
                }
                continue

            try:
                dfp = pd.read_csv(path)
                y_true = dfp.get('y_true')
                prob = dfp.get('prob')
                pred = dfp.get('y_pred', (prob >= 0.5).astype(int) if prob is not None else None)

                roc_auc = None
                pr_auc = None
                if prob is not None and not np.all((prob == 0) | (prob == 1)):
                    roc_auc = float(roc_auc_score(y_true, prob))
                    pr_auc = float(average_precision_score(y_true, prob))
                else:
                    self.logger.warning("%s: probability array invalid or missing; skipping AUC metrics.", name)


                metrics = {
                    'Accuracy': float(accuracy_score(y_true, pred)) if pred is not None else None,
                    'F1': float(f1_score(y_true, pred)) if pred is not None else None,
                    'Precision': float(precision_score(y_true, pred)) if pred is not None else None,
                    'Recall': float(recall_score(y_true, pred)) if pred is not None else None,
                    'ROC AUC': roc_auc,
                    'PR AUC': pr_auc
                }
                models[name] = metrics
            except Exception as e:
                self.logger.warning("Failed computing metrics for %s: %s", name, e)
                models[name] = {m: None for m in ['Accuracy', 'F1', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']}

        comp_df = pd.DataFrame(models).T
        score_df = comp_df.copy()

        # Normalize columns
        def norm_col(s):
            s = pd.to_numeric(s, errors='coerce')
            if s.isnull().all():
                return s
            return (s - s.min()) / (s.max() - s.min()) if s.max() != s.min() else 1.0

        for c in ['ROC AUC', 'PR AUC', 'F1', 'Precision', 'Recall']:
            score_df[c + '_norm'] = norm_col(score_df[c])

        score_df['final_score'] = (
            0.5 * score_df['ROC AUC_norm'] +
            0.2 * score_df['PR AUC_norm'] +
            0.2 * score_df['F1_norm']
        )
        if 'Recall' in score_df.columns:
            score_df.loc[score_df['Recall'] < 0.3, 'final_score'] -= 0.2

        score_df['final_score'] = score_df['final_score'].fillna(-999)
        selected = score_df['final_score'].idxmax()
        recommendation = {
            'recommended_model': selected,
            'reason': 'Highest weighted score (ROC AUC primary, PR AUC + F1 secondary)',
            'scores': score_df.loc[selected].to_dict()
        }

        outp = self.results_dir / "final_model_selection_summary.json"
        summary = {
            #'comparison_table': score_df.round(4).fillna('').to_dict(),
            'comparison_table': comp_df.round(4).fillna('').to_dict(),
            'recommendation': recommendation
        }

        with open(outp, 'w') as f:
            json.dump(summary, f, indent=2)
        self.logger.info("Saved final_model_selection_summary -> %s", outp)

        return str(outp)

    # ----------------------------
    # Optuna best params save helpers
    # ----------------------------
    def save_optuna_best(self, estimator: str, best_params: dict):
        """Save optuna best params JSON as optuna_best_{estimator}.json"""
        try:
            outp = self.results_dir / f"optuna_best_{estimator}.json"
            with open(outp, "w") as fh:
                json.dump(best_params or {}, fh, indent=2)
            self.logger.info("Saved optuna best for %s -> %s", estimator, outp)
            return str(outp)
        except Exception as e:
            self.logger.warning("Failed to save optuna best params: %s", e)
            return None

    # ----------------------------
    # SHAP values CSV export (shap_values.csv)
    # ----------------------------
    def save_shap_values_csv(self, model_name: str = "xgb"):
        """
        Save shap_values.csv (full or sampled) to results_dir/shap_values.csv
        If results/shap_values_{model_name}.csv exists, copy/rename it.
        Otherwise compute SHAP (guarded) and save.
        """
        try:
            # priority: existing per-model file
            existing = self.results_dir / f"shap_values_{model_name}.csv"
            outp = self.results_dir / "shap_values.csv"
            if existing.exists():
                df = pd.read_csv(existing)
                df.to_csv(outp, index=False)
                self.logger.info("Copied existing shap_values_%s to shap_values.csv", model_name)
                return str(outp)

            # else compute shap
            import shap  # type: ignore
            model_obj = getattr(self, f"{model_name}_model", None)
            if model_obj is None:
                self.logger.warning("No model available to compute SHAP for %s", model_name)
                return None
            if self.X_test is None:
                self.logger.warning("No X_test to compute SHAP on.")
                return None

            # transform X_test
            if self.preprocessor is not None:
                X_input = self.preprocessor.transform(self.X_test)
            else:
                X_input = self.X_test.fillna(0).values
            # extract raw model
            raw_model = model_obj.named_steps['clf'] if hasattr(model_obj, 'named_steps') and 'clf' in model_obj.named_steps else model_obj
            expl = shap.TreeExplainer(raw_model)
            shap_vals = expl.shap_values(X_input)
            # choose representative array
            if isinstance(shap_vals, list):
                if len(shap_vals) == 0:
                    self.logger.warning("SHAP returned empty list.")
                    return None
                shap_arr = np.asarray(shap_vals[0])
            else:
                shap_arr = np.asarray(shap_vals)
            feat_names = getattr(self, 'numeric_features', []) + getattr(self, 'categorical_features', [])
            if not feat_names and isinstance(self.X_test, pd.DataFrame):
                feat_names = list(self.X_test.columns)
            n_feats = shap_arr.shape[1]
            if len(feat_names) != n_feats:
                self.logger.warning(
                    "Feature name length (%d) != SHAP value width (%d). Adjusting...",
                    len(feat_names), n_feats
                )
                if len(feat_names) > n_feats:
                    feat_names = feat_names[:n_feats]
                else:
                    feat_names = list(feat_names) + [f"extra_{i}" for i in range(n_feats - len(feat_names))]

            df_shap = pd.DataFrame(shap_arr, columns=feat_names)
            df_shap.to_csv(outp, index=False)
            self.logger.info("Saved shap_values.csv -> %s", outp)
            return str(outp)
        except Exception as e:
            self.logger.warning("Failed to save shap_values.csv: %s", e)
            return None

    # ----------------------------
    # Target distribution normalized CSV (target_distribution_normalized.csv)
    # ----------------------------
    def save_target_distribution_normalized(self):
        """
        Save a CSV with normalized counts/proportions for the target column.
        Columns: value, count, proportion
        """
        try:
            if self.df is None or self.target_col not in self.df.columns:
                self.logger.warning("No dataframe/target for target_distribution_normalized.")
                return None
            vc = self.df[self.target_col].value_counts().rename_axis('value').reset_index(name='count')
            total = vc['count'].sum()
            vc['proportion'] = (vc['count'] / total).round(6)
            outp = self.results_dir / "target_distribution_normalized.csv"
            vc.to_csv(outp, index=False)
            self.logger.info("Saved target_distribution_normalized -> %s", outp)
            return str(outp)
        except Exception as e:
            self.logger.warning("Failed to save target_distribution_normalized.csv: %s", e)
            return None

    # ----------------------------
    # Convenience method: produce all requested files if models/predictions exist
    # ----------------------------
    def generate_standard_outputs(self):
        """
        Run a sequence of the above methods to produce the files requested in the notebook:
        balanced_rf_predictions(.csv,_confidence), easy_ensemble_predictions(.csv,_confidence),
        lr_predictions(.csv,_confidence), nn_predictions(.csv,_confidence), tabnet_predictions(.csv,_confidence),
        xgb_predictions(.csv,_confidence),
        external_test_eval.csv, feature_engineering_cv_comparison.csv (placeholder),
        final_model_selection_summary.json, model_comparison_metrics.csv,
        mutual_info_scores.csv, optuna_best_*.json (if present), shap_values.csv,
        target_distribution_normalized.csv
        """
        outputs = {}

        # 1) Ensure predictions exist in results_dir from training steps (otherwise call training)
        # Try to aggregate available predictions by tags
        tags = ['brf', 'eec', 'lr', 'nn', 'tabnet', 'xgb']
        for tag in tags:
            # attempt to read existing preds; if not present but model exists, generate predictions
            pred_path = self.results_dir / f"{tag}_predictions.csv"
            if not pred_path.exists():
                # attempt to auto-generate if model and X_test available
                model_obj = getattr(self, f"{tag}_model", None)
                if model_obj is not None and self.X_test is not None:
                    try:
                        # attempt model.predict on raw X_test or transformed
                        try:
                            preds = model_obj.predict(self.X_test)
                            probs = model_obj.predict_proba(self.X_test)[:, 1] if hasattr(model_obj, 'predict_proba') else np.array(preds, dtype=float)
                        except Exception:
                            # try transform
                            Xproc = self.preprocessor.transform(self.X_test) if self.preprocessor is not None else self.X_test.fillna(0).values
                            preds = model_obj.predict(Xproc)
                            probs = model_obj.predict_proba(Xproc)[:, 1] if hasattr(model_obj, 'predict_proba') else np.array(preds, dtype=float)
                        self.save_prediction_with_confidence(tag, self.X_test, self.y_test, preds, probs)
                    except Exception as e:
                        self.logger.warning("Auto-generate predictions failed for %s: %s", tag, e)
            else:
                self.logger.info("Predictions for %s already exist.", tag)

        # 2) Aggregate model metrics and prepare final selection
        outputs['model_comparison_metrics'] = self.aggregate_model_comparison_metrics()
        outputs['final_model_summary'] = self.prepare_final_model_selection_summary()
        # 3) Mutual info
        outputs['mutual_info_scores'] = self.compute_and_save_mutual_info()
        # 4) Copy/save generic shap_values.csv
        outputs['shap_values'] = self.save_shap_values_csv()
        # 5) Optuna files: if self.results_dir contains optuna outputs, keep them (no-op)
        # (these are produced by tune_with_optuna())
        optuna_files = list(self.results_dir.glob("optuna_best_*.json"))
        outputs['optuna_files'] = [str(p) for p in optuna_files] if optuna_files else None
        # 6) external_test_eval: if exists keep, else create placeholder from best model
        external_path = self.results_dir / "external_test_eval.csv"
        if not external_path.exists():
            # if X_test and best model exist, create external_test_eval from selected model predictions
            sel = None
            try:
                sel = json.load(open(self.results_dir / "final_model_selection_summary.json"))['selected_model']
            except Exception:
                sel = None
            if sel and getattr(self, f"{sel}_model", None) is not None:
                model_obj = getattr(self, f"{sel}_model")
                try:
                    preds = model_obj.predict(self.X_test)
                    probs = model_obj.predict_proba(self.X_test)[:, 1] if hasattr(model_obj, 'predict_proba') else np.array(preds, dtype=float)
                    self.create_external_test_eval(preds=preds, probs=probs, y_true=self.y_test, ids=self.X_test.get('id').astype(str).tolist() if 'id' in self.X_test.columns else None)
                    outputs['external_test_eval'] = str(external_path)
                except Exception as e:
                    self.logger.warning("Failed to create external_test_eval from selected model: %s", e)
        else:
            outputs['external_test_eval'] = str(external_path)

        # 7) Target distribution normalized
        outputs['target_distribution_normalized'] = self.save_target_distribution_normalized()

        self.logger.info("Standard outputs generation complete.")
        return outputs

    def load_model(self, name: str, path: Optional[str] = None):
        """Load a saved model artifact and attach to context by name."""
        if path is None:
            path = self.model_dir / f"{name}.pkl"
        else:
            path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model path not found: {path}")
        obj = joblib.load(path)
        setattr(self, name, obj)
        self.logger.info("Loaded model %s from %s", name, path)
        return obj

    # ----------------------------
    # Utility: automatic pipeline for a single customer
    # ----------------------------
    def predict_single_nope(self, customer_row: pd.Series = None, customer_dict: dict = None, by_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Predict renewal for a single customer.
        - by_id: will lookup in X_test if available and ID column exists.
        - customer_row: pandas Series of raw features (excluding target)
        - customer_dict: dict of raw features
        Returns: dict with prediction and source.
        """
        if self.artifacts is None or not self.artifacts:
            self.prepare_artifacts()
        model = self.artifacts.get('xgb_model') or self.artifacts.get('lr_model') or self.artifacts.get('nn_model')
        if model is None:
            raise ValueError("No model available. Train or load a model first.")

        # lookup by id
        if by_id and self.X_test is not None and 'id' in self.X_test.columns:
            row = self.X_test[self.X_test['id'].astype(str) == str(by_id)]
            if not row.empty:
                try:
                    pred = model.predict(row)[0]
                    prob = model.predict_proba(row)[0][1] if hasattr(model, 'predict_proba') else None
                except Exception:
                    # try transform then predict
                    Xr = self.preprocessor.transform(row)
                    pred = model.predict(Xr)[0]
                    prob = model.predict_proba(Xr)[0][1] if hasattr(model, 'predict_proba') else None
                return {'prediction': int(pred), 'prob': float(prob) if prob is not None else None, 'source': 'by_id'}

        # manual dict/series
        if customer_row is None and customer_dict is not None:
            row = pd.DataFrame([customer_dict])
        elif customer_row is not None:
            row = pd.DataFrame([customer_row])
        else:
            raise ValueError("Provide by_id or customer_dict/row")

        # Align columns (use feature_names)
        try:
            row = row.reindex(columns=self.artifacts['feature_names']).fillna(0)
        except Exception:
            row = row.fillna(0)

        try:
            pred = model.predict(row)[0]
            prob = model.predict_proba(row)[0][1] if hasattr(model, 'predict_proba') else None
        except Exception:
            # transform then predict
            Xr = self.preprocessor.transform(row)
            pred = model.predict(Xr)[0]
            prob = model.predict_proba(Xr)[0][1] if hasattr(model, 'predict_proba') else None

        return {'prediction': int(pred), 'prob': float(prob) if prob is not None else None, 'source': 'manual'}

    def predict_external_test(self, test_csv_path: str) -> str:
        """
        Predict on an external test CSV using the best model from training.
        Automatically re-applies feature engineering and preprocessing to
        ensure full parity with training.
        Saves predictions to external_test_eval.csv.
        """
        import json
        import pandas as pd
        import numpy as np
        import joblib
        from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score

        self.logger.info(f"Running external test prediction for {test_csv_path}")
        test_path = Path(test_csv_path)
        if not test_path.exists():
            raise FileNotFoundError(f"Test CSV not found: {test_path}")

        # 1️⃣ Load external test data
        df_ext = pd.read_csv(test_path)
        self.logger.info(f"Loaded test data with shape {df_ext.shape}")

        # 2️⃣ Apply same feature engineering as training
        try:
            if hasattr(self, "run_fe"):
                #df_ext = self.run_fe(df_ext.copy())
                self.df = df_ext.copy()
                self.run_fe()
                df_ext = self.df.copy()

                self.logger.info("Applied feature engineering on test data (run_fe).")
            else:
                self.logger.warning("run_fe() not found in context; skipping feature engineering.")
        except Exception as e:
            self.logger.error(f"Feature engineering failed on test data: {e}")
            raise

        # 3️⃣ Split features/labels
        y_ext = None
        if "renewal" in df_ext.columns:
            y_ext = df_ext["renewal"]
        elif "y_true" in df_ext.columns:
            y_ext = df_ext["y_true"]

        X_ext = df_ext.drop(columns=["renewal", "y_true"], errors="ignore")

        # 4️⃣ Load best model name from final_model_selection_summary.json
        best_model_name = None
        final_summary_path = self.results_dir / "final_model_selection_summary.json"
        if final_summary_path.exists():
            with open(final_summary_path, "r") as f:
                summary = json.load(f)
                best_model_name = (
                    summary.get("recommendation", {}).get("recommended_model")
                    or summary.get("selected_model")
                )
        if not best_model_name:
            raise RuntimeError("No best model found in summary JSON. Please run training first.")

        best_model_name = best_model_name.lower()
        self.logger.info(f"Best model selected for external test: {best_model_name}")

        # 5️⃣ Locate and load corresponding trained model
        model_file = None
        for p in self.model_dir.glob("*"):
            if best_model_name in p.name.lower() and p.suffix in (".pkl", ".joblib"):
                model_file = p
                break
        if not model_file:
            raise FileNotFoundError(f"No saved model found for {best_model_name} in {self.model_dir}")

        self.logger.info(f"Using model file: {model_file}")
        model = joblib.load(model_file)

        # 6️⃣ Ensure preprocessing consistency
        X_for_pred = X_ext.copy()
        try:
            # Case 1: Model is pipeline with preprocessor
            if hasattr(model, "named_steps") and "preproc" in model.named_steps:
                X_for_pred = model.named_steps["preproc"].transform(X_ext)
                self.logger.info("Applied preprocessor from model pipeline.")

            # Case 2: Use context preprocessor if available
            elif hasattr(self, "preprocessor") and self.preprocessor is not None:
                X_for_pred = self.preprocessor.transform(X_ext)
                self.logger.info("Applied preprocessor from current context.")

            # Case 3: Try loading saved preprocessor from disk
            else:
                preproc_path = self.model_dir / "trained_preprocessor.pkl"
                if preproc_path.exists():
                    preproc = joblib.load(preproc_path)
                    X_for_pred = preproc.transform(X_ext)
                    self.logger.info("Applied saved preprocessor from disk.")
                else:
                    self.logger.warning("No preprocessor found — using raw test data.")
        except Exception as e:
            self.logger.error(f"Preprocessor transform failed: {e}")
            raise

        # 7️⃣ Predict probabilities and labels
        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_for_pred)[:, 1]
            elif hasattr(model, "decision_function"):
                scores = model.decision_function(X_for_pred)
                probs = (scores - scores.min()) / (scores.max() - scores.min())
            else:
                preds = model.predict(X_for_pred)
                probs = np.clip(preds.astype(float), 0, 1)

            preds = (probs >= 0.5).astype(int)
        except Exception as e:
            self.logger.error(f"Prediction failed on test data: {e}")
            raise

        # 8️⃣ Evaluate if ground truth exists
        metrics = {}
        if y_ext is not None:
            try:
                metrics = {
                    "accuracy": float(accuracy_score(y_ext, preds)),
                    "f1": float(f1_score(y_ext, preds)),
                    "roc_auc": float(roc_auc_score(y_ext, probs)),
                    "pr_auc": float(average_precision_score(y_ext, probs)),
                }
                self.logger.info(f"External test metrics: {metrics}")
            except Exception as e:
                self.logger.warning(f"Failed to compute metrics for external test: {e}")

        # 9️⃣ Save predictions to results/external_test_eval.csv
        ids = df_ext["id"].astype(str).tolist() if "id" in df_ext.columns else [str(i) for i in range(len(X_ext))]
        df_out = pd.DataFrame({
            "Id": ids,
            "y_true": y_ext if y_ext is not None else [None] * len(preds),
            "y_pred": preds,
            "prob": probs,
        })
        outp = self.results_dir / "external_test_eval.csv"
        df_out.to_csv(outp, index=False)
        self.logger.info(f"Saved external test predictions to {outp}")

        # 🔟 Save metrics JSON if available
        if metrics:
            metrics_out = self.results_dir / "external_test_metrics.json"
            with open(metrics_out, "w") as f:
                json.dump(metrics, f, indent=2)
            self.logger.info(f"Saved external test metrics to {metrics_out}")

        return {
            "message": "External test predictions completed successfully.",
            "metrics": metrics,
            "preview": df_out.to_dict(orient="records"),
            "output_path": str(outp),
        }
        
        #return str(outp)
    
    ## DO NOT USE
    def predict_single_dontuse(self, user_data: pd.DataFrame):
        """Apply same FE + preprocessing + prediction for one record."""
        self.df = user_data.copy()
        self.run_fe()
        df_user = self.df.copy()
        X_user = df_user.drop(columns=['renewal'], errors='ignore')

        model_file = self.model_dir / "best_model.pkl"  # or auto-detect logic
        import joblib
        model = joblib.load(model_file)

        # apply preprocessor if available
        if hasattr(model, 'named_steps') and 'preproc' in model.named_steps:
            X_user_t = model.named_steps['preproc'].transform(X_user)
        elif hasattr(self, 'preprocessor'):
            X_user_t = self.preprocessor.transform(X_user)
        else:
            X_user_t = X_user.values

        probs = model.predict_proba(X_user_t)[:, 1]
        preds = (probs >= 0.5).astype(int)
        return probs, preds
    
    def predict_single(self, customer_row: pd.Series = None, customer_dict: dict = None, by_id: Optional[str] = None) -> dict:
        """
        Predict renewal probability for a single customer (by dict, row, or id).
        Returns consistent structure for backend API.
        """
        import numpy as np
        import pandas as pd
        import joblib
        import json
        import traceback

        try:
            # Step 1️⃣: Determine input data
            if customer_dict is not None:
                df = pd.DataFrame([customer_dict])
            elif customer_row is not None:
                df = pd.DataFrame([customer_row.to_dict()])
            elif by_id is not None and hasattr(self, "test_df"):
                df = self.test_df[self.test_df["id"].astype(str) == str(by_id)].copy()
                if df.empty:
                    raise ValueError(f"No record found for ID {by_id}")
            else:
                raise ValueError("No customer data provided for prediction.")

            self.logger.info(f"Received input with {df.shape[1]} columns: {list(df.columns)}")

            # Step 2️⃣: Feature engineering
            #df_fe = self.run_fe(df.copy())
            self.df = df.copy()
            self.run_fe()
            df = self.df
            self.logger.info(f"Feature engineering complete. Shape after FE: {df.shape}")
            
            self.logger.info(f"Incoming DF columns: {df.columns.tolist()}")
            #self.logger.info(f"Expected features: {expected_features}")

            # Step 3️⃣: Align with training schema
            feature_path = self.results_dir / "feature_names.npy"
            if feature_path.exists():
                expected_features = list(np.load(feature_path, allow_pickle=True))
                self.logger.info(f"Loaded expected training schema with {len(expected_features)} features.")
            else:
                self.logger.warning("No feature_names.npy found — using available columns.")
                expected_features = df.columns.tolist()

            # Handle missing and extra columns
            missing_cols = [col for col in expected_features if col not in df.columns]
            extra_cols = [col for col in df.columns if col not in expected_features]


            if missing_cols:
                self.logger.warning(f"Adding {len(missing_cols)} missing columns: {missing_cols}")
                for col in missing_cols:
                    df[col] = 0
            if extra_cols:
                self.logger.warning(f"Ignoring {len(extra_cols)} unexpected columns: {extra_cols}")

            # Reorder columns to match training schema
            df = df[expected_features]

            # Step 4️⃣: Load best model
            summary_path = self.results_dir / "final_model_selection_summary.json"
            if not summary_path.exists():
                raise FileNotFoundError("final_model_selection_summary.json not found.")

            with open(summary_path, "r") as f:
                summary = json.load(f)

            model_name = summary.get("recommendation", {}).get("recommended_model")
            if not model_name:
                raise ValueError("No recommended model found in final_model_selection_summary.json")

            model_file = None
            for p in self.model_dir.glob("*"):
                if model_name.lower() in p.name.lower() and p.suffix in (".pkl", ".joblib"):
                    model_file = p
                    break
            if not model_file:
                raise FileNotFoundError(f"Model file for {model_name} not found in {self.model_dir}")

            model = joblib.load(model_file)
            self.logger.info(f"Loaded model: {model_name} from {model_file.name}")

            # Step 5️⃣: Apply preprocessing (if any)
            if hasattr(model, "named_steps") and "preproc" in model.named_steps:
                X = model.named_steps["preproc"].transform(df)
            elif hasattr(self, "preprocessor") and self.preprocessor is not None:
                X = self.preprocessor.transform(df)
            else:
                self.logger.warning("No preprocessing pipeline found — using numeric columns only.")
                X = df.select_dtypes(include=[np.number])

            if X.shape[1] == 0:
                raise ValueError("No numeric columns available for prediction.")

            self.logger.info(f"Predicting with {X.shape[1]} features, expected shape matches: {X.shape[1] == len(expected_features)}")
 
            # Step 6️⃣: Type cast for NN models
            X = np.array(X).astype("float32")

            #  Adjust feature shape for NN models
            if hasattr(model, "input_shape"):
                expected_dim = model.input_shape[-1]
                actual_dim = X.shape[1]
                self.logger.info(f'expected_dim = {expected_dim}')
                self.logger.info(f'actual_dim = {actual_dim}')
                if actual_dim != expected_dim:
                    self.logger.warning(f"Adjusting input shape: expected {expected_dim}, got {actual_dim}")
                    if actual_dim < expected_dim:
                        pad_width = expected_dim - actual_dim
                        self.logger.info(f'pad_width = {pad_width}')
                        X = np.pad(X, ((0, 0), (0, pad_width)), mode='constant')
                    else:
                        X = X[:, :expected_dim]

            # Step 7️⃣: Predict
            if hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(X)[:, 1][0])
                pred = int(prob >= 0.5)
            else:
                y_pred = model.predict(X)
                if isinstance(y_pred, (list, np.ndarray)):
                    pred = int(y_pred[0])
                else:
                    pred = int(y_pred)
                prob = float(pred)


        except Exception as e:
            tb = traceback.format_exc()
            self.logger.error(f"Prediction failed in predict_single: {e}\n{tb}")
            raise

        # Step 8️⃣: Return consistent structured response
        return {
            "message": f"Prediction completed successfully using model: {model_name}",
            "model": model_name,
            "prediction": int(pred),
            "probability": round(prob, 4),
            "preview": df.head(1).to_dict(orient="records")
        }


# End of MLContext

# ----------------------------
# If run as script, provide a small demo/CLI
# ----------------------------
if __name__ == "__main__":
    ctx = MLContext(base_dir="C:/D/Work/Learn/AI_ML/Capstone/Source/2024aiml049/Capstone_insurance_renewal/test")
    ctx.logger.info("ml_core.py demo start")
    demo_csv = ctx.data_dir / "demo_train.csv"
    if demo_csv.exists():
        ctx.load_data_from_csv(str(demo_csv))
        ctx.run_eda()
        ctx.run_fe()
        #ctx.save_feature_engineering_cv_comparison() do we need this?
        ctx.prepare_feature_config()
        ctx.build_preprocessor()
        ctx.split_train_test()
        ctx.train_all_models()
        #ctx.adasyn_experiment()
        #ctx.tune_with_optuna('rf')
        #ctx.tune_with_optuna('xgb')
        ctx.export_shap()
        ctx.prepare_artifacts()
        saved = ctx.save_models(prefix="demo")
        # Finally, generate all the plots
        ctx.generate_all_plots(shap_model="xgb")
        ctx.generate_standard_outputs()
        ctx.predict_external_test('data/test_uploaded.csv')
        ctx.logger.info("Saved artifacts: %s", saved)
    else:
        ctx.logger.info("No demo CSV found at %s. Place a CSV named demo_train.csv under data/ to run demo.", demo_csv)
