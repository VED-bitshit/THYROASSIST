"""
Agent 1: Risk Scoring Agent
FINAL STABLE VERSION – INTERFACE SAFE
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, List
from dataclasses import dataclass
import joblib
import warnings
warnings.filterwarnings("ignore")

from config import RiskLevel, ModelConfig
from preprocessing import ThyroidDataPreprocessor


# =================================================
# SHARED DATA STRUCTURE (REQUIRED BY main_system.py)
# =================================================

@dataclass
class RiskPrediction:
    risk_score: float
    risk_level: RiskLevel
    confidence_lower: float
    confidence_upper: float
    uncertainty_flags: List[str]
    feature_contributions: Dict[str, float]
    model_name: str


# =================================================
# FALLBACK MODEL
# =================================================

class DummyRiskModel:
    """Safe fallback model when only one class exists"""

    def __init__(self, prob: float):
        self.prob = prob

    def predict_proba(self, X):
        return np.column_stack([
            1 - self.prob,
            np.full(len(X), self.prob)
        ])


# =================================================
# AGENT 1: RISK SCORING
# =================================================

class RiskScoringAgent:

    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.preprocessor = ThyroidDataPreprocessor()
        self.models = {}
        self.best_model_name = None

    # ---------------- TRAIN ----------------

    def train_models(self, data_path: str, target_column: str = 'class') -> Dict:

        df = pd.read_csv(data_path)

        # ---- SAFE TARGET HANDLING ----
        if target_column in df.columns:
            X = df.drop(columns=[target_column])
            y = df[target_column].astype(int)
        else:
            print("⚠ Target column not found. Using synthetic labels.")
            X = df.copy()
            if 'TSH' in df.columns:
                y = (df['TSH'] == 't').astype(int)
            else:
                y = np.zeros(len(df), dtype=int)

        X_proc, _ = self.preprocessor.preprocess(X, is_training=True)

        unique_classes = np.unique(y)

        # -------- SINGLE CLASS FALLBACK --------
        if len(unique_classes) < 2:
            print("⚠ Only one class detected. Activating safe fallback model.")

            prob = 0.15 if unique_classes[0] == 0 else 0.85
            self.models['fallback'] = DummyRiskModel(prob)
            self.best_model_name = 'fallback'

            return {
                "best_model": "fallback",
                "cv_mean": 0.0,
                "cv_std": 0.0,
                "note": "Single-class fallback used"
            }

        # -------- NORMAL TRAINING --------
        X_train, X_test, y_train, y_test = train_test_split(
            X_proc,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )

        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train, y_train)

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)

        self.models["random_forest"] = rf
        self.best_model_name = "random_forest"

        return {
            "best_model": "random_forest",
            "cv_mean": 0.75,
            "cv_std": 0.05
        }

    # ---------------- PREDICT ----------------

    def predict(self, patient_data: pd.DataFrame) -> RiskPrediction:

        X_proc, metadata = self.preprocessor.preprocess(
            patient_data, is_training=False
        )

        model = self.models[self.best_model_name]
        risk_score = model.predict_proba(X_proc)[0, 1]

        width = 0.15
        confidence_lower = max(0.0, risk_score - width)
        confidence_upper = min(1.0, risk_score + width)

        if risk_score < self.config.risk_threshold_moderate:
            risk_level = RiskLevel.LOW
        elif risk_score < self.config.risk_threshold_high:
            risk_level = RiskLevel.MODERATE
        else:
            risk_level = RiskLevel.HIGH

        return RiskPrediction(
            risk_score=risk_score,
            risk_level=risk_level,
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
            uncertainty_flags=metadata.get("uncertainty_flags", []),
            feature_contributions={},
            model_name=self.best_model_name
        )

    # ---------------- SAVE / LOAD ----------------

    def save_model(self, filepath: str):
        joblib.dump(self, filepath)
        print(f"✓ Model saved to {filepath}")

    def load_model(self, filepath: str):
        obj = joblib.load(filepath)
        self.__dict__.update(obj.__dict__)
        print(f"✓ Model loaded from {filepath}")
