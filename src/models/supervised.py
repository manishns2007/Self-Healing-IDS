"""
Supervised Models — Random Forest and XGBoost classifiers for intrusion detection.
"""

import numpy as np
import joblib
from pathlib import Path
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix
)
import xgboost as xgb
import yaml


def load_config() -> dict:
    with open("configs/model_config.yaml") as f:
        return yaml.safe_load(f)


class RandomForestDetector:
    """Binary intrusion detector using Random Forest."""

    def __init__(self, config: dict | None = None):
        cfg = (config or load_config())["random_forest"]
        self.model = RandomForestClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            min_samples_split=cfg["min_samples_split"],
            min_samples_leaf=cfg["min_samples_leaf"],
            class_weight=cfg["class_weight"],
            random_state=cfg["random_state"],
            n_jobs=-1,
        )
        self.is_fitted = False
        self.model_name = "random_forest"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestDetector":
        logger.info(f"Training RandomForest on {X.shape} ...")
        self.model.fit(X, y)
        self.is_fitted = True
        logger.success("RandomForest training complete")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns probability of attack class [P(normal), P(attack)]."""
        return self.model.predict_proba(X)

    def attack_probability(self, X: np.ndarray) -> np.ndarray:
        """Returns P(attack) for each sample."""
        return self.predict_proba(X)[:, 1]

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        y_pred = self.predict(X)
        y_prob = self.attack_probability(X)
        metrics = {
            "f1": f1_score(y, y_pred, zero_division=0),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y, y_prob),
        }
        logger.info(f"[RandomForest] F1={metrics['f1']:.4f} | AUC={metrics['roc_auc']:.4f}")
        return metrics

    def feature_importances(self) -> np.ndarray:
        return self.model.feature_importances_

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.success(f"RandomForest saved → {path}")

    @staticmethod
    def load(path: str) -> "RandomForestDetector":
        m = joblib.load(path)
        logger.info(f"RandomForest loaded ← {path}")
        return m


class XGBoostDetector:
    """Binary intrusion detector using XGBoost."""

    def __init__(self, config: dict | None = None):
        cfg = (config or load_config())["xgboost"]
        self.model = xgb.XGBClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            subsample=cfg["subsample"],
            colsample_bytree=cfg["colsample_bytree"],
            eval_metric=cfg["eval_metric"],
            random_state=cfg["random_state"],
            n_jobs=-1,
            verbosity=0,
        )
        self.is_fitted = False
        self.model_name = "xgboost"

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: np.ndarray | None = None, y_val: np.ndarray | None = None) -> "XGBoostDetector":
        logger.info(f"Training XGBoost on {X.shape} ...")
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(
            X, y,
            eval_set=eval_set,
            verbose=False,
        )
        self.is_fitted = True
        logger.success("XGBoost training complete")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def attack_probability(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        y_pred = self.predict(X)
        y_prob = self.attack_probability(X)
        metrics = {
            "f1": f1_score(y, y_pred, zero_division=0),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y, y_prob),
        }
        logger.info(f"[XGBoost] F1={metrics['f1']:.4f} | AUC={metrics['roc_auc']:.4f}")
        return metrics

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.success(f"XGBoost saved → {path}")

    @staticmethod
    def load(path: str) -> "XGBoostDetector":
        m = joblib.load(path)
        logger.info(f"XGBoost loaded ← {path}")
        return m
