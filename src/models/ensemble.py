"""
Ensemble Detector — Combines RF + XGBoost + IsolationForest + Autoencoder.
Uses weighted voting on probability scores for final prediction.
"""

import numpy as np
import joblib
from pathlib import Path
from loguru import logger
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import yaml

from src.models.supervised import RandomForestDetector, XGBoostDetector
from src.models.anomaly import IsolationForestDetector, AutoencoderDetector


def load_config() -> dict:
    with open("configs/model_config.yaml") as f:
        return yaml.safe_load(f)


class EnsembleDetector:
    """
    Weighted ensemble of supervised + anomaly detection models.

    Final score = w_rf * P_rf + w_xgb * P_xgb + w_if * S_if + w_ae * S_ae
    where S_* are normalized anomaly scores ∈ [0, 1].
    """

    def __init__(
        self,
        rf: RandomForestDetector | None = None,
        xgb: XGBoostDetector | None = None,
        iso: IsolationForestDetector | None = None,
        ae: AutoencoderDetector | None = None,
        config: dict | None = None,
    ):
        cfg = (config or load_config())["ensemble"]
        self.rf = rf
        self.xgb = xgb
        self.iso = iso
        self.ae = ae
        self.w_rf = cfg["rf_weight"]
        self.w_xgb = cfg["xgb_weight"]
        self.w_if = cfg["if_weight"]
        self.w_ae = cfg["ae_weight"]
        self.threshold = cfg["final_threshold"]
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: np.ndarray | None = None, y_val: np.ndarray | None = None) -> "EnsembleDetector":
        """Fit all sub-models."""
        logger.info("=== Training Ensemble ===")

        if self.rf is None:
            self.rf = RandomForestDetector()
        self.rf.fit(X, y)

        if self.xgb is None:
            self.xgb = XGBoostDetector()
        self.xgb.fit(X, y, X_val, y_val)

        if self.iso is None:
            self.iso = IsolationForestDetector()
        self.iso.fit(X, y)

        if self.ae is None:
            logger.info("No Autoencoder provided — skipping AE training")
            self.w_ae = 0.0  # zero out weight so ensemble score is unaffected
        else:
            self.ae.fit(X, y)

        self.is_fitted = True
        logger.success("Ensemble training complete!")
        return self

    def _compute_scores(self, X: np.ndarray) -> np.ndarray:
        """Returns weighted ensemble score ∈ [0, 1] for each sample."""
        scores = np.zeros(len(X))

        if self.rf and self.rf.is_fitted:
            scores += self.w_rf * self.rf.attack_probability(X)

        if self.xgb and self.xgb.is_fitted:
            scores += self.w_xgb * self.xgb.attack_probability(X)

        if self.iso and self.iso.is_fitted:
            scores += self.w_if * self.iso.anomaly_score(X)

        if self.ae and self.ae.is_fitted:
            scores += self.w_ae * self.ae.normalized_score(X)

        # Re-normalize by actual total weight (in case some models are disabled)
        total_weight = (
            (self.w_rf if (self.rf and self.rf.is_fitted) else 0)
            + (self.w_xgb if (self.xgb and self.xgb.is_fitted) else 0)
            + (self.w_if if (self.iso and self.iso.is_fitted) else 0)
            + (self.w_ae if (self.ae and self.ae.is_fitted) else 0)
        )
        return scores / max(total_weight, 1e-8)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns binary prediction: 1=attack, 0=normal."""
        scores = self._compute_scores(X)
        return (scores >= self.threshold).astype(int)

    def predict_with_score(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (predictions, scores) for each sample."""
        scores = self._compute_scores(X)
        return (scores >= self.threshold).astype(int), scores

    def predict_single(self, x: np.ndarray) -> dict:
        """
        Full prediction for a single feature vector.
        Returns a rich dict with all model scores.
        """
        x2d = x.reshape(1, -1)
        ensemble_score = float(self._compute_scores(x2d)[0])
        is_attack = ensemble_score >= self.threshold

        breakdown = {}
        if self.rf and self.rf.is_fitted:
            breakdown["random_forest"] = float(self.rf.attack_probability(x2d)[0])
        if self.xgb and self.xgb.is_fitted:
            breakdown["xgboost"] = float(self.xgb.attack_probability(x2d)[0])
        if self.iso and self.iso.is_fitted:
            breakdown["isolation_forest"] = float(self.iso.anomaly_score(x2d)[0])
        if self.ae and self.ae.is_fitted:
            breakdown["autoencoder"] = float(self.ae.normalized_score(x2d)[0])

        return {
            "is_attack": bool(is_attack),
            "ensemble_score": ensemble_score,
            "threshold": self.threshold,
            "model_scores": breakdown,
        }

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        y_pred, y_score = self.predict_with_score(X)
        metrics = {
            "f1": f1_score(y, y_pred, zero_division=0),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y, y_score),
            "threshold": self.threshold,
        }
        logger.info(
            f"[Ensemble] F1={metrics['f1']:.4f} | "
            f"Precision={metrics['precision']:.4f} | "
            f"Recall={metrics['recall']:.4f} | "
            f"AUC={metrics['roc_auc']:.4f}"
        )
        return metrics

    def save(self, base_dir: str = "data/processed/models") -> None:
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        if self.rf:
            self.rf.save(f"{base_dir}/random_forest.pkl")
        if self.xgb:
            self.xgb.save(f"{base_dir}/xgboost.pkl")
        if self.iso:
            self.iso.save(f"{base_dir}/isolation_forest.pkl")
        if self.ae:
            self.ae.save(f"{base_dir}/autoencoder.pt")
        # Save ensemble metadata
        meta = {
            "w_rf": self.w_rf, "w_xgb": self.w_xgb,
            "w_if": self.w_if, "w_ae": self.w_ae,
            "threshold": self.threshold,
        }
        joblib.dump(meta, f"{base_dir}/ensemble_meta.pkl")
        logger.success(f"Ensemble saved → {base_dir}")

    @staticmethod
    def load(base_dir: str = "data/processed/models") -> "EnsembleDetector":
        meta = joblib.load(f"{base_dir}/ensemble_meta.pkl")
        rf = RandomForestDetector.load(f"{base_dir}/random_forest.pkl") if Path(f"{base_dir}/random_forest.pkl").exists() else None
        xgb = XGBoostDetector.load(f"{base_dir}/xgboost.pkl") if Path(f"{base_dir}/xgboost.pkl").exists() else None
        iso = IsolationForestDetector.load(f"{base_dir}/isolation_forest.pkl") if Path(f"{base_dir}/isolation_forest.pkl").exists() else None
        ae = AutoencoderDetector.load(f"{base_dir}/autoencoder.pt") if Path(f"{base_dir}/autoencoder.pt").exists() else None
        ens = EnsembleDetector(rf=rf, xgb=xgb, iso=iso, ae=ae)
        ens.w_rf = meta["w_rf"]
        ens.w_xgb = meta["w_xgb"]
        ens.w_if = meta["w_if"]
        ens.w_ae = meta["w_ae"]
        ens.threshold = meta["threshold"]
        ens.is_fitted = True
        logger.success(f"Ensemble loaded ← {base_dir}")
        return ens
