"""
Drift Detector — Monitors for data drift and model performance degradation.
Uses statistical tests (KS test + PSI) to detect distribution shifts.
"""

import time
import numpy as np
import pandas as pd
from collections import deque
from scipy import stats
from loguru import logger
from pathlib import Path
import yaml
import json


def load_config() -> dict:
    with open("configs/healing_config.yaml") as f:
        return yaml.safe_load(f)


def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index — measures distribution shift."""
    eps = 1e-8

    def _pct(x, buckets):
        breakpoints = np.linspace(np.percentile(x, 0), np.percentile(x, 100), buckets + 1)
        counts = np.histogram(x, bins=breakpoints)[0]
        return (counts + eps) / (len(x) + eps * buckets)

    exp_pct = _pct(expected, buckets)
    act_pct = _pct(actual, buckets)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


class DriftDetector:
    """
    Monitors incoming traffic data for concept drift.
    Maintains a sliding window of recent predictions and compares
    feature distributions against the training baseline.
    """

    def __init__(self, reference_data: np.ndarray | None = None, config: dict | None = None):
        self.cfg = (config or load_config())["drift_detection"]
        self.reference_data = reference_data  # Training set features
        self.window = deque(maxlen=max(self.cfg["min_samples_for_check"] * 2, 500))
        self.performance_window = deque(maxlen=200)  # (y_true, y_pred) pairs
        self._drift_history: list[dict] = []
        self._last_check = 0.0

    def set_reference(self, X: np.ndarray) -> None:
        """Set the training data baseline for drift comparison."""
        self.reference_data = X
        logger.info(f"DriftDetector reference set: {X.shape}")

    def add_sample(self, features: np.ndarray, y_true: int | None = None, y_pred: int | None = None) -> None:
        """Add a new incoming sample to the monitoring window."""
        self.window.append(features)
        if y_true is not None and y_pred is not None:
            self.performance_window.append((y_true, y_pred))

    def add_batch(self, X: np.ndarray, y_true: np.ndarray | None = None, y_pred: np.ndarray | None = None) -> None:
        """Add a batch of samples."""
        for i, x in enumerate(X):
            yt = int(y_true[i]) if y_true is not None else None
            yp = int(y_pred[i]) if y_pred is not None else None
            self.add_sample(x, yt, yp)

    def check_feature_drift(self) -> dict:
        """Run KS test and PSI for each feature between reference and window."""
        if self.reference_data is None:
            return {"drift_detected": False, "reason": "no_reference"}

        n_window = len(self.window)
        if n_window < self.cfg["min_samples_for_check"]:
            return {"drift_detected": False, "reason": f"insufficient_samples ({n_window})"}

        window_data = np.array(list(self.window))
        n_features = min(self.reference_data.shape[1], window_data.shape[1])

        ks_results = []
        psi_results = []
        drifted_features = []

        for i in range(n_features):
            ref_col = self.reference_data[:, i]
            win_col = window_data[:, i]

            # KS test
            ks_stat, p_value = stats.ks_2samp(ref_col, win_col)
            ks_results.append(p_value)

            # PSI
            psi = compute_psi(ref_col, win_col)
            psi_results.append(psi)

            if p_value < self.cfg["drift_threshold"] or psi > self.cfg["psi_threshold"]:
                drifted_features.append({
                    "feature_idx": i,
                    "ks_p_value": round(p_value, 6),
                    "psi": round(psi, 6),
                })

        drift_detected = len(drifted_features) > 0
        result = {
            "drift_detected": drift_detected,
            "n_drifted_features": len(drifted_features),
            "n_total_features": n_features,
            "drifted_features": drifted_features,
            "mean_ks_p_value": float(np.mean(ks_results)),
            "max_psi": float(np.max(psi_results)),
            "window_size": n_window,
            "timestamp": time.time(),
        }

        if drift_detected:
            logger.warning(
                f"🌊 Data drift detected! {len(drifted_features)}/{n_features} features drifted"
            )
            self._drift_history.append(result)

        self._last_check = time.time()
        return result

    def check_performance_drift(self) -> dict:
        """Check if model F1 has dropped below acceptable threshold."""
        if len(self.performance_window) < 50:
            return {"drift_detected": False, "reason": "insufficient_labels"}

        from sklearn.metrics import f1_score
        y_true = [p[0] for p in self.performance_window]
        y_pred = [p[1] for p in self.performance_window]
        f1 = f1_score(y_true, y_pred, zero_division=0)

        threshold = self.cfg["performance_drop_threshold"]
        # Drift if F1 falls below the configured minimum acceptable floor.
        # e.g. threshold=0.05 means we need F1 >= 0.05; anything below triggers retraining.
        drift = f1 < threshold

        return {
            "drift_detected": drift,
            "current_f1": round(f1, 4),
            "min_acceptable_f1": round(1.0 - threshold, 4),
            "window_size": len(self.performance_window),
            "timestamp": time.time(),
        }

    def full_check(self) -> dict:
        """Run all drift checks and return combined result."""
        feature_result = self.check_feature_drift()
        perf_result = self.check_performance_drift()

        any_drift = feature_result.get("drift_detected") or perf_result.get("drift_detected")

        return {
            "drift_detected": any_drift,
            "feature_drift": feature_result,
            "performance_drift": perf_result,
            "drift_history_count": len(self._drift_history),
            "timestamp": time.time(),
        }

    def get_drift_history(self) -> list[dict]:
        return self._drift_history
