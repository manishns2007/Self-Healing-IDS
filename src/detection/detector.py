"""
Detector — Real-time inference engine.
Loads the ensemble model and provides single/batch prediction.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from typing import Optional
import time

from src.models.ensemble import EnsembleDetector
from src.preprocessing.feature_engineering import FeatureEngineer

MODEL_DIR = "data/processed/models"
FE_PATH = "data/processed/feature_engineer.pkl"

# Attack category guesser based on raw features
def guess_attack_category(record: dict, is_attack: bool) -> str:
    if not is_attack:
        return "normal"
    count = record.get("count", 0)
    src_bytes = record.get("src_bytes", 0)
    srv_serror_rate = record.get("srv_serror_rate", 0)
    logged_in = record.get("logged_in", 1)
    root_shell = record.get("root_shell", 0)

    if count > 300 or srv_serror_rate > 0.7:
        return "dos"
    elif root_shell > 0 or record.get("su_attempted", 0) > 0:
        return "u2r"
    elif src_bytes < 100 and logged_in == 0:
        return "probe"
    else:
        return "r2l"


class Detector:
    """
    Runtime detector: load once, predict many.
    Thread-safe for use in FastAPI.
    """

    _instance: Optional["Detector"] = None

    def __init__(self):
        self.ensemble: Optional[EnsembleDetector] = None
        self.fe: Optional[FeatureEngineer] = None
        self._loaded = False
        self._load_time: Optional[float] = None
        self._predictions_count = 0
        self._attacks_detected = 0

    @classmethod
    def get_instance(cls) -> "Detector":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self, model_dir: str = MODEL_DIR, fe_path: str = FE_PATH) -> bool:
        """Load models from disk. Returns True on success."""
        try:
            self.fe = FeatureEngineer.load(fe_path)
            self.ensemble = EnsembleDetector.load(model_dir)
            self._loaded = True
            self._load_time = time.time()
            logger.success(f"Detector loaded: {self.ensemble.rf.model_name} ensemble")
            return True
        except Exception as e:
            logger.error(f"Detector load failed: {e}")
            return False

    @property
    def is_ready(self) -> bool:
        return self._loaded and self.ensemble is not None and self.fe is not None

    def predict(self, record: dict) -> dict:
        """Single-record prediction. Returns full prediction dict."""
        if not self.is_ready:
            raise RuntimeError("Detector not loaded. Call detector.load() first.")

        start = time.perf_counter()
        X = self.fe.transform_record(record)
        result = self.ensemble.predict_single(X)
        latency_ms = (time.perf_counter() - start) * 1000

        self._predictions_count += 1
        if result["is_attack"]:
            self._attacks_detected += 1

        # Enrich result
        result["attack_category"] = guess_attack_category(record, result["is_attack"])
        result["latency_ms"] = round(latency_ms, 2)
        result["prediction_id"] = self._predictions_count

        return result

    def predict_batch(self, records: list[dict]) -> list[dict]:
        """Batch prediction. More efficient than calling predict() in a loop."""
        if not self.is_ready:
            raise RuntimeError("Detector not loaded.")

        df = pd.DataFrame(records)
        X = self.fe.transform(df)
        preds, scores = self.ensemble.predict_with_score(X)

        results = []
        for i, (pred, score) in enumerate(zip(preds, scores)):
            is_attack = bool(pred)
            self._predictions_count += 1
            if is_attack:
                self._attacks_detected += 1
            results.append({
                "is_attack": is_attack,
                "ensemble_score": float(score),
                "threshold": self.ensemble.threshold,
                "attack_category": guess_attack_category(records[i], is_attack),
                "prediction_id": self._predictions_count,
            })
        return results

    @property
    def stats(self) -> dict:
        return {
            "loaded": self._loaded,
            "load_time": self._load_time,
            "total_predictions": self._predictions_count,
            "total_attacks_detected": self._attacks_detected,
            "attack_rate": self._attacks_detected / max(self._predictions_count, 1),
        }
