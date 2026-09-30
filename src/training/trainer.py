"""
Trainer — MLflow-tracked training pipeline.
Handles train/validation split, model fitting, metric logging, and model registration.
"""

import time
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from loguru import logger
from sklearn.model_selection import train_test_split

from src.ingestion.data_loader import load_nslkdd, load_or_generate_synthetic
from src.preprocessing.feature_engineering import FeatureEngineer
from src.models.ensemble import EnsembleDetector

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "Self-Healing-IDS"
MODEL_DIR = "data/processed/models"
FE_PATH = "data/processed/feature_engineer.pkl"


def train(
    use_synthetic: bool = False,
    register_model: bool = True,
    run_name: str | None = None,
) -> dict:
    """
    Full training pipeline with MLflow tracking.

    Returns:
        dict: Evaluation metrics and run ID.
    """
    logger.info("=" * 60)
    logger.info("Starting training pipeline ...")
    start_time = time.time()

    # ── 1. Load data ─────────────────────────────────────────────
    if use_synthetic:
        df = load_or_generate_synthetic()
        df_test = df.sample(frac=0.2, random_state=42)
        df_train = df.drop(df_test.index)
    else:
        try:
            df_train = load_nslkdd("train")
            df_test = load_nslkdd("test")
        except Exception as e:
            logger.warning(f"NSL-KDD load failed ({e}), using synthetic data")
            df = load_or_generate_synthetic()
            df_test = df.sample(frac=0.2, random_state=42)
            df_train = df.drop(df_test.index)

    y_train = df_train["is_attack"].values
    y_test = df_test["is_attack"].values

    # ── 2. Feature engineering ────────────────────────────────────
    fe = FeatureEngineer()
    X_train = fe.fit_transform(df_train)
    X_test = fe.transform(df_test)
    fe.save(FE_PATH)

    # Validation split from training
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )

    # ── 3. MLflow run ─────────────────────────────────────────────
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    except Exception:
        pass  # Use local file tracking if server not available
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name or f"training_{int(time.time())}") as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run: {run_id}")

        # Log params
        with open("configs/model_config.yaml") as f:
            cfg = yaml.safe_load(f)
        mlflow.log_params({
            "n_train": len(X_tr),
            "n_val": len(X_val),
            "n_test": len(X_test),
            "n_features": fe.n_features,
            "rf_n_estimators": cfg["random_forest"]["n_estimators"],
            "xgb_n_estimators": cfg["xgboost"]["n_estimators"],
            "if_contamination": cfg["isolation_forest"]["contamination"],
            "ae_epochs": cfg["autoencoder"]["epochs"],
        })

        # ── 4. Train ensemble ─────────────────────────────────────
        ensemble = EnsembleDetector()
        ensemble.fit(X_tr, y_tr, X_val, y_val)

        # ── 5. Evaluate ───────────────────────────────────────────
        logger.info("Evaluating on test set ...")
        metrics = ensemble.evaluate(X_test, y_test)

        # Individual model metrics
        rf_metrics = ensemble.rf.evaluate(X_test, y_test)
        xgb_metrics = ensemble.xgb.evaluate(X_test, y_test)
        iso_metrics = ensemble.iso.evaluate(X_test, y_test)
        ae_metrics = ensemble.ae.evaluate(X_test, y_test)

        # Log all metrics
        mlflow.log_metrics({
            "ensemble_f1": metrics["f1"],
            "ensemble_precision": metrics["precision"],
            "ensemble_recall": metrics["recall"],
            "ensemble_roc_auc": metrics["roc_auc"],
            "rf_f1": rf_metrics["f1"],
            "xgb_f1": xgb_metrics["f1"],
            "if_f1": iso_metrics["f1"],
            "ae_f1": ae_metrics["f1"],
            "training_time_s": time.time() - start_time,
        })

        # ── 6. Save models ────────────────────────────────────────
        ensemble.save(MODEL_DIR)

        # Log artifacts
        mlflow.log_artifacts(MODEL_DIR, artifact_path="models")
        mlflow.log_artifact(FE_PATH, artifact_path="preprocessor")
        mlflow.log_artifact("configs/model_config.yaml", artifact_path="configs")

        # ── 7. Register model ─────────────────────────────────────
        if register_model:
            try:
                mlflow.sklearn.log_model(
                    ensemble.rf.model,
                    artifact_path="rf_model",
                    registered_model_name="IDS-RandomForest",
                )
                logger.success("Model registered in MLflow Model Registry")
            except Exception as e:
                logger.warning(f"Model registration skipped: {e}")

        duration = time.time() - start_time
        logger.success(
            f"Training complete in {duration:.1f}s | "
            f"Ensemble F1={metrics['f1']:.4f} | AUC={metrics['roc_auc']:.4f}"
        )

        return {
            "run_id": run_id,
            "metrics": metrics,
            "training_time_s": duration,
            "model_dir": MODEL_DIR,
            "fe_path": FE_PATH,
        }


if __name__ == "__main__":
    result = train(use_synthetic=False)
    print(result)
