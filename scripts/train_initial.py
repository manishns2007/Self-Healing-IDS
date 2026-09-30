"""
Initial training script — Run this first to download data and train the ensemble model.
Usage:
    python scripts/train_initial.py              # Use NSL-KDD dataset (auto-download)
    python scripts/train_initial.py --synthetic  # Use synthetic data (no internet needed)
"""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from src.ingestion.data_loader import download_nslkdd
from src.training.trainer import train

def main():
    parser = argparse.ArgumentParser(description="Train the Self-Healing IDS model")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data instead of NSL-KDD")
    parser.add_argument("--no-register", action="store_true", help="Skip MLflow model registration")
    parser.add_argument("--no-autoencoder", action="store_true", help="Skip Autoencoder training (much faster, recommended for first run)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Self-Healing IDS — Initial Training")
    logger.info("=" * 60)

    if not args.synthetic:
        logger.info("Downloading NSL-KDD dataset ...")
        try:
            download_nslkdd()
        except Exception as e:
            logger.warning(f"Download failed: {e}. Falling back to synthetic data.")
            args.synthetic = True

    result = train(
        use_synthetic=args.synthetic,
        register_model=not args.no_register,
        run_name="initial_training",
    )

    logger.success("=" * 60)
    logger.success(f"Training complete!")
    logger.success(f"  Ensemble F1    : {result['metrics']['f1']:.4f}")
    logger.success(f"  Ensemble AUC   : {result['metrics']['roc_auc']:.4f}")
    logger.success(f"  Precision      : {result['metrics']['precision']:.4f}")
    logger.success(f"  Recall         : {result['metrics']['recall']:.4f}")
    logger.success(f"  Training time  : {result['training_time_s']:.1f}s")
    logger.success(f"  Model saved to : {result['model_dir']}")
    logger.success("=" * 60)
    logger.success("Next step: python -m uvicorn src.api.main:app --reload")


if __name__ == "__main__":
    main()
