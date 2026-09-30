"""
Anomaly Detection Models — Isolation Forest and Autoencoder.
Used to detect unknown/zero-day attacks without labeled data.
"""

import numpy as np
import joblib
import torch
import torch.nn as nn
from pathlib import Path
from loguru import logger
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
import yaml


def load_config() -> dict:
    with open("configs/model_config.yaml") as f:
        return yaml.safe_load(f)


# ── Isolation Forest ─────────────────────────────────────────────────────────

class IsolationForestDetector:
    """
    Unsupervised anomaly detector using Isolation Forest.
    Trained on normal traffic only; anomalies = intrusions.
    """

    def __init__(self, config: dict | None = None):
        cfg = (config or load_config())["isolation_forest"]
        self.model = IsolationForest(
            n_estimators=cfg["n_estimators"],
            contamination=cfg["contamination"],
            max_samples=cfg["max_samples"],
            random_state=cfg["random_state"],
            n_jobs=-1,
        )
        self.is_fitted = False
        self.model_name = "isolation_forest"

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "IsolationForestDetector":
        """Train on all data or normal-only subset."""
        if y is not None:
            X_normal = X[y == 0]
            logger.info(f"Training IsolationForest on {X_normal.shape} (normal only) ...")
            self.model.fit(X_normal)
        else:
            logger.info(f"Training IsolationForest on {X.shape} ...")
            self.model.fit(X)
        self.is_fitted = True
        logger.success("IsolationForest training complete")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns 1 (attack) or 0 (normal)."""
        raw = self.model.predict(X)  # -1 = anomaly, 1 = normal
        return (raw == -1).astype(int)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Returns anomaly score ∈ [0, 1]; higher = more anomalous."""
        scores = -self.model.score_samples(X)  # negate: higher = more anomalous
        # Normalize to [0, 1]
        s_min, s_max = scores.min(), scores.max()
        if s_max > s_min:
            return (scores - s_min) / (s_max - s_min)
        return np.zeros_like(scores)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        y_pred = self.predict(X)
        y_score = self.anomaly_score(X)
        from sklearn.metrics import f1_score, precision_score, recall_score
        metrics = {
            "f1": f1_score(y, y_pred, zero_division=0),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y, y_score),
        }
        logger.info(f"[IsolationForest] F1={metrics['f1']:.4f} | AUC={metrics['roc_auc']:.4f}")
        return metrics

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "IsolationForestDetector":
        return joblib.load(path)


# ── Autoencoder ──────────────────────────────────────────────────────────────

class _AutoencoderNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout_rate: float):
        super().__init__()
        # Encoder
        enc_layers = []
        prev = input_dim
        for h in hidden_dims[: len(hidden_dims) // 2 + 1]:
            enc_layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout_rate)]
            prev = h
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers = []
        for h in hidden_dims[len(hidden_dims) // 2 + 1:] + [input_dim]:
            dec_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        dec_layers[-1] = nn.Linear(hidden_dims[len(hidden_dims) // 2], input_dim)  # no ReLU at output
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        recon = self.forward(x)
        return ((x - recon) ** 2).mean(dim=1)


class AutoencoderDetector:
    """
    Deep Autoencoder for anomaly detection.
    Trained on normal traffic; high reconstruction error → attack.
    """

    def __init__(self, input_dim: int | None = None, config: dict | None = None):
        self.cfg = (config or load_config())["autoencoder"]
        self.input_dim = input_dim
        self.net: _AutoencoderNet | None = None
        self.threshold: float | None = self.cfg["reconstruction_threshold"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_fitted = False
        self.model_name = "autoencoder"

    def _build_net(self, input_dim: int) -> None:
        self.input_dim = input_dim
        self.net = _AutoencoderNet(
            input_dim=input_dim,
            hidden_dims=self.cfg["hidden_dims"],
            dropout_rate=self.cfg["dropout_rate"],
        ).to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "AutoencoderDetector":
        """Train on normal traffic only. y is used to filter normals if provided."""
        self._build_net(X.shape[1])
        X_normal = X[y == 0] if y is not None else X
        logger.info(f"Training Autoencoder on {X_normal.shape} (normal only, device={self.device}) ...")

        dataset = torch.tensor(X_normal, dtype=torch.float32)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.cfg["batch_size"], shuffle=True
        )
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.cfg["learning_rate"])

        self.net.train()
        for epoch in range(self.cfg["epochs"]):
            total_loss = 0.0
            for batch in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                recon = self.net(batch)
                loss = nn.MSELoss()(recon, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                logger.debug(f"  Epoch {epoch+1}/{self.cfg['epochs']} | loss={total_loss/len(loader):.6f}")

        # Compute threshold from reconstruction errors on normal data
        self.net.eval()
        with torch.no_grad():
            errors = self.net.reconstruction_error(dataset.to(self.device)).cpu().numpy()
        self.threshold = float(np.percentile(errors, 95))
        logger.success(f"Autoencoder trained | threshold={self.threshold:.6f}")
        self.is_fitted = True
        return self

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            errors = self.net.reconstruction_error(tensor).cpu().numpy()
        return errors

    def predict(self, X: np.ndarray) -> np.ndarray:
        errors = self.anomaly_score(X)
        return (errors > self.threshold).astype(int)

    def normalized_score(self, X: np.ndarray) -> np.ndarray:
        """Normalize reconstruction error to [0, 1] for ensemble."""
        scores = self.anomaly_score(X)
        s_min, s_max = 0.0, self.threshold * 3  # cap at 3x threshold
        return np.clip((scores - s_min) / (s_max - s_min + 1e-8), 0.0, 1.0)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        from sklearn.metrics import f1_score, precision_score, recall_score
        y_pred = self.predict(X)
        y_score = self.normalized_score(X)
        metrics = {
            "f1": f1_score(y, y_pred, zero_division=0),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y, y_score),
        }
        logger.info(f"[Autoencoder] F1={metrics['f1']:.4f} | AUC={metrics['roc_auc']:.4f}")
        return metrics

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "cfg": self.cfg,
            "input_dim": self.input_dim,
            "threshold": self.threshold,
            "state_dict": self.net.state_dict() if self.net else None,
        }
        torch.save(state, path)
        logger.success(f"Autoencoder saved → {path}")

    @staticmethod
    def load(path: str) -> "AutoencoderDetector":
        state = torch.load(path, map_location="cpu")
        det = AutoencoderDetector(input_dim=state["input_dim"], config={"autoencoder": state["cfg"]})
        det._build_net(state["input_dim"])
        det.net.load_state_dict(state["state_dict"])
        det.threshold = state["threshold"]
        det.is_fitted = True
        logger.info(f"Autoencoder loaded ← {path}")
        return det
