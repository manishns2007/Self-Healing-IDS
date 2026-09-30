"""
Feature Engineering — Transforms raw NSL-KDD records into ML-ready features.
Handles encoding, scaling, and feature selection.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from loguru import logger
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder

PROCESSED_DIR = Path("data/processed")

# Categorical columns to encode
CATEGORICAL_COLS = ["protocol_type", "service", "flag"]

# Columns to drop before training (non-features)
DROP_COLS = ["label", "is_attack", "attack_category", "_timestamp", "_simulated", "_injected"]

# Known categories for each column (from full NSL-KDD dataset)
KNOWN_CATEGORIES = {
    "protocol_type": ["icmp", "tcp", "udp"],
    "service": [
        "aol", "auth", "bgp", "courier", "csnet_ns", "ctf", "daytime", "discard",
        "domain", "domain_u", "echo", "eco_i", "ecr_i", "efs", "exec", "finger",
        "ftp", "ftp_data", "gopher", "harvest", "hostnames", "http", "http_2784",
        "http_443", "http_8001", "imap4", "IRC", "iso_tsap", "klogin", "kshell",
        "ldap", "link", "login", "mtp", "name", "netbios_dgm", "netbios_ns",
        "netbios_ssn", "netstat", "nnsp", "nntp", "ntp_u", "other", "pm_dump",
        "pop_2", "pop_3", "printer", "private", "red_i", "remote_job", "rje",
        "shell", "smtp", "sql_net", "ssh", "sunrpc", "supdup", "systat", "telnet",
        "tftp_u", "tim_i", "time", "urh_i", "urp_i", "uucp", "uucp_path",
        "vmnet", "whois", "X11", "Z39_50",
    ],
    "flag": ["OTH", "REJ", "RSTO", "RSTOS0", "RSTR", "S0", "S1", "S2", "S3", "SF", "SH"],
}


class FeatureEngineer:
    """
    Full preprocessing pipeline: encode → engineer → scale.
    Designed to be fit on training data and reused for inference.
    """

    def __init__(self):
        self.ordinal_encoder = OrdinalEncoder(
            categories=[KNOWN_CATEGORIES[c] for c in CATEGORICAL_COLS],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        self.scaler = StandardScaler()
        self.feature_names_: list[str] = []
        self._fitted = False

    def _drop_non_features(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = [c for c in DROP_COLS if c in df.columns]
        return df.drop(columns=cols_to_drop)

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        df = df.copy()
        for col in CATEGORICAL_COLS:
            if col not in df.columns:
                df[col] = "other"  # fallback for missing columns

        if fit:
            encoded = self.ordinal_encoder.fit_transform(df[CATEGORICAL_COLS])
        else:
            encoded = self.ordinal_encoder.transform(df[CATEGORICAL_COLS])

        encoded_df = pd.DataFrame(
            encoded, columns=CATEGORICAL_COLS, index=df.index
        )
        df[CATEGORICAL_COLS] = encoded_df
        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features that improve detection accuracy."""
        df = df.copy()

        # Traffic ratios
        df["bytes_ratio"] = np.where(
            df["dst_bytes"] > 0, df["src_bytes"] / (df["dst_bytes"] + 1), 0
        )

        # Error rate compound
        df["total_error_rate"] = (
            df.get("serror_rate", 0) + df.get("rerror_rate", 0)
        ) / 2.0

        # Host diversity index
        df["host_srv_diversity"] = df.get("dst_host_diff_srv_rate", 0) * df.get("dst_host_count", 0)

        # Connection intensity score
        df["conn_intensity"] = (
            df.get("count", 0) * df.get("same_srv_rate", 0)
        )

        return df

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit on training data and return transformed features."""
        df = self._drop_non_features(df)
        df = self._encode_categoricals(df, fit=True)
        df = self._engineer_features(df)
        self.feature_names_ = list(df.columns)
        X = df.values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        X = self.scaler.fit_transform(X)
        self._fitted = True
        logger.info(f"FeatureEngineer fitted: {X.shape[1]} features on {X.shape[0]:,} samples")
        return X

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using the fitted encoder/scaler."""
        if not self._fitted:
            raise RuntimeError("FeatureEngineer must be fitted before calling transform().")
        df = self._drop_non_features(df)
        df = self._encode_categoricals(df, fit=False)
        df = self._engineer_features(df)

        # Align columns with training
        for col in self.feature_names_:
            if col not in df.columns:
                df[col] = 0.0
        df = df[self.feature_names_]

        X = df.values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        return self.scaler.transform(X)

    def transform_record(self, record: dict) -> np.ndarray:
        """Transform a single record dict for real-time inference."""
        df = pd.DataFrame([record])
        return self.transform(df)

    def save(self, path: str = "data/processed/feature_engineer.pkl") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.success(f"FeatureEngineer saved → {path}")

    @staticmethod
    def load(path: str = "data/processed/feature_engineer.pkl") -> "FeatureEngineer":
        fe = joblib.load(path)
        logger.info(f"FeatureEngineer loaded ← {path}")
        return fe

    @property
    def n_features(self) -> int:
        return len(self.feature_names_)
