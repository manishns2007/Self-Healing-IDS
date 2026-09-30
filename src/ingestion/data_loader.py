"""
Data Loader — Downloads and loads the NSL-KDD dataset.
NSL-KDD is a cleaned version of the classic KDD Cup 1999 dataset.
"""

import os
import urllib.request
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

# ── NSL-KDD column definitions ──────────────────────────────────────────────
NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"
]

# ── Attack category mapping ──────────────────────────────────────────────────
ATTACK_MAP = {
    "normal": "normal",
    # DoS
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos",
    "smurf": "dos", "teardrop": "dos", "apache2": "dos", "udpstorm": "dos",
    "processtable": "dos", "worm": "dos",
    # Probe
    "ipsweep": "probe", "nmap": "probe", "portsweep": "probe",
    "satan": "probe", "mscan": "probe", "saint": "probe",
    # R2L
    "ftp_write": "r2l", "guess_passwd": "r2l", "imap": "r2l", "multihop": "r2l",
    "phf": "r2l", "spy": "r2l", "warezclient": "r2l", "warezmaster": "r2l",
    "sendmail": "r2l", "named": "r2l", "snmpgetattack": "r2l", "snmpguess": "r2l",
    "xlock": "r2l", "xsnoop": "r2l", "httptunnel": "r2l",
    # U2R
    "buffer_overflow": "u2r", "loadmodule": "u2r", "perl": "u2r",
    "rootkit": "u2r", "sqlattack": "u2r", "xterm": "u2r", "ps": "u2r",
}

DOWNLOAD_URLS = {
    "train": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt",
    "test": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt",
}

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def download_nslkdd(force: bool = False) -> dict[str, Path]:
    """Download NSL-KDD train and test files if not already present."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, url in DOWNLOAD_URLS.items():
        dest = RAW_DIR / f"nslkdd_{split}.txt"
        paths[split] = dest
        if dest.exists() and not force:
            logger.info(f"[{split}] Already exists: {dest}")
            continue
        logger.info(f"[{split}] Downloading from {url} ...")
        try:
            urllib.request.urlretrieve(url, dest)
            logger.success(f"[{split}] Saved to {dest}")
        except Exception as e:
            logger.error(f"[{split}] Download failed: {e}")
            raise
    return paths


def load_nslkdd(split: str = "train") -> pd.DataFrame:
    """
    Load NSL-KDD data into a DataFrame.
    Adds binary label (0=normal, 1=attack) and attack category.
    """
    path = RAW_DIR / f"nslkdd_{split}.txt"
    if not path.exists():
        logger.warning(f"{path} not found. Downloading ...")
        download_nslkdd()

    df = pd.read_csv(path, header=None, names=NSL_KDD_COLUMNS)
    df.drop(columns=["difficulty"], inplace=True)

    # Normalize label
    df["label"] = df["label"].str.strip().str.lower()
    df["attack_category"] = df["label"].map(ATTACK_MAP).fillna("unknown")
    df["is_attack"] = (df["label"] != "normal").astype(int)

    logger.info(
        f"[{split}] Loaded {len(df):,} records | "
        f"Attacks: {df['is_attack'].sum():,} | "
        f"Normal: {(~df['is_attack'].astype(bool)).sum():,}"
    )
    return df


def load_or_generate_synthetic(n_samples: int = 50000) -> pd.DataFrame:
    """
    Fallback: generate synthetic network traffic data if dataset download fails.
    Useful for quick testing without internet access.
    """
    logger.warning("Generating synthetic dataset as fallback ...")
    np.random.seed(42)

    n_normal = int(n_samples * 0.7)
    n_attack = n_samples - n_normal

    def _make_records(n, is_attack):
        base = {
            "duration": np.random.exponential(5, n),
            "protocol_type": np.random.choice(["tcp", "udp", "icmp"], n),
            "service": np.random.choice(["http", "ftp", "smtp", "ssh", "dns"], n),
            "flag": np.random.choice(["SF", "S0", "REJ", "RSTO"], n),
            "src_bytes": np.random.exponential(5000, n) if not is_attack else np.random.exponential(50000, n),
            "dst_bytes": np.random.exponential(3000, n),
            "land": np.zeros(n, dtype=int),
            "wrong_fragment": np.random.poisson(0.01, n),
            "urgent": np.zeros(n, dtype=int),
            "hot": np.random.poisson(0.5, n),
            "num_failed_logins": np.zeros(n, dtype=int),
            "logged_in": np.ones(n, dtype=int) if not is_attack else np.random.randint(0, 2, n),
            "num_compromised": np.zeros(n, dtype=int),
            "root_shell": np.zeros(n, dtype=int),
            "su_attempted": np.zeros(n, dtype=int),
            "num_root": np.zeros(n, dtype=int),
            "num_file_creations": np.zeros(n, dtype=int),
            "num_shells": np.zeros(n, dtype=int),
            "num_access_files": np.zeros(n, dtype=int),
            "num_outbound_cmds": np.zeros(n, dtype=int),
            "is_host_login": np.zeros(n, dtype=int),
            "is_guest_login": np.zeros(n, dtype=int),
            "count": np.random.poisson(10, n) if not is_attack else np.random.poisson(500, n),
            "srv_count": np.random.poisson(10, n),
            "serror_rate": np.random.uniform(0, 0.1, n) if not is_attack else np.random.uniform(0.5, 1.0, n),
            "srv_serror_rate": np.random.uniform(0, 0.1, n),
            "rerror_rate": np.random.uniform(0, 0.1, n),
            "srv_rerror_rate": np.random.uniform(0, 0.1, n),
            "same_srv_rate": np.random.uniform(0.8, 1.0, n),
            "diff_srv_rate": np.random.uniform(0, 0.2, n),
            "srv_diff_host_rate": np.random.uniform(0, 0.1, n),
            "dst_host_count": np.random.randint(1, 255, n),
            "dst_host_srv_count": np.random.randint(1, 255, n),
            "dst_host_same_srv_rate": np.random.uniform(0.5, 1.0, n),
            "dst_host_diff_srv_rate": np.random.uniform(0, 0.3, n),
            "dst_host_same_src_port_rate": np.random.uniform(0, 1.0, n),
            "dst_host_srv_diff_host_rate": np.random.uniform(0, 0.3, n),
            "dst_host_serror_rate": np.random.uniform(0, 0.1, n),
            "dst_host_srv_serror_rate": np.random.uniform(0, 0.1, n),
            "dst_host_rerror_rate": np.random.uniform(0, 0.1, n),
            "dst_host_srv_rerror_rate": np.random.uniform(0, 0.1, n),
            "is_attack": int(is_attack),
            "attack_category": "normal" if not is_attack else np.random.choice(["dos", "probe", "r2l", "u2r"], n),
        }
        return pd.DataFrame(base)

    df = pd.concat([_make_records(n_normal, False), _make_records(n_attack, True)], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    logger.success(f"Synthetic dataset: {len(df):,} records")
    return df


if __name__ == "__main__":
    download_nslkdd()
    df_train = load_nslkdd("train")
    df_test = load_nslkdd("test")
    print(df_train.head())
    print(df_train["attack_category"].value_counts())
