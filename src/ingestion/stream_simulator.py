"""
Stream Simulator — Simulates live network traffic for real-time IDS evaluation.
Replays NSL-KDD records as a streaming source with configurable rate.
"""

import time
import queue
import threading
import random
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from typing import Iterator, Optional

from src.ingestion.data_loader import load_nslkdd, load_or_generate_synthetic


class TrafficStreamSimulator:
    """
    Simulates a real-time network traffic stream.
    Supports normal playback, attack injection, and drift simulation.
    """

    def __init__(
        self,
        records_per_second: float = 10.0,
        attack_burst_probability: float = 0.05,
        drift_after_n_records: Optional[int] = None,
        use_synthetic: bool = False,
    ):
        self.rps = records_per_second
        self.attack_burst_prob = attack_burst_probability
        self.drift_after = drift_after_n_records
        self.use_synthetic = use_synthetic
        self._stop_event = threading.Event()
        self._stream_queue: queue.Queue = queue.Queue(maxsize=1000)
        self._records_sent = 0
        self._thread: Optional[threading.Thread] = None

        logger.info(f"TrafficStreamSimulator: {rps=:.1f} rec/s, attack_burst_prob={attack_burst_probability:.2%}")

    def _load_data(self) -> pd.DataFrame:
        if self.use_synthetic:
            return load_or_generate_synthetic()
        try:
            return load_nslkdd("train")
        except Exception:
            logger.warning("Falling back to synthetic data")
            return load_or_generate_synthetic()

    def _produce(self, df: pd.DataFrame):
        """Producer thread: pushes records into queue at configured rate."""
        interval = 1.0 / self.rps
        indices = list(range(len(df)))

        while not self._stop_event.is_set():
            # Drift simulation: shuffle feature distributions after threshold
            if self.drift_after and self._records_sent >= self.drift_after:
                df = self._apply_drift(df)
                self.drift_after = None  # Only apply once
                logger.warning("🌊 Data drift applied to stream!")

            idx = random.choice(indices)
            record = df.iloc[idx].to_dict()
            record["_timestamp"] = time.time()
            record["_simulated"] = True

            # Inject attack burst occasionally
            if random.random() < self.attack_burst_prob and record["is_attack"] == 0:
                record = self._inject_attack(record)

            try:
                self._stream_queue.put_nowait(record)
                self._records_sent += 1
            except queue.Full:
                pass  # Drop if consumer is slow

            time.sleep(interval)

    def _apply_drift(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simulate concept drift by skewing numeric features."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        drift_cols = random.sample(numeric_cols, min(5, len(numeric_cols)))
        df = df.copy()
        for col in drift_cols:
            df[col] = df[col] * random.uniform(1.5, 3.0)
        logger.debug(f"Drift applied to columns: {drift_cols}")
        return df

    def _inject_attack(self, record: dict) -> dict:
        """Modify a normal record to look like a DoS attack."""
        record["is_attack"] = 1
        record["attack_category"] = "dos"
        record["src_bytes"] = random.randint(100000, 1000000)
        record["count"] = random.randint(400, 512)
        record["serror_rate"] = random.uniform(0.8, 1.0)
        record["_injected"] = True
        return record

    def start(self) -> None:
        """Start the background producer thread."""
        df = self._load_data()
        self._thread = threading.Thread(target=self._produce, args=(df,), daemon=True)
        self._thread.start()
        logger.success("Traffic stream started")

    def stop(self) -> None:
        """Stop the producer."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info(f"Stream stopped. Total records sent: {self._records_sent:,}")

    def stream(self, batch_size: int = 1) -> Iterator[list[dict]]:
        """
        Generator: yields batches of records from the stream.
        Blocks until records are available.
        """
        batch = []
        while not self._stop_event.is_set():
            try:
                record = self._stream_queue.get(timeout=0.5)
                batch.append(record)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            except queue.Empty:
                if batch:
                    yield batch
                    batch = []

    @property
    def stats(self) -> dict:
        return {
            "records_sent": self._records_sent,
            "queue_size": self._stream_queue.qsize(),
            "running": not self._stop_event.is_set(),
        }


if __name__ == "__main__":
    # Demo: stream 30 records at 5 rec/s
    sim = TrafficStreamSimulator(records_per_second=5.0, drift_after_n_records=15)
    sim.start()

    count = 0
    for batch in sim.stream(batch_size=5):
        for record in batch:
            print(f"[{record['_timestamp']:.2f}] is_attack={record['is_attack']} | {record['attack_category']}")
            count += 1
        if count >= 30:
            break

    sim.stop()
