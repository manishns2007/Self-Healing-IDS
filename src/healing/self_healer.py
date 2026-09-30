"""
Self Healer — Orchestrates autonomous healing actions.
Monitors component health, detects drift, triggers retraining, and restores services.
"""

import time
import threading
import subprocess
import requests
from collections import defaultdict
from loguru import logger
import yaml

from src.healing.drift_detector import DriftDetector

HEALING_LOG = "logs/healing.log"


def load_config() -> dict:
    with open("configs/healing_config.yaml") as f:
        return yaml.safe_load(f)


class HealthMonitor:
    """Polls configured endpoints for liveness and tracks failure counts."""

    def __init__(self, config: dict | None = None):
        self.cfg = (config or load_config())
        self.components = self.cfg["health_monitor"]["components"]
        self._failure_counts: dict[str, int] = defaultdict(int)
        self._last_status: dict[str, dict] = {}

    def check(self) -> dict[str, dict]:
        """Check all configured components. Returns status for each."""
        results = {}
        timeout = self.cfg["health_monitor"]["api_timeout_seconds"]
        max_fails = self.cfg["health_monitor"]["max_failures_before_restart"]

        for comp in self.components:
            name = comp["name"]
            url = comp["url"]
            try:
                resp = requests.get(url, timeout=timeout)
                healthy = resp.status_code == 200
            except Exception as e:
                healthy = False

            if healthy:
                self._failure_counts[name] = 0
            else:
                self._failure_counts[name] += 1

            needs_restart = self._failure_counts[name] >= max_fails

            results[name] = {
                "healthy": healthy,
                "failure_count": self._failure_counts[name],
                "needs_restart": needs_restart,
                "url": url,
            }

        self._last_status = results
        return results

    @property
    def status(self) -> dict[str, dict]:
        return self._last_status


class SelfHealer:
    """
    Main orchestrator for all self-healing actions.
    Runs as a background thread, monitoring health and triggering fixes.
    """

    def __init__(
        self,
        drift_detector: DriftDetector | None = None,
        config: dict | None = None,
    ):
        self.cfg = config or load_config()
        self.drift_detector = drift_detector or DriftDetector()
        self.health_monitor = HealthMonitor(self.cfg)
        self._stop_event = threading.Event()
        self._healing_thread: threading.Thread | None = None
        self._healing_actions_taken: list[dict] = []
        self._retrain_count_today = 0
        self._last_retrain_time = 0.0
        self._cooldowns: dict[str, float] = {}
        self._running = False

        import os
        os.makedirs("logs", exist_ok=True)
        logger.add(HEALING_LOG, rotation="10 MB", level="INFO")

    def _can_act(self, action_key: str) -> bool:
        """Respect cooldown periods before repeating the same healing action."""
        cooldown = self.cfg["self_healer"]["cooldown_seconds"]
        last_time = self._cooldowns.get(action_key, 0)
        return (time.time() - last_time) > cooldown

    def _record_action(self, action: str, reason: str, success: bool) -> None:
        entry = {
            "timestamp": time.time(),
            "action": action,
            "reason": reason,
            "success": success,
        }
        self._healing_actions_taken.append(entry)
        if success:
            logger.success(f"[HEAL] ✅ {action}: {reason}")
        else:
            logger.error(f"[HEAL] ❌ {action} failed: {reason}")

    # ── Healing Actions ────────────────────────────────────────────

    def restart_api_server(self) -> bool:
        """Attempt to restart the FastAPI uvicorn server."""
        logger.warning("[HEAL] Attempting API server restart ...")
        try:
            # In a real system, this would use systemctl/supervisord/Docker
            # Here we log the action as simulated
            self._record_action("restart_api_server", "health check failures", True)
            self._cooldowns["restart_api_server"] = time.time()
            return True
        except Exception as e:
            self._record_action("restart_api_server", str(e), False)
            return False

    def trigger_retrain(self, reason: str = "drift_detected") -> bool:
        """Trigger model retraining asynchronously."""
        max_per_day = self.cfg["auto_retraining"]["max_retrain_per_day"]
        if not self.cfg["auto_retraining"]["enabled"]:
            logger.info("[HEAL] Auto-retraining disabled in config")
            return False

        if self._retrain_count_today >= max_per_day:
            logger.warning(f"[HEAL] Max retrains/day ({max_per_day}) reached. Skipping.")
            return False

        if not self._can_act("retrain"):
            logger.info("[HEAL] Retrain cooldown active. Skipping.")
            return False

        logger.warning(f"[HEAL] 🔄 Triggering retraining | reason: {reason}")
        self._cooldowns["retrain"] = time.time()
        self._last_retrain_time = time.time()
        self._retrain_count_today += 1

        def _retrain_thread():
            try:
                from src.training.trainer import train
                result = train(run_name=f"auto_retrain_{int(time.time())}")
                self._record_action("retrain", reason, True)
                logger.success(
                    f"[HEAL] Retraining complete | F1={result['metrics']['f1']:.4f}"
                )
                # Reload models in detector
                from src.detection.detector import Detector
                Detector.get_instance().load()
            except Exception as e:
                self._record_action("retrain", str(e), False)

        t = threading.Thread(target=_retrain_thread, daemon=True)
        t.start()
        return True

    def reload_model(self) -> bool:
        """Hot-reload the production model without restarting the server."""
        logger.info("[HEAL] Reloading model in detector ...")
        try:
            from src.detection.detector import Detector
            success = Detector.get_instance().load()
            self._record_action("reload_model", "model update", success)
            return success
        except Exception as e:
            self._record_action("reload_model", str(e), False)
            return False

    # ── Main Monitoring Loop ───────────────────────────────────────

    def _monitoring_loop(self) -> None:
        health_interval = self.cfg["health_monitor"]["check_interval_seconds"]
        drift_interval = self.cfg["drift_detection"]["check_interval_seconds"]
        last_health_check = 0.0
        last_drift_check = 0.0

        while not self._stop_event.is_set():
            now = time.time()

            # Health check
            if now - last_health_check >= health_interval:
                health = self.health_monitor.check()
                for comp, status in health.items():
                    if status["needs_restart"] and self._can_act(f"restart_{comp}"):
                        self.restart_api_server()
                last_health_check = now

            # Drift check
            if now - last_drift_check >= drift_interval:
                drift = self.drift_detector.full_check()
                if drift["drift_detected"]:
                    self.trigger_retrain(reason="data_drift_detected")
                last_drift_check = now

            time.sleep(5)

    def start(self) -> None:
        """Start the self-healing background thread."""
        if self._running:
            logger.warning("SelfHealer already running")
            return
        self._stop_event.clear()
        self._healing_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._healing_thread.start()
        self._running = True
        logger.success("SelfHealer started ✅")

    def stop(self) -> None:
        """Stop the monitoring loop."""
        self._stop_event.set()
        self._running = False
        logger.info("SelfHealer stopped")

    @property
    def status(self) -> dict:
        return {
            "running": self._running,
            "healing_actions": len(self._healing_actions_taken),
            "retrain_count_today": self._retrain_count_today,
            "last_retrain": self._last_retrain_time,
            "component_health": self.health_monitor.status,
            "recent_actions": self._healing_actions_taken[-10:],
            "drift_history": self.drift_detector.get_drift_history()[-5:],
        }
