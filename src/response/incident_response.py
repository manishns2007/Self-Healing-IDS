"""
Incident Response Engine — Automatically takes action on detected intrusions.
Simulates firewall rules, IP blocking, rate limiting, and admin alerts.
"""

import time
import json
from collections import defaultdict
from pathlib import Path
from loguru import logger
import yaml

BLOCK_LOG_PATH = "logs/blocked_ips.json"


def load_config() -> dict:
    with open("configs/response_config.yaml") as f:
        return yaml.safe_load(f)


class FirewallEngine:
    """
    Simulated firewall / IP blocker.
    In production, replace _exec_block / _exec_unblock with real iptables/cloud API calls.
    """

    def __init__(self, config: dict | None = None):
        self.cfg = (config or load_config())["firewall"]
        self.simulated = self.cfg["simulated"]
        self.whitelist = set(self.cfg["whitelist"])
        self._blocked: dict[str, dict] = {}  # ip -> {blocked_at, duration}
        self._block_log_path = Path(BLOCK_LOG_PATH)
        self._block_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_block_log()

    def _load_block_log(self) -> None:
        if self._block_log_path.exists():
            with open(self._block_log_path) as f:
                self._blocked = json.load(f)

    def _save_block_log(self) -> None:
        with open(self._block_log_path, "w") as f:
            json.dump(self._blocked, f, indent=2)

    def _exec_block(self, ip: str) -> None:
        if self.simulated:
            logger.warning(f"🔥 [SIMULATED] BLOCK IP: {ip}")
        else:
            import subprocess
            subprocess.run(["iptables", "-A", "INPUT", "-s", ip, "-j", "DROP"], check=True)
            logger.warning(f"🔥 [REAL] BLOCKED IP: {ip}")

    def _exec_unblock(self, ip: str) -> None:
        if self.simulated:
            logger.info(f"✅ [SIMULATED] UNBLOCK IP: {ip}")
        else:
            import subprocess
            subprocess.run(["iptables", "-D", "INPUT", "-s", ip, "-j", "DROP"], check=True)

    def block_ip(self, ip: str, duration_s: int | None = None) -> dict:
        if ip in self.whitelist:
            return {"action": "skipped", "reason": "whitelisted", "ip": ip}
        if ip in self._blocked:
            return {"action": "already_blocked", "ip": ip}

        duration = duration_s or self.cfg["block_duration_seconds"]
        self._blocked[ip] = {"blocked_at": time.time(), "duration": duration}
        self._exec_block(ip)
        self._save_block_log()
        return {"action": "blocked", "ip": ip, "duration_s": duration}

    def unblock_ip(self, ip: str) -> dict:
        if ip not in self._blocked:
            return {"action": "not_blocked", "ip": ip}
        del self._blocked[ip]
        self._exec_unblock(ip)
        self._save_block_log()
        return {"action": "unblocked", "ip": ip}

    def auto_unblock_expired(self) -> list[str]:
        """Unblock IPs whose block duration has expired."""
        now = time.time()
        to_unblock = [
            ip for ip, info in self._blocked.items()
            if now - info["blocked_at"] > info["duration"]
        ]
        for ip in to_unblock:
            self.unblock_ip(ip)
        return to_unblock

    def is_blocked(self, ip: str) -> bool:
        return ip in self._blocked

    @property
    def blocked_ips(self) -> list[dict]:
        result = []
        for ip, info in self._blocked.items():
            result.append({
                "ip": ip,
                "blocked_at": info["blocked_at"],
                "duration_s": info["duration"],
                "expires_at": info["blocked_at"] + info["duration"],
                "remaining_s": max(0, info["blocked_at"] + info["duration"] - time.time()),
            })
        return result


class RateLimiter:
    """Simple sliding-window rate limiter."""

    def __init__(self, config: dict | None = None):
        self.cfg = (config or load_config())["rate_limiting"]
        self._windows: dict[str, list[float]] = defaultdict(list)

    def check(self, ip: str) -> dict:
        """Check if IP exceeds rate limit. Returns {allowed, count, limit}."""
        now = time.time()
        window = self._windows[ip]
        # Remove timestamps older than 60 seconds
        self._windows[ip] = [t for t in window if now - t < 60]
        self._windows[ip].append(now)
        count = len(self._windows[ip])
        allowed = count <= self.cfg["max_requests_per_minute"]
        return {"allowed": allowed, "count": count, "limit": self.cfg["max_requests_per_minute"]}


class IncidentResponder:
    """
    Orchestrates all response actions based on alert severity.
    Reads action rules from response_config.yaml.
    """

    def __init__(self, config: dict | None = None):
        self.cfg = config or load_config()
        self.firewall = FirewallEngine(self.cfg)
        self.rate_limiter = RateLimiter(self.cfg)
        self._incidents: list[dict] = []

    def respond(self, alert: dict) -> list[str]:
        """
        Execute automated response for an alert.
        Returns list of actions taken.
        """
        severity = alert.get("severity", "LOW").lower()
        source_ip = alert.get("source_ip", "unknown")
        attack_cat = alert.get("attack_category", "")
        actions_taken = []

        # Determine action set from config
        severity_rules = self.cfg.get("severity_rules", {})
        rule = severity_rules.get(severity.upper(), severity_rules.get("low", {}))
        action_list = rule.get("actions", ["log_incident"])

        for action in action_list:
            if action == "block_ip" and source_ip != "unknown":
                result = self.firewall.block_ip(source_ip)
                actions_taken.append(f"block_ip:{source_ip}({result['action']})")

            elif action == "rate_limit_ip" and source_ip != "unknown":
                result = self.rate_limiter.check(source_ip)
                actions_taken.append(f"rate_limit:{source_ip}(count={result['count']})")

            elif action == "kill_connection":
                actions_taken.append(f"kill_connection:{source_ip}[simulated]")
                logger.warning(f"[RESPONSE] Kill connection: {source_ip}")

            elif action == "alert_admin":
                msg = (
                    f"🚨 SECURITY ALERT [{alert.get('severity')}]\n"
                    f"  Category: {attack_cat}\n"
                    f"  Source IP: {source_ip}\n"
                    f"  Score: {alert.get('ensemble_score', 0):.3f}\n"
                    f"  Time: {alert.get('datetime', 'unknown')}"
                )
                logger.critical(msg)
                actions_taken.append("alert_admin:logged")

            elif action == "log_incident":
                self._log_incident(alert)
                actions_taken.append("log_incident")

            elif action == "increase_monitoring":
                actions_taken.append(f"increase_monitoring:{source_ip}")

        logger.info(f"[RESPONSE] Actions for {severity.upper()}: {actions_taken}")
        return actions_taken

    def _log_incident(self, alert: dict) -> None:
        self._incidents.append({**alert, "responded_at": time.time()})

    @property
    def incident_summary(self) -> dict:
        return {
            "total_incidents": len(self._incidents),
            "blocked_ips": self.firewall.blocked_ips,
            "n_blocked": len(self.firewall.blocked_ips),
        }
