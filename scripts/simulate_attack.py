"""
Attack simulation script — Fires simulated attacks at the running IDS API.
Run the API first: python -m uvicorn src.api.main:app --reload
Then: python scripts/simulate_attack.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import random
import requests
import argparse
from loguru import logger

API_BASE = "http://localhost:8000/api/v1"

SCENARIOS = {
    "normal": f"{API_BASE}/simulate/normal",
    "dos":    f"{API_BASE}/simulate/dos",
    "probe":  f"{API_BASE}/simulate/probe",
    "burst":  f"{API_BASE}/simulate/burst",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=list(SCENARIOS.keys()), default="burst")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--delay", type=float, default=0.3)
    args = parser.parse_args()

    logger.info(f"Simulating {args.count}x '{args.scenario}' attacks ...")

    if args.scenario == "burst":
        resp = requests.post(SCENARIOS["burst"])
        logger.success(f"Burst triggered: {resp.json()}")
        return

    for i in range(args.count):
        try:
            resp = requests.post(SCENARIOS[args.scenario])
            data = resp.json()
            icon = "🚨" if data.get("is_attack") else "✅"
            logger.info(
                f"[{i+1}/{args.count}] {icon} "
                f"score={data.get('ensemble_score', 0):.3f} | "
                f"category={data.get('attack_category', '?')} | "
                f"severity={data.get('severity', '?')}"
            )
        except Exception as e:
            logger.error(f"Request failed: {e}")
        time.sleep(args.delay)

if __name__ == "__main__":
    main()
