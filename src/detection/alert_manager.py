"""
Alert Manager — Generates, scores, stores, and retrieves security alerts.
Uses SQLite for persistence via SQLAlchemy.
"""

import uuid
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from loguru import logger
from sqlalchemy import create_engine, Column, String, Float, Integer, Boolean, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, Session

DB_PATH = "data/alerts.db"
Base = declarative_base()

SEVERITY_THRESHOLDS = {
    "CRITICAL": 0.90,
    "HIGH": 0.75,
    "MEDIUM": 0.60,
    "LOW": 0.50,
}

ATTACK_SEVERITY_OVERRIDE = {
    "dos": "HIGH",
    "ddos": "CRITICAL",
    "u2r": "CRITICAL",
    "r2l": "HIGH",
    "probe": "MEDIUM",
}


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(Float, nullable=False)
    datetime_str = Column(String, nullable=False)
    severity = Column(String, nullable=False)
    is_attack = Column(Boolean, nullable=False)
    ensemble_score = Column(Float, nullable=False)
    attack_category = Column(String, nullable=True)
    source_ip = Column(String, nullable=True)
    protocol = Column(String, nullable=True)
    service = Column(String, nullable=True)
    model_scores = Column(Text, nullable=True)  # JSON string
    response_taken = Column(Text, nullable=True)  # JSON list
    acknowledged = Column(Boolean, default=False)
    notes = Column(Text, nullable=True)


class AlertManager:
    def __init__(self, db_path: str = DB_PATH):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        logger.info(f"AlertManager initialized | DB: {db_path}")

    def _get_severity(self, score: float, attack_category: str | None = None) -> str:
        if attack_category and attack_category in ATTACK_SEVERITY_OVERRIDE:
            override = ATTACK_SEVERITY_OVERRIDE[attack_category]
            # Only override if score supports it
            if score >= SEVERITY_THRESHOLDS.get(override, 0.5):
                return override

        for sev, thresh in SEVERITY_THRESHOLDS.items():
            if score >= thresh:
                return sev
        return "INFORMATIONAL"

    def create_alert(
        self,
        prediction: dict,
        raw_record: dict | None = None,
        response_taken: list[str] | None = None,
    ) -> Alert:
        """Create and persist a new alert from a prediction dict."""
        ts = time.time()
        score = prediction.get("ensemble_score", 0.0)
        attack_cat = raw_record.get("attack_category") if raw_record else None

        alert = Alert(
            id=str(uuid.uuid4()),
            timestamp=ts,
            datetime_str=datetime.fromtimestamp(ts).isoformat(),
            severity=self._get_severity(score, attack_cat),
            is_attack=bool(prediction.get("is_attack", False)),
            ensemble_score=score,
            attack_category=attack_cat,
            source_ip=raw_record.get("_source_ip", "unknown") if raw_record else "unknown",
            protocol=raw_record.get("protocol_type", "unknown") if raw_record else "unknown",
            service=raw_record.get("service", "unknown") if raw_record else "unknown",
            model_scores=json.dumps(prediction.get("model_scores", {})),
            response_taken=json.dumps(response_taken or []),
            acknowledged=False,
        )

        with self.SessionLocal() as session:
            session.add(alert)
            session.commit()
            session.refresh(alert)

        if alert.is_attack:
            logger.warning(
                f"🚨 [{alert.severity}] Attack detected | "
                f"score={score:.3f} | category={attack_cat or 'unknown'} | id={alert.id[:8]}"
            )

        return alert

    def get_recent_alerts(self, limit: int = 50, only_attacks: bool = False) -> list[dict]:
        with self.SessionLocal() as session:
            query = session.query(Alert).order_by(Alert.timestamp.desc()).limit(limit)
            alerts = query.all()

        result = []
        for a in alerts:
            if only_attacks and not a.is_attack:
                continue
            result.append({
                "id": a.id,
                "timestamp": a.timestamp,
                "datetime": a.datetime_str,
                "severity": a.severity,
                "is_attack": a.is_attack,
                "ensemble_score": a.ensemble_score,
                "attack_category": a.attack_category,
                "source_ip": a.source_ip,
                "protocol": a.protocol,
                "service": a.service,
                "model_scores": json.loads(a.model_scores or "{}"),
                "response_taken": json.loads(a.response_taken or "[]"),
                "acknowledged": a.acknowledged,
            })
        return result

    def get_stats(self) -> dict:
        with self.SessionLocal() as session:
            total = session.query(Alert).count()
            attacks = session.query(Alert).filter(Alert.is_attack == True).count()
            by_severity = {}
            for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
                by_severity[sev] = session.query(Alert).filter(Alert.severity == sev).count()

        return {
            "total_alerts": total,
            "total_attacks": attacks,
            "total_normal": total - attacks,
            "attack_rate": attacks / max(total, 1),
            "by_severity": by_severity,
        }

    def acknowledge_alert(self, alert_id: str, notes: str = "") -> bool:
        with self.SessionLocal() as session:
            alert = session.query(Alert).filter(Alert.id == alert_id).first()
            if alert:
                alert.acknowledged = True
                alert.notes = notes
                session.commit()
                return True
        return False
