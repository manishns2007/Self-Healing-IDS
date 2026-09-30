"""Pydantic schemas for the IDS API."""

from pydantic import BaseModel, Field
from typing import Optional, Any


class TrafficRecord(BaseModel):
    """Represents a single network traffic record for detection."""
    duration: float = Field(0.0, ge=0)
    protocol_type: str = "tcp"
    service: str = "http"
    flag: str = "SF"
    src_bytes: float = Field(0.0, ge=0)
    dst_bytes: float = Field(0.0, ge=0)
    land: int = Field(0, ge=0, le=1)
    wrong_fragment: int = Field(0, ge=0)
    urgent: int = Field(0, ge=0)
    hot: int = Field(0, ge=0)
    num_failed_logins: int = Field(0, ge=0)
    logged_in: int = Field(1, ge=0, le=1)
    num_compromised: int = Field(0, ge=0)
    root_shell: int = Field(0, ge=0, le=1)
    su_attempted: int = Field(0, ge=0)
    num_root: int = Field(0, ge=0)
    num_file_creations: int = Field(0, ge=0)
    num_shells: int = Field(0, ge=0)
    num_access_files: int = Field(0, ge=0)
    num_outbound_cmds: int = Field(0, ge=0)
    is_host_login: int = Field(0, ge=0, le=1)
    is_guest_login: int = Field(0, ge=0, le=1)
    count: int = Field(0, ge=0)
    srv_count: int = Field(0, ge=0)
    serror_rate: float = Field(0.0, ge=0, le=1)
    srv_serror_rate: float = Field(0.0, ge=0, le=1)
    rerror_rate: float = Field(0.0, ge=0, le=1)
    srv_rerror_rate: float = Field(0.0, ge=0, le=1)
    same_srv_rate: float = Field(1.0, ge=0, le=1)
    diff_srv_rate: float = Field(0.0, ge=0, le=1)
    srv_diff_host_rate: float = Field(0.0, ge=0, le=1)
    dst_host_count: int = Field(255, ge=0, le=255)
    dst_host_srv_count: int = Field(255, ge=0, le=255)
    dst_host_same_srv_rate: float = Field(1.0, ge=0, le=1)
    dst_host_diff_srv_rate: float = Field(0.0, ge=0, le=1)
    dst_host_same_src_port_rate: float = Field(0.0, ge=0, le=1)
    dst_host_srv_diff_host_rate: float = Field(0.0, ge=0, le=1)
    dst_host_serror_rate: float = Field(0.0, ge=0, le=1)
    dst_host_srv_serror_rate: float = Field(0.0, ge=0, le=1)
    dst_host_rerror_rate: float = Field(0.0, ge=0, le=1)
    dst_host_srv_rerror_rate: float = Field(0.0, ge=0, le=1)
    # Optional metadata
    source_ip: Optional[str] = "192.168.1.1"

    model_config = {"extra": "ignore"}


class DetectionResult(BaseModel):
    is_attack: bool
    ensemble_score: float
    threshold: float
    attack_category: str
    severity: str
    latency_ms: float
    model_scores: dict[str, float]
    alert_id: Optional[str] = None
    actions_taken: list[str] = []


class BatchDetectionRequest(BaseModel):
    records: list[TrafficRecord]


class TrainRequest(BaseModel):
    use_synthetic: bool = False
    run_name: Optional[str] = None


class AcknowledgeRequest(BaseModel):
    alert_id: str
    notes: str = ""
