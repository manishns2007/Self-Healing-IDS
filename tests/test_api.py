"""
Tests for FastAPI endpoints using TestClient.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_state():
    """Mock the app state so we don't need a real model."""
    mock_detector = MagicMock()
    mock_detector.is_ready = True
    mock_detector.predict.return_value = {
        "is_attack": True,
        "ensemble_score": 0.87,
        "threshold": 0.5,
        "attack_category": "dos",
        "latency_ms": 2.3,
        "model_scores": {"random_forest": 0.9, "xgboost": 0.85},
        "prediction_id": 1,
    }
    mock_detector.stats = {"total_predictions": 10, "total_attacks_detected": 3, "attack_rate": 0.3, "loaded": True}

    mock_alert_mgr = MagicMock()
    mock_alert = MagicMock()
    mock_alert.id = "test-alert-id-123"
    mock_alert_mgr.create_alert.return_value = mock_alert
    mock_alert_mgr.get_recent_alerts.return_value = []
    mock_alert_mgr.get_stats.return_value = {"total_alerts": 10, "total_attacks": 3, "total_normal": 7, "attack_rate": 0.3, "by_severity": {}}

    mock_incident = MagicMock()
    mock_incident.respond.return_value = ["block_ip:10.0.0.1(blocked)"]
    mock_incident.firewall = MagicMock()
    mock_incident.firewall.blocked_ips = []
    mock_incident.incident_summary = {"total_incidents": 1, "blocked_ips": [], "n_blocked": 0}

    mock_drift = MagicMock()
    mock_drift.full_check.return_value = {"drift_detected": False, "feature_drift": {}, "performance_drift": {}}

    mock_healer = MagicMock()
    mock_healer.status = {"running": True, "healing_actions": 0, "retrain_count_today": 0, "last_retrain": 0, "component_health": {}, "recent_actions": [], "drift_history": []}

    state = MagicMock()
    state.detector = mock_detector
    state.alert_manager = mock_alert_mgr
    state.incident_responder = mock_incident
    state.drift_detector = mock_drift
    state.self_healer = mock_healer
    state.start_time = 0.0
    return state


def test_health_endpoint():
    """Test /health returns 200 without model loaded."""
    with patch("src.detection.detector.Detector.get_instance") as mock:
        mock.return_value = MagicMock(is_ready=False, load=lambda: False)
        with patch("src.detection.alert_manager.AlertManager"):
            with patch("src.response.incident_response.IncidentResponder"):
                with patch("src.healing.self_healer.SelfHealer"):
                    with patch("src.healing.drift_detector.DriftDetector"):
                        from src.api.main import app
                        client = TestClient(app, raise_server_exceptions=False)
                        resp = client.get("/health")
                        assert resp.status_code == 200
                        data = resp.json()
                        assert "status" in data


def test_alert_stats_structure():
    """Verify alert stats schema."""
    from src.detection.alert_manager import AlertManager
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    mgr = None
    try:
        mgr = AlertManager(db_path=db_path)
        stats = mgr.get_stats()
        assert "total_alerts" in stats
        assert "total_attacks" in stats
        assert "by_severity" in stats
    finally:
        if mgr:
            mgr.close()
        if os.path.exists(db_path):
            os.unlink(db_path)
