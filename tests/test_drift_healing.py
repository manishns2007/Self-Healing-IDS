"""
Unit tests for Drift Detection, Incident Response, and Alert Management.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import pytest
from src.healing.drift_detector import DriftDetector, compute_psi
from src.response.incident_response import IncidentResponder, MockFirewall
from src.detection.alert_manager import AlertManager


def test_compute_psi():
    # Identical distributions should have PSI very close to 0
    np.random.seed(42)
    expected = np.random.normal(0, 1, 1000)
    actual = np.random.normal(0, 1, 1000)
    psi = compute_psi(expected, actual)
    assert psi < 0.1

    # Shifted distributions should have high PSI
    shifted = np.random.normal(3, 1, 1000)
    psi_shifted = compute_psi(expected, shifted)
    assert psi_shifted > 0.25


def test_drift_detector_feature_drift():
    np.random.seed(42)
    ref = np.random.normal(0, 1, (200, 5))
    detector = DriftDetector(reference_data=ref)

    # Add samples from same distribution
    same_data = np.random.normal(0, 1, (120, 5))
    detector.add_batch(same_data)
    result = detector.check_feature_drift()
    assert result["drift_detected"] is False

    # Create new detector with shifted data
    detector_shifted = DriftDetector(reference_data=ref)
    shifted_data = np.random.normal(5, 1, (120, 5))
    detector_shifted.add_batch(shifted_data)
    result_shifted = detector_shifted.check_feature_drift()
    assert result_shifted["drift_detected"] is True
    assert result_shifted["n_drifted_features"] > 0


def test_drift_detector_performance_drift():
    detector = DriftDetector()
    # High performance: all correct predictions
    for _ in range(60):
        detector.add_sample(np.zeros(5), y_true=1, y_pred=1)
    res_high = detector.check_performance_drift()
    assert res_high["drift_detected"] is False
    assert res_high["current_f1"] == 1.0

    # Low performance: false negatives / false positives
    detector_low = DriftDetector()
    for _ in range(60):
        detector_low.add_sample(np.zeros(5), y_true=1, y_pred=0)
    res_low = detector_low.check_performance_drift()
    assert res_low["drift_detected"] is True
    assert res_low["current_f1"] < 0.80


def test_mock_firewall_and_incident_responder():
    firewall = MockFirewall()
    responder = IncidentResponder(firewall=firewall)

    # Critical DoS attack should block IP
    actions = responder.respond(
        is_attack=True,
        attack_category="dos",
        source_ip="192.168.1.100",
        severity="CRITICAL",
        ensemble_score=0.95,
    )
    assert any("block_ip" in a for a in actions)
    assert "192.168.1.100" in responder.firewall.blocked_ips

    # Test auto unblock expiration
    responder.firewall.blocked_ips["192.168.1.100"] = time.time() - 10000  # expired
    unblocked = responder.firewall.auto_unblock_expired()
    assert "192.168.1.100" in unblocked
    assert "192.168.1.100" not in responder.firewall.blocked_ips


def test_alert_manager_lifecycle():
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    mgr = None
    try:
        mgr = AlertManager(db_path=db_path)
        alert = mgr.create_alert(
            is_attack=True,
            ensemble_score=0.88,
            attack_category="probe",
            raw_record={"source_ip": "10.0.0.5", "protocol_type": "tcp", "service": "http"},
            model_scores={"rf": 0.85, "xgb": 0.90},
            response_taken=["log_alert"],
        )
        assert alert.id is not None
        assert alert.source_ip == "10.0.0.5"
        assert alert.is_attack is True

        recent = mgr.get_recent_alerts(limit=10)
        assert len(recent) == 1
        assert recent[0]["id"] == alert.id

        # Acknowledge
        ack_res = mgr.acknowledge_alert(alert.id, notes="Investigated by SecOps")
        assert ack_res is True

        stats = mgr.get_stats()
        assert stats["total_alerts"] == 1
        assert stats["total_attacks"] == 1
    finally:
        if mgr:
            mgr.close()
        if os.path.exists(db_path):
            os.unlink(db_path)
