"""Detection router — /detect and /detect/batch endpoints."""

from fastapi import APIRouter, HTTPException, Request
from src.api.models import TrafficRecord, DetectionResult, BatchDetectionRequest

router = APIRouter()


@router.post("/detect", response_model=DetectionResult, summary="Detect intrusion in a single traffic record")
def detect(record: TrafficRecord, request: Request):
    state = request.app.state.ids
    if not state.detector.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    rec_dict = record.model_dump()
    prediction = state.detector.predict(rec_dict)

    # Determine severity
    score = prediction["ensemble_score"]
    severity = "INFORMATIONAL"
    for sev, thresh in [("CRITICAL", 0.90), ("HIGH", 0.75), ("MEDIUM", 0.60), ("LOW", 0.50)]:
        if score >= thresh:
            severity = sev
            break

    # Incident response
    alert_dict = {
        "severity": severity,
        "ensemble_score": score,
        "attack_category": prediction.get("attack_category", "unknown"),
        "source_ip": record.source_ip or "unknown",
        "datetime": None,
    }
    actions = []
    if prediction["is_attack"]:
        actions = state.incident_responder.respond(alert_dict)

    # Persist alert
    alert = state.alert_manager.create_alert(prediction, rec_dict, actions)

    # Feed to drift detector
    try:
        X = state.detector.fe.transform_record(rec_dict)
        state.drift_detector.add_sample(X.flatten())
    except Exception:
        pass

    return DetectionResult(
        is_attack=prediction["is_attack"],
        ensemble_score=prediction["ensemble_score"],
        threshold=prediction["threshold"],
        attack_category=prediction.get("attack_category", "unknown"),
        severity=severity,
        latency_ms=prediction.get("latency_ms", 0.0),
        model_scores=prediction.get("model_scores", {}),
        alert_id=alert.id,
        actions_taken=actions,
    )


@router.post("/detect/batch", summary="Detect intrusions in a batch of records")
def detect_batch(body: BatchDetectionRequest, request: Request):
    state = request.app.state.ids
    if not state.detector.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    records = [r.model_dump() for r in body.records]
    predictions = state.detector.predict_batch(records)
    return {"results": predictions, "count": len(predictions)}
