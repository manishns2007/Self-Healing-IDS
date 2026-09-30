"""Healing router — self-healing status and manual trigger endpoints."""

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/healing/status", summary="Self-healing system status")
def healing_status(request: Request):
    state = request.app.state.ids
    return state.self_healer.status


@router.get("/healing/drift", summary="Latest drift detection results")
def drift_status(request: Request):
    state = request.app.state.ids
    return state.drift_detector.full_check()


@router.post("/healing/retrain", summary="Manually trigger model retraining")
def manual_retrain(request: Request):
    state = request.app.state.ids
    triggered = state.self_healer.trigger_retrain(reason="manual_trigger")
    return {"triggered": triggered, "message": "Retraining started" if triggered else "Retrain not triggered (cooldown/limit)"}


@router.post("/healing/reload-model", summary="Reload production model without restart")
def reload_model(request: Request):
    state = request.app.state.ids
    success = state.self_healer.reload_model()
    return {"success": success}


@router.post("/healing/unblock/{ip}", summary="Manually unblock an IP")
def unblock_ip(ip: str, request: Request):
    state = request.app.state.ids
    result = state.incident_responder.firewall.unblock_ip(ip)
    return result
