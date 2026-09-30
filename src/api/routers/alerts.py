"""Alerts router — query, acknowledge, and summarize alerts."""

from fastapi import APIRouter, Request, HTTPException
from src.api.models import AcknowledgeRequest

router = APIRouter()


@router.get("/alerts", summary="Get recent alerts")
def get_alerts(request: Request, limit: int = 50, attacks_only: bool = False):
    state = request.app.state.ids
    return {
        "alerts": state.alert_manager.get_recent_alerts(limit=limit, only_attacks=attacks_only),
        "total": state.alert_manager.get_stats().get("total_alerts", 0),
    }


@router.get("/alerts/stats", summary="Alert statistics")
def alert_stats(request: Request):
    return request.app.state.ids.alert_manager.get_stats()


@router.post("/alerts/acknowledge", summary="Acknowledge an alert")
def acknowledge(body: AcknowledgeRequest, request: Request):
    success = request.app.state.ids.alert_manager.acknowledge_alert(body.alert_id, body.notes)
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"acknowledged": True, "alert_id": body.alert_id}
