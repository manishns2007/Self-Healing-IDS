"""Metrics router — model performance, detector stats, and system metrics."""

import time
from fastapi import APIRouter, Request, BackgroundTasks
from src.api.models import TrainRequest

router = APIRouter()


@router.get("/metrics/detector", summary="Real-time detector statistics")
def detector_metrics(request: Request):
    state = request.app.state.ids
    return state.detector.stats if state.detector else {"error": "detector not loaded"}


@router.get("/metrics/system", summary="System-level metrics")
def system_metrics(request: Request):
    state = request.app.state.ids
    return {
        "uptime_s": round(time.time() - state.start_time, 1),
        "model_loaded": state.detector.is_ready,
        "blocked_ips": state.incident_responder.firewall.blocked_ips,
        "incident_summary": state.incident_responder.incident_summary,
    }


@router.post("/train", summary="Trigger model training")
def trigger_training(body: TrainRequest, background_tasks: BackgroundTasks, request: Request):
    """Kick off training asynchronously and return immediately."""
    def _train():
        from src.training.trainer import train
        result = train(use_synthetic=body.use_synthetic, run_name=body.run_name)
        # Reload model after training
        request.app.state.ids.detector.load()

    background_tasks.add_task(_train)
    return {"status": "training_started", "use_synthetic": body.use_synthetic}
