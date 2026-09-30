"""
FastAPI Application — Main entry point for the Self-Healing IDS API.
Provides endpoints for detection, alerts, metrics, healing status, and model management.
"""

import time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.detection.detector import Detector
from src.detection.alert_manager import AlertManager
from src.response.incident_response import IncidentResponder
from src.healing.self_healer import SelfHealer
from src.healing.drift_detector import DriftDetector

from src.api.routers import detection, alerts, metrics, healing, simulation


# ── Shared app state ──────────────────────────────────────────────────────────

class AppState:
    detector: Detector = None
    alert_manager: AlertManager = None
    incident_responder: IncidentResponder = None
    drift_detector: DriftDetector = None
    self_healer: SelfHealer = None
    start_time: float = 0.0


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all components on startup, clean up on shutdown."""
    logger.info("🚀 Self-Healing IDS starting up ...")
    state.start_time = time.time()

    # Initialize components
    state.detector = Detector.get_instance()
    state.alert_manager = AlertManager()
    state.incident_responder = IncidentResponder()
    state.drift_detector = DriftDetector()
    state.self_healer = SelfHealer(drift_detector=state.drift_detector)

    # Load model (if available)
    loaded = state.detector.load()
    if not loaded:
        logger.warning("⚠️  No trained model found. Train first using: python scripts/train_initial.py")

    # Start self-healing background thread
    state.self_healer.start()

    # Attach state to app
    app.state.ids = state

    logger.success("✅ Self-Healing IDS ready!")
    yield

    # Shutdown
    logger.info("Shutting down Self-Healing IDS ...")
    state.self_healer.stop()


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Self-Healing IDS API",
    description=(
        "A Machine Learning-powered Intrusion Detection System with "
        "automated drift detection, model retraining, and incident response."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow React dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(detection.router, prefix="/api/v1", tags=["Detection"])
app.include_router(alerts.router, prefix="/api/v1", tags=["Alerts"])
app.include_router(metrics.router, prefix="/api/v1", tags=["Metrics"])
app.include_router(healing.router, prefix="/api/v1", tags=["Healing"])
app.include_router(simulation.router, prefix="/api/v1", tags=["Simulation"])


# ── Health & root ─────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    return {
        "status": "healthy",
        "model_loaded": state.detector.is_ready if state.detector else False,
        "uptime_s": round(time.time() - state.start_time, 1) if state.start_time else 0,
    }


@app.get("/", tags=["System"])
def root():
    return {
        "name": "Self-Healing IDS",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
