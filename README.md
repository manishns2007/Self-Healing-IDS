# 🛡️ Self-Healing IDS — MLOps Intrusion Detection System

A full-stack, production-grade **Intrusion Detection System** with autonomous self-healing capabilities, built for MLOps.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **Ensemble ML** | Random Forest + XGBoost + Isolation Forest + Autoencoder |
| 📊 **MLflow Tracking** | Full experiment tracking, model registry, artifact storage |
| 🌊 **Drift Detection** | KS Test + PSI-based feature drift + performance drop monitoring |
| 🔄 **Auto-Retraining** | Drift-triggered model retraining without human intervention |
| 🔥 **Incident Response** | Automated IP blocking, rate limiting, severity-based escalation |
| 🩺 **Self-Healing** | Component health monitoring + automatic restart + hot model reload |
| 🚀 **FastAPI Backend** | REST API with 15+ endpoints for detection, alerts, and healing |
| 🎨 **React Dashboard** | Live-updating cybersecurity dashboard with charts and alert feed |
| 🐳 **Dockerized** | Full Docker Compose stack for one-command deployment |

---

## 🏗️ Architecture

```
NSL-KDD Dataset
     │
     ▼
Data Ingestion → Feature Engineering → Ensemble Training (MLflow)
                                              │
                              ┌───────────────┴───────────────┐
                              │                               │
                    Supervised Models              Anomaly Models
                  Random Forest + XGBoost    Isolation Forest + Autoencoder
                              │                               │
                              └───────────┬───────────────────┘
                                          │ Weighted Ensemble Score
                                          ▼
                                   FastAPI Detector
                                          │
                         ┌────────────────┴────────────────┐
                         │                                 │
                  Incident Responder               Self-Healing Loop
               Block IPs, Rate Limit          Drift Detect → Retrain
                         │                                 │
                         └────────────┬────────────────────┘
                                      │
                               React Dashboard
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 20+
- (Optional) Docker + Docker Compose

### Option A — Local Development

**Step 1: Install Python dependencies**
```bash
pip install -r requirements.txt
```

**Step 2: Train the model** (downloads NSL-KDD automatically)
```bash
python scripts/train_initial.py
# Or with synthetic data (no internet needed):
python scripts/train_initial.py --synthetic
```

**Step 3: Start MLflow tracking server** (optional, in a separate terminal)
```bash
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow_data/mlflow.db --default-artifact-root ./mlflow_data/artifacts
```

**Step 4: Start the FastAPI backend**
```bash
python -m uvicorn src.api.main:app --reload --port 8000
```

**Step 5: Start the React dashboard** (in a separate terminal)
```bash
cd dashboard
npm install
npm run dev
```

**Step 6: Simulate attacks** (in a separate terminal)
```bash
python scripts/simulate_attack.py --scenario burst
```

### Option B — Docker Compose (one command)
```bash
# Build and start all services
docker-compose up --build

# Services:
#   API:       http://localhost:8000
#   Dashboard: http://localhost:3000
#   MLflow:    http://localhost:5000
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | System health check |
| `POST` | `/api/v1/detect` | Detect intrusion in a traffic record |
| `POST` | `/api/v1/detect/batch` | Batch detection |
| `GET`  | `/api/v1/alerts` | Get recent alerts |
| `GET`  | `/api/v1/alerts/stats` | Alert statistics |
| `POST` | `/api/v1/alerts/acknowledge` | Acknowledge an alert |
| `GET`  | `/api/v1/metrics/detector` | Detector runtime stats |
| `GET`  | `/api/v1/metrics/system` | System metrics |
| `POST` | `/api/v1/train` | Trigger training |
| `GET`  | `/api/v1/healing/status` | Self-healer status |
| `GET`  | `/api/v1/healing/drift` | Drift detection results |
| `POST` | `/api/v1/healing/retrain` | Manual retrain trigger |
| `POST` | `/api/v1/healing/reload-model` | Hot-reload model |
| `POST` | `/api/v1/simulate/dos` | Simulate DoS attack |
| `POST` | `/api/v1/simulate/burst` | Mixed 50-record burst |

Interactive API docs: **http://localhost:8000/docs**

---

## 🤖 Self-Healing Loop

```
Every 5 min: Check for data drift (KS test + PSI)
     │
     ├─ Drift detected? → Trigger auto-retraining
     │                          │
     │                    New model trained
     │                          │
     │                    Hot-reload in API (zero downtime)
     │
Every 1 min: Health check all components
     │
     ├─ API offline? → Restart sequence
     └─ Model degraded? → Trigger retrain
```

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run only model tests
pytest tests/test_models.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 📁 Project Structure

```
Self Healing IDS/
├── src/
│   ├── ingestion/      # NSL-KDD loader + stream simulator
│   ├── preprocessing/  # Feature engineering + encoding
│   ├── models/         # RF, XGBoost, IsoForest, Autoencoder, Ensemble
│   ├── training/       # MLflow-tracked training pipeline
│   ├── detection/      # Runtime detector + alert manager
│   ├── response/       # Incident response + firewall engine
│   ├── healing/        # Drift detector + self-healer
│   └── api/            # FastAPI app + routers
├── dashboard/          # React + Recharts frontend
├── scripts/            # Train + simulate scripts
├── tests/              # pytest test suite
├── configs/            # YAML configurations
├── data/               # Datasets + trained models
├── docker-compose.yml
└── requirements.txt
```

---

## 📊 Dataset

Uses **NSL-KDD** (41 features, 4 attack categories):
- **DoS** — Denial of Service (Neptune, Smurf, Land, ...)
- **Probe** — Reconnaissance (Satan, Ipsweep, Nmap, ...)
- **R2L** — Remote to Local (FTP Write, Guess Passwd, ...)
- **U2R** — User to Root (Rootkit, Buffer Overflow, ...)

Auto-downloaded from GitHub on first run. Synthetic data fallback available.

---

## 🛡️ Attack Severity Mapping

| Severity | Score Threshold | Attack Types | Actions |
|----------|----------------|--------------|---------|
| CRITICAL | ≥ 0.90 | DoS, DDoS, Rootkit | Block IP + Kill Connection + Alert |
| HIGH     | ≥ 0.75 | Probe, R2L, U2R | Rate Limit + Alert |
| MEDIUM   | ≥ 0.60 | Probe, FTP Write | Log + Increase Monitoring |
| LOW      | ≥ 0.50 | Any | Log Only |

---

## 📄 License
MIT License — Built for educational and research purposes.
