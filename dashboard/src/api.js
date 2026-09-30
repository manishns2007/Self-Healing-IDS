// API client — all calls to the FastAPI backend
import axios from 'axios';

const API_BASE = 'http://localhost:8000/api/v1';
const api = axios.create({ baseURL: API_BASE, timeout: 10000 });

export const fetchAlerts = (limit = 50, attacksOnly = false) =>
  api.get('/alerts', { params: { limit, attacks_only: attacksOnly } }).then(r => r.data);

export const fetchAlertStats = () =>
  api.get('/alerts/stats').then(r => r.data);

export const fetchDetectorMetrics = () =>
  api.get('/metrics/detector').then(r => r.data);

export const fetchSystemMetrics = () =>
  api.get('/metrics/system').then(r => r.data);

export const fetchHealingStatus = () =>
  api.get('/healing/status').then(r => r.data);

export const fetchDriftStatus = () =>
  api.get('/healing/drift').then(r => r.data);

export const triggerRetrain = () =>
  api.post('/healing/retrain').then(r => r.data);

export const reloadModel = () =>
  api.post('/healing/reload-model').then(r => r.data);

export const simulateDos = () =>
  api.post('/simulate/dos').then(r => r.data);

export const simulateProbe = () =>
  api.post('/simulate/probe').then(r => r.data);

export const simulateNormal = () =>
  api.post('/simulate/normal').then(r => r.data);

export const simulateBurst = () =>
  api.post('/simulate/burst').then(r => r.data);

export const fetchHealth = () =>
  axios.get('http://localhost:8000/health').then(r => r.data);

export default api;
