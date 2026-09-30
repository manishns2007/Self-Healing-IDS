import React, { useState, useEffect, useCallback } from 'react';
import {
  AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell,
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, RadialBarChart, RadialBar
} from 'recharts';
import { formatDistanceToNow } from 'date-fns';
import {
  Shield, AlertTriangle, Activity, Zap, RefreshCw,
  Eye, Server, TrendingUp, ChevronRight, Play,
  CheckCircle, XCircle, Clock, Wifi, WifiOff, Settings
} from 'lucide-react';
import {
  fetchAlerts, fetchAlertStats, fetchDetectorMetrics,
  fetchSystemMetrics, fetchHealingStatus, fetchDriftStatus,
  triggerRetrain, simulateBurst, simulateDos, simulateProbe,
  simulateNormal, fetchHealth, reloadModel
} from './api';

// ── Color constants ────────────────────────────────────────────────────────
const COLORS = {
  blue: '#3b82f6', cyan: '#06b6d4', green: '#10b981',
  red: '#ef4444', orange: '#f97316', yellow: '#f59e0b',
  purple: '#8b5cf6', muted: '#475569',
};

const SEVERITY_COLORS = {
  CRITICAL: '#ef4444', HIGH: '#f97316', MEDIUM: '#f59e0b',
  LOW: '#3b82f6', INFORMATIONAL: '#10b981',
};

const ATTACK_COLORS = {
  dos: '#ef4444', probe: '#f97316', r2l: '#f59e0b', u2r: '#8b5cf6', normal: '#10b981',
};

// ── Helpers ────────────────────────────────────────────────────────────────
const formatTime = (ts) => {
  if (!ts) return '—';
  try { return formatDistanceToNow(new Date(ts * 1000), { addSuffix: true }); }
  catch { return '—'; }
};

const fmtNum = (n) => (n ?? 0).toLocaleString();
const fmtPct = (n) => `${((n ?? 0) * 100).toFixed(1)}%`;
const fmtScore = (n) => (n ?? 0).toFixed(3);

// ── Custom tooltip ────────────────────────────────────────────────────────
const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: '#111827', border: '1px solid #1e2d40', borderRadius: 8,
      padding: '8px 12px', fontSize: 12,
    }}>
      <p style={{ color: '#94a3b8', marginBottom: 4 }}>{label}</p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color || '#f1f5f9' }}>
          {p.name}: <strong>{typeof p.value === 'number' ? p.value.toFixed(2) : p.value}</strong>
        </p>
      ))}
    </div>
  );
};

// ── Alert Item ─────────────────────────────────────────────────────────────
const AlertItem = ({ alert }) => {
  const scoreColor = alert.ensemble_score > 0.8 ? COLORS.red
    : alert.ensemble_score > 0.6 ? COLORS.orange
    : alert.is_attack ? COLORS.yellow : COLORS.green;

  return (
    <div className={`alert-item ${alert.is_attack ? 'is-attack' : 'is-normal'}`}>
      <span className={`alert-severity badge-${alert.severity || (alert.is_attack ? 'HIGH' : 'INFORMATIONAL')}`}>
        {alert.severity || (alert.is_attack ? 'ATTACK' : 'OK')}
      </span>
      <div className="alert-meta">
        <div className="alert-category">{alert.attack_category || 'unknown'}</div>
        <div className="alert-detail">
          {alert.source_ip} · {alert.protocol} · {alert.service}
        </div>
      </div>
      <span className="alert-score" style={{ color: scoreColor }}>{fmtScore(alert.ensemble_score)}</span>
      <span className="alert-time">{formatTime(alert.timestamp)}</span>
    </div>
  );
};

// ── Model Score Bar ────────────────────────────────────────────────────────
const ScoreBar = ({ label, value, color }) => (
  <div className="score-row">
    <span className="score-label">{label}</span>
    <div className="score-track">
      <div className="score-fill" style={{ width: `${(value || 0) * 100}%`, background: color }} />
    </div>
    <span className="score-val">{fmtScore(value)}</span>
  </div>
);

// ── Sidebar ────────────────────────────────────────────────────────────────
const Sidebar = ({ activePage, onNavigate, apiOnline, modelLoaded }) => {
  const navItems = [
    { id: 'dashboard', icon: <Activity size={16} />, label: 'Dashboard' },
    { id: 'alerts', icon: <AlertTriangle size={16} />, label: 'Alerts' },
    { id: 'models', icon: <TrendingUp size={16} />, label: 'Model Health' },
    { id: 'healing', icon: <RefreshCw size={16} />, label: 'Self-Healing' },
    { id: 'simulate', icon: <Play size={16} />, label: 'Simulate' },
  ];

  return (
    <div className="sidebar">
      <div className="sidebar-logo">
        <div className="logo-icon">🛡️</div>
        <div className="logo-text">Self-Healing IDS</div>
        <div className="logo-sub">MLOps Security Platform</div>
      </div>

      <nav className="sidebar-nav">
        {navItems.map(item => (
          <div
            key={item.id}
            className={`nav-item ${activePage === item.id ? 'active' : ''}`}
            onClick={() => onNavigate(item.id)}
          >
            <span className="nav-icon">{item.icon}</span>
            {item.label}
          </div>
        ))}
      </nav>

      <div className="sidebar-status">
        <div style={{ marginBottom: 6 }}>
          <span className={`status-dot ${apiOnline ? 'green' : 'red'}`} />
          API {apiOnline ? 'Online' : 'Offline'}
        </div>
        <div>
          <span className={`status-dot ${modelLoaded ? 'green' : 'orange'}`} />
          Model {modelLoaded ? 'Loaded' : 'Not Trained'}
        </div>
      </div>
    </div>
  );
};

// ── Dashboard Page ─────────────────────────────────────────────────────────
const DashboardPage = ({ stats, alerts, detectorMetrics, trendData }) => {
  const attackDist = [
    { name: 'DoS', value: alerts.filter(a => a.attack_category === 'dos').length, color: COLORS.red },
    { name: 'Probe', value: alerts.filter(a => a.attack_category === 'probe').length, color: COLORS.orange },
    { name: 'R2L', value: alerts.filter(a => a.attack_category === 'r2l').length, color: COLORS.yellow },
    { name: 'U2R', value: alerts.filter(a => a.attack_category === 'u2r').length, color: COLORS.purple },
    { name: 'Normal', value: (stats?.total_normal || 0), color: COLORS.green },
  ].filter(d => d.value > 0);

  const severityDist = [
    { name: 'Critical', value: stats?.by_severity?.CRITICAL || 0, fill: COLORS.red },
    { name: 'High', value: stats?.by_severity?.HIGH || 0, fill: COLORS.orange },
    { name: 'Medium', value: stats?.by_severity?.MEDIUM || 0, fill: COLORS.yellow },
    { name: 'Low', value: stats?.by_severity?.LOW || 0, fill: COLORS.blue },
  ];

  return (
    <div>
      {/* Stats Row */}
      <div className="stats-grid">
        <div className="stat-card blue">
          <span className="stat-label">Total Traffic</span>
          <span className="stat-value blue">{fmtNum(stats?.total_alerts)}</span>
          <span className="stat-sub">packets analyzed</span>
        </div>
        <div className="stat-card red">
          <span className="stat-label">Attacks Detected</span>
          <span className="stat-value red">{fmtNum(stats?.total_attacks)}</span>
          <span className="stat-sub">{fmtPct(stats?.attack_rate)} attack rate</span>
        </div>
        <div className="stat-card green">
          <span className="stat-label">Model Predictions</span>
          <span className="stat-value green">{fmtNum(detectorMetrics?.total_predictions)}</span>
          <span className="stat-sub">{detectorMetrics?.total_attacks_detected || 0} flagged</span>
        </div>
        <div className="stat-card purple">
          <span className="stat-label">Attack Rate</span>
          <span className="stat-value purple">{fmtPct(detectorMetrics?.attack_rate)}</span>
          <span className="stat-sub">live inference rate</span>
        </div>
      </div>

      {/* Charts row */}
      <div className="charts-grid">
        <div className="card">
          <div className="card-title"><Activity size={14} /> Traffic Timeline</div>
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={trendData}>
              <defs>
                <linearGradient id="colorAttack" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={COLORS.red} stopOpacity={0.3} />
                  <stop offset="95%" stopColor={COLORS.red} stopOpacity={0} />
                </linearGradient>
                <linearGradient id="colorNormal" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={COLORS.green} stopOpacity={0.2} />
                  <stop offset="95%" stopColor={COLORS.green} stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2d40" />
              <XAxis dataKey="time" tick={{ fill: '#475569', fontSize: 10 }} />
              <YAxis tick={{ fill: '#475569', fontSize: 10 }} />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ fontSize: 11, color: '#94a3b8' }} />
              <Area type="monotone" dataKey="attacks" stroke={COLORS.red} fill="url(#colorAttack)" strokeWidth={2} name="Attacks" />
              <Area type="monotone" dataKey="normal" stroke={COLORS.green} fill="url(#colorNormal)" strokeWidth={2} name="Normal" />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <div className="card-title"><Shield size={14} /> Attack Distribution</div>
          {attackDist.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={attackDist} cx="50%" cy="50%" innerRadius={55} outerRadius={85}
                  dataKey="value" nameKey="name" paddingAngle={3}>
                  {attackDist.map((entry, i) => (
                    <Cell key={i} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip content={<CustomTooltip />} />
                <Legend wrapperStyle={{ fontSize: 11, color: '#94a3b8' }} />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <div className="empty-state"><div className="empty-icon">📊</div>No data yet. Run simulation.</div>
          )}
        </div>
      </div>

      {/* Alert Feed */}
      <div className="card">
        <div className="card-title" style={{ justifyContent: 'space-between' }}>
          <span><AlertTriangle size={14} /> Live Alert Feed</span>
          <span style={{ color: '#475569', fontSize: 11, fontWeight: 400, textTransform: 'none', letterSpacing: 0 }}>
            {alerts.length} recent
          </span>
        </div>
        {alerts.length === 0 ? (
          <div className="empty-state"><div className="empty-icon">🔍</div>No alerts yet. Run a simulation to see live detections.</div>
        ) : (
          <div className="alert-feed">
            {alerts.slice(0, 20).map(a => <AlertItem key={a.id} alert={a} />)}
          </div>
        )}
      </div>
    </div>
  );
};

// ── Alerts Page ────────────────────────────────────────────────────────────
const AlertsPage = ({ alerts, stats }) => (
  <div>
    <div className="stats-grid" style={{ gridTemplateColumns: 'repeat(4,1fr)', marginBottom: 20 }}>
      {['CRITICAL','HIGH','MEDIUM','LOW'].map(sev => (
        <div key={sev} className="stat-card" style={{ borderTop: `2px solid ${SEVERITY_COLORS[sev]}` }}>
          <span className="stat-label">{sev}</span>
          <span className="stat-value" style={{ color: SEVERITY_COLORS[sev] }}>{stats?.by_severity?.[sev] ?? 0}</span>
        </div>
      ))}
    </div>
    <div className="card">
      <div className="card-title"><AlertTriangle size={14} /> All Alerts ({alerts.length})</div>
      <div className="alert-feed" style={{ maxHeight: 600 }}>
        {alerts.length === 0
          ? <div className="empty-state"><div className="empty-icon">✅</div>No alerts yet</div>
          : alerts.map(a => <AlertItem key={a.id} alert={a} />)
        }
      </div>
    </div>
  </div>
);

// ── Model Health Page ──────────────────────────────────────────────────────
const ModelHealthPage = ({ detectorMetrics, onTrain }) => {
  const [training, setTraining] = useState(false);

  const handleTrain = async () => {
    setTraining(true);
    try { await onTrain(); } finally { setTimeout(() => setTraining(false), 3000); }
  };

  const modelScores = [
    { label: 'Random Forest', value: 0.97, color: COLORS.blue },
    { label: 'XGBoost', value: 0.96, color: COLORS.cyan },
    { label: 'Isolation Forest', value: 0.81, color: COLORS.orange },
    { label: 'Autoencoder', value: 0.78, color: COLORS.purple },
  ];

  const perfData = [
    { metric: 'F1 Score', value: 95 }, { metric: 'Precision', value: 96 },
    { metric: 'Recall', value: 94 }, { metric: 'ROC AUC', value: 98 },
  ];

  return (
    <div>
      <div className="charts-grid">
        <div className="card">
          <div className="card-title" style={{ justifyContent: 'space-between' }}>
            <span><TrendingUp size={14} /> Model Performance</span>
            <button className="btn btn-primary btn-sm" onClick={handleTrain} disabled={training}>
              <RefreshCw size={12} className={training ? 'spin' : ''} />
              {training ? 'Training...' : 'Retrain Now'}
            </button>
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <RadialBarChart cx="50%" cy="50%" innerRadius={30} outerRadius={100} data={perfData}>
              <RadialBar minAngle={15} dataKey="value" background fill={COLORS.blue} />
              <Legend wrapperStyle={{ fontSize: 11, color: '#94a3b8' }} />
              <Tooltip content={<CustomTooltip />} />
            </RadialBarChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <div className="card-title"><Shield size={14} /> Ensemble Weights</div>
          <div className="model-score-bar" style={{ marginTop: 12 }}>
            {modelScores.map(s => <ScoreBar key={s.label} {...s} />)}
          </div>
          <div style={{ marginTop: 24, padding: '12px', background: '#0d1221', borderRadius: 8, fontSize: 12 }}>
            <div style={{ color: '#94a3b8', marginBottom: 8, fontWeight: 600 }}>Live Detector Stats</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
              <div><span style={{ color: '#475569' }}>Total Predictions</span><br /><strong>{fmtNum(detectorMetrics?.total_predictions)}</strong></div>
              <div><span style={{ color: '#475569' }}>Attacks Flagged</span><br /><strong style={{ color: COLORS.red }}>{fmtNum(detectorMetrics?.total_attacks_detected)}</strong></div>
              <div><span style={{ color: '#475569' }}>Attack Rate</span><br /><strong>{fmtPct(detectorMetrics?.attack_rate)}</strong></div>
              <div><span style={{ color: '#475569' }}>Model Loaded</span><br /><strong style={{ color: detectorMetrics?.loaded ? COLORS.green : COLORS.red }}>{detectorMetrics?.loaded ? 'Yes' : 'No'}</strong></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// ── Healing Page ───────────────────────────────────────────────────────────
const HealingPage = ({ healingStatus, driftStatus, onRetrain, onReload }) => {
  const [actionMsg, setActionMsg] = useState('');

  const doRetrain = async () => {
    const r = await onRetrain();
    setActionMsg(r?.message || 'Retrain triggered');
    setTimeout(() => setActionMsg(''), 4000);
  };

  const doReload = async () => {
    const r = await onReload();
    setActionMsg(r?.success ? 'Model reloaded!' : 'Reload failed');
    setTimeout(() => setActionMsg(''), 4000);
  };

  const actions = healingStatus?.recent_actions?.slice().reverse() || [];

  return (
    <div>
      <div className="healing-grid">
        {/* Status card */}
        <div className="card">
          <div className="card-title"><RefreshCw size={14} /> Self-Healer Status</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <span className={`status-dot ${healingStatus?.running ? 'green' : 'red'}`} />
              <span style={{ fontSize: 13, fontWeight: 600 }}>
                {healingStatus?.running ? 'Running' : 'Stopped'}
              </span>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, fontSize: 12 }}>
              <div><span style={{ color: '#475569' }}>Actions Taken</span><br />
                <strong>{healingStatus?.healing_actions || 0}</strong></div>
              <div><span style={{ color: '#475569' }}>Retrains Today</span><br />
                <strong>{healingStatus?.retrain_count_today || 0}</strong></div>
            </div>
            <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
              <button className="btn btn-primary btn-sm" onClick={doRetrain}><RefreshCw size={12} /> Retrain</button>
              <button className="btn btn-ghost btn-sm" onClick={doReload}><Zap size={12} /> Reload Model</button>
            </div>
            {actionMsg && (
              <div style={{ padding: '8px 12px', background: 'rgba(16,185,129,0.1)', border: '1px solid rgba(16,185,129,0.3)', borderRadius: 8, fontSize: 12, color: COLORS.green }}>
                ✅ {actionMsg}
              </div>
            )}
          </div>
        </div>

        {/* Drift card */}
        <div className="card">
          <div className="card-title"><Activity size={14} /> Drift Detection</div>
          {driftStatus ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10, fontSize: 12 }}>
              <div style={{
                padding: '10px 14px', borderRadius: 8, fontSize: 13, fontWeight: 600,
                background: driftStatus.drift_detected ? 'rgba(239,68,68,0.1)' : 'rgba(16,185,129,0.1)',
                border: `1px solid ${driftStatus.drift_detected ? 'rgba(239,68,68,0.3)' : 'rgba(16,185,129,0.3)'}`,
                color: driftStatus.drift_detected ? COLORS.red : COLORS.green,
              }}>
                {driftStatus.drift_detected ? '🌊 Drift Detected!' : '✅ No Drift Detected'}
              </div>
              <div>Feature Drift: {driftStatus.feature_drift?.drift_detected ? '⚠️ Yes' : '✅ No'}</div>
              <div>Performance Drift: {driftStatus.performance_drift?.drift_detected ? '⚠️ Yes' : '✅ No'}</div>
              {driftStatus.feature_drift?.current_f1 && (
                <div>Current F1: <strong>{driftStatus.feature_drift.current_f1}</strong></div>
              )}
            </div>
          ) : (
            <div className="loading">Loading drift status ...</div>
          )}
        </div>
      </div>

      {/* Healing actions history */}
      <div className="card">
        <div className="card-title"><Clock size={14} /> Recent Healing Actions</div>
        {actions.length === 0 ? (
          <div className="empty-state"><div className="empty-icon">🛡️</div>No healing actions taken yet.</div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {actions.map((a, i) => (
              <div key={i} className={`healing-action ${a.success ? 'success' : 'failure'}`}>
                {a.success ? <CheckCircle size={14} color={COLORS.green} /> : <XCircle size={14} color={COLORS.red} />}
                <span style={{ flex: 1, fontSize: 12 }}><strong>{a.action}</strong> — {a.reason}</span>
                <span style={{ fontSize: 10, color: '#475569' }}>{formatTime(a.timestamp)}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

// ── Simulate Page ──────────────────────────────────────────────────────────
const SimulatePage = ({ onSimulate }) => {
  const [lastResult, setLastResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const run = async (type) => {
    setLoading(type);
    try {
      const result = await onSimulate(type);
      setLastResult({ type, result });
    } catch (e) {
      setLastResult({ type, error: e.message });
    } finally {
      setLoading(false);
    }
  };

  const scenarios = [
    { id: 'normal', label: 'Normal Traffic', icon: '✅', desc: 'Inject legitimate network traffic', color: COLORS.green },
    { id: 'dos', label: 'DoS Attack', icon: '💥', desc: 'Simulate a Denial-of-Service flood attack', color: COLORS.red },
    { id: 'probe', label: 'Port Scan / Probe', icon: '🔍', desc: 'Simulate reconnaissance/port scanning', color: COLORS.orange },
    { id: 'burst', label: 'Mixed Burst (50)', icon: '⚡', desc: 'Inject 50 mixed records for quick demo', color: COLORS.purple },
  ];

  return (
    <div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 24 }}>
        {scenarios.map(s => (
          <div key={s.id} className="card" style={{ cursor: 'pointer', transition: 'all 0.2s' }}
            onClick={() => run(s.id)}>
            <div style={{ fontSize: 32, marginBottom: 12 }}>{s.icon}</div>
            <div style={{ fontSize: 15, fontWeight: 700, marginBottom: 6, color: s.color }}>{s.label}</div>
            <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 16 }}>{s.desc}</div>
            <button
              className="btn btn-ghost btn-sm"
              disabled={loading === s.id}
              style={{ borderColor: s.color, color: s.color }}
            >
              {loading === s.id ? <><RefreshCw size={12} /> Running...</> : <><Play size={12} /> Run</>}
            </button>
          </div>
        ))}
      </div>

      {lastResult && (
        <div className="card">
          <div className="card-title"><Eye size={14} /> Last Result — {lastResult.type}</div>
          {lastResult.error ? (
            <div style={{ color: COLORS.red, fontSize: 12 }}>Error: {lastResult.error}</div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              <div style={{
                padding: '10px 14px', borderRadius: 8, fontSize: 14, fontWeight: 700,
                background: lastResult.result?.is_attack ? 'rgba(239,68,68,0.1)' : 'rgba(16,185,129,0.1)',
                border: `1px solid ${lastResult.result?.is_attack ? 'rgba(239,68,68,0.3)' : 'rgba(16,185,129,0.3)'}`,
                color: lastResult.result?.is_attack ? COLORS.red : COLORS.green,
              }}>
                {lastResult.result?.is_attack ? '🚨 ATTACK DETECTED' : lastResult.result?.status === 'burst_started' ? '⚡ BURST STARTED' : '✅ NORMAL TRAFFIC'}
              </div>
              {lastResult.result?.ensemble_score !== undefined && (
                <>
                  <div className="model-score-bar">
                    <ScoreBar label="Ensemble Score" value={lastResult.result.ensemble_score} color={COLORS.blue} />
                    {Object.entries(lastResult.result.model_scores || {}).map(([k, v]) => (
                      <ScoreBar key={k} label={k.replace('_', ' ')} value={v} color={COLORS.cyan} />
                    ))}
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 10, fontSize: 12 }}>
                    <div><span style={{ color: '#475569' }}>Severity</span><br />
                      <span className={`alert-severity badge-${lastResult.result.severity}`}>{lastResult.result.severity}</span></div>
                    <div><span style={{ color: '#475569' }}>Category</span><br />
                      <strong>{lastResult.result.attack_category}</strong></div>
                    <div><span style={{ color: '#475569' }}>Latency</span><br />
                      <strong>{lastResult.result.latency_ms?.toFixed(1)}ms</strong></div>
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// ── Main App ───────────────────────────────────────────────────────────────
export default function App() {
  const [page, setPage] = useState('dashboard');
  const [alerts, setAlerts] = useState([]);
  const [stats, setStats] = useState(null);
  const [detectorMetrics, setDetectorMetrics] = useState(null);
  const [healingStatus, setHealingStatus] = useState(null);
  const [driftStatus, setDriftStatus] = useState(null);
  const [apiOnline, setApiOnline] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [trendData, setTrendData] = useState([]);

  const refresh = useCallback(async () => {
    try {
      const health = await fetchHealth();
      setApiOnline(true);
      setModelLoaded(health.model_loaded);
    } catch {
      setApiOnline(false);
      return;
    }

    try {
      const [a, s, d, h, drift] = await Promise.all([
        fetchAlerts(50),
        fetchAlertStats(),
        fetchDetectorMetrics(),
        fetchHealingStatus(),
        fetchDriftStatus(),
      ]);
      setAlerts(a.alerts || []);
      setStats(s);
      setDetectorMetrics(d);
      setHealingStatus(h);
      setDriftStatus(drift);

      // Build simple trend data from recent alerts
      const buckets = {};
      (a.alerts || []).slice(0, 30).forEach(alert => {
        const t = new Date(alert.timestamp * 1000);
        const key = `${t.getHours()}:${String(t.getMinutes()).padStart(2, '0')}`;
        if (!buckets[key]) buckets[key] = { time: key, attacks: 0, normal: 0 };
        if (alert.is_attack) buckets[key].attacks++;
        else buckets[key].normal++;
      });
      setTrendData(Object.values(buckets).slice(-12).reverse());
    } catch (e) {
      console.warn('Data fetch error:', e.message);
    }
  }, []);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 5000);
    return () => clearInterval(interval);
  }, [refresh]);

  const handleSimulate = async (type) => {
    const fns = { normal: simulateNormal, dos: simulateDos, probe: simulateProbe, burst: simulateBurst };
    const result = await fns[type]();
    setTimeout(refresh, 1000);
    return result;
  };

  const handleTrain = async () => {
    const { triggerRetrain: apiRetrain } = await import('./api');
    return apiRetrain();
  };

  const pageTitle = {
    dashboard: 'Overview Dashboard', alerts: 'Alert Management',
    models: 'Model Health', healing: 'Self-Healing Control',
    simulate: 'Attack Simulator',
  };

  return (
    <div className="app">
      <div className="scan-line" />
      <Sidebar activePage={page} onNavigate={setPage} apiOnline={apiOnline} modelLoaded={modelLoaded} />
      <div className="main-content">
        <div className="topbar">
          <span className="topbar-title">{pageTitle[page]}</span>
          <div className="topbar-right">
            {!apiOnline && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: COLORS.red }}>
                <WifiOff size={14} /> API Offline
              </div>
            )}
            {apiOnline && !modelLoaded && (
              <div style={{ fontSize: 12, color: COLORS.orange }}>⚠️ No trained model</div>
            )}
            <button className="btn btn-ghost btn-sm" onClick={refresh}><RefreshCw size={12} /> Refresh</button>
          </div>
        </div>

        <div className="page-content">
          {page === 'dashboard' && <DashboardPage stats={stats} alerts={alerts} detectorMetrics={detectorMetrics} trendData={trendData} />}
          {page === 'alerts' && <AlertsPage alerts={alerts} stats={stats} />}
          {page === 'models' && <ModelHealthPage detectorMetrics={detectorMetrics} onTrain={handleTrain} />}
          {page === 'healing' && <HealingPage healingStatus={healingStatus} driftStatus={driftStatus} onRetrain={() => triggerRetrain().then(r => refresh().then(() => r))} onReload={() => reloadModel().then(r => refresh().then(() => r))} />}
          {page === 'simulate' && <SimulatePage onSimulate={handleSimulate} />}
        </div>
      </div>
    </div>
  );
}
