/**
 * EdgeShield backend — real-time gateway (Node + Express + Socket.IO). Port 4000.
 *
 * Responsibilities: scenario engine, ML relay, threat scoring, autonomous
 * response, incident management, persistence, analytics, live-FL relay, and
 * broadcasting everything to the dashboard. NO ML logic and NO metric
 * fabrication: every number passing through here originated in the inference
 * service, ml-engine artifacts, SQL aggregates over persisted events, or
 * ground-truth-vs-prediction bookkeeping (live accuracy).
 *
 * Socket channels: init, metrics, prediction, threat, response, incident,
 * device-status, scenario-status, scenario-reset, federated-round,
 * federated-status, explanation, service-status.
 */

require('dotenv').config({ path: require('path').join(__dirname, '..', '.env') });
const path = require('path');
const fs = require('fs');
const express = require('express');
const http = require('http');
const cors = require('cors');
const axios = require('axios');
const { Server } = require('socket.io');
const { Scenario } = require('./demo/scenario');
const dbmod = require('./db');
const { threatScore } = require('./scoring');
const { explanationText } = require('./nlg');
const { ResponseEngine } = require('./response');
const { IncidentManager } = require('./incidents');
const { FlRunner } = require('./fl-runner');
const { Notifier } = require('./services/notifier');

const PORT = Number(process.env.BACKEND_PORT || 4000);
const INFERENCE_URL = process.env.INFERENCE_URL || 'http://127.0.0.1:8100';
const SIMULATOR_URL = process.env.SIMULATOR_URL || 'http://127.0.0.1:8200';
const REPO = path.join(__dirname, '..');
const STARTED_AT = Date.now();

const inference = axios.create({ baseURL: INFERENCE_URL, timeout: 10000 });
const simulator = axios.create({ baseURL: SIMULATOR_URL, timeout: 10000 });

const app = express();
app.use(cors());
app.use(express.json());
const server = http.createServer(app);
const io = new Server(server, { cors: { origin: '*' } });

const stmts = dbmod.init();
const RUN_ID = Date.now();

// ---------------- in-memory state (rebuilt every run; Reset clears it) -------
const state = {
  devices: [],
  dataSource: 'unknown',
  predictions: [],   // last 400
  threats: [],       // last 200
  flowCache: new Map(), // flow_id -> { features, meta } (this run; DB has all)
  metrics: {
    totalFlows: 0,
    threatsDetected: 0,
    latencySumMs: 0,      // -> avg = sum/totalFlows (real per-request latencies)
    confidenceSum: 0,
    normalCount: 0,
    maliciousCount: 0,
    liveCorrect: 0,       // prediction === simulator ground truth
    liveTotal: 0,
  },
  modelInfo: null,
  serviceStatus: { inference: 'unknown', simulator: 'unknown' },
};

function metricsSnapshot() {
  const m = state.metrics;
  const resp = responseEngine.snapshot();
  const openIncidents = stmts.raw.prepare(
    "SELECT COUNT(*) AS n FROM incidents WHERE status IN ('Open','Investigating')").get().n;
  return {
    totalDevices: state.devices.length,
    devicesOnline: state.devices.filter((d) => d.status !== 'offline' && d.status !== 'quarantined').length,
    devicesQuarantined: resp.quarantined.length,
    activeConnections: io.engine ? io.engine.clientsCount : 0,
    totalFlows: m.totalFlows,
    threatsDetected: m.threatsDetected,
    blockedFlows: resp.blockedFlows,
    normalFlows: m.normalCount,
    maliciousFlows: m.maliciousCount,
    activeIncidents: openIncidents,
    avgLatencyMs: m.totalFlows ? +(m.latencySumMs / m.totalFlows).toFixed(3) : null,
    avgConfidence: m.totalFlows ? +(m.confidenceSum / m.totalFlows).toFixed(4) : null,
    // "Live AI accuracy" = predictions matching the simulator's ground-truth
    // class label for each flow, measured during this run. Real, not scripted.
    liveAccuracy: m.liveTotal ? +(m.liveCorrect / m.liveTotal).toFixed(4) : null,
    liveSamples: m.liveTotal,
    uptimeSec: Math.round((Date.now() - STARTED_AT) / 1000),
    dataSource: state.dataSource,
    simulated: true,
  };
}

function broadcast(event, payload) {
  // book-keeping on the way through
  if (event === 'prediction') {
    const { meta, prediction } = payload;
    // threat scoring (deterministic, from real model outputs + flow metadata)
    const { score, severity } = threatScore(prediction, meta);
    payload.threat_score = score;
    payload.severity = severity;

    state.metrics.totalFlows += 1;
    state.metrics.latencySumMs += prediction.latency_ms;
    state.metrics.confidenceSum += prediction.confidence;
    if (prediction.label === 'Normal') state.metrics.normalCount += 1;
    else state.metrics.maliciousCount += 1;
    state.metrics.liveTotal += 1;
    if (prediction.label === meta.true_class) state.metrics.liveCorrect += 1;
    state.predictions.push(payload);
    if (state.predictions.length > 400) state.predictions.shift();

    // persist flow + prediction (features kept so /explain/{flow_id} works)
    state.flowCache.set(meta.flow_id, { features: payload.features, meta });
    if (state.flowCache.size > 2000) {
      state.flowCache.delete(state.flowCache.keys().next().value);
    }
    try {
      stmts.insertFlow.run({
        flow_id: meta.flow_id, device_id: meta.device_id, ts: meta.timestamp,
        features: JSON.stringify(payload.features),
      });
      stmts.insertPrediction.run({
        flow_id: meta.flow_id, ts: meta.timestamp, device_id: meta.device_id,
        prediction: prediction.label, confidence: prediction.confidence,
        latency_ms: prediction.latency_ms, true_class: meta.true_class ?? null,
        run_id: RUN_ID,
      });
    } catch (e) { console.error('[db]', e.message); }
  }
  if (event === 'threat') {
    state.metrics.threatsDetected += 1;
    state.threats.push(payload);
    if (state.threats.length > 200) state.threats.shift();
    try {
      stmts.insertThreat.run({
        flow_id: payload.meta.flow_id, ts: payload.meta.timestamp,
        device_id: payload.meta.device_id, device_name: payload.meta.device_name,
        source_ip: payload.meta.src_ip, destination_ip: payload.meta.dst_ip,
        prediction: payload.prediction.label, confidence: payload.prediction.confidence,
        latency_ms: payload.prediction.latency_ms,
        threat_score: payload.threat_score, severity: payload.severity,
        run_id: RUN_ID,
      });
    } catch (e) { console.error('[db]', e.message); }
    // autonomous response: quarantine / status transitions / incidents
    responseEngine.onDetection(payload, payload.tick ?? -1);
    // notifications: severity-based email / SMS / browser / alarm dispatch
    notifier.handleThreat(payload,
      responseEngine.openIncidentFor(payload.meta.device_id, payload.prediction.label));
  }
  if (event === 'device-status') {
    const d = state.devices.find((x) => x.id === payload.deviceId);
    if (d) {
      d.status = payload.status;
      try {
        stmts.upsertDevice.run({
          id: d.id, device_name: d.name, ip: d.ip, status: d.status,
        });
      } catch (e) { console.error('[db]', e.message); }
    }
  }
  if (event === 'scenario-reset') {
    state.predictions = [];
    state.threats = [];
    state.flowCache.clear();
    state.metrics = {
      totalFlows: 0, threatsDetected: 0, latencySumMs: 0, confidenceSum: 0,
      normalCount: 0, maliciousCount: 0, liveCorrect: 0, liveTotal: 0,
    };
    responseEngine.reset();
    state.devices.forEach((d) => { d.status = 'online'; });
    io.emit('metrics', metricsSnapshot());
    io.emit('devices', state.devices);
  }
  io.emit(event, payload);
  if (event === 'prediction' || event === 'threat' || event === 'response') {
    io.emit('metrics', metricsSnapshot());
  }
}

const incidents = new IncidentManager({ stmts, onEvent: broadcast });
const responseEngine = new ResponseEngine({ onEvent: broadcast, incidents });
const flRunner = new FlRunner({ onEvent: (ev, p) => io.emit(ev, p), stmts });
const notifier = new Notifier({ stmts, onEvent: (ev, p) => io.emit(ev, p) });

const scenario = new Scenario({
  simulator, inference, onEvent: broadcast,
  hooks: {
    isBlocked: (deviceId) => responseEngine.isBlocked(deviceId),
    onTick: (t) => responseEngine.onTick(t),
    onBlocked: (n) => { for (let i = 0; i < n; i += 1) responseEngine.countBlockedFlow(); },
  },
});

// periodic metrics persistence (only while flows are moving)
let lastPersistedFlows = -1;
setInterval(() => {
  const m = metricsSnapshot();
  if (m.totalFlows > 0 && m.totalFlows !== lastPersistedFlows) {
    lastPersistedFlows = m.totalFlows;
    try {
      stmts.insertMetricsSample.run({
        ts: Date.now() / 1000, total_flows: m.totalFlows, threats: m.threatsDetected,
        avg_latency_ms: m.avgLatencyMs, avg_confidence: m.avgConfidence,
        live_accuracy: m.liveAccuracy,
      });
    } catch (e) { console.error('[db]', e.message); }
  }
}, 10000);

// ---------------- bootstrap: connect to the two Python services --------------
async function bootstrap(retries = 60) {
  for (let i = 0; i < retries; i += 1) {
    try {
      const dev = await simulator.get('/devices');
      state.devices = dev.data.devices.map((d) => ({ ...d, status: 'online' }));
      state.dataSource = dev.data.data_source;
      state.serviceStatus.simulator = 'ok';
      const mi = await inference.get('/model-info');
      state.modelInfo = mi.data;
      state.serviceStatus.inference = 'ok';
      responseEngine.setDeviceNames(
        Object.fromEntries(state.devices.map((d) => [d.id, d.name])));
      state.devices.forEach((d) => stmts.upsertDevice.run({
        id: d.id, device_name: d.name, ip: d.ip, status: 'online',
      }));
      console.log(`[backend] connected: ${state.devices.length} devices, ` +
        `model=${mi.data.model_file} (${mi.data.size_kb} KB), data=${state.dataSource}`);
      return;
    } catch (e) {
      state.serviceStatus = { inference: 'waiting', simulator: 'waiting' };
      if (i % 5 === 0) console.log(`[backend] waiting for python services... (${e.code || e.message})`);
      await new Promise((r) => setTimeout(r, 2000));
    }
  }
  console.error('[backend] could not reach python services — REST still up, demo disabled');
}

// ---------------- REST ------------------------------------------------------
app.get('/health', (req, res) => res.json({
  status: 'ok', services: state.serviceStatus, db: dbmod.DB_PATH,
  uptimeSec: Math.round((Date.now() - STARTED_AT) / 1000),
}));
app.get('/devices', (req, res) => res.json({ devices: state.devices, dataSource: state.dataSource }));
app.get('/threats', (req, res) => res.json({ threats: state.threats.slice(-100) }));
app.get('/predictions', (req, res) => res.json({ predictions: state.predictions.slice(-100) }));
app.get('/metrics', (req, res) => res.json(metricsSnapshot()));
app.get('/model-performance', (req, res) => {
  res.json({ modelInfo: state.modelInfo });
});

// ---- explainability ----
async function explainByFeatures(features, meta, extras = {}) {
  const r = await inference.post('/explain', { features });
  const text = explanationText({
    label: r.data.label, confidence: r.data.confidence,
    deviceName: meta?.device_name, shapFeatures: r.data.features,
    severity: extras.severity, score: extras.threat_score,
  });
  return { ...r.data, explanation_text: text };
}

// preferred: explain a stored flow by id (features come from persistence)
app.post('/explain/:flowId', async (req, res) => {
  const { flowId } = req.params;
  try {
    let features; let meta = null;
    const cached = state.flowCache.get(flowId);
    if (cached) ({ features, meta } = cached);
    else {
      const row = stmts.getFlow.get(flowId);
      if (!row) return res.status(404).json({ error: `flow ${flowId} not found` });
      features = JSON.parse(row.features);
    }
    const threat = state.threats.find((t) => t.meta.flow_id === flowId);
    const out = await explainByFeatures(features, meta || threat?.meta, {
      severity: threat?.severity, threat_score: threat?.threat_score,
    });
    try {
      out.features.forEach((f, i) => stmts.insertExplanation.run({
        flow_id: flowId, feature: f.feature, contribution: f.impact,
        value: f.value ?? null, explanation: i === 0 ? out.explanation_text : null,
      }));
    } catch (e) { console.error('[db]', e.message); }
    io.emit('explanation', { flow_id: flowId, ...out });
    res.json({ flow_id: flowId, ...out });
  } catch (e) {
    res.status(502).json({ error: `inference service: ${e.message}` });
  }
});

// legacy: explain raw features (kept for compatibility)
app.post('/explain', async (req, res) => {
  try {
    res.json(await explainByFeatures(req.body.features, req.body.meta));
  } catch (e) {
    res.status(502).json({ error: `inference service: ${e.message}` });
  }
});

// ---- incidents ----
app.get('/incidents', (req, res) => {
  res.json({ incidents: incidents.list(Number(req.query.limit) || 50) });
});
app.patch('/incidents/:id', (req, res) => {
  try {
    const inc = incidents.update(Number(req.params.id), req.body.status);
    if (!inc) return res.status(404).json({ error: 'incident not found' });
    res.json({ incident: inc });
  } catch (e) {
    res.status(400).json({ error: e.message });
  }
});
app.get('/responses', (req, res) => res.json(responseEngine.snapshot()));

// ---- notifications ----
app.get('/notifications', (req, res) => {
  res.json({ notifications: notifier.list(Number(req.query.limit) || 50) });
});
app.get('/settings/notifications', (req, res) => res.json(notifier.publicSettings()));
app.put('/settings/notifications', (req, res) => {
  try {
    const patch = { ...req.body };
    // masked secrets round-tripped from the UI must not overwrite real values
    if (patch.smtp?.pass === '••••••') delete patch.smtp.pass;
    if (patch.twilio?.authToken === '••••••') delete patch.twilio.authToken;
    res.json(notifier.saveSettings(patch));
  } catch (e) { res.status(400).json({ error: e.message }); }
});
app.post('/notify/test-email', async (req, res) => res.json(await notifier.testEmail(req.body?.to)));
app.post('/notify/test-sms', async (req, res) => res.json(await notifier.testSms(req.body?.to)));

// inject one real attack flow through the full pipeline (simulator -> model ->
// scoring -> response -> notifications) without running the whole scenario
app.post('/test/attack', async (req, res) => {
  if (!state.devices.length) return res.status(503).json({ error: 'services not connected yet' });
  const critical = Boolean(req.body?.critical);
  // home-05 (Malware) and cam-01 (DDoS) classify >90% => Critical path;
  // router-03 (Scanning) classifies mid-confidence => High/Medium path
  const deviceId = req.body?.deviceId || (critical ? 'home-05' : 'router-03');
  try {
    await scenario._emitFlows(deviceId, 'attack', 1);
    // release the device + contain the incident shortly after (test hygiene)
    setTimeout(() => {
      try { responseEngine.onTick((scenario.tick < 0 ? 0 : scenario.tick) + 1000); } catch { /* noop */ }
    }, 15000);
    res.json({ ok: true, deviceId, critical });
  } catch (e) { res.status(502).json({ error: e.message }); }
});

// ---- analytics ----
app.get('/analytics', (req, res) => {
  res.json(dbmod.analytics(stmts, req.query.window || 'day'));
});
app.get('/analytics/export.csv', (req, res) => {
  res.setHeader('Content-Type', 'text/csv');
  res.setHeader('Content-Disposition', 'attachment; filename="edgeshield-threats.csv"');
  res.send(dbmod.threatsCsv(stmts));
});

// ---- federated learning ----
app.get('/federated-status', (req, res) => {
  const p = path.join(REPO, 'federated-learning', 'history.json');
  if (!fs.existsSync(p)) {
    return res.json({ available: false, note: 'run federated-learning/run_federated.py' });
  }
  return res.json({ available: true, live: flRunner.status(), ...JSON.parse(fs.readFileSync(p, 'utf8')) });
});
app.post('/federated/run', (req, res) => {
  const out = flRunner.start({
    rounds: Number(req.body?.rounds) || 5,
    localEpochs: Number(req.body?.localEpochs) || 1,
    engine: req.body?.engine === 'flower' ? 'flower' : 'builtin',
  });
  res.status(out.ok ? 200 : 409).json(out);
});
app.post('/federated/stop', (req, res) => res.json(flRunner.stop()));

app.get('/benchmarks', (req, res) => {
  const p = path.join(REPO, 'benchmarks', 'results.json');
  if (!fs.existsSync(p)) {
    return res.json({ available: false, note: 'run benchmarks/compare_baseline.py' });
  }
  return res.json({ available: true, ...JSON.parse(fs.readFileSync(p, 'utf8')) });
});

// scenario controls
app.post('/scenario/start', async (req, res) => {
  if (!state.devices.length) {
    return res.status(503).json({ error: 'python services not connected yet — try again in a few seconds' });
  }
  return res.json(await scenario.start(state.devices.map((d) => d.id)));
});
app.post('/scenario/pause', (req, res) => res.json(scenario.pause()));
app.post('/scenario/reset', async (req, res) => res.json(await scenario.reset()));
app.get('/scenario/status', (req, res) => res.json(scenario.status()));

// ---------------- sockets ---------------------------------------------------
io.on('connection', (socket) => {
  socket.emit('init', {
    devices: state.devices,
    metrics: metricsSnapshot(),
    modelInfo: state.modelInfo,
    scenario: scenario.status(),
    threats: state.threats.slice(-50),
    predictions: state.predictions.slice(-50),
    responses: responseEngine.snapshot().responses,
    incidents: incidents.list(20),
    federatedLive: flRunner.status(),
    notifications: notifier.list(30),
    notificationSettings: notifier.publicSettings(),
    dataSource: state.dataSource,
  });
});

server.listen(PORT, () => {
  console.log(`[backend] listening on :${PORT} · db=${dbmod.DB_PATH}`);
  bootstrap();
});
