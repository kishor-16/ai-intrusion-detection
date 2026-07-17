/**
 * EdgeShield backend — real-time gateway (Node + Express + Socket.IO). Port 4000.
 *
 * Responsibilities: host the deterministic demo scenario engine, relay
 * simulator flows through the ML inference service, broadcast everything to
 * the dashboard, expose REST snapshots. NO ML logic and NO metric fabrication:
 * every number passing through here originated in the inference service,
 * ml-engine artifacts, or ground-truth-vs-prediction bookkeeping (live accuracy).
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

const PORT = Number(process.env.BACKEND_PORT || 4000);
const INFERENCE_URL = process.env.INFERENCE_URL || 'http://127.0.0.1:8100';
const SIMULATOR_URL = process.env.SIMULATOR_URL || 'http://127.0.0.1:8200';
const REPO = path.join(__dirname, '..');

const inference = axios.create({ baseURL: INFERENCE_URL, timeout: 10000 });
const simulator = axios.create({ baseURL: SIMULATOR_URL, timeout: 10000 });

const app = express();
app.use(cors());
app.use(express.json());
const server = http.createServer(app);
const io = new Server(server, { cors: { origin: '*' } });

// ---------------- in-memory state (rebuilt every run; Reset clears it) -------
const state = {
  devices: [],
  dataSource: 'unknown',
  predictions: [],   // last 400
  threats: [],       // last 200
  metrics: {
    totalFlows: 0,
    threatsDetected: 0,
    latencySumMs: 0,      // -> avg = sum/totalFlows (real per-request latencies)
    liveCorrect: 0,       // prediction === simulator ground truth
    liveTotal: 0,
  },
  modelInfo: null,
  serviceStatus: { inference: 'unknown', simulator: 'unknown' },
};

function metricsSnapshot() {
  const m = state.metrics;
  return {
    totalDevices: state.devices.length,
    activeConnections: io.engine ? io.engine.clientsCount : 0,
    totalFlows: m.totalFlows,
    threatsDetected: m.threatsDetected,
    avgLatencyMs: m.totalFlows ? +(m.latencySumMs / m.totalFlows).toFixed(3) : null,
    // "Live AI accuracy" = predictions matching the simulator's ground-truth
    // class label for each flow, measured during this run. Real, not scripted.
    liveAccuracy: m.liveTotal ? +(m.liveCorrect / m.liveTotal).toFixed(4) : null,
    liveSamples: m.liveTotal,
    dataSource: state.dataSource,
    simulated: true,
  };
}

function broadcast(event, payload) {
  // book-keeping on the way through
  if (event === 'prediction') {
    const { meta, prediction } = payload;
    state.metrics.totalFlows += 1;
    state.metrics.latencySumMs += prediction.latency_ms;
    state.metrics.liveTotal += 1;
    if (prediction.label === meta.true_class) state.metrics.liveCorrect += 1;
    state.predictions.push(payload);
    if (state.predictions.length > 400) state.predictions.shift();
  }
  if (event === 'threat') {
    state.metrics.threatsDetected += 1;
    state.threats.push(payload);
    if (state.threats.length > 200) state.threats.shift();
  }
  if (event === 'device-status') {
    const d = state.devices.find((x) => x.id === payload.deviceId);
    if (d) d.status = payload.status;
  }
  if (event === 'scenario-reset') {
    state.predictions = [];
    state.threats = [];
    state.metrics = { totalFlows: 0, threatsDetected: 0, latencySumMs: 0, liveCorrect: 0, liveTotal: 0 };
    state.devices.forEach((d) => { d.status = 'online'; });
    io.emit('metrics', metricsSnapshot());
    io.emit('devices', state.devices);
  }
  io.emit(event, payload);
  if (event === 'prediction' || event === 'threat') io.emit('metrics', metricsSnapshot());
}

const scenario = new Scenario({ simulator, inference, onEvent: broadcast });

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
app.get('/health', (req, res) => res.json({ status: 'ok', services: state.serviceStatus }));
app.get('/devices', (req, res) => res.json({ devices: state.devices, dataSource: state.dataSource }));
app.get('/threats', (req, res) => res.json({ threats: state.threats.slice(-100) }));
app.get('/predictions', (req, res) => res.json({ predictions: state.predictions.slice(-100) }));
app.get('/metrics', (req, res) => res.json(metricsSnapshot()));
app.get('/model-performance', (req, res) => {
  res.json({ modelInfo: state.modelInfo });
});

app.get('/federated-status', (req, res) => {
  const p = path.join(REPO, 'federated-learning', 'history.json');
  if (!fs.existsSync(p)) {
    return res.json({ available: false, note: 'run federated-learning/run_federated.py' });
  }
  return res.json({ available: true, ...JSON.parse(fs.readFileSync(p, 'utf8')) });
});

app.get('/benchmarks', (req, res) => {
  const p = path.join(REPO, 'benchmarks', 'results.json');
  if (!fs.existsSync(p)) {
    return res.json({ available: false, note: 'run benchmarks/compare_baseline.py' });
  }
  return res.json({ available: true, ...JSON.parse(fs.readFileSync(p, 'utf8')) });
});

// live per-instance SHAP relay (Explanation Center)
app.post('/explain', async (req, res) => {
  try {
    const r = await inference.post('/explain', { features: req.body.features });
    res.json(r.data);
  } catch (e) {
    res.status(502).json({ error: `inference service: ${e.message}` });
  }
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
    dataSource: state.dataSource,
  });
});

server.listen(PORT, () => {
  console.log(`[backend] listening on :${PORT}`);
  bootstrap();
});
