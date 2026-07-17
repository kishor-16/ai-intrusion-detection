/**
 * Persistence layer — embedded SQL (better-sqlite3, WAL mode, zero-install).
 * Schema mirrors the platform spec (devices / threats / explanations /
 * fl_rounds / incidents / metrics_samples / flows). Everything the dashboard
 * shows can be traced back to a row here. The engine is embedded so the demo
 * stays offline + one-command; the schema is plain SQL, so swapping to
 * PostgreSQL is a driver change, not a redesign.
 */

const path = require('path');
const Database = require('better-sqlite3');

const DB_PATH = process.env.EDGESHIELD_DB
  || path.join(__dirname, 'data', 'edgeshield.db');

let db;

function init() {
  const fs = require('fs');
  fs.mkdirSync(path.dirname(DB_PATH), { recursive: true });
  db = new Database(DB_PATH);
  db.pragma('journal_mode = WAL');
  db.exec(`
    CREATE TABLE IF NOT EXISTS devices (
      id         TEXT PRIMARY KEY,
      device_name TEXT NOT NULL,
      ip         TEXT,
      status     TEXT NOT NULL DEFAULT 'online',
      last_seen  TEXT
    );
    CREATE TABLE IF NOT EXISTS flows (
      flow_id   TEXT PRIMARY KEY,
      device_id TEXT,
      ts        REAL,
      features  TEXT NOT NULL          -- JSON array, raw schema order
    );
    CREATE TABLE IF NOT EXISTS threats (
      flow_id        TEXT PRIMARY KEY,
      ts             REAL NOT NULL,
      device_id      TEXT,
      device_name    TEXT,
      source_ip      TEXT,
      destination_ip TEXT,
      prediction     TEXT NOT NULL,
      confidence     REAL NOT NULL,
      latency_ms     REAL,
      threat_score   INTEGER,
      severity       TEXT,
      run_id         INTEGER
    );
    CREATE TABLE IF NOT EXISTS predictions_log (
      flow_id    TEXT PRIMARY KEY,
      ts         REAL NOT NULL,
      device_id  TEXT,
      prediction TEXT NOT NULL,
      confidence REAL NOT NULL,
      latency_ms REAL,
      true_class TEXT,
      run_id     INTEGER
    );
    CREATE TABLE IF NOT EXISTS explanations (
      flow_id     TEXT NOT NULL,
      feature     TEXT NOT NULL,
      contribution REAL NOT NULL,
      value       REAL,
      explanation TEXT,               -- NL text stored on the first row
      created_at  TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS incidents (
      id          INTEGER PRIMARY KEY AUTOINCREMENT,
      flow_id     TEXT,
      device_id   TEXT,
      device_name TEXT,
      attack      TEXT,
      severity    TEXT,
      threat_score INTEGER,
      status      TEXT NOT NULL DEFAULT 'Open',
      recommended_action TEXT,
      created_at  REAL NOT NULL,
      updated_at  REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS fl_rounds (
      run_started REAL,
      round       INTEGER,
      accuracy    REAL,
      f1_macro    REAL,
      engine      TEXT
    );
    CREATE TABLE IF NOT EXISTS notifications (
      id          INTEGER PRIMARY KEY AUTOINCREMENT,
      incident_id INTEGER,
      flow_id     TEXT,
      attack      TEXT,
      severity    TEXT,
      channel     TEXT NOT NULL,       -- email | sms | browser | alarm | toast
      recipient   TEXT,
      status      TEXT NOT NULL,       -- queued | sent | simulated | failed
      attempts    INTEGER DEFAULT 0,
      last_error  TEXT,
      created_at  REAL NOT NULL,
      sent_at     REAL
    );
    CREATE TABLE IF NOT EXISTS settings (
      key   TEXT PRIMARY KEY,
      value TEXT NOT NULL              -- JSON
    );
    CREATE TABLE IF NOT EXISTS metrics_samples (
      ts               REAL PRIMARY KEY,
      total_flows      INTEGER,
      threats          INTEGER,
      avg_latency_ms   REAL,
      avg_confidence   REAL,
      live_accuracy    REAL
    );
    CREATE INDEX IF NOT EXISTS idx_threats_ts ON threats (ts);
    CREATE INDEX IF NOT EXISTS idx_threats_device ON threats (device_id);
    CREATE INDEX IF NOT EXISTS idx_pred_ts ON predictions_log (ts);
  `);

  return {
    upsertDevice: db.prepare(`
      INSERT INTO devices (id, device_name, ip, status, last_seen)
      VALUES (@id, @device_name, @ip, @status, datetime('now'))
      ON CONFLICT(id) DO UPDATE SET status = excluded.status, last_seen = excluded.last_seen`),
    insertFlow: db.prepare(`
      INSERT OR REPLACE INTO flows (flow_id, device_id, ts, features)
      VALUES (@flow_id, @device_id, @ts, @features)`),
    insertThreat: db.prepare(`
      INSERT OR REPLACE INTO threats
      (flow_id, ts, device_id, device_name, source_ip, destination_ip,
       prediction, confidence, latency_ms, threat_score, severity, run_id)
      VALUES (@flow_id, @ts, @device_id, @device_name, @source_ip, @destination_ip,
       @prediction, @confidence, @latency_ms, @threat_score, @severity, @run_id)`),
    insertPrediction: db.prepare(`
      INSERT OR REPLACE INTO predictions_log
      (flow_id, ts, device_id, prediction, confidence, latency_ms, true_class, run_id)
      VALUES (@flow_id, @ts, @device_id, @prediction, @confidence, @latency_ms, @true_class, @run_id)`),
    insertExplanation: db.prepare(`
      INSERT INTO explanations (flow_id, feature, contribution, value, explanation)
      VALUES (@flow_id, @feature, @contribution, @value, @explanation)`),
    insertIncident: db.prepare(`
      INSERT INTO incidents
      (flow_id, device_id, device_name, attack, severity, threat_score,
       status, recommended_action, created_at, updated_at)
      VALUES (@flow_id, @device_id, @device_name, @attack, @severity, @threat_score,
       'Open', @recommended_action, @created_at, @updated_at)`),
    updateIncident: db.prepare(
      'UPDATE incidents SET status = @status, updated_at = @updated_at WHERE id = @id'),
    getIncidents: db.prepare('SELECT * FROM incidents ORDER BY id DESC LIMIT ?'),
    getIncident: db.prepare('SELECT * FROM incidents WHERE id = ?'),
    getFlow: db.prepare('SELECT * FROM flows WHERE flow_id = ?'),
    insertFlRound: db.prepare(`
      INSERT INTO fl_rounds (run_started, round, accuracy, f1_macro, engine)
      VALUES (@run_started, @round, @accuracy, @f1_macro, @engine)`),
    insertNotification: db.prepare(`
      INSERT INTO notifications
      (incident_id, flow_id, attack, severity, channel, recipient, status, attempts, last_error, created_at, sent_at)
      VALUES (@incident_id, @flow_id, @attack, @severity, @channel, @recipient, @status, @attempts, @last_error, @created_at, @sent_at)`),
    updateNotification: db.prepare(`
      UPDATE notifications SET status = @status, attempts = @attempts,
        last_error = @last_error, sent_at = @sent_at WHERE id = @id`),
    getNotifications: db.prepare('SELECT * FROM notifications ORDER BY id DESC LIMIT ?'),
    getSetting: db.prepare('SELECT value FROM settings WHERE key = ?'),
    setSetting: db.prepare(`
      INSERT INTO settings (key, value) VALUES (@key, @value)
      ON CONFLICT(key) DO UPDATE SET value = excluded.value`),
    insertMetricsSample: db.prepare(`
      INSERT OR REPLACE INTO metrics_samples
      (ts, total_flows, threats, avg_latency_ms, avg_confidence, live_accuracy)
      VALUES (@ts, @total_flows, @threats, @avg_latency_ms, @avg_confidence, @live_accuracy)`),
    raw: db,
  };
}

// ---------- analytics queries (computed from persisted rows) ----------
const WINDOWS = { hour: 3600, day: 86400, week: 604800, month: 2592000, all: 1e12 };

function analytics(stmts, windowName = 'day') {
  const seconds = WINDOWS[windowName] || WINDOWS.day;
  const since = Date.now() / 1000 - seconds;
  const d = stmts.raw;
  const dist = d.prepare(`
    SELECT prediction AS attack, COUNT(*) AS count, ROUND(AVG(confidence), 4) AS avg_confidence,
           ROUND(AVG(threat_score), 1) AS avg_score
    FROM threats WHERE ts >= ? GROUP BY prediction ORDER BY count DESC`).all(since);
  const byDevice = d.prepare(`
    SELECT device_name AS device, COUNT(*) AS attacks, ROUND(AVG(confidence), 4) AS avg_confidence
    FROM threats WHERE ts >= ? GROUP BY device_id ORDER BY attacks DESC`).all(since);
  const timeline = d.prepare(`
    SELECT CAST(ts / 60 AS INTEGER) * 60 AS minute, COUNT(*) AS threats
    FROM threats WHERE ts >= ? GROUP BY minute ORDER BY minute`).all(since);
  const latency = d.prepare(`
    SELECT ROUND(AVG(latency_ms), 3) AS mean, MIN(latency_ms) AS min, MAX(latency_ms) AS max,
           COUNT(*) AS n
    FROM predictions_log WHERE ts >= ?`).get(since);
  const traffic = d.prepare(`
    SELECT SUM(prediction = 'Normal') AS normal, SUM(prediction != 'Normal') AS malicious,
           ROUND(AVG(confidence), 4) AS avg_confidence,
           ROUND(AVG(prediction = true_class) * 1.0, 4) AS accuracy_vs_ground_truth
    FROM predictions_log WHERE ts >= ?`).get(since);
  const incidents = d.prepare(`
    SELECT severity, status, COUNT(*) AS count FROM incidents
    WHERE created_at >= ? GROUP BY severity, status`).all(since);
  return {
    window: windowName,
    generated_at: new Date().toISOString(),
    attack_distribution: dist,
    attacks_by_device: byDevice,
    threat_timeline: timeline,
    latency_trend: latency,
    traffic,
    incidents_summary: incidents,
    source: 'SQL aggregation over persisted predictions/threats/incidents',
  };
}

function threatsCsv(stmts, limit = 10000) {
  const rows = stmts.raw.prepare(
    'SELECT * FROM threats ORDER BY ts DESC LIMIT ?').all(limit);
  const cols = ['flow_id', 'ts', 'device_id', 'device_name', 'source_ip', 'destination_ip',
    'prediction', 'confidence', 'latency_ms', 'threat_score', 'severity'];
  const esc = (v) => (v == null ? '' : /[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));
  return [cols.join(','), ...rows.map((r) => cols.map((c) => esc(r[c])).join(','))].join('\n');
}

module.exports = { init, analytics, threatsCsv, DB_PATH };
