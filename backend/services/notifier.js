/**
 * Notification orchestrator — turns real detections into notifications by
 * severity policy, with persistence, cooldown de-duplication, retries and
 * live socket events.
 *
 *   Low      -> dashboard feed only (already broadcast)
 *   Medium   -> + notification record (DB) + bell/toast event
 *   High     -> + email
 *   Critical -> + email + SMS + browser notification + alarm
 *
 * The "threshold" setting gates the outbound channels:
 *   'critical' = email/SMS only for Critical
 *   'high'     = email for High+Critical (default)
 *   'all'      = email from Medium up
 */

const { sendAlertEmail, isConfigured: emailConfigured } = require('./emailService');
const { sendAlertSms, isConfigured: smsConfigured } = require('./smsService');
const { recommendedAction } = require('../scoring');

const RETRY_MAX = 3;
const RETRY_DELAY_MS = 10000;
const COOLDOWN_MS = 60000;   // one email/SMS per device+attack per minute

const DEFAULT_SETTINGS = {
  emailRecipients: (process.env.EMAIL_TO || '').split(',').map((s) => s.trim()).filter(Boolean),
  phoneNumbers: (process.env.ALERT_PHONE || '').split(',').map((s) => s.trim()).filter(Boolean),
  threshold: 'high',
  alarmEnabled: true,
  alarmVolume: 0.6,
  browserNotifications: true,
  smtp: {},     // optional overrides of env SMTP_*
  twilio: {},   // optional overrides of env TWILIO_*
};

class Notifier {
  constructor({ stmts, onEvent }) {
    this.stmts = stmts;
    this.onEvent = onEvent;   // (event, payload) => io.emit
    this.cooldown = new Map();
    this.settings = this._loadSettings();
  }

  // ---------------- settings ----------------
  _loadSettings() {
    try {
      const rowValue = this.stmts.getSetting.get('notifications');
      if (rowValue) return { ...DEFAULT_SETTINGS, ...JSON.parse(rowValue.value) };
    } catch (e) { console.error('[notify] settings load:', e.message); }
    return { ...DEFAULT_SETTINGS };
  }

  saveSettings(patch) {
    this.settings = { ...this.settings, ...patch };
    this.stmts.setSetting.run({ key: 'notifications', value: JSON.stringify(this.settings) });
    this.onEvent('notification-settings', this.publicSettings());
    return this.publicSettings();
  }

  /** Settings safe to send to the dashboard (secrets masked). */
  publicSettings() {
    const s = this.settings;
    const mask = (v) => (v ? '••••••' : '');
    return {
      ...s,
      smtp: { ...s.smtp, pass: mask(s.smtp?.pass) },
      twilio: { ...s.twilio, authToken: mask(s.twilio?.authToken) },
      channels: {
        emailConfigured: emailConfigured(s),
        smsConfigured: smsConfigured(s),
      },
    };
  }

  // ---------------- persistence ----------------
  _record(fields) {
    const info = this.stmts.insertNotification.run({
      incident_id: fields.incident_id ?? null, flow_id: fields.flow_id ?? null,
      attack: fields.attack ?? null, severity: fields.severity ?? null,
      channel: fields.channel, recipient: fields.recipient ?? null,
      status: fields.status, attempts: fields.attempts ?? 0,
      last_error: fields.last_error ?? null,
      created_at: Date.now() / 1000, sent_at: fields.sent_at ?? null,
    });
    const rec = { id: info.lastInsertRowid, ...fields, created_at: Date.now() / 1000 };
    this.onEvent('notification', rec);
    return rec;
  }

  _update(rec, { status, attempts, last_error }) {
    this.stmts.updateNotification.run({
      id: rec.id, status, attempts,
      last_error: last_error ?? null,
      sent_at: status === 'sent' || status === 'simulated' ? Date.now() / 1000 : null,
    });
    this.onEvent('notification', { ...rec, status, attempts, last_error });
  }

  list(limit = 50) { return this.stmts.getNotifications.all(limit); }

  // ---------------- dispatch with retry ----------------
  async _dispatch(rec, sendFn) {
    for (let attempt = 1; attempt <= RETRY_MAX; attempt += 1) {
      try {
        const out = await sendFn();
        this._update(rec, { status: out.status, attempts: attempt });
        return;
      } catch (e) {
        console.error(`[notify] ${rec.channel} attempt ${attempt}/${RETRY_MAX} failed:`, e.message);
        this._update(rec, {
          status: attempt < RETRY_MAX ? 'queued' : 'failed',
          attempts: attempt, last_error: e.message,
        });
        if (attempt < RETRY_MAX) {
          await new Promise((r) => setTimeout(r, RETRY_DELAY_MS * attempt));
        }
      }
    }
  }

  // ---------------- alert building ----------------
  buildAlert(payload) {
    const { meta, prediction } = payload;
    return {
      attack: prediction.label,
      severity: payload.severity || 'Low',
      score: payload.threat_score ?? 0,
      confidence: (prediction.confidence * 100).toFixed(1),
      deviceName: meta.device_name,
      deviceId: meta.device_id,
      flowId: meta.flow_id,
      time: new Date((meta.timestamp || Date.now() / 1000) * 1000).toLocaleString(),
      srcIp: meta.src_ip, dstIp: meta.dst_ip,
      protocol: meta.protocol, packetRate: meta.packet_rate,
      latencyMs: prediction.latency_ms,
      shapFeatures: prediction.shap_top || [],
      recommendedAction: recommendedAction(prediction.label),
      explanation: `The model classified this flow as ${prediction.label} with `
        + `${(prediction.confidence * 100).toFixed(1)}% confidence`
        + (prediction.shap_top?.length
          ? `; top contributing features: ${prediction.shap_top.slice(0, 4).map((f) => f.feature).join(', ')}.`
          : '.'),
    };
  }

  _emailAllowed(severity) {
    const t = this.settings.threshold;
    if (t === 'all') return ['Medium', 'High', 'Critical'].includes(severity);
    if (t === 'critical') return severity === 'Critical';
    return severity === 'High' || severity === 'Critical';
  }

  // ---------------- main entry: called for every threat -------------------
  handleThreat(payload, incidentId = null) {
    const severity = payload.severity || 'Low';
    if (severity === 'Low' || severity === 'None') return;   // dashboard feed only

    const alert = this.buildAlert(payload);
    const base = {
      incident_id: incidentId, flow_id: alert.flowId,
      attack: alert.attack, severity,
    };

    // bell/toast record for every Medium+ detection (DB-persisted)
    this._record({ ...base, channel: 'toast', recipient: 'dashboard', status: 'sent', sent_at: Date.now() / 1000 });

    // cooldown: at most one outbound email/SMS per device+attack per minute
    const key = `${alert.deviceId}:${alert.attack}`;
    const last = this.cooldown.get(key) || 0;
    const now = Date.now();
    const inCooldown = now - last < COOLDOWN_MS;

    if (this._emailAllowed(severity) && !inCooldown) {
      for (const to of this.settings.emailRecipients.length
        ? this.settings.emailRecipients : ['(no recipients configured)']) {
        const rec = this._record({ ...base, channel: 'email', recipient: to, status: 'queued' });
        this._dispatch(rec, () => sendAlertEmail({ to, alert, settings: this.settings }));
      }
    }

    if (severity === 'Critical') {
      if (!inCooldown) {
        for (const to of this.settings.phoneNumbers.length
          ? this.settings.phoneNumbers : ['(no phone configured)']) {
          const rec = this._record({ ...base, channel: 'sms', recipient: to, status: 'queued' });
          this._dispatch(rec, () => sendAlertSms({ to, alert, settings: this.settings }));
        }
      }
      if (this.settings.browserNotifications) {
        this._record({ ...base, channel: 'browser', recipient: 'dashboard', status: 'sent', sent_at: now / 1000 });
        this.onEvent('browser-notify', {
          title: '🚨 Critical Threat Detected',
          body: `Attack: ${alert.attack}\nDevice: ${alert.deviceName}\nConfidence: ${alert.confidence}%`,
          flowId: alert.flowId,
        });
      }
      if (this.settings.alarmEnabled) {
        this._record({ ...base, channel: 'alarm', recipient: 'dashboard', status: 'sent', sent_at: now / 1000 });
        this.onEvent('alarm', {
          incident_id: incidentId, device_id: alert.deviceId, device_name: alert.deviceName,
          attack: alert.attack, confidence: alert.confidence,
          volume: this.settings.alarmVolume,
        });
      }
    }
    if (!inCooldown) this.cooldown.set(key, now);
  }

  // ---------------- tests ----------------
  sampleAlert(severity = 'Critical') {
    return {
      attack: 'DDoS', severity, score: severity === 'Critical' ? 92 : 71,
      confidence: '98.2', deviceName: 'Smart Camera', deviceId: 'cam-01',
      flowId: 'test-0001', time: new Date().toLocaleString(),
      srcIp: '185.220.101.42', dstIp: '10.0.1.11', protocol: 'TCP',
      packetRate: 2154.7, latencyMs: 0.021,
      shapFeatures: [{ feature: 'Protocol Type' }, { feature: 'Header_Length' },
        { feature: 'IAT' }, { feature: 'rst_count' }],
      recommendedAction: recommendedAction('DDoS'),
      explanation: 'TEST notification — verifying the EdgeShield AI alert delivery pipeline.',
    };
  }

  async testEmail(to) {
    const recipient = to || this.settings.emailRecipients[0] || '(no recipients configured)';
    const rec = this._record({
      attack: 'DDoS', severity: 'Critical', channel: 'email',
      recipient, status: 'queued', flow_id: 'test-0001',
    });
    try {
      const out = await sendAlertEmail({ to: recipient, alert: this.sampleAlert(), settings: this.settings });
      this._update(rec, { status: out.status, attempts: 1 });
      return { ok: true, ...out };
    } catch (e) {
      this._update(rec, { status: 'failed', attempts: 1, last_error: e.message });
      return { ok: false, error: e.message };
    }
  }

  async testSms(to) {
    const recipient = to || this.settings.phoneNumbers[0] || '(no phone configured)';
    const rec = this._record({
      attack: 'DDoS', severity: 'Critical', channel: 'sms',
      recipient, status: 'queued', flow_id: 'test-0001',
    });
    try {
      const out = await sendAlertSms({ to: recipient, alert: this.sampleAlert(), settings: this.settings });
      this._update(rec, { status: out.status, attempts: 1 });
      return { ok: true, ...out };
    } catch (e) {
      this._update(rec, { status: 'failed', attempts: 1, last_error: e.message });
      return { ok: false, error: e.message };
    }
  }
}

module.exports = { Notifier };
