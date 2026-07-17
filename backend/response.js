/**
 * Autonomous Response Engine — drives device security state from REAL model
 * detections (not the scenario script). Deterministic rule set:
 *
 *   confidence > 0.90            -> QUARANTINE device, block its traffic at the
 *                                   gateway, open an incident, notify the feed
 *   confidence 0.60–0.90         -> mark UNDER ATTACK
 *   confidence < 0.60            -> mark SUSPICIOUS (a lead, not a conviction)
 *   no detections for 4 ticks    -> recover to PROTECTED (quarantine: 8 ticks)
 *   protected + 6 quiet ticks    -> back to ONLINE
 *
 * Device states: online | suspicious | under-attack | quarantined | protected
 * (offline is reserved for simulator loss). Every transition is emitted as a
 * device-status event and every response action as a 'response' feed event.
 */

const { responseText } = require('./nlg');

const QUARANTINE_CONF = Number(process.env.QUARANTINE_CONF || 0.90);
const ATTACK_CONF = 0.60;
const RECOVER_TICKS = 4;
const RELEASE_TICKS = 8;
const ONLINE_TICKS = 6;

class ResponseEngine {
  constructor({ onEvent, incidents }) {
    this.onEvent = onEvent;
    this.incidents = incidents;
    this.reset();
  }

  reset() {
    this.devices = {};        // id -> { status, lastDetTick, quietSince, openIncidents: {label: incidentId} }
    this.blockedFlows = 0;
    this.responses = [];      // recent response feed entries
  }

  _dev(id) {
    if (!this.devices[id]) {
      this.devices[id] = { status: 'online', lastDetTick: -1, quietSince: -1, openIncidents: {} };
    }
    return this.devices[id];
  }

  isBlocked(deviceId) { return this._dev(deviceId).status === 'quarantined'; }

  countBlockedFlow() { this.blockedFlows += 1; }

  _setStatus(deviceId, status) {
    const d = this._dev(deviceId);
    if (d.status === status) return;
    d.status = status;
    this.onEvent('device-status', { deviceId, status });
  }

  _respond(action, meta, prediction, extra = {}) {
    const entry = {
      kind: 'response',
      id: `resp-${Date.now()}-${meta.device_id}-${action}`,
      action,
      device_id: meta.device_id,
      device_name: meta.device_name,
      label: prediction?.label,
      confidence: prediction?.confidence,
      ts: Date.now() / 1000,
      text: responseText({
        action, deviceName: meta.device_name,
        label: prediction?.label, confidence: prediction?.confidence,
      }),
      ...extra,
    };
    this.responses.push(entry);
    if (this.responses.length > 100) this.responses.shift();
    this.onEvent('response', entry);
    return entry;
  }

  /** Called for every threat detection (label ≠ Normal). */
  onDetection(payload, tick) {
    const { meta, prediction, threat_score: score, severity } = payload;
    const d = this._dev(meta.device_id);
    // tick is -1 for detections outside a scenario run (e.g. test attacks);
    // clamp to 0 so the recovery loop still sees and releases the device
    d.lastDetTick = tick < 0 ? 0 : tick;

    if (prediction.confidence > QUARANTINE_CONF && d.status !== 'quarantined') {
      this._setStatus(meta.device_id, 'quarantined');
      const resp = this._respond('quarantine', meta, prediction, { threat_score: score, severity });
      const incident = this.incidents.create({
        flow_id: meta.flow_id,
        device_id: meta.device_id,
        device_name: meta.device_name,
        attack: prediction.label,
        severity: severity || 'Critical',
        threat_score: score,
      });
      d.openIncidents[prediction.label] = incident.id;
      resp.incident_id = incident.id;
    } else if (d.status !== 'quarantined') {
      if (prediction.confidence >= ATTACK_CONF) {
        this._setStatus(meta.device_id, 'under-attack');
      } else if (d.status === 'online' || d.status === 'protected') {
        this._setStatus(meta.device_id, 'suspicious');
        this._respond('watch', meta, prediction);
      }
      // High-risk without quarantine still opens ONE incident per device+class
      if (severity === 'Critical' && !d.openIncidents[prediction.label]) {
        const incident = this.incidents.create({
          flow_id: meta.flow_id,
          device_id: meta.device_id,
          device_name: meta.device_name,
          attack: prediction.label,
          severity,
          threat_score: score,
        });
        d.openIncidents[prediction.label] = incident.id;
      }
    }
  }

  /** Called once per scenario tick — recovery / release logic. */
  onTick(tick) {
    for (const [id, d] of Object.entries(this.devices)) {
      const quiet = tick - d.lastDetTick;
      if (d.lastDetTick < 0) continue;
      if (d.status === 'quarantined' && quiet >= RELEASE_TICKS) {
        this._setStatus(id, 'protected');
        d.quietSince = tick;
        this._respond('release', { device_id: id, device_name: this.names?.[id] || id }, null);
        for (const [label, incId] of Object.entries(d.openIncidents)) {
          this.incidents.update(incId, 'Contained');
          delete d.openIncidents[label];
        }
      } else if ((d.status === 'under-attack' || d.status === 'suspicious') && quiet >= RECOVER_TICKS) {
        this._setStatus(id, 'protected');
        d.quietSince = tick;
        this._respond('recover', { device_id: id, device_name: this.names?.[id] || id }, null);
        for (const [label, incId] of Object.entries(d.openIncidents)) {
          this.incidents.update(incId, 'Contained');
          delete d.openIncidents[label];
        }
      } else if (d.status === 'protected' && d.quietSince >= 0 && tick - d.quietSince >= ONLINE_TICKS) {
        this._setStatus(id, 'online');
      }
    }
  }

  setDeviceNames(map) { this.names = map; }

  /** Open incident id for a device+attack, if one exists (for notifications). */
  openIncidentFor(deviceId, label) {
    return this._dev(deviceId).openIncidents[label] ?? null;
  }

  snapshot() {
    return {
      blockedFlows: this.blockedFlows,
      responses: this.responses.slice(-50),
      quarantined: Object.entries(this.devices)
        .filter(([, d]) => d.status === 'quarantined').map(([id]) => id),
    };
  }
}

module.exports = { ResponseEngine, QUARANTINE_CONF };
