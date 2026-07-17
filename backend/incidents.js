/**
 * Incident Management — every high-risk detection opens an incident with a
 * lifecycle (Open -> Investigating -> Contained -> Resolved). Persisted to the
 * incidents table; every create/update is broadcast on the 'incident' channel.
 */

const { recommendedAction } = require('./scoring');

const STATUSES = ['Open', 'Investigating', 'Contained', 'Resolved'];

class IncidentManager {
  constructor({ stmts, onEvent }) {
    this.stmts = stmts;
    this.onEvent = onEvent;
  }

  create({ flow_id, device_id, device_name, attack, severity, threat_score }) {
    const now = Date.now() / 1000;
    const info = this.stmts.insertIncident.run({
      flow_id, device_id, device_name, attack,
      severity, threat_score: threat_score ?? null,
      recommended_action: recommendedAction(attack),
      created_at: now, updated_at: now,
    });
    const incident = this.stmts.getIncident.get(info.lastInsertRowid);
    this.onEvent('incident', { type: 'created', incident });
    return incident;
  }

  update(id, status) {
    if (!STATUSES.includes(status)) throw new Error(`invalid status: ${status}`);
    this.stmts.updateIncident.run({ id, status, updated_at: Date.now() / 1000 });
    const incident = this.stmts.getIncident.get(id);
    if (incident) this.onEvent('incident', { type: 'updated', incident });
    return incident;
  }

  list(limit = 50) { return this.stmts.getIncidents.all(limit); }
}

module.exports = { IncidentManager, STATUSES };
