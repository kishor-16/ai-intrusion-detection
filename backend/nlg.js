/**
 * Natural-language explanation builder. Deterministic templating over REAL
 * values (model confidence, real SHAP attributions, real flow metadata) —
 * no generation, no randomness: the text is a readable rendering of measured
 * numbers, and every number in it exists in the structured payload too.
 */

const { recommendedAction } = require('./scoring');

const listify = (names) => (names.length <= 1 ? names.join('')
  : `${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}`);

/** Text for a detection given per-instance SHAP features (from /explain). */
function explanationText({ label, confidence, deviceName, shapFeatures, severity, score }) {
  const top = (shapFeatures || []).slice(0, 4).map((f) => f.feature);
  const conf = (confidence * 100).toFixed(1);
  const parts = [];
  parts.push(`${deviceName || 'The device'} was classified as ${label} with ${conf}% confidence.`);
  if (top.length) {
    parts.push(`The most influential features for this verdict were ${listify(top)} `
      + '(per-instance SHAP attribution on the trained model).');
  }
  if (severity && score != null) {
    parts.push(`Threat score: ${score}/100 (${severity}).`);
  }
  if (label !== 'Normal') {
    parts.push(`Recommended action: ${recommendedAction(label)}`);
  }
  return parts.join(' ');
}

/** Short feed line for an autonomous response action. */
function responseText({ action, deviceName, label, confidence }) {
  const conf = confidence != null ? ` (${(confidence * 100).toFixed(1)}% confidence)` : '';
  switch (action) {
    case 'quarantine':
      return `AUTO-RESPONSE: ${deviceName} quarantined after high-confidence ${label} detection${conf}. Communication blocked at the gateway.`;
    case 'watch':
      return `AUTO-RESPONSE: ${deviceName} flagged suspicious after ${label} detection${conf}. Monitoring tightened.`;
    case 'release':
      return `AUTO-RESPONSE: ${deviceName} released from quarantine — no malicious flows in the cool-down window.`;
    case 'recover':
      return `${deviceName} returned to normal operation — threat contained.`;
    default:
      return `AUTO-RESPONSE: ${action} on ${deviceName}.`;
  }
}

module.exports = { explanationText, responseText };
