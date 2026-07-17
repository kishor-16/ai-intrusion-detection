/**
 * Threat Scoring Engine — deterministic 0–100 score from REAL model outputs
 * and flow metadata. No randomness: same detection → same score.
 *
 *   score = confidence·55 + class_weight (≤20) + packet_rate factor (≤15)
 *         + anomaly (1 − P(Normal))·10
 */

const CLASS_WEIGHT = {
  DDoS: 20, Malware: 20, 'Brute Force': 14, Spoofing: 14, Scanning: 8, Normal: 0,
};

const RECOMMENDED_ACTION = {
  DDoS: 'Rate-limit the device uplink and drop flows from the offending source at the gateway.',
  Malware: 'Isolate the device from the network and inspect outbound communications for C2 beaconing.',
  'Brute Force': 'Lock the targeted account/service, enforce backoff, and rotate credentials.',
  Spoofing: 'Verify ARP/DNS integrity, re-authenticate the device, and enable port security.',
  Scanning: 'Block the scanning source and audit exposed services on the scanned device.',
};

function threatScore(prediction, meta) {
  if (prediction.label === 'Normal') return { score: 0, severity: 'None' };
  const conf = prediction.confidence || 0;
  const classW = CLASS_WEIGHT[prediction.label] ?? 10;
  const pktRate = Number(meta?.packet_rate) || 0;
  const rateW = Math.min(pktRate / 2000, 1) * 15;
  const anomaly = (1 - (prediction.probs?.Normal ?? 0)) * 10;
  const score = Math.round(Math.min(conf * 55 + classW + rateW + anomaly, 100));
  const severity = score > 80 ? 'Critical' : score > 60 ? 'High' : score > 30 ? 'Medium' : 'Low';
  return { score, severity };
}

const recommendedAction = (label) =>
  RECOMMENDED_ACTION[label] || 'Review the flow and monitor the device.';

module.exports = { threatScore, recommendedAction };
