/**
 * Shared presentation metadata for the six model classes.
 * MITRE ATT&CK technique mappings are the standard ones for each attack family —
 * they are UI annotations for SOC context, not model output.
 */
export const ATTACK_META = {
  DDoS: { color: '#ff3b5c', sev: 'critical', icon: '🌊', mitre: 'T1498', mitreName: 'Network Denial of Service' },
  'Brute Force': { color: '#ff8a3d', sev: 'high', icon: '🔑', mitre: 'T1110', mitreName: 'Brute Force' },
  Scanning: { color: '#ffb020', sev: 'medium', icon: '📡', mitre: 'T1046', mitreName: 'Network Service Discovery' },
  Spoofing: { color: '#b06bff', sev: 'high', icon: '🎭', mitre: 'T1557', mitreName: 'Adversary-in-the-Middle' },
  Malware: { color: '#ff4fa3', sev: 'critical', icon: '🦠', mitre: 'T1071', mitreName: 'C2 / Application Layer Protocol' },
  Normal: { color: '#22ff88', sev: 'none', icon: '✓', mitre: null, mitreName: null },
};

export const metaFor = (label) => ATTACK_META[label] || { color: '#8b96b8', sev: 'medium', icon: '⚠', mitre: null };

/** Confidence tier for an attack verdict: how sure the model is. */
export function confTier(conf) {
  if (conf >= 0.85) return 'high';
  if (conf >= 0.6) return 'medium';
  return 'low';
}

/** Match a scenario/device attack label (e.g. "Botnet/Malware Beaconing") to a class key. */
export function classFromScenarioLabel(text) {
  if (!text) return null;
  const t = text.toLowerCase();
  if (t.includes('ddos') || t.includes('flood')) return 'DDoS';
  if (t.includes('brute')) return 'Brute Force';
  if (t.includes('scan')) return 'Scanning';
  if (t.includes('spoof')) return 'Spoofing';
  if (t.includes('malware') || t.includes('botnet') || t.includes('beacon')) return 'Malware';
  return null;
}
