/**
 * SMS notification service — Twilio, behind a small provider interface so it
 * can be swapped for MSG91 / Fast2SMS later (implement send() and rename).
 * Credentials from env / saved settings, never hardcoded.
 * Fallback mode: without credentials, sends are simulated (logged + recorded
 * as "simulated") so the platform keeps working out of the box.
 */

function twilioConfig(settings = {}) {
  const s = settings.twilio || {};
  return {
    accountSid: s.accountSid || process.env.TWILIO_ACCOUNT_SID || '',
    authToken: s.authToken || process.env.TWILIO_AUTH_TOKEN || '',
    phoneNumber: s.phoneNumber || process.env.TWILIO_PHONE_NUMBER || '',
  };
}

const isConfigured = (settings) => {
  const c = twilioConfig(settings);
  return Boolean(c.accountSid && c.authToken && c.phoneNumber);
};

/** Compact alert SMS body (~2 segments max). */
function alertSmsBody(alert) {
  return [
    '🚨 EdgeShield AI',
    `${alert.severity} ${alert.attack} detected.`,
    `Device: ${alert.deviceName}`,
    `Confidence: ${alert.confidence}%`,
    'Immediate investigation required.',
  ].join('\n');
}

// ---- provider: Twilio ------------------------------------------------------
let client = null;
let clientSid = '';
function getClient(cfg) {
  if (!client || clientSid !== cfg.accountSid) {
    // lazy require: the app must boot even if the package were missing
    // eslint-disable-next-line global-require
    const twilio = require('twilio');
    client = twilio(cfg.accountSid, cfg.authToken);
    clientSid = cfg.accountSid;
  }
  return client;
}

/**
 * Send an alert SMS. Returns { status: 'sent'|'simulated', sid?|body }.
 * Throws only on real provider errors (caller handles retry).
 */
async function sendAlertSms({ to, alert, settings }) {
  const body = alertSmsBody(alert);
  if (!isConfigured(settings)) {
    console.warn(`[sms] Twilio not configured — SMS Simulation Successful → ${to}:\n${body}`);
    return { status: 'simulated', body };
  }
  const cfg = twilioConfig(settings);
  const msg = await getClient(cfg).messages.create({
    body, from: cfg.phoneNumber, to,
  });
  console.log(`[sms] sent to ${to}: ${msg.sid}`);
  return { status: 'sent', sid: msg.sid, body };
}

module.exports = { sendAlertSms, isConfigured, alertSmsBody };
