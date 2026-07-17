/**
 * Email notification service — Nodemailer over SMTP (Gmail by default).
 * Credentials come from env / saved settings, never hardcoded.
 * Fallback mode: if SMTP is not configured, sends are recorded as
 * "simulated" with a logged warning — the platform never crashes on it.
 */

const nodemailer = require('nodemailer');

const SEV_COLOR = {
  Critical: '#ff3b5c', High: '#ff8a3d', Medium: '#ffb020', Low: '#8b96b8',
};

function smtpConfig(settings = {}) {
  const s = settings.smtp || {};
  return {
    host: s.host || process.env.SMTP_HOST || 'smtp.gmail.com',
    port: Number(s.port || process.env.SMTP_PORT || 587),
    user: s.user || process.env.SMTP_USER || '',
    pass: s.pass || process.env.SMTP_PASS || '',
    from: s.from || process.env.EMAIL_FROM || s.user || process.env.SMTP_USER || '',
  };
}

const isConfigured = (settings) => {
  const c = smtpConfig(settings);
  return Boolean(c.host && c.user && c.pass);
};

let transporter = null;
let transporterKey = '';
function getTransporter(settings) {
  const c = smtpConfig(settings);
  const key = `${c.host}:${c.port}:${c.user}`;
  if (!transporter || transporterKey !== key) {
    transporter = nodemailer.createTransport({
      host: c.host, port: c.port,
      secure: c.port === 465,
      auth: { user: c.user, pass: c.pass },
    });
    transporterKey = key;
  }
  return transporter;
}

const esc = (v) => String(v ?? '—')
  .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

const row = (label, value, mono = false) => `
  <tr>
    <td style="padding:7px 12px;color:#8b96b8;font-size:12px;letter-spacing:.5px;
        text-transform:uppercase;white-space:nowrap;">${esc(label)}</td>
    <td style="padding:7px 12px;color:#e6ecff;font-size:13px;
        ${mono ? "font-family:Consolas,monospace;" : ''}">${value}</td>
  </tr>`;

/** Responsive dark HTML alert email. All values come from the real detection. */
function alertHtml(alert) {
  const color = SEV_COLOR[alert.severity] || SEV_COLOR.Low;
  const shap = (alert.shapFeatures || []).slice(0, 5)
    .map((f) => `<span style="display:inline-block;background:rgba(0,212,255,.12);
      color:#00d4ff;border-radius:5px;padding:2px 8px;margin:2px;font-size:12px;
      font-family:Consolas,monospace;">${esc(f.feature ?? f)}</span>`).join('');
  const url = process.env.DASHBOARD_URL || 'http://localhost:5173';
  return `<!doctype html><html><body style="margin:0;padding:0;background:#050816;">
  <div style="max-width:620px;margin:0 auto;padding:24px 14px;
       font-family:'Segoe UI',Arial,sans-serif;">
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0"
      style="background:#0b1026;border:1px solid rgba(120,160,255,.2);border-radius:14px;overflow:hidden;">
      <tr><td style="padding:20px 24px;border-bottom:3px solid ${color};">
        <table role="presentation" width="100%"><tr>
          <td style="font-size:22px;">⬡</td>
          <td style="padding-left:10px;">
            <div style="color:#e6ecff;font-weight:700;font-size:16px;">EdgeShield AI</div>
            <div style="color:#8b96b8;font-size:10px;letter-spacing:2px;">IOT SECURITY OPERATIONS CENTER</div>
          </td>
          <td align="right">
            <span style="background:${color}22;color:${color};border:1px solid ${color};
              border-radius:999px;padding:4px 14px;font-size:12px;font-weight:700;
              letter-spacing:1px;">${esc(alert.severity).toUpperCase()}</span>
          </td>
        </tr></table>
      </td></tr>
      <tr><td style="padding:22px 24px 8px;">
        <div style="color:${color};font-size:19px;font-weight:700;">
          🚨 ${esc(alert.attack)} detected on ${esc(alert.deviceName)}</div>
        <div style="color:#8b96b8;font-size:13px;margin-top:6px;line-height:1.6;">
          ${esc(alert.explanation || 'Automated alert from the EdgeShield AI intrusion detection model.')}</div>
      </td></tr>
      <tr><td style="padding:10px 12px;">
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0"
          style="background:rgba(255,255,255,.03);border-radius:10px;">
          ${row('Attack type', `<b style="color:${color}">${esc(alert.attack)}</b>`)}
          ${row('Severity / score', `${esc(alert.severity)} · ${esc(alert.score)}/100`)}
          ${row('Model confidence', `${esc(alert.confidence)}%`)}
          ${row('Affected device', esc(alert.deviceName))}
          ${row('Detection time', esc(alert.time))}
          ${row('Source IP', esc(alert.srcIp), true)}
          ${row('Destination IP', esc(alert.dstIp), true)}
          ${row('Protocol', esc(alert.protocol), true)}
          ${row('Packet rate', `${esc(alert.packetRate)} pkt/s`, true)}
          ${row('Inference latency', `${esc(alert.latencyMs)} ms`, true)}
        </table>
      </td></tr>
      ${shap ? `<tr><td style="padding:6px 24px;">
        <div style="color:#8b96b8;font-size:11px;letter-spacing:1px;text-transform:uppercase;
          margin-bottom:6px;">Top SHAP features (why the model decided this)</div>${shap}
      </td></tr>` : ''}
      ${alert.recommendedAction ? `<tr><td style="padding:14px 24px 4px;">
        <div style="background:rgba(255,176,32,.08);border:1px solid rgba(255,176,32,.35);
          border-radius:10px;padding:12px 14px;color:#ffb020;font-size:13px;line-height:1.6;">
          <b>Recommended action:</b> <span style="color:#e6ecff">${esc(alert.recommendedAction)}</span></div>
      </td></tr>` : ''}
      <tr><td align="center" style="padding:22px;">
        <a href="${esc(url)}" style="background:${color};color:#fff;text-decoration:none;
          padding:11px 30px;border-radius:9px;font-weight:700;font-size:14px;
          display:inline-block;">Open SOC Dashboard</a>
      </td></tr>
      <tr><td style="padding:14px 24px;border-top:1px solid rgba(120,160,255,.12);
        color:#5a6486;font-size:11px;line-height:1.6;">
        Automated notification from EdgeShield AI. Detection produced by real int8 ONNX
        model inference on ${esc(alert.time)}; traffic in this demo environment is simulated.
      </td></tr>
    </table>
  </div></body></html>`;
}

/**
 * Send an alert email. Returns { status: 'sent'|'simulated', info?|error? }.
 * Throws only on real transport errors (caller handles retry).
 */
async function sendAlertEmail({ to, alert, settings }) {
  const subject = `🚨 ${alert.severity.toUpperCase()} SECURITY ALERT — ${alert.attack} on ${alert.deviceName} — EdgeShield AI`;
  if (!isConfigured(settings)) {
    console.warn('[email] SMTP not configured — simulating send to', to,
      '(set SMTP_USER/SMTP_PASS in .env or the Notification Settings page)');
    return { status: 'simulated', subject };
  }
  const c = smtpConfig(settings);
  const info = await getTransporter(settings).sendMail({
    from: `"EdgeShield AI" <${c.from}>`,
    to,
    subject,
    html: alertHtml(alert),
    text: `${alert.severity.toUpperCase()} — ${alert.attack} on ${alert.deviceName} `
      + `(confidence ${alert.confidence}%). ${alert.recommendedAction || ''}`,
  });
  console.log(`[email] sent to ${to}: ${info.messageId}`);
  return { status: 'sent', messageId: info.messageId, subject };
}

module.exports = { sendAlertEmail, isConfigured, alertHtml };
