import React, { useEffect, useState } from 'react';
import { useStore } from '../socket.jsx';
import { useTestAlarm } from '../notifications.jsx';

const Field = ({ label, children, hint }) => (
  <label className="set-field">
    <span className="set-label">{label}</span>
    {children}
    {hint && <span className="set-hint">{hint}</span>}
  </label>
);

export default function NotificationSettings() {
  const { state, api } = useStore();
  const testAlarm = useTestAlarm();
  const [s, setS] = useState(null);
  const [msg, setMsg] = useState(null);
  const [busy, setBusy] = useState('');

  useEffect(() => {
    if (state.notifSettings) setS(state.notifSettings);
    else api.getNotifSettings().then(setS).catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (!s) return <div className="panel"><div className="empty">Loading settings…</div></div>;

  const upd = (patch) => setS({ ...s, ...patch });
  const updNested = (key, patch) => setS({ ...s, [key]: { ...(s[key] || {}), ...patch } });

  const flash = (text, ok = true) => {
    setMsg({ text, ok });
    setTimeout(() => setMsg(null), 6000);
  };

  async function run(name, fn) {
    setBusy(name);
    try {
      const r = await fn();
      if (r.ok === false || r.error) flash(`${name} failed: ${r.error || 'unknown error'}`, false);
      else if (r.status === 'simulated') flash(`${name}: simulated (credentials not configured — see hints below)`);
      else flash(`${name}: OK`);
    } catch (e) { flash(`${name} failed: ${e.message}`, false); }
    setBusy('');
  }

  async function save() {
    setBusy('save');
    const out = await api.saveNotifSettings({
      emailRecipients: s.emailRecipients, phoneNumbers: s.phoneNumbers,
      threshold: s.threshold, alarmEnabled: s.alarmEnabled, alarmVolume: s.alarmVolume,
      browserNotifications: s.browserNotifications, smtp: s.smtp, twilio: s.twilio,
    });
    setS(out);
    setBusy('');
    flash('Settings saved');
  }

  const csv = (arr) => (arr || []).join(', ');
  const parseCsv = (v) => v.split(',').map((x) => x.trim()).filter(Boolean);

  return (
    <div className="grid" style={{ gap: 16, maxWidth: 980 }}>
      {msg && (
        <div className={`panel set-flash ${msg.ok ? 'ok' : 'fail'}`}>{msg.text}</div>
      )}

      <div className="panel">
        <h3>
          <span>Notification channels</span>
          <span className="h3-note">
            email: <b style={{ color: s.channels?.emailConfigured ? 'var(--green)' : 'var(--amber)' }}>
              {s.channels?.emailConfigured ? 'configured' : 'simulation mode'}</b>
            {' · '}sms: <b style={{ color: s.channels?.smsConfigured ? 'var(--green)' : 'var(--amber)' }}>
              {s.channels?.smsConfigured ? 'configured' : 'simulation mode'}</b>
          </span>
        </h3>
        <div className="grid cols-2">
          <Field label="Email recipients" hint="comma-separated; receives High + Critical alerts">
            <input className="set-input" value={csv(s.emailRecipients)}
              placeholder="admin@company.com, soc@company.com"
              onChange={(e) => upd({ emailRecipients: parseCsv(e.target.value) })} />
          </Field>
          <Field label="Phone numbers (SMS)" hint="E.164 format (+91…); Critical alerts only">
            <input className="set-input" value={csv(s.phoneNumbers)}
              placeholder="+911234567890"
              onChange={(e) => upd({ phoneNumbers: parseCsv(e.target.value) })} />
          </Field>
          <Field label="Notification threshold" hint="which severities go outbound (email)">
            <select className="set-input" value={s.threshold}
              onChange={(e) => upd({ threshold: e.target.value })}>
              <option value="critical">Critical only</option>
              <option value="high">High + Critical</option>
              <option value="all">All (Medium and up)</option>
            </select>
          </Field>
          <Field label="Browser notifications" hint="native OS notifications for Critical threats">
            <span style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
              <input type="checkbox" checked={s.browserNotifications}
                onChange={(e) => upd({ browserNotifications: e.target.checked })} />
              <button className="btn ghost inc-action" onClick={() => {
                if (typeof Notification !== 'undefined') {
                  Notification.requestPermission().then((p) => flash(`Browser permission: ${p}`));
                }
              }}>
                Request permission {typeof Notification !== 'undefined' ? `(${Notification.permission})` : '(unsupported)'}
              </button>
            </span>
          </Field>
        </div>
      </div>

      <div className="panel">
        <h3><span>Alarm</span></h3>
        <div className="grid cols-3">
          <Field label="Alarm on Critical threats">
            <span style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
              <input type="checkbox" checked={s.alarmEnabled}
                onChange={(e) => upd({ alarmEnabled: e.target.checked })} />
              <span className="det">{s.alarmEnabled ? 'enabled' : 'muted'}</span>
            </span>
          </Field>
          <Field label={`Volume — ${Math.round((s.alarmVolume ?? 0.6) * 100)}%`}>
            <input type="range" min="0" max="1" step="0.05" value={s.alarmVolume ?? 0.6}
              onChange={(e) => upd({ alarmVolume: Number(e.target.value) })} />
          </Field>
          <Field label="Try it">
            <button className="btn ghost" onClick={() => testAlarm(s.alarmVolume ?? 0.6)}>
              🚨 Test alarm (4s)
            </button>
          </Field>
        </div>
      </div>

      <div className="grid cols-2">
        <div className="panel">
          <h3><span>SMTP (Gmail)</span> <span className="h3-note">blank = use .env values</span></h3>
          <Field label="Host"><input className="set-input" value={s.smtp?.host || ''}
            placeholder="smtp.gmail.com" onChange={(e) => updNested('smtp', { host: e.target.value })} /></Field>
          <Field label="Port"><input className="set-input" value={s.smtp?.port || ''}
            placeholder="587" onChange={(e) => updNested('smtp', { port: e.target.value })} /></Field>
          <Field label="User"><input className="set-input" value={s.smtp?.user || ''}
            placeholder="you@gmail.com" onChange={(e) => updNested('smtp', { user: e.target.value })} /></Field>
          <Field label="App password" hint="Gmail → Security → 2-Step Verification → App passwords">
            <input className="set-input" type="password" value={s.smtp?.pass || ''}
              placeholder="16-char app password" onChange={(e) => updNested('smtp', { pass: e.target.value })} />
          </Field>
        </div>
        <div className="panel">
          <h3><span>Twilio (SMS)</span> <span className="h3-note">blank = use .env / simulation</span></h3>
          <Field label="Account SID"><input className="set-input" value={s.twilio?.accountSid || ''}
            placeholder="AC…" onChange={(e) => updNested('twilio', { accountSid: e.target.value })} /></Field>
          <Field label="Auth token"><input className="set-input" type="password" value={s.twilio?.authToken || ''}
            onChange={(e) => updNested('twilio', { authToken: e.target.value })} /></Field>
          <Field label="From number"><input className="set-input" value={s.twilio?.phoneNumber || ''}
            placeholder="+1…" onChange={(e) => updNested('twilio', { phoneNumber: e.target.value })} /></Field>
          <div className="set-hint" style={{ marginTop: 8 }}>
            Provider is swappable (MSG91 / Fast2SMS) — see backend/services/smsService.js.
          </div>
        </div>
      </div>

      <div className="panel">
        <h3><span>Test the full pipeline</span> <span className="h3-note">real simulator flow → real model inference → scoring → notifications</span></h3>
        <div className="chip-row" style={{ gap: 10 }}>
          <button className="btn" disabled={!!busy}
            onClick={() => run('Test attack', () => api.testAttack(false))}>
            ⚔ Generate test attack
          </button>
          <button className="btn primary" disabled={!!busy}
            onClick={() => run('Critical attack', () => api.testAttack(true))}>
            💥 Generate CRITICAL attack
          </button>
          <button className="btn" disabled={!!busy}
            onClick={() => run('Test email', () => api.testEmail())}>
            ✉ Send test email
          </button>
          <button className="btn" disabled={!!busy}
            onClick={() => run('Test SMS', () => api.testSms())}>
            📱 Send test SMS
          </button>
          <button className="btn primary" disabled={busy === 'save'} onClick={save}
            style={{ marginLeft: 'auto' }}>
            💾 Save settings
          </button>
        </div>
        <div className="src">
          test attacks inject one real attack flow through simulator → model → scoring →
          response engine → notifications; without SMTP/Twilio credentials, sends are simulated and logged
        </div>
      </div>
    </div>
  );
}
