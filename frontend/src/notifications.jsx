/**
 * Notification Center UI: topbar bell + dropdown, toast stack, critical-alarm
 * banner with a WebAudio siren (synthesized — no audio files, fully offline),
 * and native browser notifications. All data arrives over Socket.IO.
 */
import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useStore } from './socket.jsx';
import { metaFor } from './attackMeta.js';

const tstr = (ts) => (ts ? new Date(ts * 1000).toLocaleTimeString() : '');

const CHANNEL_ICON = { email: '✉', sms: '📱', browser: '🔔', alarm: '🚨', toast: '🛡' };
const STATUS_CLASS = { sent: 'ok', simulated: 'sim', failed: 'fail', queued: 'wait' };

// ---------- WebAudio siren (two-tone sweep) ----------
function startSiren(volume) {
  const Ctx = window.AudioContext || window.webkitAudioContext;
  if (!Ctx) return null;
  const ctx = new Ctx();
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  const lfo = ctx.createOscillator();
  const lfoGain = ctx.createGain();
  osc.type = 'square';
  osc.frequency.value = 760;
  lfo.type = 'sine';
  lfo.frequency.value = 2.2;          // sweep rate
  lfoGain.gain.value = 220;           // sweep depth
  lfo.connect(lfoGain).connect(osc.frequency);
  gain.gain.value = Math.max(0, Math.min(volume, 1)) * 0.12;
  osc.connect(gain).connect(ctx.destination);
  osc.start(); lfo.start();
  return {
    setVolume: (v) => { gain.gain.value = Math.max(0, Math.min(v, 1)) * 0.12; },
    stop: () => { try { osc.stop(); lfo.stop(); ctx.close(); } catch { /* closed */ } },
  };
}

/** Hook: exposes a test-alarm trigger for the settings page. */
export function useTestAlarm() {
  const { dispatch } = useStore();
  return (volume = 0.6) => dispatch({
    type: 'alarm',
    p: {
      attack: 'TEST', device_name: 'Alarm Test', confidence: '—',
      volume, test: true,
    },
  });
}

// ---------- alarm banner ----------
export function AlarmBanner() {
  const { state, dispatch } = useStore();
  const { alarm, incidents, notifSettings } = state;
  const sirenRef = useRef(null);

  // start/stop siren with the alarm state
  useEffect(() => {
    if (alarm && (notifSettings?.alarmEnabled ?? true)) {
      sirenRef.current = startSiren(alarm.volume ?? notifSettings?.alarmVolume ?? 0.6);
    }
    return () => { sirenRef.current?.stop(); sirenRef.current = null; };
  }, [alarm, notifSettings?.alarmEnabled]);

  // auto-stop when the linked incident is contained/resolved
  useEffect(() => {
    if (!alarm?.incident_id) return;
    const inc = incidents.find((i) => i.id === alarm.incident_id);
    if (inc && inc.status !== 'Open' && inc.status !== 'Investigating') {
      dispatch({ type: 'alarm-dismiss' });
    }
  }, [incidents, alarm, dispatch]);

  // test alarms stop themselves after 4s
  useEffect(() => {
    if (!alarm?.test) return undefined;
    const t = setTimeout(() => dispatch({ type: 'alarm-dismiss' }), 4000);
    return () => clearTimeout(t);
  }, [alarm, dispatch]);

  if (!alarm) return null;
  return (
    <div className="alarm-banner" role="alert">
      <span className="alarm-icon" aria-hidden="true">🚨</span>
      <span>
        <b>CRITICAL THREAT{alarm.test ? ' — ALARM TEST' : ''}:</b>{' '}
        {alarm.attack} on <b>{alarm.device_name}</b>
        {alarm.confidence !== '—' && <> · {alarm.confidence}% confidence</>}
        {alarm.incident_id != null && <> · incident #{alarm.incident_id}</>}
      </span>
      <button className="btn alarm-dismiss" onClick={() => dispatch({ type: 'alarm-dismiss' })}>
        ✕ Dismiss alarm
      </button>
    </div>
  );
}

// ---------- native browser notifications ----------
export function BrowserNotifier() {
  const { state } = useStore();
  const { browserNotify, notifSettings } = state;
  const lastSeq = useRef(0);
  useEffect(() => {
    if (!browserNotify || browserNotify.seq === lastSeq.current) return;
    lastSeq.current = browserNotify.seq;
    if (!(notifSettings?.browserNotifications ?? true)) return;
    if (typeof Notification === 'undefined' || Notification.permission !== 'granted') return;
    const n = new Notification(browserNotify.title, {
      body: browserNotify.body, tag: 'edgeshield-critical',
    });
    n.onclick = () => { window.focus(); n.close(); };
  }, [browserNotify, notifSettings]);
  return null;
}

// ---------- toast stack ----------
export function Toasts() {
  const { state } = useStore();
  const { notifications } = state;
  const [visible, setVisible] = useState([]);
  const seen = useRef(new Set());
  const mounted = useRef(false);

  useEffect(() => {
    if (!mounted.current) {
      // don't replay history as toasts on first mount
      notifications.forEach((n) => seen.current.add(n.id));
      mounted.current = true;
      return;
    }
    const fresh = notifications.filter(
      (n) => n.channel === 'toast' && !seen.current.has(n.id));
    if (!fresh.length) return;
    fresh.forEach((n) => seen.current.add(n.id));
    setVisible((v) => [...fresh, ...v].slice(0, 4));
    fresh.forEach((n) => setTimeout(
      () => setVisible((v) => v.filter((x) => x.id !== n.id)), 6000));
  }, [notifications]);

  if (!visible.length) return null;
  return (
    <div className="toast-stack" aria-live="polite">
      {visible.map((n) => (
        <div key={n.id} className="toast" style={{ '--sev-color': metaFor(n.attack).color }}>
          <span aria-hidden="true">{metaFor(n.attack).icon}</span>
          <span><b>{n.attack}</b> · {n.severity} — notification dispatched</span>
        </div>
      ))}
    </div>
  );
}

// ---------- bell + dropdown ----------
export function NotificationBell() {
  const { state } = useStore();
  const { notifications } = state;
  const [open, setOpen] = useState(false);
  const [readAt, setReadAt] = useState(() => Date.now() / 1000);
  const nav = useNavigate();
  const panelRef = useRef(null);

  const unread = useMemo(
    () => notifications.filter((n) => (n.created_at || 0) > readAt).length,
    [notifications, readAt]);

  useEffect(() => {
    if (!open) return undefined;
    const close = (e) => { if (!panelRef.current?.contains(e.target)) setOpen(false); };
    document.addEventListener('mousedown', close);
    return () => document.removeEventListener('mousedown', close);
  }, [open]);

  return (
    <div className="bell-wrap" ref={panelRef}>
      <button className="btn ghost bell-btn" aria-label={`Notifications (${unread} unread)`}
        onClick={() => { setOpen(!open); if (!open) setReadAt(Date.now() / 1000); }}>
        🔔
        {unread > 0 && <span className="bell-badge">{unread > 99 ? '99+' : unread}</span>}
      </button>
      {open && (
        <div className="bell-panel">
          <div className="bell-head">
            <span>Notification Center</span>
            <button className="btn ghost inc-action"
              onClick={() => { setOpen(false); nav('/settings'); }}>⚙ Settings</button>
          </div>
          <div className="bell-list scroll-slim">
            {notifications.length === 0 && (
              <div className="empty" style={{ padding: 20 }}>
                No notifications yet — they appear when the model detects Medium+ threats.
              </div>
            )}
            {notifications.slice(0, 20).map((n) => (
              <div key={n.id} className="bell-item">
                <span className="bell-ch" aria-hidden="true">{CHANNEL_ICON[n.channel] || '•'}</span>
                <span className="bell-body">
                  <span>
                    <b style={{ color: metaFor(n.attack).color }}>{n.attack || 'System'}</b>
                    {n.severity && <span className={`tag sev-${(n.severity || '').toLowerCase()}`}>{n.severity}</span>}
                  </span>
                  <span className="det">
                    {n.channel}{n.recipient && n.recipient !== 'dashboard' ? ` → ${n.recipient}` : ''} · {tstr(n.created_at)}
                  </span>
                </span>
                <span className={`notif-status ${STATUS_CLASS[n.status] || ''}`}>{n.status}</span>
              </div>
            ))}
          </div>
          <div className="bell-foot src">
            source: notifications table — every entry is a real dispatch record
          </div>
        </div>
      )}
    </div>
  );
}
