/**
 * Socket.IO store — the ONLY data source for the dashboard.
 * Every displayed value arrives from the backend (which got it from the
 * inference service / ml-engine artifacts). The frontend computes NOTHING.
 */
import React, { createContext, useContext, useEffect, useReducer, useMemo } from 'react';
import { io } from 'socket.io-client';

export const BACKEND = import.meta.env.VITE_BACKEND_URL || 'http://127.0.0.1:4000';

const initial = {
  connected: false,
  devices: [],
  metrics: {},
  modelInfo: null,
  threats: [],
  predictions: [],
  responses: [],       // autonomous response engine feed entries
  incidents: [],       // incident management (persisted backend-side)
  flLive: { running: false, rounds: [] },  // live federated training events
  notifications: [],   // notification records (email/sms/browser/alarm/toast)
  notifSettings: null, // masked notification settings from the backend
  alarm: null,         // active critical alarm payload (null = silent)
  browserNotify: null, // last browser-notification request {title, body, seq}
  scenario: { running: false, finished: false, tick: -1 },
  dataSource: 'unknown',
  serviceMessage: null,
};

function reducer(state, a) {
  switch (a.type) {
    case 'connected': return { ...state, connected: true };
    case 'disconnected': return { ...state, connected: false };
    case 'init': return {
      ...state,
      devices: a.p.devices || [],
      metrics: a.p.metrics || {},
      modelInfo: a.p.modelInfo,
      scenario: a.p.scenario || state.scenario,
      threats: a.p.threats || [],
      predictions: a.p.predictions || [],
      responses: a.p.responses || [],
      incidents: a.p.incidents || [],
      flLive: a.p.federatedLive || state.flLive,
      notifications: a.p.notifications || [],
      notifSettings: a.p.notificationSettings || state.notifSettings,
      dataSource: a.p.dataSource || 'unknown',
    };
    case 'notification': {
      const rest = state.notifications.filter((n) => n.id !== a.p.id);
      return { ...state, notifications: [a.p, ...rest].slice(0, 100) };
    }
    case 'notification-settings': return { ...state, notifSettings: a.p };
    case 'alarm': return { ...state, alarm: { ...a.p, startedAt: Date.now() } };
    case 'alarm-dismiss': return { ...state, alarm: null };
    case 'browser-notify': return {
      ...state, browserNotify: { ...a.p, seq: (state.browserNotify?.seq || 0) + 1 },
    };
    case 'response': return {
      ...state, responses: [...state.responses.slice(-79), a.p],
    };
    case 'incident': {
      const inc = a.p.incident;
      const rest = state.incidents.filter((x) => x.id !== inc.id);
      return { ...state, incidents: [inc, ...rest].slice(0, 50) };
    }
    case 'federated-round': {
      const fl = state.flLive;
      const rounds = a.p.type === 'round' ? [...fl.rounds, a.p] : fl.rounds;
      return {
        ...state,
        flLive: {
          ...fl,
          running: a.p.type !== 'done',
          rounds: a.p.type === 'start' ? [] : rounds,
          clients: a.p.type === 'start' ? a.p.clients : fl.clients,
          lastEvent: a.p,
        },
      };
    }
    case 'federated-status': return {
      ...state,
      flLive: { ...state.flLive, running: a.p.running, rounds: a.p.rounds || state.flLive.rounds },
    };
    case 'metrics': return { ...state, metrics: a.p };
    // 'batch' applies a ~200ms window of streamed events in ONE state update,
    // so a burst of per-flow events costs one re-render instead of dozens.
    case 'batch': {
      const next = { ...state };
      if (a.p.predictions.length) {
        next.predictions = [...state.predictions, ...a.p.predictions].slice(-60);
      }
      if (a.p.threats.length) {
        next.threats = [...state.threats, ...a.p.threats].slice(-100);
      }
      if (a.p.metrics) next.metrics = a.p.metrics;
      return next;
    }
    case 'device-status': return {
      ...state,
      devices: state.devices.map((d) => (d.id === a.p.deviceId ? { ...d, status: a.p.status } : d)),
    };
    case 'scenario-status': return { ...state, scenario: a.p };
    case 'scenario-reset': return { ...state, threats: [], predictions: [], responses: [] };
    case 'service-status': return { ...state, serviceMessage: a.p };
    default: return state;
  }
}

const Ctx = createContext(null);

export function SocketProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, initial);

  useEffect(() => {
    const s = io(BACKEND, { transports: ['websocket', 'polling'] });
    s.on('connect', () => dispatch({ type: 'connected' }));
    s.on('disconnect', () => dispatch({ type: 'disconnected' }));

    // High-frequency stream events are buffered and flushed on one timer;
    // low-frequency control events dispatch immediately.
    const buf = { predictions: [], threats: [], metrics: null };
    const flush = setInterval(() => {
      if (buf.predictions.length || buf.threats.length || buf.metrics) {
        dispatch({
          type: 'batch',
          p: { predictions: buf.predictions.splice(0), threats: buf.threats.splice(0), metrics: buf.metrics },
        });
        buf.metrics = null;
      }
    }, 200);
    s.on('prediction', (p) => buf.predictions.push(p));
    s.on('threat', (p) => buf.threats.push(p));
    s.on('metrics', (p) => { buf.metrics = p; });
    s.on('scenario-reset', (p) => {
      buf.predictions.length = 0; buf.threats.length = 0; buf.metrics = null;
      dispatch({ type: 'scenario-reset', p });
    });
    ['init', 'device-status', 'scenario-status', 'service-status',
      'response', 'incident', 'federated-round', 'federated-status',
      'notification', 'notification-settings', 'alarm', 'browser-notify'].forEach((ev) =>
      s.on(ev, (p) => dispatch({ type: ev, p })));
    return () => { clearInterval(flush); s.disconnect(); };
  }, []);

  const api = useMemo(() => ({
    start: () => fetch(`${BACKEND}/scenario/start`, { method: 'POST' }),
    pause: () => fetch(`${BACKEND}/scenario/pause`, { method: 'POST' }),
    reset: () => fetch(`${BACKEND}/scenario/reset`, { method: 'POST' }),
    explain: (features) => fetch(`${BACKEND}/explain`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features }),
    }).then((r) => r.json()),
    explainFlow: (flowId) => fetch(`${BACKEND}/explain/${encodeURIComponent(flowId)}`, {
      method: 'POST',
    }).then((r) => r.json()),
    runFederated: (opts = {}) => fetch(`${BACKEND}/federated/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(opts),
    }).then((r) => r.json()),
    updateIncident: (id, status) => fetch(`${BACKEND}/incidents/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status }),
    }).then((r) => r.json()),
    analyticsCsvUrl: `${BACKEND}/analytics/export.csv`,
    getNotifSettings: () => fetch(`${BACKEND}/settings/notifications`).then((r) => r.json()),
    saveNotifSettings: (s) => fetch(`${BACKEND}/settings/notifications`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(s),
    }).then((r) => r.json()),
    testEmail: (to) => fetch(`${BACKEND}/notify/test-email`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ to }),
    }).then((r) => r.json()),
    testSms: (to) => fetch(`${BACKEND}/notify/test-sms`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ to }),
    }).then((r) => r.json()),
    testAttack: (critical) => fetch(`${BACKEND}/test/attack`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ critical }),
    }).then((r) => r.json()),
    getBenchmarks: () => fetch(`${BACKEND}/benchmarks`).then((r) => r.json()),
    getFederated: () => fetch(`${BACKEND}/federated-status`).then((r) => r.json()),
  }), []);

  return <Ctx.Provider value={{ state, api, dispatch }}>{children}</Ctx.Provider>;
}

export const useStore = () => useContext(Ctx);
