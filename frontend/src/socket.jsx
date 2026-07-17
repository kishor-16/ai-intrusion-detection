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
      dataSource: a.p.dataSource || 'unknown',
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
    case 'scenario-reset': return { ...state, threats: [], predictions: [] };
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
    ['init', 'device-status', 'scenario-status', 'service-status'].forEach((ev) =>
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
    getBenchmarks: () => fetch(`${BACKEND}/benchmarks`).then((r) => r.json()),
    getFederated: () => fetch(`${BACKEND}/federated-status`).then((r) => r.json()),
  }), []);

  return <Ctx.Provider value={{ state, api }}>{children}</Ctx.Provider>;
}

export const useStore = () => useContext(Ctx);
