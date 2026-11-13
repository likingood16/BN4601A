// Server calls, the signed-in person, live updates and shared clinical helpers.
import { useCallback, useEffect, useRef, useState } from "react";

async function call(method, url, body) {
  const opts = { method };
  if (body instanceof FormData) opts.body = body;
  else if (body !== undefined) {
    opts.body = JSON.stringify(body);
    opts.headers = { "Content-Type": "application/json" };
  }
  const r = await fetch(url, opts);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || `Something went wrong (${r.status})`);
  return data;
}
export const api = {
  get: (u) => call("GET", u),
  post: (u, b) => call("POST", u, b),
  put: (u, b) => call("PUT", u, b),
  patch: (u, b) => call("PATCH", u, b),
  del: (u) => call("DELETE", u),
};

// ---------------------------------------------------------------- who is using this screen (no real login)
const ME_KEY = "ed-me-v2";
function readMe() {
  try { return JSON.parse(localStorage.getItem(ME_KEY)) || { who: "", role: "Triage nurse" }; }
  catch { return { who: "", role: "Triage nurse" }; }
}
const meListeners = new Set();
let meNow = readMe();
export function useMe() {
  const [me, setMe] = useState(meNow);
  useEffect(() => { meListeners.add(setMe); return () => meListeners.delete(setMe); }, []);
  const save = (m) => {
    meNow = m;
    try { localStorage.setItem(ME_KEY, JSON.stringify(m)); } catch { /* private window */ }
    meListeners.forEach((f) => f(m));
  };
  return [me, save];
}
export const EDITORS = ["Triage nurse", "Organising doctor"];
export const canEditBoard = (me) => EDITORS.includes(me.role);

// ---------------------------------------------------------------- live data: one event stream for the whole tab
const liveListeners = new Set();
let source = null;
let connected = false;
const statusListeners = new Set();
function setConnected(v) { connected = v; statusListeners.forEach((f) => f(v)); }
function ensureStream() {
  if (source) return;
  source = new EventSource("/api/events");
  source.onopen = () => setConnected(true);
  source.onmessage = () => liveListeners.forEach((f) => f());
  source.onerror = () => setConnected(false);          // the browser reconnects by itself
}
export function useConnected() {
  const [c, setC] = useState(connected);
  useEffect(() => { statusListeners.add(setC); ensureStream(); return () => statusListeners.delete(setC); }, []);
  return c;
}

// Load `url` now and again whenever the server says something changed. Returns [data, error, reload].
export function useLive(url) {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const busy = useRef(false);
  const again = useRef(false);
  const load = useCallback(async () => {
    if (!url) return;
    if (busy.current) { again.current = true; return; }
    busy.current = true;
    try {
      const d = await api.get(url);
      setData(d);
      setError("");
    } catch (e) { setError(e.message); }
    busy.current = false;
    if (again.current) { again.current = false; load(); }
  }, [url]);
  useEffect(() => {
    setData(null);
    ensureStream();
    load();
    liveListeners.add(load);
    return () => liveListeners.delete(load);
  }, [load]);
  return [data, error, load];
}

export function useNow(ms = 15000) {
  const [now, setNow] = useState(Date.now() / 1000);
  useEffect(() => { const t = setInterval(() => setNow(Date.now() / 1000), ms); return () => clearInterval(t); }, [ms]);
  return now;
}

// ---------------------------------------------------------------- clinical display helpers
export const LEVEL = {
  1: { name: "Resuscitation", colour: "var(--l1)", target: 0 },
  2: { name: "Emergent", colour: "var(--l2)", target: 10 },
  3: { name: "Urgent", colour: "var(--l3)", target: 30 },
  4: { name: "Less urgent", colour: "var(--l4)", target: 60 },
  5: { name: "Non-urgent", colour: "var(--l5)", target: 120 },
};

export function minutes(created, now) { return Math.max(0, Math.floor((now - created) / 60)); }
export function hm(m) { return m < 60 ? `${m}m` : `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, "0")}m`; }

// NEWS2 single-parameter bands (RCP 2017): "red" = 3 points, "amber" = 1-2 points.
export function concern(key, v) {
  if (v === null || v === undefined || v === "") return "";
  if (key === "mental") return v === "alert" ? "" : "red";
  v = Number(v);
  switch (key) {
    case "hr": return v <= 40 || v >= 131 ? "red" : v <= 50 || v >= 91 ? "amber" : "";
    case "sbp": return v <= 90 || v >= 220 ? "red" : v <= 110 ? "amber" : "";
    case "rr": return v <= 8 || v >= 25 ? "red" : v >= 21 || v <= 11 ? "amber" : "";
    case "spo2": return v <= 91 ? "red" : v <= 95 ? "amber" : "";
    case "temp_c": return v <= 35 ? "red" : v >= 39.1 || v <= 36 || v >= 38.1 ? "amber" : "";
    case "nrs_pain": return v >= 7 ? "amber" : "";
    default: return "";
  }
}

export const TILES = [
  { key: "hr", label: "HR", unit: "/min" },
  { key: "bp", label: "BP mmHg", unit: "" },
  { key: "rr", label: "RR", unit: "/min" },
  { key: "spo2", label: "SpO₂", unit: "%" },
  { key: "temp_c", label: "Temp", unit: "°C" },
  { key: "nrs_pain", label: "Pain", unit: "/10" },
  { key: "mental", label: "AVPU", unit: "" },
];
const AVPU = { alert: "A", verbal: "V", pain: "P", unresponsive: "U" };

export function tileValue(key, f) {
  if (key === "bp") return f.sbp != null ? `${fmt(f.sbp)}/${f.dbp != null ? fmt(f.dbp) : "?"}` : null;
  if (key === "mental") return f.mental ? AVPU[f.mental] || f.mental : null;
  return f[key] != null ? fmt(f[key]) : null;
}
export function tileConcern(key, f) {
  if (key === "bp") return concern("sbp", f.sbp);
  return concern(key, f[key]);
}
// What a picture is decides what checks it: bone X-ray -> fracture detector (+ outline), chest X-ray -> 18-finding check.
export const IMAGE_TYPES = [
  ["", "AI decides"], ["bone_xray", "Bone X-ray (fracture check)"], ["chest_xray", "Chest X-ray (18-finding check)"],
  ["ct_slice", "CT slice"], ["mri_slice", "MRI slice"], ["ultrasound", "Ultrasound"], ["ecg", "ECG"], ["other", "Other / photo"],
];

export const fmt =(n) => (Number.isInteger(n) ? String(n) : String(Math.round(n * 10) / 10));

export function whoLine(f) {
  const sex = { male: "M", female: "F" }[f.sex] || "";
  const age = f.age != null ? Math.round(f.age) : "";
  return [age, sex].filter(Boolean).join(" ") || "Age/sex not recorded";
}
