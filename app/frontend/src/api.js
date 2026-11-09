// Talking to server.py, plus the signed-in person (kept in this browser only).
import { useEffect, useRef, useState } from "react";

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
};

export function useMe() {
  const read = () => {
    try { return JSON.parse(localStorage.getItem("ed-me")) || { who: "", role: "Triage nurse" }; }
    catch { return { who: "", role: "Triage nurse" }; }
  };
  const [me, setMe] = useState(read);
  const save = (m) => {
    setMe(m);
    try { localStorage.setItem("ed-me", JSON.stringify(m)); } catch { /* private window: keep in memory */ }
  };
  return [me, save];
}

// Poll a URL every `ms` (faster while the AI is working). Returns [data, error, reload].
export function usePoll(url, ms = 2500) {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const timer = useRef();
  const load = async () => {
    if (!url) return;
    try {
      const d = await api.get(url);
      setData(d);
      setError("");
      return d;
    } catch (e) { setError(e.message); }
  };
  useEffect(() => {
    let stop = false;
    const loop = async () => {
      const d = await load();
      if (!stop) timer.current = setTimeout(loop, d && d.status === "working" ? 1000 : ms);
    };
    setData(null);
    loop();
    return () => { stop = true; clearTimeout(timer.current); };
  }, [url]);
  return [data, error, load];
}

export const LEVEL_COLOUR = { 1: "#d32f2f", 2: "#f57c00", 3: "#fbc02d", 4: "#388e3c", 5: "#1976d2" };

export function waited(created, now) {
  const m = Math.max(0, Math.floor(((now || Date.now() / 1000) - created) / 60));
  return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, "0")}m`;
}

// NEWS2-style colouring of one number: "red", "amber" or "".
export function concern(key, v) {
  if (v === null || v === undefined || v === "") return "";
  v = Number(v);
  const r = {
    hr: v <= 40 || v >= 131 ? "red" : v <= 50 || v >= 111 ? "amber" : "",
    sbp: v <= 90 || v >= 220 ? "red" : v <= 110 ? "amber" : "",
    rr: v <= 8 || v >= 25 ? "red" : v >= 21 ? "amber" : "",
    spo2: v <= 91 ? "red" : v <= 95 ? "amber" : "",
    temp_c: v <= 35 || v >= 39.1 ? "red" : v >= 38.1 ? "amber" : "",
    nrs_pain: v >= 7 ? "amber" : "",
  };
  return r[key] || "";
}
