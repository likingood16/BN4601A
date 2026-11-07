import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { api, useConnected, useMe } from "./api.js";
import Chart from "./Chart.jsx";
import Case from "./Case.jsx";
import Board from "./Board.jsx";
import "./styles.css";

function useHash() {
  const [hash, setHash] = useState(location.hash || "#/");
  useEffect(() => {
    const f = () => setHash(location.hash || "#/");
    addEventListener("hashchange", f);
    return () => removeEventListener("hashchange", f);
  }, []);
  return hash;
}

export function useMeta() {
  const [meta, setMeta] = useState(null);
  useEffect(() => { api.get("/api/meta").then(setMeta).catch(() => {}); }, []);
  return meta;
}

function TopBar({ page, meta }) {
  const [me, setMe] = useMe();
  const online = useConnected();
  const roles = meta?.roles || ["Triage nurse", "Organising doctor", "Doctor", "Nurse", "Ward"];
  const link = (href, label, key) => (
    <a href={href} className={page === key ? "tab on" : "tab"}>{label}</a>
  );
  return (
    <header className={`topbar ${page === "board" ? "dark" : ""}`}>
      <div className="brand"><span className="cross" aria-hidden>+</span> ED Case Files</div>
      <nav>
        {link("#/", "New patient", "chart")}
        {link("#/board", "Whiteboard", "board")}
      </nav>
      {meta?.test && <span className="pill test" title="Writing is faked; numbers, rules and the vitals model are real">TEST MODE</span>}
      <span className={`dot ${online ? "ok" : "bad"}`} title={online ? "Live" : "Reconnecting…"} />
      <div className="me">
        <input aria-label="Your name" placeholder="Your name" value={me.who}
               className={!me.who ? "need" : ""}
               onChange={(e) => setMe({ ...me, who: e.target.value })} />
        <select aria-label="Your role" value={me.role} onChange={(e) => setMe({ ...me, role: e.target.value })}>
          {roles.map((r) => <option key={r}>{r}</option>)}
        </select>
      </div>
    </header>
  );
}

function App() {
  const hash = useHash();
  const meta = useMeta();
  const m = hash.match(/^#\/case\/(\w+)/);
  const page = m ? "case" : hash.startsWith("#/board") ? "board" : "chart";
  useEffect(() => {
    document.body.className = page === "board" ? "board-page" : "";
    if (page === "chart") document.title = "New patient · ED";
    if (page === "board") document.title = "Whiteboard · ED";
  }, [page]);
  return (
    <>
      <TopBar page={page} meta={meta} />
      {page === "case" ? <Case id={m[1]} meta={meta} /> : page === "board" ? <Board meta={meta} left={hash.includes("left")} /> : <Chart />}
    </>
  );
}

createRoot(document.getElementById("root")).render(<App />);
