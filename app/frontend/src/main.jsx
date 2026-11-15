import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";
import { api, useMe } from "./api";
import Chart from "./Chart";
import Case from "./Case";
import Board from "./Board";

// Pages by the part after #: #/ charting, #/case/<id> case file, #/board whiteboard.
function useRoute() {
  const [hash, setHash] = useState(location.hash);
  useEffect(() => {
    const f = () => setHash(location.hash);
    addEventListener("hashchange", f);
    return () => removeEventListener("hashchange", f);
  }, []);
  return hash.replace(/^#\/?/, "").split("/");
}

function TopBar({ me, setMe, meta, page }) {
  return (
    <div className="top">
      <span className="brand">ED charting</span>
      <nav>
        <a href="#/" className={page === "" ? "on" : ""}>Charting</a>
        <a href="#/board" target="ed-board" className={page === "board" ? "on" : ""}>Whiteboard</a>
      </nav>
      {meta?.test && <span className="test-flag">Test mode: AI answers are made up</span>}
      <div className="me">
        <input placeholder="Your name" value={me.who} style={{ width: 150 }}
          onChange={(e) => setMe({ ...me, who: e.target.value })} aria-label="Your name" />
        <select value={me.role} onChange={(e) => setMe({ ...me, role: e.target.value })} aria-label="Your role">
          {(meta?.roles || [me.role]).map((r) => <option key={r}>{r}</option>)}
        </select>
      </div>
    </div>
  );
}

function App() {
  const [page, id] = useRoute();
  const [me, setMe] = useMe();
  const [meta, setMeta] = useState(null);
  useEffect(() => { api.get("/api/meta").then(setMeta).catch(() => {}); }, []);
  useEffect(() => {                         // the case file sets its own title ("P1 · case file")
    if (page === "board") document.title = "ED whiteboard";
    else if (page !== "case") document.title = "ED charting";
  }, [page]);
  const props = { me, setMe, meta };
  return (
    <div className={page === "board" ? "board-page" : ""}>
      <TopBar me={me} setMe={setMe} meta={meta} page={page} />
      {page === "case" ? <Case id={id} {...props} /> : page === "board" ? <Board {...props} /> : <Chart {...props} />}
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
