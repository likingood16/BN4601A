// Screen 1: the triage nurse types everything in one box. Numbers appear as tiles WHILE typing (fixed rules), so a
// wrong number is caught before Submit. Submit opens the case file in a new window and puts the patient on the board.
import React, { useEffect, useRef, useState } from "react";
import { api, LEVEL, TILES, tileConcern, tileValue, useLive, useMe, whoLine } from "./api.js";

const EXAMPLE = "e.g. 67M CP since 0600, sweaty, radiating to L arm. HR 112 BP 90/60 RR 22 sats 94% RA T 37.2 pain 8/10. " +
  "hx DM, HTN on metformin. NKDA. BIBA";

export function Tiles({ fields, source = {}, big = false }) {
  const f = fields || {};
  return (
    <div className={`tiles ${big ? "big" : ""}`}>
      {TILES.map((t) => {
        const v = tileValue(t.key, f);
        const c = v == null ? "missing" : tileConcern(t.key, f);
        const srcKey = t.key === "bp" ? "sbp" : t.key;
        const src = source[srcKey];
        return (
          <div key={t.key} className={`tile ${c}`} title={src?.text ? `From: “${src.text}”` : "Not in the note"}>
            <div className="tl">{t.label}</div>
            <div className="tv">{v ?? "—"}<span className="tu">{v != null ? t.unit : ""}</span></div>
            <div className="ts">{src?.text && !src.prehospital ? `“${src.text}”` : v == null ? "not recorded" : ""}</div>
          </div>
        );
      })}
    </div>
  );
}

function Pictures({ files, setFiles }) {
  const [over, setOver] = useState(false);
  const input = useRef();
  const add = (list) => setFiles((fs) => [...fs, ...[...list].filter((f) => f.type.startsWith("image/"))]);
  return (
    <div className={`drop ${over ? "over" : ""}`}
         onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
         onDrop={(e) => { e.preventDefault(); setOver(false); add(e.dataTransfer.files); }}>
      <button type="button" className="ghost" onClick={() => input.current.click()}>Add X-rays or photos</button>
      <span className="hint">or drop / paste them here</span>
      <input ref={input} type="file" accept="image/*" multiple hidden onChange={(e) => { add(e.target.files); e.target.value = ""; }} />
      {files.length > 0 && (
        <div className="thumbs">
          {files.map((f, i) => (
            <figure key={i}>
              <img src={URL.createObjectURL(f)} alt={f.name} />
              <button type="button" className="x" aria-label={`Remove ${f.name}`}
                      onClick={() => setFiles((fs) => fs.filter((_, j) => j !== i))}>&times;</button>
            </figure>
          ))}
        </div>
      )}
    </div>
  );
}

export default function Chart() {
  const [me] = useMe();
  const [text, setText] = useState("");
  const [files, setFiles] = useState([]);
  const [preview, setPreview] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [sent, setSent] = useState(null);
  const [board] = useLive("/api/board");
  const box = useRef();

  useEffect(() => {                                     // live tiles, 250 ms after the last key
    if (!text.trim()) { setPreview(null); return; }
    const t = setTimeout(() => api.post("/api/extract", { text }).then(setPreview).catch(() => {}), 250);
    return () => clearTimeout(t);
  }, [text]);

  useEffect(() => {                                     // paste a screenshot / photo straight into the note
    const onPaste = (e) => {
      const imgs = [...(e.clipboardData?.files || [])].filter((f) => f.type.startsWith("image/"));
      if (imgs.length) { e.preventDefault(); setFiles((fs) => [...fs, ...imgs]); }
    };
    addEventListener("paste", onPaste);
    return () => removeEventListener("paste", onPaste);
  }, []);

  const submit = async (e) => {
    e?.preventDefault();
    setError("");
    if (!me.who.trim()) { setError("Type your name at the top right first, so the note is signed."); return; }
    if (!text.trim()) { setError("Type the note first."); box.current.focus(); return; }
    setBusy(true);
    const form = new FormData();
    form.append("text", text);
    form.append("who", me.who);
    files.forEach((f) => form.append("files", f));
    try {
      const out = await api.post("/api/patients", form);
      setSent(out);
      setText(""); setFiles([]); setPreview(null);
      window.open(`#/case/${out.id}`, `case-${out.id}`);
      box.current.focus();
    } catch (err) { setError(err.message); }
    setBusy(false);
  };

  const f = preview?.fields || {};
  const recent = (board?.patients || []).slice().sort((a, b) => b.created - a.created).slice(0, 8);
  const arrival = f.ambulance === 1 ? "Ambulance" : f.ambulance === 0 ? "Walk-in" : "Arrival not recorded";

  return (
    <main className="chart">
      <form className="card note" onSubmit={submit}>
        <div className="card-h">
          <h1>New patient</h1>
          <span className="hint">Write it however you like: numbers, history, comments. The AI tidies it; you check it.</span>
        </div>
        <textarea ref={box} autoFocus value={text} placeholder={EXAMPLE} aria-label="Triage note"
                  onChange={(e) => setText(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) submit(); }} />
        <Pictures files={files} setFiles={setFiles} />
        {error && <div className="error" role="alert">{error}</div>}
        <div className="submit-row">
          <span className="hint">Signed as <b>{me.who || "—"}</b> ({me.role}) &middot; Ctrl+Enter sends</span>
          <button className="primary" disabled={busy}>{busy ? "Sending…" : "Submit and open case file"}</button>
        </div>
        {sent && (
          <div className="sent" role="status">
            <b>{sent.bed}</b> sent. The case file opened in a new window and the patient is on the whiteboard.{" "}
            <a href={`#/case/${sent.id}`} target={`case-${sent.id}`}>Open again</a>
          </div>
        )}
      </form>

      <section className="card check">
        <div className="card-h">
          <h2>Numbers found</h2>
          <span className="hint">Check them now. Grey = not in the note.</span>
        </div>
        <div className="who-row">
          <span className={f.age != null || f.sex ? "" : "missing"}>{whoLine(f)}</span>
          <span className={f.ambulance != null ? "" : "missing"}>{arrival}</span>
        </div>
        <Tiles fields={f} source={preview?.source} />
        {(preview?.warnings || []).map((w, i) => <div key={i} className="warn">{w}</div>)}
        {!preview && <p className="empty">Start typing: the numbers show up here.</p>}

        <h2 className="mt">Recent patients</h2>
        <ul className="recent">
          {recent.map((p) => (
            <li key={p.id}>
              <a href={`#/case/${p.id}`} target={`case-${p.id}`}>
                <span className="lvl" style={{ background: p.level ? LEVEL[p.level].colour : "var(--muted)" }}>{p.level ?? "…"}</span>
                <b>{p.bed}</b> {p.who} &middot; {p.complaint || "…"}
                <span className={`status ${p.status}`}>{p.status === "working" ? p.progress : p.status === "error" ? "AI failed" : ""}</span>
              </a>
            </li>
          ))}
          {!recent.length && <li className="empty">None yet.</li>}
        </ul>
      </section>
    </main>
  );
}
