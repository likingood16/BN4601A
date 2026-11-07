// Screen B: the case file. Opens by itself after Submit. Big numbers on top, the tidy (editable) report,
// notes as typed, files, and on the right the AI's questions/to-dos and the chat. Add-info bar at the bottom.
import { useEffect, useState } from "react";
import { api, concern, LEVEL_COLOUR, usePoll, waited } from "./api";
import { form, ImagePicker, Questions } from "./Chart";

const TILES = [["hr", "HR"], ["bp", "BP"], ["rr", "RR"], ["spo2", "SpO₂ %"], ["temp_c", "Temp °C"], ["nrs_pain", "Pain /10"],
  ["mental", "AVPU"], ["age", "Age"]];

function Vitals({ p, me, reload }) {
  const [edit, setEdit] = useState(null);
  const f = p.fields, src = p.found?.found || {};
  const show = (k) => k === "bp" ? (f.sbp != null ? `${f.sbp}/${f.dbp ?? "?"}` : null) : f[k];
  const save = async () => {
    try { await api.put(`/api/patients/${p.id}/fields`, { fields: edit, who: me.who || "not recorded" }); setEdit(null); reload(); }
    catch (e) { alert(e.message); }
  };
  if (edit) {
    const keys = ["age", "sex", "hr", "sbp", "dbp", "rr", "spo2", "temp_c", "nrs_pain", "mental", "ambulance", "injured", "chief_complaint"];
    return (
      <div className="card" style={{ marginBottom: 14 }}>
        <h3>Fix the numbers (the urgency and level are scored again)</h3>
        <div className="vitals">
          {keys.map((k) => (
            <div className="vital" key={k}>
              <div className="k">{p.field_names[k] || "Complaint"}</div>
              <input value={edit[k] ?? ""} onChange={(e) => setEdit({ ...edit, [k]: e.target.value })} />
            </div>
          ))}
        </div>
        <div className="row"><button className="primary" onClick={save}>Save and re-score</button><button onClick={() => setEdit(null)}>Cancel</button></div>
      </div>
    );
  }
  return (
    <>
      <div className="vitals">
        {TILES.map(([k, label]) => {
          const v = show(k);
          const c = k === "bp" ? concern("sbp", f.sbp) : concern(k, v);
          const from = k === "bp" ? src.sbp : src[k];
          return (
            <div key={k} className={`vital ${v == null ? "none" : c}`} title={from ? `From the note: "${from}"` : "Not in the notes"}>
              <div className="k">{label}</div>
              <div className="v">{v ?? "not recorded"}</div>
              <div className="src">{from ? `“${from}”` : " "}</div>
            </div>
          );
        })}
      </div>
      <div className="row muted" style={{ marginTop: -6, marginBottom: 12 }}>
        Numbers pulled out of the note by fixed rules{p.found?.checked_by ? `, checked by ${p.found.checked_by}` : ""}.
        {(p.found?.warnings || []).map((w) => <span key={w} className="pill red">{w}</span>)}
        <button className="small" onClick={() => setEdit({ ...p.fields })} disabled={p.status === "working"}>Fix numbers</button>
      </div>
    </>
  );
}

// Tiny markdown: ## headings, - bullets, _italic_ lines.
function Report({ text }) {
  if (!text) return <p className="muted">The case file appears here when the AI has finished.</p>;
  const out = [];
  let list = [];
  const flush = () => { if (list.length) out.push(<ul key={out.length}>{list}</ul>); list = []; };
  text.split("\n").forEach((line, i) => {
    if (line.startsWith("## ")) { flush(); out.push(<h2 key={i}>{line.slice(3)}</h2>); }
    else if (/^\s*[-*] /.test(line)) list.push(<li key={i}>{line.replace(/^\s*[-*] /, "")}</li>);
    else if (line.trim()) { flush(); out.push(<p key={i} className={/^_.*_$/.test(line) ? "muted" : ""}>{line.replace(/^_|_$/g, "")}</p>); }
  });
  flush();
  return <div className="report">{out}</div>;
}

function Files({ p }) {
  const [zoom, setZoom] = useState(null);
  const images = p.notes.flatMap((n) => n.images.map((img) => ({ ...img, who: n.who, time: n.time })));
  return (
    <div>
      <div className="row" style={{ marginBottom: 10 }}>
        <a href={`/api/patients/${p.id}/case_file`} download={`case_file_${p.bed}.txt`}><button>⬇ Download the whole case file</button></a>
        <span className="muted">Timeline, tidy case file and the notes as typed, in one text file.</span>
      </div>
      {!images.length && <p className="muted">No images yet. Add them with the bar at the bottom.</p>}
      {images.map((img) => (
        <div key={img.file} style={{ borderTop: "1px solid var(--line)", paddingTop: 10 }}>
          <b>{img.name}</b> <span className="muted">· {img.kind ? img.kind.replace("_", " ") : "waiting for the AI"} · added by {img.who}, {img.time}</span>
          <div className="file">
            {[["file", "As uploaded"], ["overlay", "AI marks"], ["outline", "AI outline"]].filter(([k]) => img[k])
              .map(([k, label]) => ({ file: img[k], label })).concat(img.extra || []).map(({ file, label }) => (
              <figure key={file} style={{ margin: 0 }}>
                <img src={`/api/images/${file}`} alt={label} onClick={() => setZoom(file)} />
                <figcaption className="muted">{label}</figcaption>
              </figure>
            ))}
          </div>
          {img.checklist && (
            <div className="checklist">
              {Object.entries(img.checklist).map(([item, v]) => (
                <span key={item} className={`check ${v.seen}`} title={v.where || ""}>
                  {v.seen === "yes" ? "⚠" : v.seen === "no" ? "✓" : "?"} {item}{v.where ? ` · ${v.where}` : ""}
                </span>
              ))}
            </div>
          )}
          <p style={{ fontSize: 14 }}>{img.text}</p>
        </div>
      ))}
      {zoom && <div className="zoom" onClick={() => setZoom(null)}><img src={`/api/images/${zoom}`} alt="" /></div>}
    </div>
  );
}

function Todos({ p, me, reload }) {
  const [text, setText] = useState("");
  const can = ["Triage nurse", "Organising doctor"].includes(me.role);
  const act = async (fn) => { try { await fn(); reload(); } catch (e) { alert(e.message); } };
  const body = { who: me.who || "not recorded", role: me.role };
  return (
    <div>
      {p.todos.map((t, i) => (
        <label key={i} className={`todo ${t.done ? "done" : ""}`}>
          <input type="checkbox" checked={t.done} disabled={!can}
            onChange={() => act(() => api.patch(`/api/patients/${p.id}/todos/${i}`, { ...body, done: !t.done }))} />
          {t.ai && <span className="ai-mark" title="Suggested by the AI">✦</span>}<span>{t.text}</span>
        </label>
      ))}
      {can ? (
        <div className="row" style={{ marginTop: 6 }}>
          <input style={{ flex: 1 }} value={text} onChange={(e) => setText(e.target.value)} placeholder="Add a to-do" />
          <button className="small" onClick={() => act(async () => { await api.post(`/api/patients/${p.id}/todos`, { ...body, text }); setText(""); })}>Add</button>
        </div>
      ) : <p className="muted">Only the triage nurse or organising doctor ticks these off.</p>}
    </div>
  );
}

function Chat({ p, me, reload }) {
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const ask = async () => {
    if (!q.trim()) return;
    setBusy(true);
    try { await api.post(`/api/patients/${p.id}/chat`, { text: q, who: me.who || "not recorded" }); setQ(""); reload(); }
    catch (e) { alert(e.message); }
    setBusy(false);
  };
  return (
    <div>
      {!p.chat.length && <p className="muted">Ask about this patient. The AI reads the whole case file, including your edits.</p>}
      {p.chat.map((m, i) => <div key={i} className={`chat-msg ${m.role}`}><span className="muted">{m.who} · {m.time}</span>{"\n"}{m.text}</div>)}
      <div className="row" style={{ marginTop: 8 }}>
        <input style={{ flex: 1 }} value={q} onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === "Enter" && ask()}
          placeholder="What else should we rule out?" disabled={!p.decision} />
        <button onClick={ask} disabled={busy || !p.decision}>{busy ? "Thinking…" : "Ask"}</button>
      </div>
    </div>
  );
}

export default function Case({ id, me }) {
  const [p, error, reload] = usePoll(`/api/patients/${id}`);
  const [tab, setTab] = useState("case");
  const [side, setSide] = useState("questions");
  const [edit, setEdit] = useState(null);
  const [text, setText] = useState("");
  const [files, setFiles] = useState([]);
  const [msg, setMsg] = useState("");
  const picker = ImagePicker({ files, setFiles, compact: true });     // a plain call: the 📎 button + thumbnails
  useEffect(() => { if (p) document.title = `${p.bed} · case file`; }, [p?.bed]);
  if (!p) return <div className="page">{error ? <div className="error">{error}</div> : "Loading…"}</div>;
  const d = p.decision;

  const add = async () => {
    try { await api.post(`/api/patients/${id}/notes`, form(text, me, files)); setText(""); setFiles([]); setMsg(""); reload(); }
    catch (e) { setMsg(e.message); }
  };
  const saveReport = async () => {
    try { await api.put(`/api/patients/${id}/report`, { text: edit, who: me.who || "not recorded" }); setEdit(null); reload(); }
    catch (e) { alert(e.message); }
  };

  return (
    <div className="page">
      <div className="head">
        <div className="level" style={{ background: d ? LEVEL_COLOUR[d.level] : "#cfd8dc" }}>{d ? d.level : "…"}</div>
        <div>
          <div className="name">{p.bed} · {p.who} · {p.complaint || "…"}</div>
          <div className="row" style={{ marginTop: 4 }}>
            {d && <span className={`pill ${d.flagged ? "red" : ""}`}>{d.flagged ? "Urgent: flagged" : "Not flagged"} · {Math.round(d.p_urgent * 100)}% chance of level 1–2</span>}
            {d && <span className="pill">Levels it can't rule out: {d.level_set.join(", ")}</span>}
            {p.raised_by && <span className="pill amber" title="The urgent flag catches 90% of urgent patients, so it also raises some who aren't. Use your judgement.">⬆ {p.raised_by}</span>}
            {d && <span className="pill">NEWS2 {d.news2}</span>}
            <span className="pill">{p.stage} · {p.doctor}</span>
            <span className="pill">Waiting {waited(p.created)}</span>
          </div>
        </div>
      </div>
      {p.status === "working" && <div className="working"><div className="spin" />{p.progress}…</div>}
      {p.status === "error" && (
        <div className="working" style={{ background: "var(--red-bg)", color: "var(--red)" }}>
          {p.progress}. <button className="small" onClick={() => api.post(`/api/patients/${id}/rerun`).then(reload)}>Re-run AI</button>
          <span className="muted">{p.error}</span>
        </div>
      )}
      <Vitals p={p} me={me} reload={reload} />
      <div className="addbar">
        <div className="card">
          <div className="row" style={{ flexWrap: "nowrap" }}>
            <input style={{ flex: 1, fontSize: 16 }} value={text} onChange={(e) => setText(e.target.value)} onKeyDown={(e) => e.key === "Enter" && add()}
              placeholder="Add information: trop 2.1, ECG ST depression V4-6, aspirin 300 given" aria-label="Add information" />
            {picker.button}
            <button className="primary" onClick={add}>Add</button>
          </div>
          {picker.thumbs}
          {msg && <div className="error">{msg}</div>}
        </div>
      </div>
      <div className="split wide">
        <div className="card">
          <div className="tabs">
            {[["case", "Case file"], ["notes", `Notes as typed (${p.notes.length})`], ["files", `Files (${p.notes.reduce((n, x) => n + x.images.length, 0)})`], ["timeline", "Timeline"]].map(([k, l]) => (
              <button key={k} className={tab === k ? "on" : ""} onClick={() => setTab(k)}>{l}</button>
            ))}
          </div>
          {tab === "case" && (edit === null ? (
            <>
              <div className="row" style={{ justifyContent: "space-between" }}>
                <span className="muted">{p.report_by === "AI" ? "Written by the AI from the notes. Check it, and edit anything." : p.report_by ? `Last edited by ${p.report_by}` : ""}</span>
                <button className="small" onClick={() => setEdit(p.report)} disabled={!p.report || p.status === "working"}>✎ Edit</button>
              </div>
              <Report text={p.report} />
            </>
          ) : (
            <>
              <textarea className="report-edit" value={edit} onChange={(e) => setEdit(e.target.value)} />
              <div className="row" style={{ marginTop: 8 }}><button className="primary" onClick={saveReport}>Save</button><button onClick={() => setEdit(null)}>Cancel</button></div>
            </>
          ))}
          {tab === "notes" && p.notes.map((n, i) => <div key={i} className="note"><span className="muted">{n.time} · {n.who}</span>{"\n"}{n.text}</div>)}
          {tab === "files" && <Files p={p} />}
          {tab === "timeline" && p.timeline.map((e, i) => <div key={i} className="note"><span className="muted">{e.time} · {e.who} · {e.what}</span>{"\n"}{e.text}</div>)}
        </div>
        <div className="card">
          <div className="tabs">
            <button className={side === "questions" ? "on" : ""} onClick={() => setSide("questions")}>Questions and to-dos</button>
            <button className={side === "chat" ? "on" : ""} onClick={() => setSide("chat")}>Chat</button>
          </div>
          {side === "questions" ? (
            <>
              <Questions p={p} onPick={(q) => setText(`${q.replace(/\s*\[TEST MODE.*\]$/, "")} → `)} />
              <h3 style={{ marginTop: 16 }}>To do (also on the whiteboard)</h3>
              <Todos p={p} me={me} reload={reload} />
            </>
          ) : <Chat p={p} me={me} reload={reload} />}
        </div>
      </div>
    </div>
  );
}
