// Screen 2: the case file. Follows the patient from triage to the doctors and the ward.
// Left: numbers + the case file (each line can be checked, edited or removed; AI lines are marked until a person
// checks them). Right: what's missing (questions), to-dos, chat, pictures, history. Bottom: add information, any time.
import React, { useEffect, useRef, useState } from "react";
import { api, canEditBoard, LEVEL, useLive, useMe, whoLine, hm, minutes, useNow } from "./api.js";
import { Tiles } from "./Chart.jsx";

const KIND = {
  ai: { badge: "✦ AI", title: "Tidied by the AI from the note. Check it." },
  typed: { badge: "note", title: "The nurse's own words, filed under this heading by fixed rules" },
  auto: { badge: "auto", title: "Written by code from the data (numbers, image tools, trained models)" },
  person: { badge: "", title: "Written or edited by a person" },
};
const NUMBER_FIELDS = [
  ["age", "Age"], ["sex", "Sex"], ["hr", "HR"], ["sbp", "BP sys"], ["dbp", "BP dia"], ["rr", "RR"], ["spo2", "SpO₂"],
  ["temp_c", "Temp °C"], ["nrs_pain", "Pain /10"], ["mental", "AVPU"], ["ambulance", "Arrival"],
];

function Header({ p, me, meta }) {
  const now = useNow();
  const d = p.decision;
  const L = d ? LEVEL[d.level] : null;
  const editor = canEditBoard(me);
  const setBoard = (change) => api.patch(`/api/patients/${p.id}/board`, { who: me.who, role: me.role, ...change })
    .catch((e) => alert(e.message));
  return (
    <div className="case-head">
      <div className="lvl-big" style={{ background: L ? L.colour : "var(--muted)" }} title={L ? L.name : "Scoring…"}>
        <span>{d ? d.level : "…"}</span><small>{L ? L.name : "scoring"}</small>
      </div>
      <div className="id">
        <h1>{p.bed} <span className="sub">{whoLine(p.fields)}</span></h1>
        <div className="complaint">{p.complaint || "Presenting complaint not recorded yet"}</div>
        {d && (
          <div className="decision">
            <b>{Math.round(d.p_urgent * 100)}%</b> chance of needing level 1–2 care
            {d.flagged ? <span className="flag">FLAGGED URGENT{d.floor_flag ? " · vital-sign safety floor" : ""}</span>
                       : <span className="ok">not flagged</span>}
            <span className="muted">NEWS2 {d.news2} &middot; levels it can't rule out: {d.level_set.join(", ")}</span>
          </div>
        )}
      </div>
      <div className="board-ctl">
        <label>Stage
          {editor ? (
            <select value={p.stage} onChange={(e) => setBoard({ stage: e.target.value })}>
              {(meta?.stages || [p.stage]).map((s) => <option key={s}>{s}</option>)}
            </select>
          ) : <b>{p.stage}</b>}
        </label>
        <label>Doctor
          {editor ? <DoctorInput value={p.doctor} onSave={(doctor) => setBoard({ doctor })} /> : <b>{p.doctor || "none"}</b>}
        </label>
        <span className="muted">Waiting {hm(minutes(p.created, now))}</span>
        <a className="ghost small" href={`/api/patients/${p.id}/export`} target="_blank" rel="noreferrer">Download case file</a>
      </div>
    </div>
  );
}

export function DoctorInput({ value, onSave, className = "" }) {
  const [v, setV] = useState(value || "");
  useEffect(() => setV(value || ""), [value]);
  return (
    <input className={className} value={v} placeholder="assign" aria-label="Doctor"
           onChange={(e) => setV(e.target.value)}
           onBlur={() => v !== (value || "") && onSave(v)}
           onKeyDown={(e) => e.key === "Enter" && e.currentTarget.blur()} />
  );
}

function Progress({ p, retry }) {
  if (p.status === "working") {
    return <div className="progress" role="status"><span className="spin" /> AI: {p.progress}. You can keep adding information.</div>;
  }
  if (p.status === "error") {
    return <div className="progress bad" role="alert">{p.progress} <button className="small" onClick={retry}>Retry</button></div>;
  }
  return null;
}

function Numbers({ p, me }) {
  const [edit, setEdit] = useState(false);
  const [vals, setVals] = useState({});
  const [err, setErr] = useState("");
  const start = () => {
    const v = {};
    NUMBER_FIELDS.forEach(([k]) => { v[k] = p.fields[k] ?? ""; });
    setVals(v); setEdit(true); setErr("");
  };
  const save = async () => {
    try {
      await api.put(`/api/patients/${p.id}/fields`, { who: me.who || "unknown", fields: vals });
      setEdit(false);
    } catch (e) { setErr(e.message); }
  };
  return (
    <section className="card">
      <div className="card-h">
        <h2>Numbers</h2>
        <span className="hint">Pulled from the note by fixed rules. Hover a tile to see the words it came from.</span>
        {!edit && <button className="ghost small" onClick={start} disabled={p.status === "working"}>Correct numbers</button>}
      </div>
      {p.rescore && !edit && (
        <div className="warn">New numbers came in after triage. The level was scored on the arrival numbers.{" "}
          <button className="small" onClick={() => api.put(`/api/patients/${p.id}/fields`, { who: me.who || "unknown", fields: p.fields })}>
            Score again with the latest numbers</button></div>
      )}
      {!edit ? <Tiles fields={p.fields} source={p.source} big /> : (
        <div className="fix">
          {NUMBER_FIELDS.map(([k, label]) => (
            <label key={k}>{label}
              {k === "sex" ? (
                <select value={vals[k]} onChange={(e) => setVals({ ...vals, [k]: e.target.value })}>
                  <option value="">not recorded</option><option value="male">male</option><option value="female">female</option>
                </select>
              ) : k === "mental" ? (
                <select value={vals[k]} onChange={(e) => setVals({ ...vals, [k]: e.target.value })}>
                  <option value="">not recorded</option><option value="alert">A</option><option value="verbal">V</option>
                  <option value="pain">P</option><option value="unresponsive">U</option>
                </select>
              ) : k === "ambulance" ? (
                <select value={vals[k]} onChange={(e) => setVals({ ...vals, [k]: e.target.value })}>
                  <option value="">not recorded</option><option value="1">ambulance</option><option value="0">walk-in</option>
                </select>
              ) : (
                <input inputMode="decimal" value={vals[k]} onChange={(e) => setVals({ ...vals, [k]: e.target.value })} />
              )}
            </label>
          ))}
          <div className="row">
            <button className="primary small" onClick={save}>Save and score again</button>
            <button className="ghost small" onClick={() => setEdit(false)}>Cancel</button>
            {err && <span className="error">{err}</span>}
          </div>
        </div>
      )}
      {p.warnings.map((w, i) => <div key={i} className="warn">{w}</div>)}
    </section>
  );
}

function Item({ it, pid, me }) {
  const [editing, setEditing] = useState(false);
  const [text, setText] = useState(it.text);
  const k = KIND[it.kind] || KIND.person;
  const patch = (body) => api.patch(`/api/patients/${pid}/items/${it.id}`, { who: me.who || "unknown", ...body })
    .catch((e) => alert(e.message));
  if (editing) {
    return (
      <li className="item editing">
        <textarea value={text} autoFocus rows={Math.max(1, Math.ceil(text.length / 70))} onChange={(e) => setText(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); patch({ text }); setEditing(false); }
                                      if (e.key === "Escape") setEditing(false); }} />
        <button className="primary small" onClick={() => { patch({ text }); setEditing(false); }}>Save</button>
        <button className="ghost small" onClick={() => { setText(it.text); setEditing(false); }}>Cancel</button>
      </li>
    );
  }
  const unchecked = it.kind === "ai" && !it.confirmed;
  return (
    <li className={`item ${it.kind} ${unchecked ? "unchecked" : ""}`}>
      <span className="txt">{it.text}</span>
      {it.quote && it.kind !== "person" && it.quote !== it.text && (
        <span className="quote" title="The words in the note this line is based on">from “{it.quote}”</span>
      )}
      <span className="meta">
        {k.badge && <span className={`badge ${it.kind}`} title={k.title}>{k.badge}{it.confirmed && it.kind === "ai" ? " ✓" : ""}</span>}
        {it.kind === "person" && <span className="by">{it.by} {it.time}</span>}
      </span>
      <span className="acts">
        {unchecked && <button className="small ok" title="Correct: mark as checked" onClick={() => patch({ confirmed: true })}>✓ Correct</button>}
        {!(it.kind === "auto" && ["Observations", "Triage decision support"].includes(it.section)) && (
          <button className="small ghost" onClick={() => { setText(it.text); setEditing(true); }}>Edit</button>)}
        {it.kind !== "auto" && <button className="small ghost" aria-label="Remove line" title="Remove"
                                       onClick={() => confirm("Remove this line from the case file?") && patch({ delete: true })}>&times;</button>}
      </span>
    </li>
  );
}

function AddLine({ pid, section, me }) {
  const [open, setOpen] = useState(false);
  const [text, setText] = useState("");
  const save = async () => {
    if (!text.trim()) return setOpen(false);
    await api.post(`/api/patients/${pid}/items`, { who: me.who || "unknown", section, text }).catch((e) => alert(e.message));
    setText(""); setOpen(false);
  };
  if (!open) return <button className="add-line" onClick={() => setOpen(true)}>+ add</button>;
  return (
    <div className="item editing">
      <input autoFocus value={text} placeholder={`Add to ${section}`} onChange={(e) => setText(e.target.value)}
             onKeyDown={(e) => { if (e.key === "Enter") save(); if (e.key === "Escape") setOpen(false); }} />
      <button className="primary small" onClick={save}>Add</button>
    </div>
  );
}

function CaseFile({ p, me }) {
  const [showRejected, setShowRejected] = useState(false);
  const unchecked = p.items.filter((i) => i.kind === "ai" && !i.confirmed);
  const checkAll = () => Promise.all(unchecked.map((i) =>
    api.patch(`/api/patients/${p.id}/items/${i.id}`, { who: me.who || "unknown", confirmed: true })));
  return (
    <section className="card casefile">
      <div className="card-h">
        <h2>Case file</h2>
        <span className="hint">Uses only what was written. Edit any line; your edits are never overwritten.</span>
        {unchecked.length > 0 && (
          <button className="small ok" onClick={checkAll}>{"✓"} All {unchecked.length} AI lines are correct</button>)}
      </div>
      {p.sections.map((s) => {
        const xs = p.items.filter((i) => i.section === s);
        return (
          <div key={s} className={`sec ${xs.length ? "" : "empty"} ${s === "Triage decision support" ? "triage" : ""}`}>
            <h3>{s}</h3>
            <ul>
              {xs.map((it) => <Item key={it.id} it={it} pid={p.id} me={me} />)}
              {!xs.length && <li className="nr">not recorded</li>}
            </ul>
            {s !== "Triage decision support" && <AddLine pid={p.id} section={s} me={me} />}
          </div>
        );
      })}
      {p.rejected.length > 0 && (
        <div className="rejected">
          <button className="link" onClick={() => setShowRejected(!showRejected)}>
            {p.rejected.length} AI line(s) rejected because the note doesn't support them {showRejected ? "▴" : "▾"}
          </button>
          {showRejected && (
            <ul>{p.rejected.map((r, i) => <li key={i}><s>{r.text}</s> <span className="muted">({r.section}: {r.why})</span></li>)}</ul>
          )}
        </div>
      )}
      <details className="raw">
        <summary>Notes exactly as typed ({p.notes.length})</summary>
        {p.notes.map((n, i) => (
          <div key={i} className="raw-note"><span className="muted">{n.time} {n.who}{n.answers ? ` — answering “${n.answers}”` : ""}</span>
            <p>{n.text || "(pictures only)"}</p></div>
        ))}
      </details>
    </section>
  );
}

function Questions({ p, me, answer }) {
  return (
    <div className="panel">
      <p className="hint">Missing information that would change care for this complaint. Answer here and it goes into the case file.</p>
      {p.questions.length === 0 && <p className="empty">{p.status === "working" ? "Checking…" : "Nothing important missing."}</p>}
      <ol className="questions">
        {p.questions.map((q) => (
          <li key={q.text}>
            <div>{q.text}</div>
            <div className="qmeta"><span className="badge">{q.by === "AI" ? "✦ AI" : q.why}</span>
              <button className="small" onClick={() => answer(q.text)}>Answer</button>
              <button className="small ghost" onClick={() => api.post(`/api/patients/${p.id}/questions/dismiss`, { who: me.who || "unknown", text: q.text })}>Not needed</button>
            </div>
          </li>
        ))}
      </ol>
    </div>
  );
}

export function Todos({ p, me, compact = false }) {
  const editor = canEditBoard(me);
  const [text, setText] = useState("");
  const todo = (t, body) => api.patch(`/api/patients/${p.id}/todos/${t.id}`, { who: me.who || "unknown", role: me.role, ...body })
    .catch((e) => alert(e.message));
  const add = async () => {
    if (!text.trim()) return;
    await api.post(`/api/patients/${p.id}/todos`, { who: me.who || "unknown", role: me.role, text }).catch((e) => alert(e.message));
    setText("");
  };
  const open = p.todos.filter((t) => !t.done), done = p.todos.filter((t) => t.done);
  return (
    <div className="panel">
      {!compact && <p className="hint">{"✦"} = suggested by the rules or AI; a person ticks it off. {editor ? "" : "Only the triage nurse or organising doctor can change these."}</p>}
      <ul className="todos">
        {[...open, ...done].map((t) => (
          <li key={t.id} className={t.done ? "done" : ""}>
            <label>
              <input type="checkbox" checked={t.done} disabled={!editor} onChange={() => todo(t, { done: !t.done })} />
              <span>{t.ai && <span className="star" title={`Suggested (${t.by}${t.why ? `: ${t.why}` : ""})`}>{"✦"}</span>}{t.text}</span>
            </label>
            {t.done && <span className="by">{t.done_by}</span>}
            {editor && <button className="small ghost" aria-label="Remove to-do" onClick={() => todo(t, { delete: true })}>&times;</button>}
          </li>
        ))}
        {!p.todos.length && <li className="empty">No to-dos.</li>}
      </ul>
      {editor && (
        <div className="row">
          <input value={text} placeholder="Add a to-do" onChange={(e) => setText(e.target.value)} onKeyDown={(e) => e.key === "Enter" && add()} />
          <button className="small" onClick={add}>Add</button>
        </div>
      )}
    </div>
  );
}

function Chat({ p, me }) {
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const end = useRef();
  useEffect(() => end.current?.scrollIntoView({ block: "end" }), [p.chat.length, busy]);
  const ask = async (text) => {
    if (!text.trim() || busy) return;
    setBusy(true); setQ("");
    await api.post(`/api/patients/${p.id}/chat`, { who: me.who || "unknown", text }).catch((e) => alert(e.message));
    setBusy(false);
  };
  const starters = ["What are the main concerns?", "What could this be? Give a differential.", "Write an SBAR handover."];
  return (
    <div className="panel chat">
      <p className="hint">The AI reads the latest case file, including people's edits. It supports, it doesn't decide.</p>
      <div className="msgs">
        {p.chat.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}><span className="muted">{m.role === "ai" ? "✦ AI" : m.who} {m.time}</span><div>{m.text}</div></div>
        ))}
        {busy && <div className="msg ai"><span className="spin" /> thinking…</div>}
        <div ref={end} />
      </div>
      {!p.chat.length && <div className="starters">{starters.map((s) => <button key={s} className="small ghost" onClick={() => ask(s)}>{s}</button>)}</div>}
      <div className="row">
        <input value={q} placeholder={p.decision ? "Ask about this patient" : "Available when the case file is ready"} disabled={!p.decision}
               onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === "Enter" && ask(q)} />
        <button className="small" disabled={!p.decision || busy} onClick={() => ask(q)}>Ask</button>
      </div>
    </div>
  );
}

function Pictures({ p }) {
  const imgs = p.notes.flatMap((n) => n.images.map((i) => ({ ...i, who: n.who, time: n.time })));
  if (!imgs.length) return <div className="panel"><p className="empty">No pictures. Add them with the bar at the bottom.</p></div>;
  return (
    <div className="panel pics">
      {imgs.map((i) => (
        <div key={i.file} className="pic">
          <div className="pic-h"><b>{i.name}</b> <span className="muted">{i.kind ? i.kind.replace("_", " ") : "reading…"} &middot; {i.who} {i.time}</span></div>
          <div className="pic-row">
            <figure><img src={`/api/images/${i.file}`} alt={i.name} /><figcaption>As uploaded</figcaption></figure>
            {i.overlay && <figure><img src={`/api/images/${i.overlay}`} alt="AI marks" /><figcaption>{"✦"} AI marks</figcaption></figure>}
            {i.outline && <figure><img src={`/api/images/${i.outline}`} alt="AI outline" /><figcaption>{"✦"} AI outline</figcaption></figure>}
          </div>
          {i.text && <p><span className="badge ai">{"✦"} AI, unchecked</span> {i.text} {i.tools}</p>}
        </div>
      ))}
    </div>
  );
}

function History({ p }) {
  return (
    <div className="panel">
      <ul className="timeline">
        {p.timeline.slice().reverse().map((e, i) => (
          <li key={i}><span className="muted">{e.time} &middot; {e.who}</span> <b>{e.what}</b> {e.text}</li>
        ))}
      </ul>
      {p.seconds != null && <p className="muted">Last AI run took {p.seconds} s.</p>}
    </div>
  );
}

function AddBar({ p, me, answering, setAnswering }) {
  const [text, setText] = useState("");
  const [files, setFiles] = useState([]);
  const [msg, setMsg] = useState("");
  const box = useRef();
  const file = useRef();
  useEffect(() => { if (answering) box.current.focus(); }, [answering]);
  const send = async () => {
    if (!text.trim() && !files.length) return;
    if (!me.who.trim()) { setMsg("Type your name at the top right first."); return; }
    const form = new FormData();
    form.append("text", text); form.append("who", me.who);
    if (answering) form.append("answers", answering);
    files.forEach((f) => form.append("files", f));
    try {
      const out = await api.post(`/api/patients/${p.id}/notes`, form);
      setText(""); setFiles([]); setAnswering("");
      setMsg(out.queued_behind_ai ? "Saved. The AI will add it as soon as it finishes the current step." : "Saved. Adding it to the case file…");
      setTimeout(() => setMsg(""), 4000);
    } catch (e) { setMsg(e.message); }
  };
  return (
    <div className="addbar">
      {answering && <div className="answering">Answering: <b>{answering}</b> <button className="link" onClick={() => setAnswering("")}>cancel</button></div>}
      <div className="row">
        <button className="ghost" title="Add pictures" aria-label="Add pictures" onClick={() => file.current.click()}>{"📎"}{files.length ? ` ${files.length}` : ""}</button>
        <input type="file" ref={file} hidden multiple accept="image/*" onChange={(e) => { setFiles([...files, ...e.target.files]); e.target.value = ""; }} />
        <textarea ref={box} rows={1} value={text} aria-label="Add information"
                  placeholder={answering ? "Type the answer" : "Add information: results, new obs, exam findings, plan, pictures… (Enter sends)"}
                  onChange={(e) => setText(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }} />
        <button className="primary" onClick={send}>Add</button>
      </div>
      {msg && <div className="hint">{msg}</div>}
    </div>
  );
}

export default function Case({ id, meta }) {
  const [me] = useMe();
  const [p, error, reload] = useLive(`/api/patients/${id}`);
  const [tab, setTab] = useState("questions");
  const [answering, setAnswering] = useState("");
  useEffect(() => { if (p) document.title = `${p.bed} ${p.complaint ? "· " + p.complaint : ""} · case file`; }, [p?.bed, p?.complaint]);
  if (error) return <main className="center"><p className="error">{error}</p><a href="#/board">Back to the whiteboard</a></main>;
  if (!p) return <main className="center"><span className="spin" /> Loading…</main>;
  const openTodos = p.todos.filter((t) => !t.done).length;
  const tabs = [["questions", `Missing (${p.questions.length})`], ["todos", `To-do (${openTodos})`], ["chat", "Ask AI"],
                ["pictures", "Pictures"], ["history", "History"]];
  return (
    <main className="case">
      <Header p={p} me={me} meta={meta} />
      <Progress p={p} retry={() => api.post(`/api/patients/${p.id}/retry`).then(reload)} />
      <div className="case-body">
        <div className="left">
          <Numbers p={p} me={me} />
          <CaseFile p={p} me={me} />
        </div>
        <aside className="card side">
          <div className="tabs" role="tablist">
            {tabs.map(([k, label]) => (
              <button key={k} role="tab" aria-selected={tab === k} className={tab === k ? "on" : ""} onClick={() => setTab(k)}>{label}</button>
            ))}
          </div>
          {tab === "questions" && <Questions p={p} me={me} answer={(q) => setAnswering(q)} />}
          {tab === "todos" && <Todos p={p} me={me} />}
          {tab === "chat" && <Chat p={p} me={me} />}
          {tab === "pictures" && <Pictures p={p} />}
          {tab === "history" && <History p={p} />}
        </aside>
      </div>
      <AddBar p={p} me={me} answering={answering} setAnswering={setAnswering} />
    </main>
  );
}
