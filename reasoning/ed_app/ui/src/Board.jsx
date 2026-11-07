// Screen 3: the whiteboard (TV or second laptop). Readable from across the room: one row per patient, most urgent
// first, level colour, big type, few columns. Waiting time turns red when it passes the level's target time.
// Only the triage nurse and the organising doctor can change it; everyone else sees a read-only board.
import React, { useState } from "react";
import { api, canEditBoard, LEVEL, hm, minutes, useLive, useMe, useNow } from "./api.js";
import { DoctorInput } from "./Case.jsx";

function TodoCell({ p, editor, me }) {
  const [open, setOpen] = useState(false);
  const [text, setText] = useState("");
  const todo = (t, body) => api.patch(`/api/patients/${p.id}/todos/${t.id}`, { who: me.who || "unknown", role: me.role, ...body })
    .catch((e) => alert(e.message));
  const left = p.todos.filter((t) => !t.done);
  const shown = open ? p.todos : left.slice(0, 3);
  const add = async () => {
    if (!text.trim()) return;
    await api.post(`/api/patients/${p.id}/todos`, { who: me.who || "unknown", role: me.role, text }).catch((e) => alert(e.message));
    setText("");
  };
  return (
    <div className="todo-cell">
      {shown.map((t) => (
        <button key={t.id} className={`chip ${t.done ? "done" : ""}`} disabled={!editor}
                title={editor ? (t.done ? "Mark not done" : "Tick off") : `${t.ai ? "Suggested" : "Added"}${t.why ? ` (${t.why})` : ""}`}
                onClick={() => todo(t, { done: !t.done })}>
          {t.done ? "✓ " : t.ai ? "✦ " : ""}{t.text}
        </button>
      ))}
      {!open && left.length > 3 && <button className="chip more" onClick={() => setOpen(true)}>+{left.length - 3} more</button>}
      {open && <button className="chip more" onClick={() => setOpen(false)}>less</button>}
      {!p.todos.length && <span className="muted">&mdash;</span>}
      {open && editor && (
        <input className="todo-add" value={text} placeholder="+ to-do" onChange={(e) => setText(e.target.value)}
               onKeyDown={(e) => e.key === "Enter" && add()} />
      )}
    </div>
  );
}

function Row({ p, meta, me, now }) {
  const editor = canEditBoard(me);
  const L = p.level ? LEVEL[p.level] : null;
  const m = minutes(p.created, now);
  const late = L && p.stage === "Waiting for doctor" && m > L.target;
  const set = (change) => api.patch(`/api/patients/${p.id}/board`, { who: me.who || "unknown", role: me.role, ...change })
    .catch((e) => alert(e.message));
  return (
    <tr className={`${p.leaving ? "leaving" : ""} ${p.flagged ? "flagged" : ""}`}>
      <td className="c-lvl"><div className="lvl" style={{ background: L ? L.colour : "var(--muted)" }}>{p.level ?? "…"}</div></td>
      <td className="c-bed">{p.bed}</td>
      <td className="c-pt">
        <a href={`#/case/${p.id}`} target={`case-${p.id}`}>
          <div className="pt-cc">{p.complaint || (p.status === "working" ? "…" : "not recorded")}</div>
          <div className="pt-who">{p.who}{p.flagged ? " · URGENT" : ""}
            {p.status === "working" && <span className="ai-busy"><span className="spin" /> {p.progress}</span>}
            {p.status === "error" && <span className="ai-err">AI failed: open to retry</span>}
            {p.questions > 0 && p.status !== "working" && <span className="q">{p.questions} missing</span>}
          </div>
        </a>
      </td>
      <td className="c-stage">
        {editor ? (
          <select value={p.stage} onChange={(e) => set({ stage: e.target.value })} aria-label={`Stage for ${p.bed}`}>
            {(meta?.stages || []).map((s) => <option key={s}>{s}</option>)}
          </select>
        ) : <span className="stage">{p.stage}</span>}
        {p.leaving && editor && <button className="clear" onClick={() => set({ cleared: true })}>{"✓"} Clear from board</button>}
      </td>
      <td className="c-doc">
        {editor ? <DoctorInput className="doc" value={p.doctor} onSave={(doctor) => set({ doctor })} />
                : <span className={p.doctor ? "" : "muted"}>{p.doctor || "—"}</span>}
      </td>
      <td className={`c-wait ${late ? "late" : ""}`} title={L ? `Target to see a doctor: ${L.target} min` : ""}>{hm(m)}</td>
      <td className="c-todo"><TodoCell p={p} editor={editor} me={me} /></td>
    </tr>
  );
}

export default function Board({ meta, left }) {
  const [me] = useMe();
  const now = useNow(15000);
  const [data, error] = useLive(`/api/board${left ? "?cleared=true" : ""}`);
  const editor = canEditBoard(me);
  if (error) return <main className="board"><p className="error">{error}</p></main>;
  if (!data) return <main className="board"><span className="spin" /> Loading…</main>;
  const ps = data.patients;
  const counts = [1, 2, 3, 4, 5].map((l) => ps.filter((p) => p.level === l).length);
  const clock = new Date(now * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  if (left) {
    return (
      <main className="board">
        <div className="board-bar"><h1>Left the ED</h1><a href="#/board" className="tab">Back to the board</a></div>
        <table className="bt">
          <tbody>
            {ps.map((p) => (
              <tr key={p.id}>
                <td className="c-lvl"><div className="lvl" style={{ background: p.level ? LEVEL[p.level].colour : "var(--muted)" }}>{p.level}</div></td>
                <td className="c-bed">{p.bed}</td>
                <td className="c-pt"><a href={`#/case/${p.id}`} target={`case-${p.id}`}><div className="pt-cc">{p.complaint}</div><div className="pt-who">{p.who}</div></a></td>
                <td>{p.stage}</td>
                <td>{editor && <button className="clear" onClick={() => api.patch(`/api/patients/${p.id}/board`, { who: me.who, role: me.role, cleared: false })}>Put back</button>}</td>
              </tr>
            ))}
            {!ps.length && <tr><td className="muted">Nobody yet.</td></tr>}
          </tbody>
        </table>
      </main>
    );
  }

  return (
    <main className="board">
      <div className="board-bar">
        <h1>{ps.length} in the ED</h1>
        <div className="counts">
          {counts.map((n, i) => n > 0 && (
            <span key={i} className="count" style={{ background: LEVEL[i + 1].colour }} title={LEVEL[i + 1].name}>L{i + 1}: {n}</span>
          ))}
        </div>
        <span className="muted">{editor ? `You (${me.role}) can change the board.` : `Read only for ${me.role}.`} {"✦"} = suggested; a person ticks it off.</span>
        <a href="#/board/left" className="tab">Left the ED ({data.n_cleared})</a>
        <span className="clock">{clock}</span>
      </div>
      <table className="bt">
        <thead>
          <tr><th>Lvl</th><th>Bed</th><th>Patient</th><th>Stage</th><th>Doctor</th><th>Waiting</th><th>To do</th></tr>
        </thead>
        <tbody>
          {ps.map((p) => <Row key={p.id} p={p} meta={meta} me={me} now={now} />)}
          {!ps.length && <tr><td colSpan={7} className="empty">No patients. New patients appear here as soon as triage submits.</td></tr>}
        </tbody>
      </table>
    </main>
  );
}
