// Screen C: the whiteboard. Put it on a second screen or laptop. Updates by itself.
// Everyone can look; only the triage nurse and the organising doctor can change it.
import { useState } from "react";
import { api, LEVEL_COLOUR, usePoll, waited } from "./api";
import { whoLine } from "./Chart";

function Row({ p, can, me, now, reload, stages }) {
  const [todo, setTodo] = useState("");
  const [doctor, setDoctor] = useState(null);
  const [dx, setDx] = useState("");                     // final diagnosis + level, for learning (optional)
  const [finalLevel, setFinalLevel] = useState("");
  const body = { who: me.who || "not recorded", role: me.role };
  const act = async (fn) => { try { await fn(); reload(); } catch (e) { alert(e.message); } };
  const mins = (now - p.created) / 60;
  const late = mins > 60 && ["Waiting for doctor", "Results pending"].includes(p.stage);
  return (
    <tr className={`${p.leaving || p.cleared ? "gone" : ""} ${p.flagged && !p.leaving && !p.cleared ? (p.needs_obs ? "needs-row" : "urgent-row") : ""}`}>
      <td><span className="lvl" style={{ background: p.level ? LEVEL_COLOUR[p.level] : "#607d8b" }}>{p.level ?? "…"}</span>
        {p.raised_by && <span className="raised" title={p.raised_by}>⬆</span>}</td>
      <td>{p.bed}</td>
      <td>
        <a href={`#/case/${p.id}`} target="ed_case_file"><b>{whoLine(p.who, p.status)}</b></a>
        <div className="sub">{p.complaint || (p.status === "working" ? "AI working…" : "")}</div>
        {p.reason && <div className={`why ${p.needs_obs ? "needs" : ""} ${p.override && !p.flagged ? "doctor" : ""}`}>{p.override ? "🩺 " : ""}{p.reason}</div>}
        {p.review && <div className="why needs">⚠ {p.review}</div>}
        {p.status === "working" && <div className="sub">⏳ {p.progress}</div>}
      </td>
      <td>
        {can && !p.cleared ? (
          <select value={p.stage} onChange={(e) => act(() => api.patch(`/api/patients/${p.id}/board`, { ...body, stage: e.target.value }))}>
            {stages.map((s) => <option key={s}>{s}</option>)}
          </select>
        ) : p.stage}
        {can && p.leaving && (
          <div style={{ marginTop: 6 }}>
            {!p.cleared && (
              <div className="outcome" title="Optional. Used to check and improve the AI; never changes this patient's record">
                <input value={dx} placeholder="Final diagnosis (optional)" aria-label="Final diagnosis"
                  onChange={(e) => setDx(e.target.value)} />
                <select value={finalLevel} aria-label="Level that was right" onChange={(e) => setFinalLevel(e.target.value)}>
                  <option value="">Right level?</option>
                  {[1, 2, 3, 4, 5].map((l) => <option key={l} value={l}>Level {l}{l === p.level ? " (as given)" : ""}</option>)}
                </select>
              </div>
            )}
            <button className="small clear-btn" onClick={() => act(() => api.patch(`/api/patients/${p.id}/board`, {
              ...body, cleared: !p.cleared,
              ...(p.cleared ? {} : { diagnosis: dx || null, final_level: finalLevel ? Number(finalLevel) : null }) }))}>
              {p.cleared ? "Put back on the board" : "✓ Clear from board"}
            </button>
          </div>
        )}
      </td>
      <td>
        {can ? (
          <input value={doctor ?? p.doctor} style={{ width: 170 }} onChange={(e) => setDoctor(e.target.value)}
            onBlur={() => doctor !== null && act(async () => { await api.patch(`/api/patients/${p.id}/board`, { ...body, doctor }); setDoctor(null); })}
            onKeyDown={(e) => e.key === "Enter" && e.target.blur()} />
        ) : p.doctor}
      </td>
      <td className={late ? "late" : ""} style={{ whiteSpace: "nowrap" }}>{waited(p.created, now)}</td>
      <td>
        {p.todos.map((t, i) => (
          <span key={i} className={`chip ${t.done ? "done" : ""}`}>
            {can ? (
              <button className="chip-btn" aria-pressed={t.done} title={t.done ? "Mark as not done" : "Tick off"}
                onClick={() => act(() => api.patch(`/api/patients/${p.id}/todos/${i}`, { ...body, done: !t.done }))}>
                {t.ai ? "✦ " : ""}{t.text}
              </button>
            ) : <span>{t.ai ? "✦ " : ""}{t.text}</span>}
            {can && <button className="chip-btn x" aria-label={`Remove ${t.text}`}
              onClick={() => act(() => api.patch(`/api/patients/${p.id}/todos/${i}`, { ...body, delete: true }))}>×</button>}
          </span>
        ))}
        {can && (
          <input value={todo} placeholder="+ to-do" style={{ width: 130, fontSize: 15 }} onChange={(e) => setTodo(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && todo.trim() && act(async () => { await api.post(`/api/patients/${p.id}/todos`, { ...body, text: todo }); setTodo(""); })} />
        )}
      </td>
    </tr>
  );
}

export default function Board({ me, meta }) {
  const [left, setLeft] = useState(false);              // show the patients who left the ED instead
  const [board, error, reload] = usePoll(left ? "/api/board?cleared=true" : "/api/board", 3000);
  const can = (meta?.board_editors || []).includes(me.role);
  const [learn] = usePoll("/api/learning", 4000);       // was 30 s: an outcome took that long to show (5q #11)
  if (!board) return <div className="page">{error || "Loading…"}</div>;
  return (
    <div className="page" style={{ maxWidth: "none" }}>
      <div className="row" style={{ color: "#9fb3c8", marginBottom: 6 }}>
        <span>{left ? `${board.patients.length} patients who left the ED` : `${board.patients.length} patients in the ED`} ·
          ✦ = suggested by the AI, a person ticks it off ·{" "}
          {can ? `You (${me.role}) can change the board` : "View only: the triage nurse or organising doctor changes the board"}</span>
        <button className="small" style={{ marginLeft: "auto" }} onClick={() => setLeft(!left)}>
          {left ? "← Back to the board" : `Left the ED (${board.n_cleared})`}
        </button>
      </div>
      {learn?.enabled && (
        <div className="learn-line" title="Corrections are collected while you work; the models only change after an update that passes the safety test (learning.py)">
          Learning from use: {learn.usable_for_learning} of {learn.needed_before_update} patients with an outcome needed for the
          next update · {learn.number_corrections} number fixes · AI models {learn.models_version}
        </div>
      )}
      {error && <div className="error">{error}</div>}
      <div className="board-wrap"><table className="board">
        <thead><tr><th>Level</th><th>Bed</th><th>Patient</th><th>Stage</th><th>Doctor</th><th>Waiting</th><th>To do</th></tr></thead>
        <tbody>
          {board.patients.map((p) => <Row key={p.id} p={p} can={can} me={me} now={board.now} reload={reload} stages={meta?.stages || []} />)}
        </tbody>
      </table></div>
      {!board.patients.length && <p style={{ fontSize: 22, color: "#9fb3c8" }}>No patients yet. They appear here when the triage nurse presses Submit.</p>}
    </div>
  );
}
