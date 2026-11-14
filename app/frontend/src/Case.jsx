// Screen B: the case file. Opens by itself after Submit. Big numbers on top, the tidy (editable) report,
// notes as typed, files, and on the right the AI's questions/to-dos and the chat. Add-info bar at the bottom.
import { useEffect, useState } from "react";
import { api, concern, LEVEL_COLOUR, usePoll, waited } from "./api";
import { form, ImagePicker, Questions, whoLine } from "./Chart";

const TILES = [["hr", "HR"], ["bp", "BP"], ["rr", "RR"], ["spo2", "SpO₂ %"], ["temp_c", "Temp °C"], ["nrs_pain", "Pain /10"],
  ["mental", "ACVPU"], ["age", "Age"]];
// "verbal" is also what new confusion (e.g. GCS 13-14) is stored as, since NEWS2 scores both 3; show both letters
const ACVPU = { alert: "A", verbal: "C/V", pain: "P", unresponsive: "U" };

function Vitals({ p, me, reload, meta }) {
  const [edit, setEdit] = useState(null);
  const [saving, setSaving] = useState(false);
  const f = p.fields, src = p.found?.found || {}, by = p.found?.by || {};
  const checks = Object.fromEntries((p.found?.check || []).map((c) => [c.field, c]));
  const checkOf = (k) => k === "bp" ? (checks.sbp || checks.dbp) : checks[k];
  const show = (k) => k === "bp" ? (f.sbp != null ? `${f.sbp}/${f.dbp ?? "?"}` : null)
    : k === "mental" && f.mental ? (ACVPU[f.mental] ?? f.mental) : f[k];
  const save = async () => {
    setSaving(true);          // it used to look like nothing happened while the AI was busy (5q #4)
    try { await api.put(`/api/patients/${p.id}/fields`, { fields: edit, who: me.who || "not recorded" }); setEdit(null); reload(); }
    catch (e) { alert(e.message); }
    setSaving(false);
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
        <div className="row"><button className="primary" onClick={save} disabled={saving}>{saving ? "Re-scoring…" : "Save and re-score"}</button><button onClick={() => setEdit(null)}>Cancel</button></div>
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
          const chk = checkOf(k);
          const who = k === "bp" ? by.sbp : by[k];
          return (
            <div key={k} className={`vital ${v == null ? "none" : c} ${chk ? "disagree" : ""}`}
              title={chk ? `Rules read ${chk.rules} from "${chk.rules_words}", AI read ${chk.ai} from "${chk.ai_words}". Press Fix numbers to settle it.`
                : from ? `From the note: "${from}"${who ? ` (read by: ${who})` : ""}` : "Not in the notes"}>
              <div className="k">{label}{chk && <span className="check-tag">check</span>}</div>
              <div className="v">{v ?? "not recorded"}</div>
              <div className="src">{chk ? `AI read ${chk.ai}` : who === "nurse" ? `fixed by ${p.found?.checked_by || "a nurse"}`
                : from ? `“${from}”` : " "}</div>
            </div>
          );
        })}
      </div>
      <div className="row muted" style={{ marginTop: -6, marginBottom: 12 }}>
        Numbers read by {meta?.test ? "fixed rules (test mode)" : "fixed rules and the AI, each checked against the note's words; where they disagree the tile says check"}{p.found?.checked_by ? `, checked by ${p.found.checked_by}` : ""}.
        {(p.found?.warnings || []).map((w) => <span key={w} className={`pill ${w.startsWith("please check") ? "amber" : "red"}`}>⚠ {w}</span>)}
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
  // keys must differ between a list and a heading: they used to clash (list keyed by count, heading by line), so after
  // an update React showed old headings twice (7 Nov stress test)
  const flush = () => { if (list.length) out.push(<ul key={`ul${out.length}`}>{list}</ul>); list = []; };
  text.split("\n").forEach((line, i) => {
    if (line.startsWith("## ")) { flush(); out.push(<h2 key={`h${i}`}>{line.slice(3)}</h2>); }
    else if (/^\s*[-*] /.test(line)) list.push(<li key={`li${i}`}>{line.replace(/^\s*[-*] /, "")}</li>);
    else if (line.trim()) { flush(); out.push(<p key={`p${i}`} className={/^_.*_$/.test(line) ? "muted" : ""}>{line.replace(/^_|_$/g, "")}</p>); }
  });
  flush();
  return <div className="report">{out}</div>;
}

// 14 Nov: a doctor changes the priority (lower or raise) with a reason; the AI never lowers by itself
const DOCTORS = ["Doctor", "Organising doctor"];
function Priority({ p, me, reload }) {
  const [open, setOpen] = useState(false);
  const [level, setLevel] = useState("");
  const [reason, setReason] = useState("");
  const [msg, setMsg] = useState("");
  const d = p.decision;
  if (!d) return null;
  const can = DOCTORS.includes(me.role);
  const send = async (body) => {
    try { await api.put(`/api/patients/${p.id}/priority`, { who: me.who || "not recorded", role: me.role, ...body }); setOpen(false); setReason(""); setMsg(""); reload(); }
    catch (e) { setMsg(e.message); }
  };
  return (
    <div className="priority">
      {p.review && <div className="review-warn">⚠ {p.review}</div>}
      <div className="row">
        {can && !open && <button className="small" onClick={() => { setLevel(String(p.level)); setOpen(true); }}>Change priority</button>}
        {can && p.override && !open && <button className="small" onClick={() => send({ clear: true })}>Back to the AI's level ({p.override.now_ai_level})</button>}
        {!can && <span className="muted">Only a doctor (Doctor or Organising doctor) can change the priority.</span>}
        {p.raised_by && !p.override && <span className="muted hint">The score and the safety rules disagree. A doctor can review and change the priority.</span>}
      </div>
      {open && (
        <div className="row priority-form">
          <label>Level <select value={level} onChange={(e) => setLevel(e.target.value)} aria-label="New level">
            {[1, 2, 3, 4, 5].map((l) => <option key={l} value={l}>{l}{l <= 2 ? " (urgent)" : ""}</option>)}
          </select></label>
          <input style={{ flex: 1 }} value={reason} onChange={(e) => setReason(e.target.value)} aria-label="Reason"
            placeholder="Reason (optional), e.g. chest pain settled, ECG and troponin normal" />
          <button className="primary" onClick={() => send({ level: Number(level), reason })}>Save</button>
          <button onClick={() => { setOpen(false); setMsg(""); }}>Cancel</button>
          <span className="muted">AI + safety rules say level {d.level}{d.flagged ? " (urgent)" : ""}.</span>
        </div>
      )}
      {msg && <div className="error">{msg}</div>}
    </div>
  );
}

// Core task 3: the SBAR handover, written by the AI from everything gathered so far (notes, answers, chat)
const SBAR_HEAD = /^\W*(Situation|Background|Assessment|Recommendations?)\b\W*$/i;
function Handover({ p, me, reload }) {
  const [busy, setBusy] = useState(false);
  const [copied, setCopied] = useState(false);
  const s = p.sbar || {};
  const write = async () => {
    setBusy(true);
    try { await api.post(`/api/patients/${p.id}/sbar`, { who: me.who || "not recorded" }); reload(); }
    catch (e) { alert(e.message); }
    setBusy(false);
  };
  const copy = async () => {
    try { await navigator.clipboard.writeText(s.text); setCopied(true); setTimeout(() => setCopied(false), 2000); }
    catch { alert("Copy didn't work here: use Download instead"); }
  };
  // "**Situation**" or "S - Situation:" on its own line becomes a heading; other ** marks are dropped
  const shown = (s.text || "").split("\n").map((l) => SBAR_HEAD.test(l.replace(/\*/g, "")) ? `## ${l.replace(/[^A-Za-z ]/g, "").trim()}`
    : /^\*\*[^*]+\*\*$/.test(l.trim()) ? `## ${l.replace(/\*\*/g, "").trim()}` : l.replace(/\*\*/g, "")).join("\n");
  const stale = s.text && (p.notes.length > s.n_notes || p.chat.length > s.n_chat);
  return (
    <div>
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 8 }}>
        <span className="muted">A handover for the next doctor (Situation, Background, Assessment, Recommendation), from the notes, every answer and number fix, and the chat.</span>
        <button className="primary" onClick={write} disabled={busy || p.status === "working" || !p.decision}>
          {busy ? "Writing… (about 30 s)" : s.text ? "Write it again" : "Write SBAR handover"}</button>
      </div>
      {s.text ? (
        <>
          <p className="muted">Written by the AI at {s.time}, asked by {s.who}. A draft: check it before you hand over.</p>
          {stale && <p className="disagree-note">New information was added after this was written: write it again to include it.</p>}
          <div className="row" style={{ marginBottom: 6 }}>
            <button className="small" onClick={copy}>{copied ? "✓ Copied" : "Copy"}</button>
            <a href={`/api/patients/${p.id}/sbar.txt`} download={`SBAR_${p.bed}.txt`}><button className="small">⬇ Download</button></a>
          </div>
          <Report text={shown} />
        </>
      ) : <p className="muted">No handover yet. Press the button when you are ready to hand over.</p>}
    </div>
  );
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
          {img.findings?.length > 0 && (
            <div className="checklist" title="Trained image model: red = above its cut-off, so look here; it is not a diagnosis">
              <span className="muted">{img.model}:</span>
              {img.findings.filter((f) => !f.type_of_bleed).map((f) => (
                <span key={f.name} className={`check ${f.flag ? "yes" : f.possible || f.also_seen ? "unclear" : "no"}`}
                  title={`model score ${f.score}${f.cut != null ? ` (cut-off ${f.cut})` : ""}, not a probability${f.possible ? "; the second model says possible: check" : ""}${f.also_seen ? "; seen, but usually not urgent" : ""}`}>
                  {f.flag ? "⚠" : f.possible ? "?" : f.also_seen ? "•" : "✓"} {f.possible ? `Possible ${f.name.toLowerCase()}` : f.name}
                </span>
              ))}
            </div>
          )}
          {img.checklist && (
            <div className="checklist">
              {Object.entries(img.checklist).map(([item, v]) => (
                <span key={item} className={`check ${v.seen}`} title={v.where || ""}>
                  {v.seen === "yes" ? "⚠" : v.seen === "no" ? "✓" : "?"} {item}{v.where ? ` · ${v.where}` : ""}
                </span>
              ))}
            </div>
          )}
          {img.disagree && <p className="disagree-note">⚠ {img.disagree}</p>}
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

export default function Case({ id, me, meta }) {
  const [p, error, reload] = usePoll(`/api/patients/${id}`);
  const [tab, setTab] = useState("case");
  const [side, setSide] = useState("questions");
  const [edit, setEdit] = useState(null);
  const [text, setText] = useState("");
  const [files, setFiles] = useState([]);
  const [msg, setMsg] = useState("");
  const picker = ImagePicker({ files, setFiles, compact: true, onReject: setMsg });     // a plain call: the 📎 button + thumbnails
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
        <div className="level" style={{ background: d ? LEVEL_COLOUR[p.level] : "#cfd8dc" }}
          title={p.override ? `Set by ${p.override.who}; AI + safety rules say ${p.override.now_ai_level}` : ""}>{d ? p.level : "…"}</div>
        <div>
          <div className="name">{p.bed} · {whoLine(p.who, p.status)} · {p.complaint || "…"}</div>
          <div className="row" style={{ marginTop: 4 }}>
            {d && p.override && <span className={`pill ${p.flagged ? "red" : ""} doctor`}>🩺 {p.reason} · {p.override.time}</span>}
            {d && p.override && <span className="pill">AI + safety rules now: level {d.level}{d.flagged ? ", urgent" : ""} · {Math.round(d.p_urgent * 100)}% chance of level 1–2</span>}
            {d && !p.override && (d.needs_obs
              ? <span className="pill needs" title="Two or more of BP, RR and HR are not recorded, so the patient can't be scored. The safety rule treats that as urgent until they are measured.">Urgent: needs vitals · {d.missing.join(" and ")} not recorded, measure {d.missing.length > 1 ? "them" : "it"}</span>
              : <span className={`pill ${d.flagged ? "red" : ""}`}>{d.flagged ? "Urgent: flagged" : `${p.group}: not flagged`} · {Math.round(d.p_urgent * 100)}% chance of level 1–2{d.flagged && p.reason ? ` · ${p.reason}` : ""}</span>)}
            {d && !p.override && d.needs_obs && <span className="pill">{Math.round(d.p_urgent * 100)}% chance of level 1–2 from what is recorded</span>}
            {d && <span className="pill">Levels it can't rule out: {d.level_set.join(", ")}</span>}
            {p.raised_by && <span className="pill amber" title="The urgent flag catches 90% of urgent patients, so it also raises some who aren't. Use your judgement.">⬆ {p.raised_by}</span>}
            {d && <span className="pill">NEWS2 {d.news2}{p.child ? " (adult score)" : ""}</span>}
            {d && p.child && (p.child_score
              ? <span className="pill amber" title="The urgency score comes from a model trained on children (US emergency visits under 16). NEWS2 is for age 16 and over, and the level model was trained on patients aged 15 and over. Use a paediatric scale and your own judgement.">Under 16: children's urgency score; NEWS2 and level from adult models</span>
              : <span className="pill amber" title="NEWS2 is for age 16 and over, and the urgency and level models were trained on patients aged 15 and over. Use a paediatric scale and your own judgement.">Under 16: scores from models trained on adults</span>)}
            <span className="pill">{p.stage} · {p.doctor}</span>
            <span className="pill">{p.timer.label} {waited(p.timer.since)}{p.timer.restarted ? ` · total ${waited(p.created)}` : ""}</span>
          </div>
          <Priority p={p} me={me} reload={reload} />
        </div>
      </div>
      {p.status === "working" && <div className="working"><div className="spin" />{p.progress}…</div>}
      {p.status === "error" && (
        <div className="working" style={{ background: "var(--red-bg)", color: "var(--red)" }}>
          {p.progress}. <button className="small" onClick={() => api.post(`/api/patients/${id}/rerun`).then(reload)}>Re-run AI</button>
          <span className="muted">{p.error}</span>
        </div>
      )}
      <Vitals p={p} me={me} reload={reload} meta={meta} />
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
            {[["case", "Case file"], ["sbar", "SBAR handover"], ["notes", `Notes as typed (${p.notes.length})`], ["files", `Files (${p.notes.reduce((n, x) => n + x.images.length, 0)})`], ["timeline", "Timeline"]].map(([k, l]) => (
              <button key={k} className={tab === k ? "on" : ""} onClick={() => setTab(k)}>{l}</button>
            ))}
          </div>
          {tab === "case" && (edit === null ? (
            <>
              <div className="row" style={{ justifyContent: "space-between" }}>
                <span className="muted">{p.report_by === "AI" ? "Written by the AI from the notes. Check it, and edit anything." : p.report_by ? `Last edited by ${p.report_by}` : ""}</span>
                <button className="edit-btn" onClick={() => setEdit(p.report)} disabled={!p.report || p.status === "working"}
                  title="Change anything in the case file; your name is saved with the edit">✎ Edit the case file</button>
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
          {tab === "sbar" && <Handover p={p} me={me} reload={reload} />}
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
              <h3>Questions for you to answer</h3>
              <p className="muted" style={{ marginTop: -6 }}>The AI is asking you. Click one, type the answer in the "Add information" bar above, press Add.</p>
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
