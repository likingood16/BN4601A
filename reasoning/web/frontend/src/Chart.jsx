// Screen A: the triage nurse types everything messy in one box, adds images, presses Submit.
// The case file opens by itself in another window; the AI's questions appear on the right.
import { useEffect, useState } from "react";
import { api, usePoll } from "./api";

const isPicture = (f) => f.type.startsWith("image/") || /\.(dcm|tif+)$/i.test(f.name);
// "?, ?" before the numbers are read looked broken (5q #11)
export const whoLine = (who) => (who === "?, ?" ? "reading the note…" : who);

export function ImagePicker({ files, setFiles, compact = false, onReject = alert }) {
  const [over, setOver] = useState(false);
  const add = (list) => {
    const bad = [...list].filter((f) => !isPicture(f));   // a PDF used to vanish without a word (5q #11)
    if (bad.length) onReject(`${bad.map((f) => f.name).join(", ")}: only pictures can be added (JPG, PNG ...)`);
    setFiles([...files, ...[...list].filter(isPicture)]);
  };
  const thumbs = files.length > 0 && (
    <div className="thumbs">
      {files.map((f, i) => (
        <div key={i} style={{ textAlign: "center" }}>
          <img src={URL.createObjectURL(f)} alt={f.name} />
          <div><button className="small" onClick={() => setFiles(files.filter((_, j) => j !== i))}>Remove</button></div>
        </div>
      ))}
    </div>
  );
  if (compact) return { button: (
    <label className="small-btn attach" title="Add X-rays or photos">
      📎 Image{files.length ? ` (${files.length})` : ""}
      <input type="file" accept="image/*,.dcm,.tif,.tiff" multiple hidden onChange={(e) => { add(e.target.files); e.target.value = ""; }} />
    </label>), thumbs };
  return (
    <>
      <label className={`drop ${over ? "over" : ""}`} style={{ display: "block", cursor: "pointer" }}
        onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
        onDrop={(e) => { e.preventDefault(); setOver(false); add(e.dataTransfer.files); }}>
        📎 Drop X-rays or photos here, or click to choose
        <input type="file" accept="image/*,.dcm,.tif,.tiff" multiple hidden onChange={(e) => add(e.target.files)} />
      </label>
      {files.length > 0 && (
        <div className="thumbs">
          {files.map((f, i) => (
            <div key={i} style={{ textAlign: "center" }}>
              <img src={URL.createObjectURL(f)} alt={f.name} />
              <div><button className="small" onClick={() => setFiles(files.filter((_, j) => j !== i))}>Remove</button></div>
            </div>
          ))}
        </div>
      )}
    </>
  );
}

export function form(text, me, files) {
  const fd = new FormData();
  fd.append("text", text);
  fd.append("who", me.who || "not recorded");
  files.forEach((f) => fd.append("files", f));
  return fd;
}

export function Questions({ p, onPick }) {
  if (!p) return <p className="muted">After you submit, the AI's questions about what's missing appear here.</p>;
  if (p.status === "working") return <p className="muted">⏳ {p.progress}…</p>;
  if (!p.questions.length) return <p className="muted">No questions from the AI.</p>;
  return p.questions.map((q, i) => (
    <div key={i} className="q" onClick={() => onPick(q)} title="Click to answer">{i + 1}. {q}</div>
  ));
}

export default function Chart({ me, setMe }) {
  const [done, setDone] = useState("");
  const [board] = usePoll("/api/board", 4000);
  const [target, setTarget] = useState("new");          // "new" or a patient id to add a note to
  const [text, setText] = useState("");
  const [files, setFiles] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [current, setCurrent] = useState(null);
  const [p] = usePoll(current ? `/api/patients/${current}` : null);
  const [answer, setAnswer] = useState("");
  const [find, setFind] = useState("");                 // search the picker: bed, age/sex or complaint words

  useEffect(() => { if (target !== "new") setCurrent(target); }, [target]);

  const label = (x) => `${x.bed} · level ${x.level ?? "…"} · ${whoLine(x.who)} · ${(x.complaint || "…").slice(0, 40)}`;
  const words = find.toLowerCase().split(/\s+/).filter(Boolean);
  const all = board?.patients || [];
  // every word must match; "P12" matches bed P12 exactly, not P120
  const matches = all.filter((x) => words.every((w) => /^p\d+$/.test(w) ? x.bed.toLowerCase() === w : label(x).toLowerCase().includes(w)));
  const shown = matches.some((x) => x.id === target) || target === "new" ? matches : [...all.filter((x) => x.id === target), ...matches];

  const submit = async () => {
    if (!text.trim() && !files.length) return setError("Type the note first");
    if (!me.who.trim()) return setError("Type your name next to Submit, so the case file shows who wrote it");
    setBusy(true); setError(""); setDone("");
    // opened now, or the browser blocks it; always the same named tab, not a new tab per patient (5q #11)
    const win = target === "new" ? window.open("", "ed_case_file") : null;
    try {
      let id = target;
      if (target === "new") id = (await api.post("/api/patients", form(text, me, files))).id;
      else await api.post(`/api/patients/${id}/notes`, form(text, me, files));
      if (win) win.location = `${location.pathname}#/case/${id}`;
      setCurrent(id); setText(""); setFiles([]);
      setDone(target === "new" ? "Sent. The case file opens in the case-file tab, and the patient is on the whiteboard."
        : "Added to the case file. The AI is updating it.");
    } catch (e) { setError(e.message); if (win) win.close(); }
    setBusy(false);
  };

  const sendAnswer = async () => {
    if (!answer.trim()) return;
    try { await api.post(`/api/patients/${current}/notes`, form(answer, me, [])); setAnswer(""); }
    catch (e) { setError(e.message); }
  };

  return (
    <div className="page">
      <div className="split">
        <div className="card">
          <div className="row patient-pick">
            <label htmlFor="which-patient" className="pick-label">Patient</label>
            <select id="which-patient" value={target} onChange={(e) => setTarget(e.target.value)} aria-label="Which patient">
              <option value="new">➕ New patient</option>
              {shown.length > 0 && (
                <optgroup label={words.length ? `${matches.length} of ${all.length} match “${find}”`
                  : `Add to a patient in the ED (${all.length}, most urgent first)`}>
                  {shown.map((x) => <option key={x.id} value={x.id}>{label(x)}</option>)}
                </optgroup>
              )}
            </select>
          </div>
          {all.length > 0 && (
            <div className="row patient-find">
              <input value={find} onChange={(e) => setFind(e.target.value)} aria-label="Find a patient"
                placeholder="Find a patient: bed (P12), age/sex (F, 72) or complaint words, then Enter"
                onKeyDown={(e) => { if (e.key === "Enter" && matches.length) { setTarget(matches[0].id); setFind(""); } }} />
              {words.length > 0 && <span className="muted">{matches.length ? `${matches.length} found` : "no match"}</span>}
            </div>
          )}
          <div className="row" style={{ marginBottom: 10 }}>
            <span className="muted">{target === "new" ? "Write it however you like. The AI tidies it up."
              : "This note is added to the chosen patient's case file."}</span>
            <label className="small-btn" style={{ marginLeft: "auto" }}>
              📄 Open a text file
              <input type="file" accept=".txt,text/plain" hidden onChange={async (e) => {
                const f = e.target.files[0];
                if (f) {
                  const loaded = (await f.text()).trim();
                  // a picture opened here filled the note with junk (5q #10)
                  if (!/\.(txt|text|md)$/i.test(f.name) || loaded.includes("\u0000") || loaded.includes("�"))
                    setError(`${f.name} isn't a text file: open a .txt note here, and add pictures in the picture box`);
                  else { setError(""); setText((t) => (t.trim() ? t.trim() + "\n" : "") + loaded); }
                }
                e.target.value = "";
              }} />
            </label>
          </div>
          <textarea className="note-box" value={text} onChange={(e) => { setText(e.target.value); setError(""); setDone(""); }}
            placeholder={"Example: 67M CP since 0600, sweaty, radiating L arm. HR 112 BP 90/60 sats 94 RA RR 22. hx DM HTN on metformin. came by amb. pain 8/10"} />
          <ImagePicker files={files} setFiles={setFiles} onReject={setError} />
          {error && <div className="error">{error}</div>}
          {done && <div className="ok-msg">{done}</div>}
          <div className="row" style={{ justifyContent: "flex-end", marginTop: 10 }}>
            <label className="muted">Written by{" "}
              <input value={me.who} onChange={(e) => { setMe({ ...me, who: e.target.value }); setError(""); }} placeholder="your name"
                style={{ width: 140 }} aria-label="Written by" />
            </label>
            <button className="primary" onClick={submit} disabled={busy}>{busy ? "Sending…" : "Submit"}</button>
          </div>
        </div>
        <div className="card">
          <h3>AI questions {p && <>for {p.bed} · {whoLine(p.who)} · <a href={`#/case/${p.id}`} target="ed_case_file">open case file</a></>}</h3>
          {/* the answer box sits ABOVE the questions so it doesn't jump down when they arrive (7 Nov stress test) */}
          {p && (
            <div className="row" style={{ marginBottom: 12 }}>
              <input style={{ flex: 1 }} value={answer} onChange={(e) => setAnswer(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && sendAnswer()} placeholder="Answer or add information" />
              <button onClick={sendAnswer}>Add</button>
            </div>
          )}
          <Questions p={p} onPick={(q) => setAnswer(`${q.replace(/\s*\[TEST MODE.*\]$/, "")} → `)} />
        </div>
      </div>
    </div>
  );
}
