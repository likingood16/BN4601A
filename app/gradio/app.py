"""ED triage app, Gradio version (9 Nov 2026). Same back end as the React app (app/server.py: the patients,
the AI job, the whiteboard rules, the learning log), with Gradio screens, so it can run as a Hugging Face Space on
ZeroGPU (ZeroGPU only works with Gradio).

    python app.py --test        # any laptop, no GPU: MedGemma's writing is faked (same as server.py --test)
    python app.py               # real AI on this computer's GPU (slow on a 4 GB GPU: see REPORT_NOTES 5q)
    On a Hugging Face Space: app.py runs with no arguments -> real AI. Set BN4601A_TEST=1 as a Space variable
    for test mode. Then open http://127.0.0.1:7860 (or the Space).

ZeroGPU: the GPU exists only inside functions marked @spaces.GPU, and each call runs in a separate process. So one
patient's whole AI run is one GPU call: the patient goes in, the updated patient comes back and is copied onto the
board. Re-scoring and chat are GPU calls of their own. Off ZeroGPU the decorator does nothing and the AI runs in a
background thread exactly like the React app. BN4601A_FAKE_ZERO=1 imitates the copy-in/copy-out on a laptop.
"""
import argparse
import copy
import html
import os
import shutil
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]                                       # app/gradio/app.py (repo and Space alike)
sys.path[:0] = [str(REPO / "app"), str(REPO / "reasoning" / "src")]

import gradio as gr                                          # noqa: E402
from fastapi import HTTPException                            # noqa: E402

import server as S                                           # noqa: E402  (the React app's back end)

# ---------------------------------------------------------------- GPU calls (ZeroGPU or plain)
FAKE_ZERO = os.environ.get("BN4601A_FAKE_ZERO") == "1"
try:
    import spaces
    from spaces.config import Config as _SpacesConfig
    ZERO = bool(getattr(_SpacesConfig, "zero_gpu", False))
    GPU = spaces.GPU
except Exception:                                            # not on a Space: the decorator does nothing
    ZERO = False

    def GPU(fn=None, **_):
        return fn if fn is not None else (lambda f: f)
if FAKE_ZERO:                                                # laptop check: arguments and results are copies
    ZERO = True

    def GPU(fn=None, **_):
        def wrap(f):
            return lambda *a, **k: copy.deepcopy(f(*copy.deepcopy(a), **copy.deepcopy(k)))
        return wrap(fn) if fn is not None else wrap

ap = argparse.ArgumentParser()
ap.add_argument("--test", action="store_true", help="fake MedGemma writing, no GPU")
ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
ap.add_argument("--board", default="", help="where the board is saved (default: case_files/board.pkl here)")
ap.add_argument("--lan", action="store_true", help="reachable from other devices on the same Wi-Fi")
ARGS, _ = ap.parse_known_args()
TEST = ARGS.test or os.environ.get("BN4601A_TEST") == "1"


def _model_files_in_place():
    """Space bundle: the CheXpert weights travel as home/.torchxrayvision/...; put them where torchxrayvision looks."""
    src = REPO / "home"
    if src.is_dir():
        for f in src.rglob("*"):
            dest = Path.home() / f.relative_to(src)
            if f.is_file() and not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(f, dest)


def _import_now():
    """Import what the AI job imports, here, before the screens start: Gradio's screen refresh and the AI job importing
    the same library at the same moment from two threads failed with "partially initialized module 'polars'" (9 Nov)."""
    import importlib
    for name in ("polars", "pandas", "scipy", "sklearn", "joblib", "ultralytics", "torchvision",
                 "charting", "learning", "cards", "ed_assessment", "vitals_model", "image_models", "image_reader",
                 "image_checklist", "image_tools", "pathway_heads"):
        try:
            importlib.import_module(name)
        except Exception as e:                               # an optional extra that isn't installed
            print(f"(not loaded at start: {name}: {type(e).__name__})", flush=True)


def _files_from_hub():
    """On the Space the trained files are not in the Space (anyone can see a public Space's files): they come from the
    group's private model repo, with the Space's HF_TOKEN secret. Returns "" when all are in place, else what failed."""
    import install_models
    if all(f.exists() for f in install_models.ai_targets().values()):
        return ""
    try:
        install_models.from_hub(token=os.environ.get("HF_TOKEN") or None)
    except SystemExit as e:
        return str(e).splitlines()[0]
    return ""


_model_files_in_place()
NO_FILES = _files_from_hub() if os.environ.get("SPACE_ID") else ""
if NO_FILES and not TEST:                                    # without its files the real AI can't start: say why
    print("model files missing, starting in test mode:", NO_FILES, flush=True)
    TEST = True
_import_now()
if TEST:
    from test_backend import TestPipeline
    AI = TestPipeline()
else:
    from pipeline import EDPipeline
    AI = EDPipeline()                                        # at module level: ZeroGPU needs the model placed here

BOARD = Path(ARGS.board) if ARGS.board else HERE / "case_files" / "board.pkl"
BOARD.parent.mkdir(parents=True, exist_ok=True)
store = S.store = S.Store(AI, BOARD)
RUN_LOCK = threading.Lock()                                  # ZeroGPU: one patient's AI run at a time
AI_FIELDS = ("notes", "fields", "found", "visit", "report", "report_by", "questions", "todos", "status", "progress",
             "error", "pending", "report_seen_by_chat")


def _work_seconds(p):
    """GPU time to ask for: ZeroGPU refuses a call that asks for more than the viewer has left today (2 min logged
    out, 5 min with a free account), so ask for what a patient needs: about 30 s on an A40 + more per picture."""
    pictures = sum(len(p.notes[i]["images"]) for i in p.pending if i < len(p.notes))
    return min(90 + 40 * pictures, 240)


@GPU(duration=_work_seconds)
def _gpu_work(p):
    """One patient's whole AI run (numbers, urgency, pictures, case file, questions), as Store._job does it."""
    store.ai = AI
    while True:
        try:
            store._work(p)
        except Exception as e:                               # the notes are saved; the AI can be re-run
            import traceback
            traceback.print_exc()
            p.status, p.progress, p.error = "error", "The AI step failed; the notes are saved", repr(e)
            return p
        if not p.pending:
            p.status, p.progress, p.error = "ready", f"Case file updated {S.now()}", ""
            return p


@GPU(duration=30)
def _gpu_triage(fields):
    return AI.triage(fields, explain=False)


@GPU(duration=60)
def _gpu_chat(visit, text):
    return AI.chat(visit, text), visit


class _RemoteAI:
    """On ZeroGPU the model only answers inside GPU calls: these send fix-numbers and chat there."""

    def __init__(self, real):
        self.real = real

    def __getattr__(self, name):
        return getattr(self.real, name)

    def triage(self, fields, explain=False):
        return _gpu_triage(fields)

    def chat(self, visit, text):
        answer, back = _gpu_chat(visit, text)
        visit.messages, visit.unseen = back.messages, back.unseen
        return answer


if ZERO:                                                    # (test mode too, so BN4601A_FAKE_ZERO checks this path)
    store.ai = _RemoteAI(AI)


def start_ai(p):
    """Off ZeroGPU: a background thread, like the React app (screens poll). On ZeroGPU: one GPU call, then copy back."""
    if not ZERO:
        store.run(p)
        return
    with RUN_LOCK:
        p.status, p.progress = "working", "AI working on the GPU"
        store.changed()
        back = _gpu_work(p)
        if back is not p:
            for k in AI_FIELDS:
                setattr(p, k, getattr(back, k))
        store.changed()


# ---------------------------------------------------------------- helpers
NEW = "new"
TILES = [("hr", "HR"), ("sbp", "BP"), ("rr", "RR"), ("spo2", "SpO₂ %"), ("temp_c", "Temp °C"), ("nrs_pain", "Pain /10"),
         ("mental", "ACVPU"), ("age", "Age")]
FIX_ROWS = [("age", "Age"), ("sex", "Sex (male/female)"), ("hr", "HR"), ("sbp", "BP sys"), ("dbp", "BP dia"),
            ("rr", "RR"), ("spo2", "SpO2 %"), ("temp_c", "Temp C"), ("nrs_pain", "Pain /10"),
            ("mental", "Consciousness (alert/verbal/pain/unresponsive)"), ("ambulance", "Ambulance (1/0)"),
            ("injured", "Injury (1/0)"), ("chief_complaint", "Complaint")]


def _call(fn, *a, **k):
    try:
        return fn(*a, **k)
    except HTTPException as e:
        raise gr.Error(str(e.detail))


def _guard(fn, n):
    """A refused action (no name, wrong file ...) shows a warning and leaves the screen as it was; a raised gr.Error
    would put a red 'Error' badge on every output box."""
    def run(*a):
        try:
            return fn(*a)
        except gr.Error as e:
            gr.Warning(getattr(e, "message", str(e)))
            return gr.skip() if n == 1 else [gr.skip()] * n
    return run


def on(event, fn, inputs, outputs):
    outputs = outputs if isinstance(outputs, list) else [outputs]
    return event(_guard(fn, len(outputs)), inputs, outputs)


def _need_name(name):
    if not (name or "").strip():
        raise gr.Error("Type your name (top right) first, so the case file shows who did it")
    return name.strip()


def _num(v):
    return f"{v:g}" if isinstance(v, (int, float)) else str(v)


def _choices(new=True, cleared=False, everyone=False):
    """Patients for a picker. everyone=True (the case-file picker): also those who left the ED, whose case files
    stay readable (a cleared patient vanished from the list while open: "not in the list of choices", 9 Nov)."""
    out = [("➕ New patient", NEW)] if new else []
    rows = S.board(cleared=cleared)["patients"]
    if everyone:
        rows = S.board(cleared=False)["patients"] + [{**r, "complaint": f"(left the ED) {r['complaint'] or ''}"}
                                                       for r in S.board(cleared=True)["patients"]]
    for r in rows:
        level = r["level"] if r["level"] else "…"
        who = "reading the note…" if r["who"] == "?, ?" else r["who"]
        out.append((f"{r['bed']} · level {level} · {who} · {(r['complaint'] or '')[:45]}", r["id"]))
    return out


def _patient(pid):
    return store.patients.get(pid) if pid and pid != NEW else None


def _header(p):
    if p is None:
        return "Pick a patient."
    d = p.visit.decision if p.visit else None
    who = S._who_line(p.fields)
    if d is None:
        line = f"### {p.bed} · {who}\n\n⏳ {p.progress}…"
        return line + (f"\n\n⚠️ {p.error}" if p.status == "error" else "")
    if d.needs_obs:                              # urgent only because vital signs are missing (10 Nov, A1)
        flag = (f"🟠 **Urgent: needs vitals** · {' and '.join(d.missing)} not recorded, measure "
                f"{'them' if len(d.missing) > 1 else 'it'} to score this patient · {d.p_urgent:.0%} chance of level 1–2 "
                "from what is recorded")
    else:
        flag = (f"🔴 **Urgent: flagged** · {d.p_urgent:.0%} chance of level 1–2 · {d.reason()}" if d.flagged
                else f"Not flagged · {d.p_urgent:.0%} chance of level 1–2")
    parts = [f"## Level {d.level} · {p.bed} · {who} · {p.complaint()}",
             f"{flag} · levels it can't rule out: {', '.join(map(str, d.level_set))} · NEWS2 {d.news2} · "
             f"{p.stage} · {p.doctor}"]
    if d.raised_by():
        parts.append(f"⬆ {d.raised_by()}")
    if S.is_child(p.fields):
        parts.append(f"🟠 {S.CHILD_NOTE}")
    if p.status == "working":
        parts.append(f"⏳ {p.progress}…")
    elif p.status == "error":
        parts.append(f"⚠️ {p.progress} ({p.error})")
    return "\n\n".join(parts)


def _numbers(p):
    if p is None:
        return [], ""
    f, quotes, by = p.fields, p.found.get("found", {}), p.found.get("by", {})
    rows = []
    for key, label in TILES:
        if key == "sbp":
            value = f"{_num(f['sbp'])}/{_num(f.get('dbp', '?'))}" if f.get("sbp") is not None else "not recorded"
        else:
            value = _num(f[key]) if f.get(key) is not None else "not recorded"
        reader = {"both": "rules + AI agree", "check": "⚠ CHECK: rules and AI disagree", "rules": "rules",
                  "AI": "AI only", "nurse": f"fixed by {p.found.get('checked_by', 'a nurse')}"}.get(by.get(key), "")
        rows.append([label, value, quotes.get(key, ""), reader])
    warnings = list(dict.fromkeys(p.found.get("warnings", [])))        # each warning once
    return rows, _warning_box(warnings)


def _warning_box(warnings):
    """A red box above nothing else can be missed in: the warnings used to be a plain bullet under the table, and a
    tester reported "HR 400 warning not shown" (group test 10 Nov, D5)."""
    if not warnings:
        return ""
    items = "".join(f"<li>{html.escape(w)}</li>" for w in warnings)
    return ('<div style="background:#fdecea;color:#b71c1c;border:2px solid #e57373;border-radius:8px;padding:8px 14px;'
            f'font-weight:600">⚠ Check these numbers<ul style="margin:4px 0 0 18px;font-weight:500">{items}</ul></div>')


def _pictures(p):
    items, texts = [], []
    for n in (p.notes if p else []):
        for img in n["images"]:
            items.append((str(store.images / img["file"]), f"{img['name']} · as uploaded"))
            if img.get("overlay"):
                items.append((str(store.images / img["overlay"]), "AI marks"))
            if img.get("outline"):
                items.append((str(store.images / img["outline"]), "AI outline"))
            for x in img.get("extra", []):
                items.append((str(store.images / x["file"]), x["label"]))
            chips = " · ".join(("⚠ " if f.get("flag") else "? " if f.get("possible") else "• " if f.get("also_seen")
                                else "✓ ") + f["name"] for f in img.get("findings", []))
            kind = (img.get("kind") or "not read yet").replace("_", " ")
            clash = f"\n\n⚠ **{img['disagree']}**" if img.get("disagree") else ""
            texts.append(f"**{img['name']}** ({kind}){': ' + chips if chips else ''}{clash}\n\n{img.get('text', '')}")
    return items, "\n\n---\n\n".join(texts) or "No pictures yet."


def _todos(p):
    if p is None:
        return gr.update(choices=[], value=[]), gr.update(choices=[], value=None)
    labels = [(("✦ " if t.get("ai") else "") + t["text"], str(i)) for i, t in enumerate(p.todos)]
    done = [str(i) for i, t in enumerate(p.todos) if t["done"]]
    return gr.update(choices=labels, value=done), gr.update(choices=labels, value=None)


def case_view(pid):
    """Everything on the case-file screen for one patient."""
    p = _patient(pid)
    rows, warnings = _numbers(p)
    pics, pic_text = _pictures(p)
    todos, remove = _todos(p)
    report = (p.report or ("_The case file appears here when the AI has finished._" if p else "")) if p else ""
    by = f"_Last edited by {p.report_by}_" if p and p.report_by and p.report_by != "AI" else (
        "_Written by the AI from the notes. Check it, and edit anything._" if p and p.report else "")
    notes = "\n\n".join(f"**{n['time']} · {n['who']}**  \n{n['text']}" for n in (p.notes if p else []))
    timeline = [[e["time"], e["who"], e["what"], e["text"]] for e in (p.visit.timeline if p and p.visit else [])]
    questions = p.questions if p else []
    chat = [{"role": "user" if m["role"] == "user" else "assistant",
             "content": (f"**{m['who']}:** " if m["role"] == "user" else "") + m["text"]} for m in (p.chat if p else [])]
    return (_header(p), rows, warnings, f"{by}\n\n{report}", notes, pics, pic_text, timeline,
            gr.update(choices=questions, value=None), todos, remove, chat, _case_file_path(p), store.version)


def _case_file_path(p):
    """The download button downloads the file it already holds (a click can't hand it one: the first try saved
    nothing, 9 Nov), so it is given the current case file whenever the screen refreshes."""
    if p is None:
        return None
    path = Path(tempfile.gettempdir()) / f"case_file_{p.bed}.txt"
    path.write_text(S.case_file(p.id), encoding="utf-8")
    return str(path)


def tick(pid, seen, labels):
    """Polling: the case file only when something changed (store.version); the patient list whenever its labels
    changed (a fast AI run finished before the first refresh, so the list kept saying "reading the note…", 9 Nov)."""
    choices = _choices(new=False, everyone=True)
    new_list = (gr.update(choices=choices), choices) if choices != labels else (gr.skip(), gr.skip())
    if store.version == seen:
        return [gr.skip()] * 14 + list(new_list)
    return (*case_view(pid), *new_list)


def _save_pictures(pid, paths):
    out = []
    for path in paths or []:
        path = Path(path)
        if path.suffix.lower() not in S.IMAGE_SUFFIXES:
            raise gr.Error(f"{path.name}: only pictures can be added (JPG, PNG ...), not videos or other files")
        name = f"{pid}_{uuid.uuid4().hex[:8]}{path.suffix.lower()}"
        shutil.copy(path, store.images / name)
        S.save_8bit(store.images / name)                     # 16-bit X-rays turned white for the AI (10 Nov, C1)
        out.append({"file": name, "name": path.name, "kind": None, "text": ""})
    return out


def _add_note(p, name, text, pictures):
    images = _save_pictures(p.id, pictures)
    if not text.strip() and not images:
        raise gr.Error("Type something or add a picture")
    note = {"time": S.now(), "who": name, "text": text.strip() or "(picture added)", "images": images}
    if not ZERO:
        store.add_note(p, note)                              # waits for the running job if there is one
        return
    if p.status == "working":
        raise gr.Error("The AI is still working on this patient; add this in a moment")
    with store.state_lock:
        p.notes.append(note)
        p.pending.append(len(p.notes) - 1)
    start_ai(p)


# ---------------------------------------------------------------- actions
def load_text_file(path):
    """'Open a text file': only plain text (a picture here used to fill the note with junk, REPORT_NOTES 5q #10)."""
    if not path:
        return gr.skip()
    data = Path(path).read_bytes()
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = None
    if text is None or "\x00" in text or Path(path).suffix.lower() not in (".txt", ".text", ".md"):
        raise gr.Error("That isn't a text file: pick a .txt note, or add pictures in the picture box")
    return text


def submit(name, pick, text, pictures):
    name = _need_name(name)
    if pick in (None, NEW):
        if not (text or "").strip():
            raise gr.Error("Type the note first")
        pid = uuid.uuid4().hex[:6]
        p = S.Patient(id=pid, bed=f"P{len(store.patients) + 1}", created=time.time())
        p.notes.append({"time": S.now(), "who": name, "text": text.strip(), "images": _save_pictures(pid, pictures)})
        p.pending = [0]
        store.patients[pid] = p
        store.changed()
        start_ai(p)
        msg = f"Sent: {p.bed} is on the whiteboard. Its case file is open in the Case file tab."
    else:
        p = _call(store.get, pick)
        _add_note(p, name, text or "", pictures)
        pid, msg = p.id, f"Added to {p.bed}'s case file."
    return (gr.update(value=""), None, gr.update(choices=_choices(), value=NEW), msg,
            gr.update(choices=_choices(new=False, everyone=True), value=pid), gr.Tabs(selected="case"))


def add_info(name, pid, text, pictures):
    name = _need_name(name)
    p = _call(store.get, pid) if pid else None
    if p is None:
        raise gr.Error("Pick a patient first")
    _add_note(p, name, text or "", pictures)
    return gr.update(value=""), None, *case_view(pid)


def pick_question(q):
    return f"{q} → " if q else gr.skip()


def save_edit(name, pid, text):
    name = _need_name(name)
    _call(S.edit_report, pid, S.ReportIn(text=text, who=name))
    return gr.update(visible=False), gr.update(visible=False), *case_view(pid)


def start_edit(pid):
    p = _patient(pid)
    if p is None or not p.report:
        raise gr.Error("Nothing to edit yet: the case file appears when the AI has finished")
    return gr.update(value=p.report, visible=True), gr.update(visible=True)


def fix_form(pid):
    p = _patient(pid)
    if p is None:
        raise gr.Error("Pick a patient first")
    return gr.update(value=[[label, _num(p.fields[k]) if p.fields.get(k) is not None else ""] for k, label in FIX_ROWS],
                     visible=True), gr.update(visible=True)


def save_fix(name, pid, table):
    name = _need_name(name)
    rows = table.values.tolist() if hasattr(table, "values") else table
    labels = {label: key for key, label in FIX_ROWS}
    fields = {labels[r[0]]: str(r[1]).strip() for r in rows if r and r[0] in labels and str(r[1]).strip()}
    for k, v in fields.items():
        if k not in ("sex", "mental", "chief_complaint"):
            try:
                float(v)
            except ValueError:
                raise gr.Error(f"{k}: '{v}' is not a number")
    _call(S.fix_numbers, pid, S.FieldsIn(fields=fields, who=name))
    return gr.update(visible=False), gr.update(visible=False), *case_view(pid)


def ask_chat(name, pid, text):
    name = _need_name(name)
    if not (text or "").strip():
        raise gr.Error("Type a question")
    _call(S.chat, pid, S.ChatIn(text=text.strip(), who=name))
    return gr.update(value=""), *case_view(pid)


def set_todos(name, role, pid, done):
    p = _patient(pid)
    if p is None:
        return gr.skip()
    name = _need_name(name)
    done = set(done or [])
    for i, t in enumerate(p.todos):
        want = str(i) in done
        if want != t["done"]:
            _call(S.change_todo, pid, i, S.TodoIn(who=name, role=role, done=want))
    return case_view(pid)


def add_todo(name, role, pid, text):
    name = _need_name(name)
    _call(S.add_todo, pid, S.TodoIn(who=name, role=role, text=text))
    return gr.update(value=""), *case_view(pid)


def remove_todo(name, role, pid, i):
    name = _need_name(name)
    if i is None:
        raise gr.Error("Pick the to-do to remove")
    _call(S.change_todo, pid, int(i), S.TodoIn(who=name, role=role, delete=True))
    return case_view(pid)



# ---------------------------------------------------------------- whiteboard
LEVEL_COLOUR = {1: "#d32f2f", 2: "#f57c00", 3: "#fbc02d", 4: "#388e3c", 5: "#1976d2"}     # same as the React board


def board_view(show_left=False):
    """The whiteboard as a coloured table (group test 10 Nov, D3: in a plain grid the urgent patients didn't stand
    out). Level badge in the level's colour; flagged rows tinted red, "needs vitals" rows amber, with the reason."""
    esc, rows = html.escape, []
    for r in S.board(cleared=show_left)["patients"]:
        todo = "".join(f'<span style="display:inline-block;margin:1px 4px 1px 0;padding:1px 8px;border-radius:10px;'
                       f'background:#e3f0fc;color:#0d3c61;{"text-decoration:line-through;opacity:.55;" if t["done"] else ""}">'
                       f'{"✦ " if t.get("ai") else ""}{esc(t["text"])}</span>' for t in r["todos"])
        ai = "" if r["status"] == "ready" else f"⏳ {esc(r['progress'])}" if r["status"] == "working" else "⚠ AI step failed"
        waited = int((time.time() - r["created"]) // 60)
        tint = "" if not r["flagged"] or r["leaving"] else "#fff3e0" if r["needs_obs"] else "#fdecea"
        why = (f'<div style="font-weight:700;color:{"#a15c00" if r["needs_obs"] else "#b71c1c"}">'
               f'{"🟠" if r["needs_obs"] else "🔴"} {esc(r["reason"])}</div>') if r["reason"] else ""
        badge = (f'<span style="display:inline-block;min-width:38px;padding:6px 0;border-radius:8px;text-align:center;'
                 f'font-weight:800;font-size:20px;color:#111;background:{LEVEL_COLOUR.get(r["level"], "#b0bec5")}">'
                 f'{r["level"] or "…"}</span>')
        who = "reading the note…" if r["who"] == "?, ?" else r["who"]
        cells = [badge, esc(r["bed"]), f"<b>{esc(who)}</b><div>{esc(r['complaint'] or '')}</div>{why}", esc(r["stage"]),
                 esc(r["doctor"]), f"{waited // 60}h {waited % 60:02d}m", todo, ai]
        rows.append(f'<tr style="background:{tint or "#ffffff"};color:#1b2733;{"opacity:.6;" if r["leaving"] else ""}">'
                    + "".join(f'<td style="padding:8px 10px;border-bottom:1px solid #dde3e9;vertical-align:top">{c}</td>'
                              for c in cells) + "</tr>")
    head = "".join(f'<th style="text-align:left;padding:8px 10px;background:#1b2733;color:#ffffff">{h}</th>' for h in BOARD_HEAD)
    # Gradio's ".prose *" gives every element the theme's text colour, so in dark mode the names on the light rows
    # were near-white (checked on screen 11 Nov); setting the variable on the table makes them dark in both themes
    table = (f'<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-size:15px;'
             f'--body-text-color:#1b2733">'
             f'<tr>{head}</tr>{"".join(rows)}</table></div>'
             + ("" if rows else "<p>No patients have left the ED yet.</p>" if show_left else
                "<p>No patients here yet. They appear when a note is submitted on the Charting tab.</p>"))
    n_rows, rows = len(rows), table
    learn = S.learning_status()
    line = (f"{n_rows} patients {'who left the ED' if show_left else 'in the ED'} · most urgent first · "
            f"🔴 flagged urgent · 🟠 urgent only because vital signs are missing · ✦ = suggested by the AI")
    if learn.get("enabled"):
        line += (f"  \nLearning from use: {learn['usable_for_learning']} of {learn['needed_before_update']} patients with "
                 f"an outcome needed for the next update · {learn['number_corrections']} number fixes · "
                 f"AI models {learn['models_version']}")
    return rows, line


def board_tick(seen, show_left):
    if store.version == seen:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()
    rows, line = board_view(show_left)
    return rows, line, gr.update(choices=_choices(new=False, cleared=show_left)), store.version


def board_pick(pid):
    p = _patient(pid)
    if p is None:
        return gr.skip(), gr.skip(), gr.update(visible=False)
    return p.stage, p.doctor, gr.update(visible=p.stage in S.LEAVING and not p.cleared)


def board_update(name, role, pid, stage, doctor, show_left):
    name = _need_name(name)                     # unlike the React board, no edits without a name (5q #9)
    if not pid:
        raise gr.Error("Pick a patient")
    _call(S.set_board, pid, S.BoardIn(who=name, role=role, stage=stage, doctor=doctor))
    rows, line = board_view(show_left)
    return rows, line, gr.update(visible=stage in S.LEAVING)


def board_clear(name, role, pid, diagnosis, right_level, show_left):
    name = _need_name(name)
    level = int(right_level[0]) if right_level and right_level[0].isdigit() else None
    _call(S.set_board, pid, S.BoardIn(who=name, role=role, cleared=True, diagnosis=diagnosis, final_level=level))
    rows, line = board_view(show_left)
    return rows, line, gr.update(choices=_choices(new=False, cleared=show_left), value=None), gr.update(visible=False)


def board_put_back(name, role, pid):
    name = _need_name(name)
    _call(S.set_board, pid, S.BoardIn(who=name, role=role, cleared=False))
    rows, line = board_view(True)
    return rows, line, gr.update(choices=_choices(new=False, cleared=True), value=None)


def board_switch(show_left):
    rows, line = board_view(show_left)
    return (rows, line, gr.update(choices=_choices(new=False, cleared=show_left), value=None),
            gr.update(visible=not show_left), gr.update(visible=show_left),
            gr.update(visible=not show_left), gr.update(visible=show_left))


# ---------------------------------------------------------------- screens
MODE = ("**Test mode:** MedGemma's writing is made up; numbers, urgency, rules, image models and whiteboard are real."
        if TEST else "**Full AI** (MedGemma + the trained models)" + (
            " on a Hugging Face GPU: each patient takes about a minute, and each visitor gets a few minutes of GPU a "
            "day (log in to Hugging Face for more)" if ZERO and not FAKE_ZERO else ""))
if NO_FILES:
    MODE += f"\n\n⚠ The full AI couldn't start: {NO_FILES} (the Space owner needs to set its HF_TOKEN secret)."
BOARD_HEAD = ["Level", "Bed", "Patient", "Stage", "Doctor", "Waiting", "To do", "AI"]

with gr.Blocks(title="ED triage app") as demo:
    with gr.Row():
        gr.Markdown(f"# ED triage app\n{MODE} · student prototype, not a medical device: never use real patients' details")
        name = gr.Textbox(label="Your name", placeholder="type your name", scale=0, min_width=180)
        role = gr.Dropdown(S.ROLES, value=S.ROLES[0], label="Role", scale=0, min_width=180)
    with gr.Tabs() as tabs:
        with gr.Tab("Charting", id="chart"):
            with gr.Row():
                with gr.Column(scale=3):
                    pick = gr.Dropdown(_choices(), value=NEW, label="Patient (type to search: bed, age/sex, complaint)",
                                       filterable=True)
                    note = gr.Textbox(label="Note: write it however you like, the AI tidies it up", lines=8,
                                      placeholder="Example: 67M CP since 0600, sweaty, radiating L arm. HR 112 BP 90/60 "
                                                  "sats 94 RA RR 22. hx DM HTN on metformin. came by amb. pain 8/10")
                    with gr.Row():
                        txt = gr.File(label="📄 Open a text file", file_types=[".txt"], type="filepath", height=90)
                        pics = gr.File(label="📎 X-rays or photos", file_count="multiple", file_types=["image"],
                                       type="filepath", height=90)
                    send = gr.Button("Submit", variant="primary")
                    sent = gr.Markdown()
                with gr.Column(scale=2):
                    gr.Markdown("### What the AI does\n1. pulls the numbers out of the note and shows where each came "
                                "from;\n2. scores urgency and a triage level (1 = most urgent);\n3. writes a tidy case "
                                "file, asks about what's missing and suggests to-dos;\n4. sorts each picture and runs the "
                                "trained model for that kind;\n5. puts the patient on the whiteboard.\n\nEverything the "
                                "AI adds is marked; a person always decides.")
        with gr.Tab("Case file", id="case"):
            seen = gr.State(-1)
            labels = gr.State([])                           # the patient list last sent to this screen
            case_pick = gr.Dropdown(_choices(new=False, everyone=True), label="Patient", filterable=True)
            header = gr.Markdown("Pick a patient.")
            numbers = gr.Dataframe(headers=["Number", "Value", "Read from the note's words", "Read by"],
                                   interactive=False, wrap=True)
            warn = gr.HTML()
            with gr.Row():
                fix_btn = gr.Button("Fix numbers", size="sm")
                dl = gr.DownloadButton("⬇ Download the whole case file", size="sm")
            fix_table = gr.Dataframe(headers=["Number", "Value"], interactive=True, visible=False, static_columns=[0],
                                     label="Fix the numbers (the urgency and level are scored again)")
            fix_save = gr.Button("Save and re-score", variant="primary", visible=False)
            with gr.Row():
                more = gr.Textbox(label="Add information", placeholder="trop 2.1, ECG ST depression V4-6, aspirin 300 "
                                                                       "given", scale=4)
                more_pics = gr.File(label="📎 Image", file_count="multiple", file_types=["image"], type="filepath",
                                    height=80, scale=1)
                more_btn = gr.Button("Add", variant="primary", scale=0)
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Tabs():
                        with gr.Tab("Case file"):
                            edit_btn = gr.Button("✎ Edit the case file (fix or add anything, then Save)", size="sm",
                                                 variant="secondary")
                            report = gr.Markdown()
                            edit_box = gr.Textbox(lines=18, visible=False, label="Edit the case file")
                            edit_save = gr.Button("Save", variant="primary", visible=False)
                        with gr.Tab("Notes as typed"):
                            notes_md = gr.Markdown()
                        with gr.Tab("Files"):
                            gallery = gr.Gallery(columns=3, height=360, label="As uploaded · AI marks · AI outline")
                            pics_md = gr.Markdown()
                        with gr.Tab("Timeline"):
                            timeline = gr.Dataframe(headers=["Time", "Who", "What", "Details"], interactive=False,
                                                    wrap=True)
                with gr.Column(scale=2):
                    with gr.Tabs():
                        with gr.Tab("Questions and to-dos"):
                            gr.Markdown("**Questions for you to answer.** The AI is asking *you* about what is "
                                        "missing: 1. click a question, 2. type the answer in the box, 3. press "
                                        "**Add my answer**. (To ask the AI something, use the **Chat** tab.)")
                            questions = gr.Radio([], label="Questions the AI wants answered")
                            answer = gr.Textbox(label="Your answer (it is added to the patient's notes)")
                            answer_btn = gr.Button("Add my answer", size="sm", variant="primary")
                            todos = gr.CheckboxGroup([], label="Suggested to-dos (also on the whiteboard): tick one when "
                                                               "it has been done; ✦ = suggested by the AI")
                            with gr.Row():
                                todo_new = gr.Textbox(placeholder="Add a to-do", show_label=False, scale=3)
                                todo_add = gr.Button("Add", size="sm", scale=1)
                            with gr.Row():
                                todo_pick = gr.Dropdown([], label="Remove a to-do", scale=3)
                                todo_del = gr.Button("Remove", size="sm", scale=1)
                        with gr.Tab("Chat"):
                            chat = gr.Chatbot(height=360, label="Ask about this patient (the AI reads the whole case "
                                                                "file, including your edits)")
                            chat_q = gr.Textbox(placeholder="What else should we rule out?", show_label=False)
                            chat_btn = gr.Button("Ask", size="sm")
        with gr.Tab("Whiteboard", id="board"):
            b_seen = gr.State(-1)
            show_left = gr.State(False)
            b_line = gr.Markdown()
            b_table = gr.HTML()
            with gr.Group():
                gr.Markdown("### Change this patient\nThese boxes **change** a patient, they don't filter the list: "
                            "1. pick the patient, 2. set the stage and the doctor, 3. press **Save the change**. Only "
                            "the **Triage nurse** and **Organising doctor** can change the board (set your name and "
                            "role at the top right); everyone else can look.")
                with gr.Row():
                    b_pick = gr.Dropdown(_choices(new=False), label="1. Which patient", scale=2)
                    with gr.Column(scale=3, visible=True) as b_edit:
                        with gr.Row():
                            b_stage = gr.Dropdown(S.STAGES, label="2. Stage (where the patient is now)")
                            b_doctor = gr.Textbox(label="Doctor looking after them")
                            b_update = gr.Button("3. Save the change", variant="primary")
                        with gr.Row(visible=False) as b_leave:
                            b_dx = gr.Textbox(label="Final diagnosis (optional)")
                            b_level = gr.Dropdown(["Right level?", "1", "2", "3", "4", "5"], value="Right level?",
                                                  label="Right level? (optional)")
                            b_clear = gr.Button("✓ Clear from board", variant="primary")
                    with gr.Column(scale=3, visible=False) as b_left_box:
                        b_back = gr.Button("Put back on the board")
            with gr.Row():
                b_show_left = gr.Button("Show patients who left the ED")
                b_show_board = gr.Button("← Back to the board", visible=False)

    case_out = [header, numbers, warn, report, notes_md, gallery, pics_md, timeline, questions, todos, todo_pick, chat,
                dl, seen]
    timer = gr.Timer(2)
    timer.tick(tick, [case_pick, seen, labels], [*case_out, case_pick, labels])
    case_pick.change(case_view, case_pick, case_out)
    on(txt.upload, load_text_file, txt, note)
    on(send.click, submit, [name, pick, note, pics], [note, pics, pick, sent, case_pick, tabs])
    on(more_btn.click, add_info, [name, case_pick, more, more_pics], [more, more_pics, *case_out])
    on(answer_btn.click, add_info, [name, case_pick, answer, more_pics], [answer, more_pics, *case_out])
    questions.change(pick_question, questions, answer)
    on(edit_btn.click, start_edit, case_pick, [edit_box, edit_save])
    on(edit_save.click, save_edit, [name, case_pick, edit_box], [edit_box, edit_save, *case_out])
    on(fix_btn.click, fix_form, case_pick, [fix_table, fix_save])
    on(fix_save.click, save_fix, [name, case_pick, fix_table], [fix_table, fix_save, *case_out])
    on(chat_btn.click, ask_chat, [name, case_pick, chat_q], [chat_q, *case_out])
    on(chat_q.submit, ask_chat, [name, case_pick, chat_q], [chat_q, *case_out])
    on(todos.input, set_todos, [name, role, case_pick, todos], case_out)
    on(todo_add.click, add_todo, [name, role, case_pick, todo_new], [todo_new, *case_out])
    on(todo_del.click, remove_todo, [name, role, case_pick, todo_pick], case_out)

    b_timer = gr.Timer(3)
    b_timer.tick(board_tick, [b_seen, show_left], [b_table, b_line, b_pick, b_seen])
    b_pick.change(board_pick, b_pick, [b_stage, b_doctor, b_leave])
    on(b_update.click, board_update, [name, role, b_pick, b_stage, b_doctor, show_left], [b_table, b_line, b_leave])
    on(b_clear.click, board_clear, [name, role, b_pick, b_dx, b_level, show_left], [b_table, b_line, b_pick, b_leave])
    on(b_back.click, board_put_back, [name, role, b_pick], [b_table, b_line, b_pick])
    b_show_left.click(lambda: True, None, show_left).then(board_switch, show_left,
                                                          [b_table, b_line, b_pick, b_edit, b_left_box, b_show_left, b_show_board])
    b_show_board.click(lambda: False, None, show_left).then(board_switch, show_left,
                                                            [b_table, b_line, b_pick, b_edit, b_left_box, b_show_left, b_show_board])
    demo.load(lambda: board_view(False), None, [b_table, b_line])

if __name__ == "__main__":
    on_space = bool(os.environ.get("SPACE_ID"))             # Hugging Face sets this; the Space picks host and port
    demo.queue(default_concurrency_limit=8).launch(
        server_name="0.0.0.0" if ARGS.lan or on_space else "127.0.0.1",
        server_port=None if on_space else ARGS.port, allowed_paths=[str(store.images)])
