"""ED app back end (FastAPI): chart first, AI second. APP_DESIGN.md has the design; the React screens are in frontend/.

    python server.py --test          # fake AI (seconds, no GPU): click the whole flow
    python server.py                 # real MedGemma (needs the GPU)
    python server.py --test --lan    # a second laptop on the same Wi-Fi can open the whiteboard
then open http://127.0.0.1:8000 (charting), /#/board (whiteboard), /#/case/<id> (case file).

One patient = notes typed messy + images -> numbers pulled out -> urgency/level (trained models) -> case file and
questions (MedGemma). The AI runs in a background thread, one job at a time (one model in memory); screens poll.
Everything is saved to case_files/board.pkl after each change, so a restart keeps the patients.
"""
import argparse
import pickle
import re
import shutil
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "src"), str(HERE.parent / "app")]

from fastapi import FastAPI, File, Form, HTTPException, UploadFile          # noqa: E402
from fastapi.responses import FileResponse, PlainTextResponse               # noqa: E402
from fastapi.staticfiles import StaticFiles                                 # noqa: E402
from pydantic import BaseModel                                              # noqa: E402

import charting                                                             # noqa: E402

# The client's actions (6 Nov 2026): where the patient is going. The last four mean the patient leaves the ED; the
# organising doctor or triage nurse then clears them off the board (the case file is kept, under "Left the ED").
STAGES = ["Waiting for doctor", "Being seen", "Results pending", "Going to surgery", "Transfer to another department",
          "Admitted to ward", "Discharged home"]
LEAVING = set(STAGES[3:])
OLD_STAGES = {"Waiting": "Waiting for doctor", "Admitted / ward": "Admitted to ward", "Discharged": "Discharged home"}
ROLES = ["Triage nurse", "Organising doctor", "Doctor", "Nurse", "Ward / other"]
BOARD_EDITORS = {"Triage nurse", "Organising doctor"}            # the client's choice: only these change the board
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".dcm"}
FIELD_NAMES = {"age": "Age", "sex": "Sex", "hr": "HR", "sbp": "BP sys", "dbp": "BP dia", "rr": "RR", "spo2": "SpO2 %",
               "temp_c": "Temp C", "nrs_pain": "Pain /10", "mental": "Consciousness", "ambulance": "Ambulance (1/0)",
               "injured": "Injury (1/0)"}


def now():
    return time.strftime("%d %b %H:%M")


@dataclass
class Patient:
    id: str
    bed: str
    created: float
    notes: list = field(default_factory=list)       # {time, who, text, images: [{file, kind, text}]} as typed
    fields: dict = field(default_factory=dict)      # numbers pulled from the notes (Pool 1 names)
    found: dict = field(default_factory=dict)       # where each number came from, + warnings
    visit: object = None                            # visit.Visit: arrival card + decision + record (for the chat)
    report: str = ""
    report_by: str = ""                             # "AI" or the person who last edited it
    questions: list = field(default_factory=list)
    todos: list = field(default_factory=list)       # {text, done, ai, by}
    chat: list = field(default_factory=list)        # {role, text, who, time}
    stage: str = "Waiting for doctor"
    cleared: bool = False                           # off the board (left the ED); case file kept
    doctor: str = "unassigned"
    status: str = "working"                         # working / ready / error
    progress: str = "Queued"
    error: str = ""
    pending: list = field(default_factory=list)     # note indexes the AI hasn't put in the case file yet
    report_seen_by_chat: str = ""

    def complaint(self):
        """The case file's 'Presenting complaint', so a person's correction shows on the header and the whiteboard.
        (The urgency score keeps the complaint it was scored with; 'Fix numbers' re-scores.)"""
        lines = self.report.splitlines()
        for i, line in enumerate(lines):
            if line.strip().lower() == "## presenting complaint":
                for nxt in lines[i + 1:]:
                    if nxt.startswith("## "):
                        break
                    text = nxt.strip().lstrip("-* ").replace("(injury)", "").strip(" .")
                    if text and text.lower() != "not recorded":
                        return text[:80]
        return self.fields.get("chief_complaint", "")

    def summary(self):
        d = self.visit.decision if self.visit else None
        return {"id": self.id, "bed": self.bed, "created": self.created, "stage": self.stage, "doctor": self.doctor, "cleared": self.cleared,
                "leaving": self.stage in LEAVING,
                "status": self.status, "progress": self.progress,
                "level": d.level if d else None, "flagged": d.flagged if d else None,
                "raised_by": d.raised_by() if d else "",
                "who": _who_line(self.fields), "complaint": self.complaint(),
                "todos": self.todos, "open_questions": len(self.questions)}

    def full(self):
        d = self.visit.decision if self.visit else None
        return {**self.summary(), "notes": self.notes, "fields": self.fields, "found": self.found,
                "field_names": FIELD_NAMES, "report": self.report, "report_by": self.report_by,
                "questions": self.questions, "chat": self.chat, "error": self.error,
                "timeline": self.visit.timeline if self.visit else [],
                "decision": None if d is None else {
                    "p_urgent": d.p_urgent, "flagged": d.flagged, "head_flag": d.head_flag, "floor_flag": d.floor_flag,
                    "level": d.level, "level_set": list(d.level_set), "news2": d.news2,
                    "floor_priority": d.floor_priority}}


def _who_line(f):
    age = f"{f['age']:.0f}" if f.get("age") is not None else "?"
    sex = {"male": "M", "female": "F"}.get(f.get("sex"), "?")
    return f"{sex}, {age}"


class Store:
    """All patients, the AI, and one lock so only one AI job runs at a time."""

    def __init__(self, ai, path):
        self.ai, self.path = ai, path               # ai: EDPipeline or TestPipeline
        self.test = not hasattr(ai, "_ask")
        self.lock, self.save_lock, self.state_lock = threading.Lock(), threading.Lock(), threading.Lock()
        self.version = 0
        self.patients = {}
        if path and path.exists():
            self.patients = pickle.loads(path.read_bytes())
            for p in self.patients.values():        # boards saved by an older version
                p.stage = OLD_STAGES.get(p.stage, p.stage)
                if not hasattr(p, "cleared"):
                    p.cleared = False
            for p in self.patients.values():        # a job cut off by a restart
                if p.status == "working":
                    p.status, p.progress = "error", "Server restarted during the AI step; press Re-run AI"
        self.images = (path.parent if path else HERE / "case_files") / "images"
        self.images.mkdir(parents=True, exist_ok=True)

    def changed(self):
        self.version += 1
        if self.path:
            with self.save_lock:
                tmp = self.path.with_suffix(".tmp")
                tmp.write_bytes(pickle.dumps(self.patients))
                tmp.replace(self.path)

    def get(self, pid):
        if pid not in self.patients:
            raise HTTPException(404, "no such patient")
        return self.patients[pid]

    # ------------------------------------------------------------ the AI job, in a background thread
    def run(self, p):
        threading.Thread(target=self._job, args=(p,), daemon=True).start()

    def _step(self, p, text):
        p.progress = text
        self.version += 1

    def add_note(self, p, note):
        """A note is always accepted, even while the AI is busy with this patient: it waits in p.pending and the
        running job does it next (an ED doesn't wait for the AI)."""
        with self.state_lock:
            p.notes.append(note)
            p.pending.append(len(p.notes) - 1)
            start = p.status != "working"
            if start:
                p.status, p.progress = "working", "Queued"
        self.changed()
        if start:
            self.run(p)

    def _job(self, p):
        with self.lock:
            while True:
                try:
                    self._work(p)
                except Exception as e:              # the nurse's notes are already saved; the AI can be re-run
                    traceback.print_exc()
                    with self.state_lock:
                        p.status, p.progress, p.error = "error", "The AI step failed; the notes are saved", repr(e)
                    break
                with self.state_lock:               # notes added while the AI worked: go round again
                    if not p.pending:
                        p.status, p.progress, p.error = "ready", f"Case file updated {now()}", ""
                        break
            self.changed()

    def _save_picture(self, img, picture, tag):
        import numpy as np
        from PIL import Image
        name = img["file"].rsplit(".", 1)[0] + f"_{tag}.png"
        Image.fromarray(np.asarray(picture).astype("uint8")).save(self.images / name)
        return name

    def _work(self, p):
        ai = None if self.test else self.ai
        doing = list(p.pending)                     # notes added from now on wait for the next round
        new = [p.notes[i] for i in doing]
        if p.visit is None:                         # first note: numbers -> urgency + level
            self._step(p, "Pulling out the numbers")
            text = "\n".join(n["text"] for n in p.notes)
            if not p.fields:
                p.fields, p.found = charting.extract_numbers(text)
            if not p.fields.get("chief_complaint"):
                self._step(p, "Writing the complaint line")
                p.fields["chief_complaint"] = charting.complaint(ai, text)
            self._step(p, "Scoring urgency (trained models)")
            p.visit = self.ai.triage(p.fields, explain=False)
            p.visit.log("Triage", f"Arrival card made from the notes of {p.notes[0]['who']}; level {p.visit.decision.level}"
                        f", urgency {p.visit.decision.p_urgent:.0%}", "AI (decision support)")
        image_lines = []
        for n in new:
            p.visit.add("notes", n["text"], n["who"])
            for img in n["images"]:
                if img.get("text"):
                    continue
                self._step(p, f"Reading image {img['name']}")
                out = self.ai.read_image(p.visit, str(self.images / img["file"]), None, n["who"])
                img["kind"], img["text"] = out["kind"], out["text"]
                found, chest = out.get("detector"), out.get("chest")
                if found is not None:
                    img["text"] += (f" Fracture detector: {found['n']} possible fracture(s) marked, check them."
                                    if found["n"] else " Fracture detector: nothing marked (it misses about 1 in 5).")
                if chest:
                    img["text"] += " Chest X-ray check: " + ", ".join(f"{k} {v:.0%}" for k, v in chest["top"]) + "."
                picture = (found or {}).get("overlay") if found and found["n"] else (chest or {}).get("overlay")
                if picture is not None:
                    img["overlay"] = self._save_picture(img, picture, "marked")
                if found and found["n"]:                # detect -> segment: the strongest box goes to MedSAM, no clicking
                    self._step(p, f"Outlining the marked area on {img['name']}")
                    box = found["boxes"][max(range(found["n"]), key=lambda i: found["confs"][i])]
                    try:
                        seg = self.ai.outline(p.visit, str(self.images / img["file"]), box, "AI (detector box)")
                        img["outline"] = self._save_picture(img, seg["overlay"], "outline")
                        img["text"] += f" Outline of the strongest mark: {seg['area_pct']:.1f}% of the picture."
                    except Exception as e:               # e.g. MedSAM not downloaded on this computer
                        img["text"] += f" (Outline not available here: {type(e).__name__}.)"
                # type-aware check (image_checklist.py): the ED checklist for this kind, outlines of positive
                # findings, and heart/lung outlines on chest X-rays
                if out.get("checklist"):
                    from image_checklist import checklist_line
                    img["checklist"] = out["checklist"]
                    anatomy = out.get("anatomy") or {}
                    img["text"] += checklist_line(out["kind"], out["checklist"], anatomy.get("ctr"))
                    img["extra"] = []
                    for i, loc in enumerate(out.get("located") or []):
                        img["extra"].append({"label": f"AI outline: {loc['finding']}"
                                             + (" (experimental)" if loc.get("experimental") else ""),
                                             "file": self._save_picture(img, loc["overlay"], f"find{i}")})
                    if anatomy.get("overlay") is not None:
                        img["extra"].append({"label": "Heart (red) and lungs (blue)"
                                             + (f", cardiothoracic ratio {anatomy['ctr']:.2f}" if anatomy.get("ctr") else ""),
                                             "file": self._save_picture(img, anatomy["overlay"], "anatomy")})
                hint = charting.body_parts(n["text"])
                if hint:
                    img["text"] += f" The note mentions: {', '.join(hint)}."
                image_lines.append(f"{img['name']} ({out['kind']}): {img['text']}")
        self._step(p, "Writing the case file")
        first = not p.report
        if first or new:                            # later notes are only ADDED into the case file (edits kept)
            p.report = charting.build_report(ai, p.visit.record(), p.report, [n["text"] for n in new],
                                             notes=[n["text"] for n in p.notes],
                                             images=[f"{i['name']} ({i.get('kind')}): {i.get('text')}"
                                                     for n in p.notes for i in n["images"]],
                                             new_images=image_lines)
            imaging = charting._split(p.report)[1].get("Imaging", [])
            if first and image_lines and all(charting._empty(l) for l in imaging):   # the AI left the readings out
                p.report = charting.merge_into(p.report, charting.file_notes([], image_lines))
            # the triage section always shows the trained models' decision, word for word
            p.report = charting.set_section(p.report, charting.TRIAGE, p.visit.decision.lines())
            if first or not p.report_by:
                p.report_by = "AI"                  # a person's name stays once they've edited it
            p.visit.log("Case file", "Written by the AI from the notes" if first else
                        f"AI added {len(new)} new note(s); nothing already there was changed")
        self._step(p, "Thinking of questions and to-dos")
        f = p.fields
        arrival = {1.0: "ambulance", 0.0: "walk-in"}.get(f.get("ambulance"), "not recorded")
        context = (f"PATIENT: age {f.get('age', 'not recorded')}, {f.get('sex', 'sex not recorded')}, arrival {arrival}.\n"
                   + "\n".join(l.strip() for l in (p.visit.ed or "").splitlines() if "Pathway flags" in l))
        questions, todo = charting.ask_questions(ai, p.report, context)
        typed = "\n".join(n["text"] for n in p.notes)
        if not self.test:                           # the small model sometimes asks for what's already recorded
            questions, todo = charting.drop_known(questions, f, typed), charting.drop_known(todo, f, typed)
        # the pathway's own questions first (fixed, specific, minus any the notes answer), then MedGemma's; at most 5
        seen, merged = set(), []
        for q in charting.drop_known(charting.pathway_questions(p.visit.ed), f, typed) + questions:
            key = re.sub(r"\W+", " ", q.lower()).strip()
            if key not in seen:
                seen.add(key)
                merged.append(q)
        p.questions = merged[:5]
        have = {t["text"].lower() for t in p.todos}
        for text, by in [(t, "ED rules") for t in charting.pathway_todos(p.visit.ed)] + [(t, "AI") for t in todo]:
            if text.lower() not in have:
                have.add(text.lower())
                p.todos.append({"text": text, "done": False, "ai": True, "by": by})
        with self.state_lock:
            p.pending = [i for i in p.pending if i not in doing]


app = FastAPI(title="ED charting")
store: Store = None


@app.get("/api/meta")
def meta():
    return {"stages": STAGES, "leaving": sorted(LEAVING), "roles": ROLES, "board_editors": sorted(BOARD_EDITORS),
            "test": store.test}


@app.get("/api/board")
def board(cleared: bool = False):
    """On the board: most urgent first, then longest waiting; patients on their way out at the bottom.
    cleared=true lists the patients who left the ED instead."""
    rows = sorted((p for p in store.patients.values() if p.cleared == cleared),
                  key=lambda p: (p.stage in LEAVING, p.visit is None, p.visit.decision.level if p.visit else 9, p.created))
    n_cleared = sum(p.cleared for p in store.patients.values())
    return {"version": store.version, "patients": [p.summary() for p in rows], "now": time.time(), "n_cleared": n_cleared}


@app.get("/api/patients/{pid}")
def patient(pid: str):
    return {"version": store.version, **store.get(pid).full()}


def _save_images(pid, files):
    out = []
    for f in files or []:
        if not f.filename:
            continue
        suffix = Path(f.filename).suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            raise HTTPException(400, f"{f.filename}: only pictures can be added (videos and other files are not read)")
        name = f"{pid}_{uuid.uuid4().hex[:8]}{suffix}"
        with open(store.images / name, "wb") as fh:
            shutil.copyfileobj(f.file, fh)
        out.append({"file": name, "name": f.filename, "kind": None, "text": ""})
    return out


@app.post("/api/patients")
def new_patient(text: str = Form(...), who: str = Form(...), files: list[UploadFile] = File(default=[])):
    if not text.strip():
        raise HTTPException(400, "Type the note first")
    pid = uuid.uuid4().hex[:6]
    p = Patient(id=pid, bed=f"P{len(store.patients) + 1}", created=time.time())
    p.notes.append({"time": now(), "who": who, "text": text.strip(), "images": _save_images(pid, files)})
    p.pending = [0]
    store.patients[pid] = p
    store.changed()
    store.run(p)
    return {"id": pid}


@app.post("/api/patients/{pid}/notes")
def add_note(pid: str, text: str = Form(""), who: str = Form(...), files: list[UploadFile] = File(default=[])):
    p = store.get(pid)
    images = _save_images(pid, files)
    if not text.strip() and not images:
        raise HTTPException(400, "Type something or add a picture")
    store.add_note(p, {"time": now(), "who": who, "text": text.strip() or "(picture added)", "images": images})
    return {"ok": True}


@app.post("/api/patients/{pid}/rerun")
def rerun(pid: str):
    p = store.get(pid)
    if p.status == "working":
        raise HTTPException(409, "already working")
    if not p.pending:
        p.pending = list(range(len(p.notes))) if p.visit is None else []
    p.status, p.progress = "working", "Queued"
    store.changed()
    store.run(p)
    return {"ok": True}


class ReportIn(BaseModel):
    text: str
    who: str


@app.put("/api/patients/{pid}/report")
def edit_report(pid: str, body: ReportIn):
    p = store.get(pid)
    if p.status == "working" and not p.report:      # the AI is writing the first version: an edit would be lost
        raise HTTPException(409, "The AI is still writing the case file; edit it when it's ready")
    p.report, p.report_by = body.text, body.who
    if p.visit:
        p.visit.log("Case file", "Edited by hand", body.who)
    store.changed()
    return {"ok": True}


class FieldsIn(BaseModel):
    fields: dict
    who: str


@app.put("/api/patients/{pid}/fields")
def fix_numbers(pid: str, body: FieldsIn):
    """The nurse corrects a number the AI pulled out -> urgency and level are scored again."""
    p = store.get(pid)
    if p.status == "working":
        raise HTTPException(409, "The AI is still working on this patient")
    clean = {}
    for k, v in body.fields.items():
        if k not in FIELD_NAMES and k != "chief_complaint":
            continue
        if v in ("", None):
            continue
        clean[k] = v if k in ("sex", "mental", "chief_complaint") else float(v)
    old = p.visit.decision.level if p.visit else None
    with store.lock:
        before = p.visit
        p.fields = clean
        p.found["checked_by"] = body.who
        p.visit = store.ai.triage(clean, explain=False)
        if before:                                  # keep everything learned since arrival; only the card + decision change
            p.visit.timeline, p.visit.results, p.visit.messages = before.timeline, before.results, before.messages
        p.visit.log("Triage", f"Numbers corrected; level {old} -> {p.visit.decision.level}", body.who)
        if p.report:
            p.report = charting.set_section(p.report, charting.TRIAGE, p.visit.decision.lines())
    store.changed()
    return {"ok": True}


class ChatIn(BaseModel):
    text: str
    who: str


@app.post("/api/patients/{pid}/chat")
def chat(pid: str, body: ChatIn):
    p = store.get(pid)
    if p.visit is None:
        raise HTTPException(409, "The case file isn't ready yet")
    with store.lock:
        if p.report and p.report != p.report_seen_by_chat:     # the chat always sees the latest case file
            p.visit.unseen.append("CURRENT CASE FILE:\n" + p.report)
            p.report_seen_by_chat = p.report
        answer = store.ai.chat(p.visit, body.text)
    p.chat += [{"role": "user", "text": body.text, "who": body.who, "time": now()},
               {"role": "ai", "text": answer, "who": "AI", "time": now()}]
    p.visit.log("Chat", f"Q ({body.who}): {body.text}")
    store.changed()
    return {"answer": answer}


class BoardIn(BaseModel):
    who: str
    role: str
    stage: str | None = None
    doctor: str | None = None
    cleared: bool | None = None


def _editor(role):
    if role not in BOARD_EDITORS:
        raise HTTPException(403, "Only the triage nurse or the organising doctor can change the board")


@app.patch("/api/patients/{pid}/board")
def set_board(pid: str, body: BoardIn):
    _editor(body.role)
    p = store.get(pid)
    change = []
    if body.stage and body.stage != p.stage:
        if body.stage not in STAGES:
            raise HTTPException(400, "unknown stage")
        change.append(f"stage {p.stage} -> {body.stage}")
        p.stage = body.stage
    if body.doctor is not None and body.doctor.strip() and body.doctor.strip() != p.doctor:
        change.append(f"doctor {p.doctor} -> {body.doctor.strip()}")
        p.doctor = body.doctor.strip()
    if body.cleared is not None and body.cleared != p.cleared:
        if body.cleared and p.stage not in LEAVING:
            raise HTTPException(400, "Set where the patient went (surgery, transfer, ward or home) before clearing")
        change.append("cleared from the board (left the ED)" if body.cleared else "put back on the board")
        p.cleared = body.cleared
    if change and p.visit:
        p.visit.log("Board", "; ".join(change), body.who)
    store.changed()
    return {"ok": True}


class TodoIn(BaseModel):
    who: str
    role: str
    text: str | None = None
    done: bool | None = None
    delete: bool = False


@app.post("/api/patients/{pid}/todos")
def add_todo(pid: str, body: TodoIn):
    _editor(body.role)
    p = store.get(pid)
    if not (body.text or "").strip():
        raise HTTPException(400, "empty to-do")
    p.todos.append({"text": body.text.strip(), "done": False, "ai": False, "by": body.who})
    store.changed()
    return {"ok": True}


@app.patch("/api/patients/{pid}/todos/{i}")
def change_todo(pid: str, i: int, body: TodoIn):
    _editor(body.role)
    p = store.get(pid)
    if not 0 <= i < len(p.todos):
        raise HTTPException(404, "no such to-do")
    t = p.todos[i]
    if body.delete:
        p.todos.pop(i)
        what = f"removed '{t['text']}'"
    else:
        t["done"] = bool(body.done)
        what = f"{'done' if t['done'] else 'not done'}: {t['text']}"
    if p.visit:
        p.visit.log("To do", what, body.who)
    store.changed()
    return {"ok": True}


@app.get("/api/patients/{pid}/case_file", response_class=PlainTextResponse)
def case_file(pid: str):
    p = store.get(pid)
    notes = "\n".join(f"{n['time']} | {n['who']}: {n['text']}" for n in p.notes)
    head = f"{p.bed} ({_who_line(p.fields)}) — stage {p.stage}, doctor {p.doctor}"
    body = p.visit.case_file(head) if p.visit else head
    return f"{body}\n\nCASE FILE (tidy, last by {p.report_by or 'nobody'})\n{p.report}\n\nNOTES AS TYPED\n{notes}\n"


@app.get("/api/images/{name}")
def image(name: str):
    path = (store.images / name).resolve()
    if path.parent != store.images.resolve() or not path.exists():
        raise HTTPException(404)
    return FileResponse(path)


@app.delete("/api/patients/{pid}")
def remove(pid: str, role: str):
    _editor(role)
    store.get(pid)
    del store.patients[pid]
    store.changed()
    return {"ok": True}


def main():
    global store
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="fake AI answers, no GPU")
    ap.add_argument("--lan", action="store_true", help="reachable from other devices on the same Wi-Fi")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--case-files", default=str(HERE / "case_files" / "board.pkl"), help="'' = don't save")
    a = ap.parse_args()
    if a.test:
        from test_backend import TestPipeline
        ai = TestPipeline()
    else:
        from pipeline import EDPipeline
        ai = EDPipeline()
    path = Path(a.case_files) if a.case_files else None
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
    store = Store(ai, path)
    dist = HERE / "frontend" / "dist"
    if dist.exists():
        @app.middleware("http")
        async def fresh_page(request, call_next):
            # the page must be re-checked on every load, or browsers keep running old screens after an update
            # (7 Nov stress test); /assets files have the version in their name, so they can be cached
            response = await call_next(request)
            if not request.url.path.startswith(("/api", "/assets")):
                response.headers["Cache-Control"] = "no-cache"
            return response
        app.mount("/", StaticFiles(directory=dist, html=True), name="screens")
    uvicorn.run(app, host="0.0.0.0" if a.lan else "127.0.0.1", port=a.port)


if __name__ == "__main__":
    main()
