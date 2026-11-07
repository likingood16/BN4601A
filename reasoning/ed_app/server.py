"""ED app v2 back end (FastAPI). PRODUCT.md has the spec; ui/ has the screens.

    python server.py --test               # fake writing, real rules + Caleb's vitals model (no GPU), port 8000
    python server.py                      # real MedGemma + trained heads (GPU)
    python server.py --lan --port 8010    # reachable from another machine (vanda job, second laptop)

One AI worker thread takes jobs from a queue (one model in memory). Every change bumps `version`; the screens listen on
/api/events (server-sent events) and reload only when something changed. Patients are saved one row each in SQLite,
so a crash or restart loses nothing.
"""
import argparse
import asyncio
import pickle
import queue
import shutil
import sqlite3
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "src"), str(HERE.parent / "app")]

from fastapi import FastAPI, File, Form, HTTPException, UploadFile          # noqa: E402
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse   # noqa: E402
from fastapi.staticfiles import StaticFiles                                 # noqa: E402
from pydantic import BaseModel                                              # noqa: E402

import casefile                                                             # noqa: E402
import extract                                                              # noqa: E402
import questions                                                            # noqa: E402

STAGES = ["Waiting for doctor", "Being seen", "Results pending", "Going to surgery", "Transfer to another department",
          "Admitted to ward", "Discharged home"]
LEAVING = set(STAGES[3:])
ROLES = ["Triage nurse", "Organising doctor", "Doctor", "Nurse", "Ward"]
BOARD_EDITORS = {"Triage nurse", "Organising doctor"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
# picture type -> name. Bone X-ray also runs the fracture detector (+ MedSAM outline); chest X-ray the 18-finding check.
IMAGE_TYPES = {"bone_xray": "Bone X-ray", "chest_xray": "Chest X-ray", "ct_slice": "CT slice", "mri_slice": "MRI slice",
               "ultrasound": "Ultrasound", "ecg": "ECG", "other": "Other / photo"}


def find_image(p, file):
    for n in p["notes"]:
        for img in n["images"]:
            if img["file"] == file:
                return n, img
    raise HTTPException(404, "No such picture")
EDITABLE_FIELDS = {"age", "sex", "hr", "sbp", "dbp", "rr", "spo2", "temp_c", "nrs_pain", "mental", "ambulance", "injured"}


def clock():
    return time.strftime("%H:%M")


def new_patient(pid, bed):
    return {"id": pid, "bed": bed, "created": time.time(), "notes": [], "ext": {"fields": {}, "source": {}, "warnings": []},
            "items": [], "rejected": [], "questions": [], "dismissed": [], "todos": [], "chat": [], "timeline": [],
            "stage": "Waiting for doctor", "doctor": "", "cleared": False, "status": "working", "progress": "Queued",
            "error": "", "pending": [], "rescore": False, "decision": None, "visit": None, "seconds": None}


class Store:
    def __init__(self, ai, test, db):
        self.ai, self.test = ai, test
        self.version, self.patients = 0, {}
        self.cond = threading.Condition()
        self.state = threading.RLock()           # guards patient dicts
        self.model_lock = threading.Lock()       # one MedGemma call at a time (worker + chat)
        self.jobs = queue.Queue()
        self.db = sqlite3.connect(db, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS patients (id TEXT PRIMARY KEY, data BLOB)")
        for pid, blob in self.db.execute("SELECT id, data FROM patients"):
            p = pickle.loads(blob)
            if p["status"] == "working":
                p["status"], p["progress"] = "error", "Server restarted during the AI step: press Retry"
            self.patients[pid] = p
        self.images = Path(db).parent / "images"
        self.images.mkdir(parents=True, exist_ok=True)
        threading.Thread(target=self._worker, daemon=True).start()

    # ------------------------------------------------------------ saving + telling the screens
    def save(self, p):
        with self.state:
            blob = pickle.dumps(p)
            self.db.execute("INSERT OR REPLACE INTO patients VALUES (?, ?)", (p["id"], blob))
            self.db.commit()
        self.bump()

    def bump(self):
        with self.cond:
            self.version += 1
            self.cond.notify_all()

    def get(self, pid):
        if pid not in self.patients:
            raise HTTPException(404, "No such patient")
        return self.patients[pid]

    def log(self, p, what, text, who="AI"):
        p["timeline"].append({"time": clock(), "who": who, "what": what, "text": text})

    # ------------------------------------------------------------ AI work
    def queue(self, p):
        with self.state:
            p["status"], p["progress"] = "working", "Queued"
        self.jobs.put(p["id"])
        self.save(p)

    def picture_job(self, p, *job):
        """('read', file, kind) or ('outline', file, box, who): picture work on the same single AI worker."""
        with self.state:
            p["status"], p["progress"] = "working", "Queued (picture)"
        self.jobs.put((p["id"], *job))
        self.save(p)

    def _picture(self, p, what, file, *args):
        n, img = find_image(p, file)
        if what == "read":
            img["text"] = ""
            for k in ("overlay", "outline", "tools"):
                img.pop(k, None)
            self._read_image(p, n, img, args[0])
        else:
            box, who = args
            self.step(p, f"Outlining the area on {img['name']}")
            seg = self.ask(self.ai.outline, p["visit"], str(self.images / file), box, who)
            img["outline"] = self._save_picture(img, seg["overlay"], "outline")
            text = f"Outline drawn from a box by {who}: {seg['area_pct']:.1f}% of the picture."
            img["tools"] = (img.get("tools", "") + " " + text).strip()
            item = next((i for i in p["items"] if i.get("image") == file and i["kind"] != "person"), None)
            if item:
                item["text"] = item["text"].rstrip() + " " + text
            self.log(p, "Imaging", f"{img['name']}: {text}", who)

    def _worker(self):
        while True:
            pid = self.jobs.get()
            if isinstance(pid, tuple):                 # picture job
                p = self.patients.get(pid[0])
                if p is None:
                    continue
                try:
                    self._picture(p, *pid[1:])
                    with self.state:
                        p["items"] = casefile.ordered(p["items"])
                        p["status"], p["progress"], p["error"] = "ready", f"Up to date {clock()}", ""
                except Exception as e:
                    traceback.print_exc()
                    with self.state:
                        p["status"], p["progress"], p["error"] = "error", f"The picture step failed: {type(e).__name__}", repr(e)
                self.save(p)
                continue
            p = self.patients.get(pid)
            if p is None or not p["pending"] and p["visit"] is not None:
                if p is not None and p["status"] == "working":
                    p["status"], p["progress"] = "ready", f"Up to date {clock()}"
                    self.save(p)
                continue
            start = time.time()
            try:
                while p["pending"]:
                    self._work(p)
                with self.state:
                    p["status"], p["progress"], p["error"] = "ready", f"Up to date {clock()}", ""
                    p["seconds"] = round(time.time() - start, 1)
            except Exception as e:                  # the notes are saved; Retry runs the AI again
                traceback.print_exc()
                with self.state:
                    p["status"], p["progress"], p["error"] = "error", "The AI step failed. The notes are saved.", repr(e)
            self.save(p)

    def step(self, p, text):
        p["progress"] = text
        self.bump()

    def ask(self, fn, *args):
        with self.model_lock:
            return fn(*args)

    def _work(self, p):
        ai = None if self.test else self.ai
        with self.state:
            doing = list(p["pending"])
        notes = [p["notes"][i] for i in doing]
        # 1. numbers (fixed rules, instant)
        for i, n in zip(doing, notes):
            if n["text"]:
                new = extract.extract(n["text"])
                for s in new["source"].values():
                    s["note"] = i
                old_fields = dict(p["ext"]["fields"])
                p["ext"] = extract.merge(p["ext"], new) if p["notes"][0] is not n else new
                if p["visit"] is not None and any(old_fields.get(k) != v for k, v in p["ext"]["fields"].items()
                                                  if k in extract.VITALS):
                    p["rescore"] = True
        # 2. the case file: AI tidies each new note, code checks every line
        sources = [s.get("text") for s in p["ext"]["source"].values()]
        for i, n in zip(doing, notes):
            if not n["text"]:
                continue
            self.step(p, "Writing the case file" if i == 0 else "Adding the new note to the case file")
            items, rejected = self.ask(casefile.file_note, ai, n["text"], i, n["who"], sources)
            with self.state:
                p["items"] += items
                p["rejected"] += rejected
            self.log(p, "Case file", f"{len(items)} line(s) filed from the note by {n['who']}"
                     + (f"; {len(rejected)} AI line(s) rejected (not supported by the note)" if rejected else ""))
        # 3. urgency + level (trained models), once, from the arrival information
        if p["visit"] is None:
            self.step(p, "Scoring urgency (trained models)")
            self._score(p)
        # 4. pictures
        for n in notes:
            for img in n["images"]:
                if not img.get("text"):
                    self._read_image(p, n, img)
        # 5. parts written by code
        with self.state:
            p["items"] = casefile.replace_auto(p["items"], "Observations",
                                               casefile.observation_items(p["ext"]["fields"], p["ext"]["source"]))
            p["items"] = casefile.ordered(p["items"])
        # 6. questions + to-dos
        self.step(p, "Checking what's missing")
        self._questions(p)
        with self.state:
            p["pending"] = [i for i in p["pending"] if i not in doing]

    def _score(self, p):
        fields = {k: v for k, v in p["ext"]["fields"].items()}
        fields["chief_complaint"] = casefile.complaint(p["items"]) or "not recorded"
        visit = self.ask(self.ai.triage, fields, False)
        if p["visit"] is not None:                 # keep the chat memory and results across a re-score
            visit.results, visit.messages, visit.timeline = p["visit"].results, p["visit"].messages, p["visit"].timeline
        p["visit"] = visit
        d = visit.decision
        p["decision"] = {"p_urgent": d.p_urgent, "flagged": d.flagged, "head_flag": d.head_flag,
                         "floor_flag": d.floor_flag, "level": d.level, "level_set": list(d.level_set),
                         "news2": d.news2, "floor_priority": d.floor_priority, "scored_with": fields}
        p["rescore"] = False
        p["items"] = casefile.replace_auto(p["items"], "Triage decision support", casefile.triage_items(d))
        self.log(p, "Triage", f"Level {d.level}, {d.p_urgent:.0%} chance of needing level 1-2 care"
                 + (" (flagged urgent)" if d.flagged else ""), "Trained models")

    def _questions(self, p):
        notes_text = "\n".join(n["text"] for n in p["notes"])
        case_text = casefile.as_text([i for i in p["items"] if i["section"] != "Triage decision support"])
        ai = None if self.test else self.ai
        qs, todos = self.ask(questions.find, ai, casefile.complaint(p["items"]), notes_text, case_text,
                             p["ext"]["fields"], p["visit"].decision if p["visit"] else None, p["dismissed"])
        with self.state:
            p["questions"] = qs
            p["alerts"] = questions.vital_actions(p["ext"]["fields"], p["visit"].decision if p["visit"] else None)
            have = {t["text"].lower() for t in p["todos"]} | {d.lower() for d in p["dismissed"]}
            for t in todos:
                if t["text"].lower() not in have:
                    p["todos"].append({"id": uuid.uuid4().hex[:6], "text": t["text"], "why": t["why"],
                                       "by": t["by"], "ai": True, "done": False, "done_by": ""})

    def _read_image(self, p, n, img, kind=None):
        """kind: the type a person chose (at upload or later); None = MedGemma sorts the picture first."""
        path = str(self.images / img["file"])
        kind = kind or img.get("chosen")
        self.step(p, f"Reading {img['name']}")
        out = self.ask(self.ai.read_image, p["visit"], path, kind, n["who"])
        img["sorted_by_ai"] = kind is None
        img["kind"], img["text"] = out["kind"], out["text"]
        found, chest = out.get("detector"), out.get("chest")
        extra = []
        if found is not None:
            extra.append(f"Fracture detector: {found['n']} possible fracture(s) marked, check them." if found["n"]
                         else "Fracture detector: nothing marked (it misses about 1 in 5).")
        if chest:
            extra.append("Chest X-ray check: " + ", ".join(f"{k} {v:.0%}" for k, v in chest["top"]) + ".")
        picture = found.get("overlay") if found and found["n"] else (chest or {}).get("overlay")
        if picture is not None:
            img["overlay"] = self._save_picture(img, picture, "marked")
        if found and found["n"]:                 # detect -> segment: strongest box to MedSAM, no clicking
            self.step(p, f"Outlining the marked area on {img['name']}")
            box = found["boxes"][max(range(found["n"]), key=lambda k: found["confs"][k])]
            try:
                seg = self.ask(self.ai.outline, p["visit"], path, box, "AI (detector box)")
                img["outline"] = self._save_picture(img, seg["overlay"], "outline")
                extra.append(f"Outline of the strongest mark: {seg['area_pct']:.1f}% of the picture.")
            except Exception as e:
                extra.append(f"(Outline not available on this computer: {type(e).__name__}.)")
        img["tools"] = " ".join(extra)
        label = IMAGE_TYPES.get(img["kind"], img["kind"])
        text = f"{img['name']} — {label}: {img['text']} {img['tools']}".strip()
        # one reading per picture: a re-read replaces the AI's earlier one (a person's edited line is kept)
        p["items"] = [i for i in p["items"] if not (i.get("image") == img["file"] and i["kind"] != "person")]
        item = casefile.new_item("Imaging", text, "ai", "Image AI", img["name"])
        item["image"] = img["file"]
        p["items"].append(item)
        self.log(p, "Imaging", f"{img['name']} read as {label}" + (" (type sorted by AI)" if img["sorted_by_ai"] else
                                                                      " (type chosen by a person)"))

    def _save_picture(self, img, picture, tag):
        import numpy as np
        from PIL import Image
        name = img["file"].rsplit(".", 1)[0] + f"_{tag}.png"
        Image.fromarray(np.asarray(picture).astype("uint8")).save(self.images / name)
        return name


# ================================================================ HTTP
app = FastAPI(title="ED app v2")
store: Store = None


def summary(p):
    d = p["decision"] or {}
    f = p["ext"]["fields"]
    sex = {"male": "M", "female": "F"}.get(f.get("sex"), "")
    age = f"{f['age']:.0f}" if f.get("age") is not None else ""
    return {"id": p["id"], "bed": p["bed"], "created": p["created"], "stage": p["stage"], "doctor": p["doctor"],
            "cleared": p["cleared"], "leaving": p["stage"] in LEAVING, "status": p["status"], "progress": p["progress"],
            "level": d.get("level"), "flagged": d.get("flagged"), "p_urgent": d.get("p_urgent"), "news2": d.get("news2"),
            "who": " ".join(x for x in (age, sex) if x) or "?", "complaint": casefile.complaint(p["items"]),
            "todos": p["todos"], "questions": len(p["questions"]), "warnings": len(p["ext"]["warnings"]),
            "seen_at": p.get("seen_at"), "alerts": p.get("alerts", [])}


def full(p):
    return {**summary(p), "notes": p["notes"], "fields": p["ext"]["fields"], "source": p["ext"]["source"],
            "warnings": p["ext"]["warnings"], "items": p["items"], "rejected": p["rejected"], "questions": p["questions"],
            "chat": p["chat"], "timeline": p["timeline"], "decision": p["decision"], "error": p["error"],
            "rescore": p["rescore"], "seconds": p["seconds"], "sections": casefile.SECTIONS, "names": extract.NAMES}


@app.get("/api/meta")
def meta():
    return {"stages": STAGES, "leaving": sorted(LEAVING), "roles": ROLES, "editors": sorted(BOARD_EDITORS),
            "image_types": IMAGE_TYPES, "test": store.test, "version": store.version}


@app.get("/api/events")
async def events():
    """Server-sent events: one line each time anything changes (the screens then reload what they show)."""
    async def stream():
        seen = -1
        while True:
            if store.version != seen:
                seen = store.version
                yield f"data: {seen}\n\n"
            else:
                yield ": keep-alive\n\n"
            await asyncio.sleep(0.4 if store.version != seen else 1.0)
    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/api/board")
def board(cleared: bool = False):
    rows = [p for p in store.patients.values() if p["cleared"] == cleared]
    rows.sort(key=lambda p: (p["stage"] in LEAVING, (p["decision"] or {}).get("level") or 9, p["created"]))
    return {"version": store.version, "now": time.time(), "patients": [summary(p) for p in rows],
            "n_cleared": sum(p["cleared"] for p in store.patients.values())}


@app.get("/api/patients/{pid}")
def patient(pid: str):
    return {"version": store.version, **full(store.get(pid))}


class TextIn(BaseModel):
    text: str


@app.post("/api/extract")
def preview(body: TextIn):
    """Live tiles while the nurse types (fixed rules only, a few ms)."""
    return extract.extract(body.text)


def _kinds(raw):
    """Form field 'kinds': a JSON list, one entry per file ('' = let the AI sort it)."""
    import json
    try:
        return [k if k in IMAGE_TYPES else None for k in json.loads(raw)] if raw else []
    except ValueError:
        return []


def _save_images(pid, files, kinds=()):
    out = []
    kinds = list(kinds)
    for idx, f in enumerate(files or []):
        if not f.filename:
            continue
        suffix = Path(f.filename).suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            raise HTTPException(400, f"{f.filename}: only pictures (PNG, JPG, ...) can be added")
        name = f"{pid}_{uuid.uuid4().hex[:8]}{suffix}"
        with open(store.images / name, "wb") as fh:
            shutil.copyfileobj(f.file, fh)
        out.append({"file": name, "name": f.filename, "kind": None, "text": "",
                    "chosen": kinds[idx] if idx < len(kinds) else None})
    return out


@app.post("/api/patients")
def create(text: str = Form(...), who: str = Form(...), kinds: str = Form(""),
           files: list[UploadFile] = File(default=[])):
    if not text.strip():
        raise HTTPException(400, "Type the note first")
    if not who.strip():
        raise HTTPException(400, "Enter your name at the top first")
    pid = uuid.uuid4().hex[:6]
    used = {p["bed"] for p in store.patients.values() if not p["cleared"]}
    bed = next(f"P{i}" for i in range(1, 999) if f"P{i}" not in used)
    p = new_patient(pid, bed)
    p["notes"].append({"time": clock(), "who": who.strip(), "text": text.strip(),
                       "images": _save_images(pid, files, _kinds(kinds))})
    p["pending"] = [0]
    store.log(p, "Arrival", "Triage note written", who.strip())
    store.patients[pid] = p
    store.queue(p)
    return {"id": pid, "bed": bed}


@app.post("/api/patients/{pid}/notes")
def add_note(pid: str, text: str = Form(""), who: str = Form(...), answers: str = Form(""), kinds: str = Form(""),
             files: list[UploadFile] = File(default=[])):
    """New information at any time, even while the AI is busy: it waits in `pending` and is done next."""
    p = store.get(pid)
    images = _save_images(pid, files, _kinds(kinds))
    if not text.strip() and not images:
        raise HTTPException(400, "Type something or add a picture")
    with store.state:
        p["notes"].append({"time": clock(), "who": who, "text": text.strip(), "images": images, "answers": answers})
        p["pending"].append(len(p["notes"]) - 1)
        if answers:
            p["dismissed"].append(answers)
            p["questions"] = [q for q in p["questions"] if q["text"] != answers]
        busy = p["status"] == "working"
    store.log(p, "Note", (f"Answered “{answers}”: " if answers else "") + (text.strip() or "picture added"), who)
    if busy:
        store.save(p)
    else:
        store.queue(p)
    return {"ok": True, "queued_behind_ai": busy}


@app.post("/api/patients/{pid}/retry")
def retry(pid: str):
    p = store.get(pid)
    if p["status"] == "working":
        raise HTTPException(409, "Already working")
    if p["visit"] is None:                          # first run failed: file the arrival note again from scratch
        p["items"] = [i for i in p["items"] if i["kind"] == "person"]
        p["pending"] = sorted(set(p["pending"]) | {0})
    store.queue(p)
    return {"ok": True}


class ItemIn(BaseModel):
    who: str
    text: str | None = None
    section: str | None = None
    confirmed: bool | None = None
    delete: bool = False


@app.post("/api/patients/{pid}/items")
def add_item(pid: str, body: ItemIn):
    p = store.get(pid)
    if body.section not in casefile.SECTIONS or not (body.text or "").strip():
        raise HTTPException(400, "Pick a heading and type the line")
    with store.state:
        p["items"].append(casefile.new_item(body.section, body.text, "person", body.who, confirmed=True))
        p["items"] = casefile.ordered(p["items"])
    store.log(p, "Case file", f"Added to {body.section}: {body.text.strip()}", body.who)
    store.save(p)
    return {"ok": True}


@app.patch("/api/patients/{pid}/items/{iid}")
def change_item(pid: str, iid: str, body: ItemIn):
    """Edit, confirm or remove one line. An edited line becomes the person's: the AI never touches it again."""
    p = store.get(pid)
    with store.state:
        item = next((i for i in p["items"] if i["id"] == iid), None)
        if item is None:
            raise HTTPException(404, "That line is gone")
        if body.delete:
            p["items"].remove(item)
            what = f"Removed from {item['section']}: {item['text']}"
        elif body.text is not None and body.text.strip() != item["text"]:
            what = f"{item['section']}: “{item['text']}” → “{body.text.strip()}”"
            item.update(text=body.text.strip(), kind="person", by=body.who, confirmed=True, time=clock())
        elif body.confirmed is not None:
            item["confirmed"], item["checked_by"] = body.confirmed, body.who
            what = f"{'Checked' if body.confirmed else 'Unchecked'}: {item['text']}"
        else:
            return {"ok": True}
    store.log(p, "Case file", what, body.who)
    store.save(p)
    return {"ok": True}


class FieldsIn(BaseModel):
    who: str
    fields: dict


@app.put("/api/patients/{pid}/fields")
def fix_numbers(pid: str, body: FieldsIn):
    """The nurse corrects or adds numbers -> urgency and level scored again, observations rewritten."""
    p = store.get(pid)
    if p["status"] == "working":
        raise HTTPException(409, "The AI is still working on this patient; try again in a moment")
    clean = {}
    for k, v in body.fields.items():
        if k not in EDITABLE_FIELDS or v in ("", None):
            continue
        try:
            clean[k] = v if k in ("sex", "mental") else float(v)
        except (TypeError, ValueError):
            raise HTTPException(400, f"{extract.NAMES.get(k, k)}: '{v}' is not a number")
    old = (p["decision"] or {}).get("level")
    with store.state:
        changed = [k for k in set(clean) | set(p["ext"]["fields"]) if clean.get(k) != p["ext"]["fields"].get(k)]
        p["ext"]["fields"] = clean
        for k in changed:
            p["ext"]["source"][k] = {"text": f"entered by {body.who}", "span": None, "prehospital": False}
        for k in [k for k in p["ext"]["source"] if k not in clean and not p["ext"]["source"][k].get("prehospital")]:
            del p["ext"]["source"][k]
    store._score(p)                                 # takes the model lock itself
    with store.state:
        p["items"] = casefile.ordered(casefile.replace_auto(
            p["items"], "Observations", casefile.observation_items(p["ext"]["fields"], p["ext"]["source"])))
    store.log(p, "Triage", f"Numbers checked ({', '.join(changed) or 'no change'}); level {old} → "
              f"{p['decision']['level']}", body.who)
    store._questions(p)
    store.save(p)
    return {"ok": True}


class QIn(BaseModel):
    who: str
    text: str


@app.post("/api/patients/{pid}/questions/dismiss")
def dismiss(pid: str, body: QIn):
    p = store.get(pid)
    with store.state:
        p["dismissed"].append(body.text)
        p["questions"] = [q for q in p["questions"] if q["text"] != body.text]
    store.log(p, "Question", f"Not needed: {body.text}", body.who)
    store.save(p)
    return {"ok": True}


@app.post("/api/patients/{pid}/chat")
def chat(pid: str, body: QIn):
    p = store.get(pid)
    if p["visit"] is None:
        raise HTTPException(409, "The case file isn't ready yet")
    case = casefile.as_text(p["items"], f"{p['bed']} {summary(p)['who']}")
    if p.get("chat_seen") != case:                 # the chat always reads the latest case file
        p["visit"].unseen.append("CURRENT CASE FILE (latest, including people's edits):\n" + case)
        p["chat_seen"] = case
    with store.model_lock:
        answer = store.ai.chat(p["visit"], body.text)
    p["chat"] += [{"role": "user", "text": body.text, "who": body.who, "time": clock()},
                  {"role": "ai", "text": answer, "who": "AI", "time": clock()}]
    store.log(p, "Chat", body.text, body.who)
    store.save(p)
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
    if body.stage and body.stage != p["stage"]:
        if body.stage not in STAGES:
            raise HTTPException(400, "Unknown stage")
        change.append(f"stage → {body.stage}")
        if body.stage == STAGES[0]:
            p["seen_at"] = None                     # back to waiting: the clock runs again from arrival
        elif p.get("seen_at") is None:
            p["seen_at"] = time.time()              # the waiting clock stops when the patient is first seen
        p["stage"] = body.stage
    if body.doctor is not None and body.doctor.strip() != p["doctor"]:
        change.append(f"doctor → {body.doctor.strip() or 'none'}")
        p["doctor"] = body.doctor.strip()
    if body.cleared is not None and body.cleared != p["cleared"]:
        if body.cleared and p["stage"] not in LEAVING:
            raise HTTPException(400, "Choose where the patient went (surgery, transfer, ward or home) first")
        change.append("cleared from the board" if body.cleared else "put back on the board")
        p["cleared"] = body.cleared
    if change:
        store.log(p, "Board", "; ".join(change), body.who)
        store.save(p)
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
        raise HTTPException(400, "Empty to-do")
    p["todos"].append({"id": uuid.uuid4().hex[:6], "text": body.text.strip(), "why": "", "by": body.who, "ai": False,
                       "done": False, "done_by": ""})
    store.log(p, "To-do", f"Added: {body.text.strip()}", body.who)
    store.save(p)
    return {"ok": True}


@app.patch("/api/patients/{pid}/todos/{tid}")
def change_todo(pid: str, tid: str, body: TodoIn):
    _editor(body.role)
    p = store.get(pid)
    t = next((t for t in p["todos"] if t["id"] == tid), None)
    if t is None:
        raise HTTPException(404, "No such to-do")
    if body.delete:
        p["todos"].remove(t)
        if t["ai"]:
            p["dismissed"].append(t["text"])           # the AI won't suggest it again
        store.log(p, "To-do", f"Removed: {t['text']}", body.who)
    else:
        t["done"], t["done_by"] = bool(body.done), body.who if body.done else ""
        store.log(p, "To-do", f"{'Done' if t['done'] else 'Not done'}: {t['text']}", body.who)
    store.save(p)
    return {"ok": True}


class PictureIn(BaseModel):
    who: str
    kind: str | None = None
    box: list[float] | None = None


@app.post("/api/patients/{pid}/images/{file}/read")
def reread(pid: str, file: str, body: PictureIn):
    """A person says what the picture is -> it is read again with that type's check (bone X-ray -> fracture
    detector + outline; chest X-ray -> 18-finding check)."""
    p = store.get(pid)
    find_image(p, file)
    if body.kind not in IMAGE_TYPES:
        raise HTTPException(400, "Unknown picture type")
    if p["visit"] is None or p["status"] == "working":
        raise HTTPException(409, "The AI is still working on this patient; try again in a moment")
    store.log(p, "Imaging", f"{file}: type set to {IMAGE_TYPES[body.kind]}, read again", body.who)
    store.picture_job(p, "read", file, body.kind)
    return {"ok": True}


@app.post("/api/patients/{pid}/images/{file}/outline")
def outline_box(pid: str, file: str, body: PictureIn):
    """A person drags a box on the picture -> MedSAM outlines what is inside it."""
    p = store.get(pid)
    find_image(p, file)
    if not body.box or len(body.box) != 4 or abs(body.box[2] - body.box[0]) < 4 or abs(body.box[3] - body.box[1]) < 4:
        raise HTTPException(400, "Drag a box around the area first")
    if p["visit"] is None or p["status"] == "working":
        raise HTTPException(409, "The AI is still working on this patient; try again in a moment")
    store.picture_job(p, "outline", file, [min(body.box[0], body.box[2]), min(body.box[1], body.box[3]),
                                            max(body.box[0], body.box[2]), max(body.box[1], body.box[3])], body.who)
    return {"ok": True}


@app.get("/api/patients/{pid}/export", response_class=PlainTextResponse)
def export(pid: str):
    p = store.get(pid)
    s = summary(p)
    head = (f"ED CASE FILE — {p['bed']} ({s['who']}) — stage: {p['stage']}, doctor: {p['doctor'] or 'none'}\n"
            "AI-tidied lines marked [AI, unchecked] have not been checked by a person.")
    notes = "\n".join(f"{n['time']} {n['who']}: {n['text']}" for n in p["notes"])
    todos = "\n".join(f"[{'x' if t['done'] else ' '}] {t['text']}" for t in p["todos"]) or "none"
    tl = "\n".join(f"{e['time']} {e['who']} — {e['what']}: {e['text']}" for e in p["timeline"])
    return f"{casefile.as_text(p['items'], head)}\n\nTO DO\n{todos}\n\nNOTES AS TYPED\n{notes}\n\nTIMELINE\n{tl}\n"


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
    with store.state:
        store.db.execute("DELETE FROM patients WHERE id = ?", (pid,))
        store.db.commit()
    store.bump()
    return {"ok": True}


def main():
    global store
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="fake writing; rules and Caleb's vitals model are real")
    ap.add_argument("--lan", action="store_true")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--db", default=str(HERE / "data" / "ed.sqlite"))
    a = ap.parse_args()
    if a.test:
        from test_backend import TestPipeline
        ai = TestPipeline()
    else:
        import pipeline
        pipeline.CAPS.update({"report": 1024, "questions": 200})
        ai = pipeline.EDPipeline()
    Path(a.db).parent.mkdir(parents=True, exist_ok=True)
    store = Store(ai, a.test, a.db)
    dist = HERE / "ui" / "dist"
    if dist.exists():
        app.mount("/", StaticFiles(directory=dist, html=True), name="ui")
    uvicorn.run(app, host="0.0.0.0" if a.lan else "127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
