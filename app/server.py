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
sys.path[:0] = [str(HERE.parent / "reasoning" / "src")]

from fastapi import FastAPI, File, Form, HTTPException, UploadFile          # noqa: E402
from fastapi.responses import FileResponse, PlainTextResponse               # noqa: E402
from fastapi.staticfiles import StaticFiles                                 # noqa: E402
from pydantic import BaseModel                                              # noqa: E402

import charting                                                             # noqa: E402
from image_io import save_8bit                                              # noqa: E402
import learning                                                             # noqa: E402

# The client's actions (6 Nov 2026): where the patient is going. The last four mean the patient leaves the ED; the
# organising doctor or triage nurse then clears them off the board (the case file is kept, under "Left the ED").
STAGES = ["Waiting for doctor", "Being seen", "Results pending", "Gone for imaging", "Going to surgery", "Transfer to another department",
          "Admitted to ward", "Discharged home"]
LEAVING = set(STAGES[4:])
OLD_STAGES = {"Waiting": "Waiting for doctor", "Admitted / ward": "Admitted to ward", "Discharged": "Discharged home"}
ROLES = ["Triage nurse", "Organising doctor", "Doctor", "Nurse", "Ward / other"]
BOARD_EDITORS = {"Triage nurse", "Organising doctor"}            # the client's choice: only these change the board
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".dcm"}
FIELD_NAMES = {"age": "Age", "sex": "Sex", "hr": "HR", "sbp": "BP sys", "dbp": "BP dia", "rr": "RR", "spo2": "SpO2 %",
               "temp_c": "Temp C", "nrs_pain": "Pain /10", "mental": "Consciousness", "ambulance": "Ambulance (1/0)",
               "injured": "Injury (1/0)"}
NEWER_READING = {"hr", "sbp", "dbp", "rr", "spo2", "temp_c", "mental", "nrs_pain"}   # vital signs: a later note updates them


GROUPS = {"Urgent": "#d32f2f", "Can wait": "#f9a825", "Minor": "#388e3c"}


def group(level, flagged):
    """Three groups for the whiteboard (Alex, 14 Nov: urgent / not urgent was too coarse). Urgent = flagged (the same
    9-in-10 cut-off) or level 1-2; Can wait = level 3; Minor = level 4-5. Only a label: nothing is re-scored."""
    if level is None:
        return ""
    return "Urgent" if flagged or level <= 2 else "Can wait" if level == 3 else "Minor"


def _read_line(added, was):
    """'Consciousness verbal (was alert), HR 128 (was 112)': what a later note changed."""
    return ", ".join(f"{FIELD_NAMES[k]} {_show(v)}" + (f" (was {_show(was[k])})" if k in was else "") for k, v in added.items())


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
    stage_since: float = 0.0                        # when the current stage began (the board timer resets, 14 Nov)
    cleared: bool = False                           # off the board (left the ED); case file kept
    doctor: str = "unassigned"
    status: str = "working"                         # working / ready / error
    progress: str = "Queued"
    error: str = ""
    pending: list = field(default_factory=list)     # note indexes the AI hasn't put in the case file yet
    report_seen_by_chat: str = ""
    early_log: list = field(default_factory=list)   # board changes made before the AI's first result
    sbar: dict = field(default_factory=dict)        # last SBAR handover: {text, who, time, n_notes, n_chat}
    # a doctor's priority (14 Nov): {level, reason, who, role, time, ai_level, ai_flagged}; the AI's decision stays
    # underneath, unchanged, for learning and for showing
    override: dict = field(default_factory=dict)

    def shown_level(self):
        """The level everyone sees and the board sorts by: the doctor's when they set one, else the AI + rules'."""
        o, d = getattr(self, "override", {}), self.visit.decision if self.visit else None
        return o["level"] if o and d else d.level if d else None

    def review(self):
        """New information since the doctor's change made the AI + rules MORE urgent than both then and the doctor's
        level: the doctor's level stays, with a warning ('' = nothing to review)."""
        o, d = getattr(self, "override", {}), self.visit.decision if self.visit else None
        if not (o and d) or d.level >= o["level"]:
            return ""
        # "more urgent than when the doctor decided": a lower level, a higher score, NEWS2 up by 2+, a stricter floor or a
        # new critical picture (the level alone rarely goes below 2, so a sicker T01 would never have warned)
        worse = (d.level < o["ai_level"] or d.p_urgent > o.get("ai_p", 1) + 0.05 or d.news2 >= o.get("ai_news2", 99) + 2
                 or d.floor_priority < o.get("ai_floor", 0) or (getattr(d, "picture", "") and not o.get("ai_picture")))
        if not worse:
            return ""
        return f"New information since {o['who']}'s change: AI and rules now say level {d.level}, please review"

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

    def timer(self):
        """The board clock (Alex, 14 Nov): waiting to be seen counts from arrival; going for imaging or a decision to
        leave the ED starts it again (how long since); other stages show the total time in the ED."""
        since = getattr(self, "stage_since", 0.0) or self.created
        if self.stage == "Waiting for doctor":
            return {"label": "Waiting to be seen", "since": self.created, "restarted": False}
        if self.stage == "Gone for imaging":
            return {"label": "At imaging", "since": since, "restarted": True}
        if self.stage in LEAVING:
            return {"label": "Since the decision to leave", "since": since, "restarted": True}
        return {"label": "In the ED", "since": self.created, "restarted": False}

    def summary(self):
        d = self.visit.decision if self.visit else None
        o = getattr(self, "override", {}) if d else {}
        out = {"id": self.id, "bed": self.bed, "created": self.created, "stage": self.stage, "timer": self.timer(), "doctor": self.doctor, "cleared": self.cleared,
               "leaving": self.stage in LEAVING,
               "status": self.status, "progress": self.progress,
               "level": d.level if d else None, "flagged": d.flagged if d else None,
               "raised_by": d.raised_by() if d else "",
               "needs_obs": bool(d and d.needs_obs), "reason": d.reason() if d else "",
               "child": is_child(self.fields), "child_score": is_child(self.fields) and child_score(),
               "who": _who_line(self.fields), "complaint": self.complaint(),
               "todos": self.todos, "open_questions": len(self.questions), "override": None, "review": ""}
        if o:                                       # a doctor set the priority: theirs is shown, the AI's underneath
            out.update(level=o["level"], flagged=o["level"] <= 2, raised_by="", needs_obs=False,
                       reason=override_line(o), review=self.review(),
                       override={**o, "now_ai_level": d.level, "now_ai_flagged": d.flagged})
        out["group"] = group(out["level"], out["flagged"])
        return out

    def full(self):
        d = self.visit.decision if self.visit else None
        return {**self.summary(), "notes": self.notes, "fields": self.fields, "found": self.found,
                "field_names": FIELD_NAMES, "report": self.report, "report_by": self.report_by,
                "questions": self.questions, "chat": self.chat, "error": self.error,
                "sbar": getattr(self, "sbar", {}),           # boards saved before 14 Nov have no SBAR
                "timeline": self.visit.timeline if self.visit else [],
                "decision": None if d is None else {
                    "p_urgent": d.p_urgent, "flagged": d.flagged, "head_flag": d.head_flag, "floor_flag": d.floor_flag,
                    "level": d.level, "level_set": list(d.level_set), "news2": d.news2, "groups": d.groups(),
                    "floor_priority": d.floor_priority, "needs_obs": d.needs_obs, "missing": list(d.missing),
                    "abnormal": list(d.abnormal), "picture": getattr(d, "picture", "")}}


CHILD_AGE = 16
CHILD_NOTE = ("Patient under 16: NEWS2, the urgency score and the triage level come from models built on adults "
              "(NEWS2 is for 16+, the triage data for 15+). Use a paediatric scale and your own judgement.")


def is_child(fields):
    age = fields.get("age")
    return age is not None and age == age and float(age) < CHILD_AGE


CHILD_NOTE_CHILD_SCORE = ("Patient under 16: the urgency score comes from a model trained on children (US emergency "
                          "visits under 16); NEWS2 and the triage level come from models built on adults (NEWS2 is for "
                          "16+). Use a paediatric scale and your own judgement.")


def child_score():
    """True when the children's urgency score is installed and in use (13 Nov 2026, REPORT_NOTES 5u)."""
    return store is not None and getattr(store.ai, "child_head", None) is not None


def child_note():
    return CHILD_NOTE_CHILD_SCORE if child_score() else CHILD_NOTE


# Picture rule (14 Nov): red critical findings from the trained image models move a patient UP, like the vital-signs
# floor. Red only (above the model's own cut-off); not amber "possible", not fractures, effusion or shadowing.
CRITICAL_PICTURES = {"Pneumothorax": "chest X-ray shows a pneumothorax",
                     "Bleeding inside the skull": "CT shows bleeding inside the skull"}


def picture_finding(p):
    """The critical red findings on this patient's pictures, in words ('' = none)."""
    found = []
    for n in p.notes:
        for img in n["images"]:
            for f in img.get("findings") or []:
                words = CRITICAL_PICTURES.get(f.get("name"))
                if words and f.get("flag") is True and words not in found:
                    found.append(words)
    return " and ".join(found)


def apply_picture_rule(p, who="AI (picture rule)"):
    """After the pictures are read, a later note or a numbers fix (each makes a new decision): put the rule back on."""
    from visit import with_picture
    finding = picture_finding(p)
    d = p.visit.decision if p.visit else None
    if not finding or d is None or getattr(d, "picture", "") == finding:
        return
    p.visit.decision = with_picture(d, finding)
    # logged once per finding (a later note or a numbers fix re-scores and the rule is simply put back on)
    if not any(f"picture rule: {finding}" in e["text"] for e in p.visit.timeline):
        p.visit.log("Triage", f"Moved up by the picture rule: {finding} (trained image model); "
                    + ("flagged urgent, " if not d.flagged else "")
                    + (f"level {d.level} -> {p.visit.decision.level}" if d.level != p.visit.decision.level
                       else f"already level {d.level}"), who)


DOCTOR_ROLES = {"Doctor", "Organising doctor"}     # who may change a patient's priority (14 Nov)


def override_line(o):
    why = f": {o['reason']}" if o.get("reason") else ""                     # the reason is optional
    return f"Level {o['level']}, set by {o['who']}{why} (AI + safety rules said {o['ai_level']})"


def triage_lines(p):
    """The case file's triage section: the trained models' decision, word for word, plus the children's warning
    (group test 10 Nov, A2: a normal 7-year-old scored NEWS2 5 with no word that the score is an adult one).
    A doctor's priority goes first, with the AI's decision kept underneath."""
    o = getattr(p, "override", {})
    doctor = ([f"PRIORITY SET BY A DOCTOR: {override_line(o)}, {o['time']}."]
              + ([f"PLEASE REVIEW: {p.review()}."] if p.review() else []) + ["The AI and safety rules said:"]) if o else []
    return doctor + p.visit.decision.lines() + ([child_note()] if is_child(p.fields) else [])


def check_override(p):
    """After any re-score (later note, answer, picture, Fix numbers): if the AI + rules now say more urgent than the
    doctor's level, the doctor's level stays and a warning shows; logged once per new AI level."""
    o, warn = getattr(p, "override", {}), p.review()
    d = p.visit.decision
    now = [d.level, round(d.p_urgent, 2), d.floor_priority, d.news2]
    if warn and o.get("review_logged") != now:
        o["review_logged"] = now
        p.visit.log("Priority", warn, "AI (decision support)")
        store.learn("priority_review", p, doctor_level=o["level"], ai_level_then=o["ai_level"],
                    ai_level_now=p.visit.decision.level, p_urgent=p.visit.decision.p_urgent)


def show_supported_pathways(p):
    """Pathways the notes don't support (poisoning from "warfarin" P11, collapse from "lightheaded" T05) are not put
    in the case file, asked about, given to-dos or read by the chat. Worked out again from the model's own flags
    (kept in visit.ed_model, logged for learning) whenever a note is added or the numbers are re-scored."""
    if p.visit is None:
        return
    p.visit.ed_model = getattr(p.visit, "ed_model", None) or p.visit.ed
    p.visit.ed = charting.supported_pathways(p.visit.ed_model, "\n".join(n["text"] for n in p.notes))


def _who_line(f):
    age = f"{f['age']:.0f}" if f.get("age") is not None else "?"
    sex = {"male": "M", "female": "F"}.get(f.get("sex"), "?")
    return f"{sex}, {age}"


class Store:
    """All patients, the AI, and one lock so only one AI job runs at a time."""

    def __init__(self, ai, path):
        self.ai, self.path = ai, path               # ai: EDPipeline or TestPipeline
        self.test = not hasattr(ai, "_ask")
        # lock: one patient's AI job at a time. The model itself is shared step by step (EDPipeline.gpu), so a
        # re-score or a chat doesn't wait for a whole job (5q #4). chat_lock: one chat answer at a time.
        self.lock, self.save_lock, self.state_lock = threading.Lock(), threading.Lock(), threading.Lock()
        self.chat_lock = threading.Lock()
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
        # learning from use (learning.py): corrections are logged next to the board; no board file = nothing saved
        self.learning = learning.FeedbackLog(path.parent / "learning") if path else None

    def learn(self, kind, p, **data):
        if self.learning is not None:
            self.learning.add(kind, p.id, **data)

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

    def _read_later_numbers(self, p, ai, new):
        """Retest 14 Nov (Caleb, T01): an answer like 'alert, GCS 15' went into the case file, but the ACVPU tile and
        the score still said 'not recorded', because only the first note's numbers were read. Later notes (answers and
        added information) are now read the same way as the first note: MedGemma and the rules both read them
        (charting.read_numbers). A number that was missing is filled in and the urgency and level are scored again.
        Caleb's final QC 14 Nov: after a doctor lowered T01, "now drowsy, BP 78/50, HR 128" changed nothing on screen
        (the old rule kept every filled tile and only made a hidden 'please check'). A vital sign in a later note is a
        NEWER reading, so it now replaces the old one (said on the timeline and in the case file: "HR 128 (was 112)")
        and the patient is scored again, which also fires the doctor's review warning. Age and sex are never read
        from a later note ("trop 85" was read as age 85, "now drowsy" as female); other differences stay 'please check'."""
        text = "\n".join(n["text"] for n in new if n["text"].strip())
        if not text:
            return
        self._step(p, "Reading numbers in the new note")
        got, info = charting.read_numbers(ai, text)
        added, was, who = {}, {}, new[-1]["who"]
        p.found.setdefault("found", {})
        p.found.setdefault("by", {})
        p.found.setdefault("warnings", [])
        for k, v in got.items():
            if k == "chief_complaint" or k not in FIELD_NAMES or k in ("age", "sex", "ambulance"):   # never change after arrival
                continue
            if p.fields.get(k) in (None, ""):
                added[k] = v
                p.found["found"][k] = info["found"].get(k, "")
                p.found["by"][k] = info["by"].get(k, "rules")
                p.found.setdefault("check", []).extend(c for c in info.get("check", []) if c["field"] == k)
            elif k in NEWER_READING and not charting._agree(k, p.fields[k], v):
                added[k] = v                         # a newer reading replaces the old one
                was[k] = p.fields[k]
                p.found["found"][k] = info["found"].get(k, "")
                p.found["by"][k] = info["by"].get(k, "rules")
            elif not charting._agree(k, p.fields[k], v):
                p.found["warnings"].append(f"please check {FIELD_NAMES[k]}: a later note says {_show(v)} "
                                           f"({info['found'].get(k, '')!r}), the tile has {_show(p.fields[k])}; "
                                           "use Fix numbers to change it")
        p.found["warnings"] += [w for w in info.get("warnings", []) if w.startswith("please check") and
                                not any(x in w for x in ("Age", "Sex", "Ambulance")) and
                                any(c["field"] in added for c in info.get("check", []))]
        if not added:
            return
        old = p.visit.decision.level
        new_visit = self.ai.triage({**p.fields, **added}, explain=False)
        with self.state_lock:
            before = p.visit
            p.fields = {**p.fields, **added}
            p.visit = new_visit
            p.visit.timeline, p.visit.results, p.visit.messages = before.timeline, before.results, before.messages
            apply_picture_rule(p)                   # a re-score keeps the picture rule
            p.visit.log("Triage", f"Numbers from a later note: {_read_line(added, was)}"
                        f"; level {old} -> {p.visit.decision.level}", who)
        if p.report:
            p.report = charting.add_line(p.report, "Observations", "From a later note: " + _read_line(added, was))
            p.report = charting.drop_not_recorded(p.report, added)

    def _work(self, p):
        ai = None if self.test else self.ai
        doing = list(p.pending)                     # notes added from now on wait for the next round
        new = [p.notes[i] for i in doing]
        later = p.visit is not None                 # these notes come after the first one: read their numbers too
        if p.visit is None:                         # first note: numbers -> urgency + level
            self._step(p, "Pulling out the numbers")
            text = "\n".join(n["text"] for n in p.notes)
            if not p.fields:                        # rules + MedGemma both read them; disagreements go to the nurse
                p.fields, p.found = charting.read_numbers(ai, text)
                p.found["ai_fields"] = dict(p.fields)   # what the AI read, kept so a correction can be learned from
            if not p.fields.get("chief_complaint"):
                self._step(p, "Writing the complaint line")
                p.fields["chief_complaint"] = charting.complaint(ai, text)
            self._step(p, "Scoring urgency (trained models)")
            p.visit = self.ai.triage(p.fields, explain=False)
            p.visit.timeline += getattr(p, "early_log", [])     # board changes made while the AI was still busy
            p.early_log = []
            p.visit.log("Triage", f"Arrival card made from the notes of {p.notes[0]['who']}; level {p.visit.decision.level}"
                        f", urgency {p.visit.decision.p_urgent:.0%}", "AI (decision support)")
        image_lines = []
        for n in new:
            p.visit.add("notes", n["text"], n["who"])
            for img in n["images"]:
                if img.get("text"):
                    continue
                self._step(p, f"Reading image {img['name']}")
                all_notes = "\n".join(x["text"] for x in p.notes)
                out = self.ai.read_image(p.visit, str(self.images / img["file"]), None, n["who"],
                                         age=p.fields.get("age"), note=all_notes)
                img["kind"], img["text"] = out["kind"], out["text"]
                m = out.get("models") or {}
                if m.get("line"):
                    img["text"] += " " + m["line"]
                if m.get("sorted_by"):
                    img["sorted_by"] = m["sorted_by"]
                # plain numbers for the screen (pictures are saved as files, not kept in the record)
                img["findings"] = [{k: v for k, v in f.items()} for f in m.get("findings", [])]
                img["model"] = m.get("model")
                if m.get("overlay") is not None:
                    img["overlay"] = self._save_picture(img, m["overlay"], "marked")
                if m.get("outline_box"):                # detect -> segment: the strongest box goes to MedSAM, no clicking
                    self._step(p, f"Outlining the marked area on {img['name']}")
                    try:
                        seg = self.ai.outline(p.visit, str(self.images / img["file"]), m["outline_box"], "AI (detector box)",
                                              fracture=m.get("outline_kind") == "fracture")
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
                    img["text"] += checklist_line(out["kind"], out["checklist"], anatomy.get("ctr"), p.fields.get("age"))
                    img["extra"] = []
                    for i, loc in enumerate(out.get("located") or []):
                        img["extra"].append({"label": f"AI outline: {loc['finding']}"
                                             + (" (experimental)" if loc.get("experimental") else ""),
                                             "file": self._save_picture(img, loc["overlay"], f"find{i}")})
                    if anatomy.get("overlay") is not None:
                        img["extra"].append({"label": "Heart (red) and lungs (blue)"
                                             + (f", cardiothoracic ratio {anatomy['ctr']:.2f}" if anatomy.get("ctr") else ""),
                                             "file": self._save_picture(img, anatomy["overlay"], "anatomy")})
                if out.get("disagree"):                 # detector and MedGemma say opposite things (10 Nov, C5)
                    img["text"] += out["disagree"]
                    img["disagree"] = out["disagree"].strip()
                hint = charting.body_parts(n["text"])
                if hint:
                    img["text"] += f" The note mentions: {', '.join(hint)}."
                image_lines.append(f"{img['name']} ({out['kind']}): {img['text']}")
        if later and new:
            self._read_later_numbers(p, ai, new)
        apply_picture_rule(p)                       # vitals + words + pictures all feed the priority
        check_override(p)                           # a doctor's level stays; warn if the AI now says more urgent
        show_supported_pathways(p)
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
            p.report = charting.tidy_numbers(p.report)
            # the triage section always shows the trained models' decision, word for word
            p.report = charting.set_section(p.report, charting.TRIAGE, triage_lines(p))
            if first or not p.report_by:
                p.report_by = "AI"                  # a person's name stays once they've edited it
            p.visit.log("Case file", "Written by the AI from the notes" if first else
                        f"AI added {len(new)} new note(s); nothing already there was changed")
        self._step(p, "Thinking of questions and to-dos")
        f = p.fields
        typed = "\n".join(n["text"] for n in p.notes)
        arrival = {1.0: "ambulance", 0.0: "walk-in"}.get(f.get("ambulance"), "not recorded")
        context = (f"PATIENT: age {f.get('age', 'not recorded')}, {f.get('sex', 'sex not recorded')}, arrival {arrival}.\n"
                   + "\n".join(l.strip() for l in (p.visit.ed or "").splitlines() if "Pathway flags" in l))
        questions, todo = charting.ask_questions(ai, p.report, context)
        kinds = [i.get("kind") for n in p.notes for i in n["images"]]
        # every question and to-do, fixed or MedGemma's, must fit THIS patient: nothing the notes already answer,
        # nothing for the wrong age, no X-ray request when the X-ray is attached (10 Nov, B3-B4)
        fit = lambda items: charting.fit_patient(items, f, typed, kinds)
        questions, todo = fit(questions), fit(todo)
        # The flagged pathways' own questions come first (a stroke always asks for the time last known well), then
        # MedGemma's, steered by the same flags in `context`. With the real AI MedGemma alone asked "When did it
        # start?" for a stroke (9 Nov, 5q #6). At most 5, no two asking the same thing.
        fixed = fit(charting.pathway_questions(p.visit.ed))
        obs = charting.missing_obs_question(f)      # 3+ observations missing: ask for them first (10 Nov, B5)
        if obs:                                     # ... and not again one by one
            questions = charting.drop_obs_questions(questions)
        merged = []
        for q in ([obs] if obs else []) + fixed + questions:
            if not any(charting.same_meaning(q, m) for m in merged):
                merged.append(q)
        p.questions = merged[:5]
        have = [t["text"] for t in p.todos]
        flags_by = "ED rules (test mode)" if self.test else "AI pathway model"
        for text, by in [(t, flags_by) for t in fit(charting.pathway_todos(p.visit.ed, typed))] + [(t, "AI") for t in todo]:
            if not any(charting.same_meaning(text, h) for h in have):    # "ECG within 10 minutes" twice (5q #3)
                have.append(text)
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
                  key=lambda p: (p.stage in LEAVING, p.visit is None, p.shown_level() or 9, p.created))
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
        save_8bit(store.images / name)              # 16-bit X-rays turned white for the AI (10 Nov, C1)
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
    ai_read = p.found.get("ai_fields", p.fields)
    changed = sorted(k for k in set(ai_read) | set(clean) if k != "chief_complaint" and ai_read.get(k) != clean.get(k))
    if changed:
        store.learn("numbers_corrected", p, notes=[n["text"] for n in p.notes], ai_read=ai_read, corrected=clean,
                    changed=changed, who=body.who)
    new_visit = store.ai.triage(clean, explain=False)       # one model step: seconds, even while other jobs run
    with store.state_lock:
        before = p.visit
        before_fields = dict(p.fields or {})
        p.fields = clean
        p.found["checked_by"] = body.who
        # a nurse has looked: readings the two readers disagreed on are now settled
        p.found["by"] = {k: ("nurse" if v == "check" or k in changed else v) for k, v in p.found.get("by", {}).items()}
        p.found["check"] = []
        p.found["warnings"] = [w for w in p.found.get("warnings", []) if not w.startswith("please check")]
        p.visit = new_visit
        if before:                                  # keep everything learned since arrival; only the card + decision change
            p.visit.timeline, p.visit.results, p.visit.messages = before.timeline, before.results, before.messages
        apply_picture_rule(p, body.who)             # a numbers fix keeps the picture rule
        check_override(p)                           # ... and a doctor's level
        show_supported_pathways(p)
        p.visit.log("Triage", f"Numbers corrected; level {old} -> {p.visit.decision.level}", body.who)
        if p.report:
            # the Observations text still had the note's numbers after a fix (Gradio check 13 Nov: tile 82/60, text
            # "BP 90/60"): say what was corrected, by whom, keeping what was typed
            was = before_fields
            fixes = [f"{FIELD_NAMES.get(k, k)} {_show(clean.get(k))} (was {_show(was.get(k))})"
                     for k in FIELD_NAMES if _show(clean.get(k)) != _show(was.get(k))]
            if fixes:
                p.report = charting.add_line(p.report, "Observations", f"Corrected by {body.who}: " + ", ".join(fixes))
            p.report = charting.set_section(p.report, charting.TRIAGE, triage_lines(p))
    store.changed()
    return {"ok": True}


def _show(v):
    """A number as the nurse would write it (90.0 -> 90), or 'not recorded'."""
    if v in (None, ""):
        return "not recorded"
    return f"{v:g}" if isinstance(v, float) else str(v)


class ChatIn(BaseModel):
    text: str
    who: str


@app.post("/api/patients/{pid}/chat")
def chat(pid: str, body: ChatIn):
    p = store.get(pid)
    if p.visit is None:
        raise HTTPException(409, "The case file isn't ready yet")
    with store.chat_lock:
        if p.report and p.report != p.report_seen_by_chat:     # the chat always sees the latest case file
            p.visit.unseen.append("CURRENT CASE FILE:\n" + p.report)
            p.report_seen_by_chat = p.report
        # the times, written out, so time windows can be worked out (5q #7): with only the record, MedGemma said "the
        # time last known well is not stated" although the note said "since 0830"
        typed = " ".join(n["text"] for n in p.notes)
        clock = re.findall(r"\b(?:since|from|at|onset|lkw|last (?:seen|known)(?: to be)? well)\s*:?\s*"
                           r"(?:[01]?\d|2[0-3]):?[0-5]\d\b", typed, re.I)
        # ... and how long ago each was: given only "since 0830", MedGemma worked the window out backwards (9 Nov)
        times = (f"Time now: {now()}. Arrived: {p.notes[0]['time']}."
                 + (f" Times written in the notes: {'; '.join(f'{c} ({_ago(c)})' for c in clock)}." if clock else ""))
        done = charting.tests_done(p.notes[1:])
        if done:                                   # Caleb's retest 14 Nov: the T01 chat asked for an ECG + troponin already done
            times += f" Tests already done (results in the notes): {', '.join(f'{k} at {v}' for k, v in done.items())}."
        if charting.st_depression_only(typed):     # Caleb's final QC: "ST depression + trop 85" was called a STEMI
            times += " The ECG shows ST depression, not ST elevation (so not a STEMI)."
        answer = charting.mark_done_tests(store.ai.chat(p.visit, f"({times}) {body.text}"), done)
        answer = charting.fix_stemi(answer, typed)
    p.chat += [{"role": "user", "text": body.text, "who": body.who, "time": now()},
               {"role": "ai", "text": answer, "who": "AI", "time": now()}]
    p.visit.log("Chat", f"Q ({body.who}): {body.text}")
    store.changed()
    return {"answer": answer}


class WhoIn(BaseModel):
    who: str


@app.post("/api/patients/{pid}/sbar")
def write_sbar(pid: str, body: WhoIn):
    """Core task 3: an SBAR handover for the next doctor, written by MedGemma from everything gathered so far (the
    record, every later note and answer, the latest case file and the chat). A draft for a person to check."""
    p = store.get(pid)
    if p.visit is None or p.status == "working":
        raise HTTPException(409, "Wait until the AI has finished reading the notes")
    with store.chat_lock:                                     # same conversation as the chat, so it uses the chat
        if p.report and p.report != p.report_seen_by_chat:
            p.visit.unseen.append("CURRENT CASE FILE:\n" + p.report)
            p.report_seen_by_chat = p.report
        # full-AI check 14 Nov (T01): the handover had the ECG and GCS but dropped "trop 85" and asked for a troponin,
        # so every later note and answer is listed again right before the request
        # (2nd check: still "please obtain troponin" with "trop 85" listed: MedGemma didn't read "trop" as troponin)
        later = [f"- {n['time']} {n['who']}: " + re.sub(r"(?i)\btrop\b", "troponin", n["text"]) for n in p.notes[1:]]
        if later:
            p.visit.unseen.append("For the handover: results and answers added after arrival (put every one of "
                                  "them in the SBAR, and don't ask for a test that is already done):\n" + "\n".join(later))
        typed = " ".join(n["text"] for n in p.notes)
        if charting.st_depression_only(typed):
            p.visit.unseen.append("The ECG shows ST depression, not ST elevation: this is not a STEMI.")
        text = charting.strip_preamble(store.ai.sbar(p.visit))
        text = charting.fix_stemi(charting.mark_done_tests(text, charting.tests_done(p.notes[1:])), typed)
    # 3rd check: even with them listed, MedGemma dropped the troponin and the ECG in one run out of three. So they are
    # also copied word for word under the AI's text: nothing typed after arrival can go missing from the handover
    asked = [f"- {m['time']} {m['who']}: {m['text']}" for m in p.chat if m["role"] == "user"]
    if later:
        text += ("\n\n**Added after arrival (copied from the notes, not written by the AI)**\n"
                 + "\n".join(f"- {n['time']} {n['who']}: {n['text']}" for n in p.notes[1:]))
    if asked:
        text += "\n\n**Asked in the chat**\n" + "\n".join(asked)
    p.sbar = {"text": text, "who": body.who, "time": now(), "n_notes": len(p.notes), "n_chat": len(p.chat)}
    p.visit.log("Handover", f"SBAR handover written by the AI (asked by {body.who}); for a person to check")
    store.changed()
    return p.sbar


@app.get("/api/patients/{pid}/sbar.txt", response_class=PlainTextResponse)
def sbar_text(pid: str):
    p = store.get(pid)
    s = getattr(p, "sbar", {})
    if not s:
        raise HTTPException(404, "No SBAR handover written yet")
    return (f"SBAR handover: {p.bed} ({_who_line(p.fields)})\nWritten by the AI at {s['time']}, asked by {s['who']}. "
            f"AI draft: check before use.\n\n{s['text']}\n")


def _ago(written, clock=None):
    """'since 0830' -> 'about 2 h 15 min ago' (a time later than now is taken as yesterday)."""
    m = re.search(r"(\d{1,2}):?(\d{2})$", written)
    if not m:
        return "time not clear"
    t = time.localtime() if clock is None else clock
    mins = (t.tm_hour * 60 + t.tm_min) - (int(m.group(1)) * 60 + int(m.group(2)))
    mins += 24 * 60 if mins < 0 else 0
    return f"about {mins // 60} h {mins % 60} min ago"


class BoardIn(BaseModel):
    who: str
    role: str
    stage: str | None = None
    doctor: str | None = None
    cleared: bool | None = None
    diagnosis: str | None = None                    # final diagnosis, typed when the patient leaves (optional)
    final_level: int | None = None                  # the level the clinician judges was right (optional)


def _editor(role, who=None):
    if role not in BOARD_EDITORS:
        raise HTTPException(403, "Only the triage nurse or the organising doctor can change the board")
    if who is not None and who.strip().lower() in ("", "not recorded"):     # an unnamed edit was saved (5q #9)
        raise HTTPException(400, "Type your name (top right) first, so the timeline shows who changed the board")


def _log(p, what, text, who):
    """On the timeline; before the AI's first result there is no timeline yet, so it waits in p.early_log (an
    unnamed board change on a queued patient was saved but never logged, 5q #9)."""
    if p.visit:
        p.visit.log(what, text, who)
    else:
        p.early_log = getattr(p, "early_log", []) + [{"time": now(), "who": who, "what": what, "text": text}]


@app.patch("/api/patients/{pid}/board")
def set_board(pid: str, body: BoardIn):
    _editor(body.role, body.who)
    p = store.get(pid)
    if body.final_level is not None and not 1 <= body.final_level <= 5:
        raise HTTPException(400, "level must be 1-5")
    change = []
    if body.stage and body.stage != p.stage:
        if body.stage not in STAGES:
            raise HTTPException(400, "unknown stage")
        change.append(f"stage {p.stage} -> {body.stage}")
        p.stage = body.stage
        p.stage_since = time.time()
    if body.doctor is not None and body.doctor.strip() and body.doctor.strip() != p.doctor:
        change.append(f"doctor {p.doctor} -> {body.doctor.strip()}")
        p.doctor = body.doctor.strip()
    if body.cleared is not None and body.cleared != p.cleared:
        change.append("cleared from the board (left the ED)" if body.cleared else "put back on the board")
        p.cleared = body.cleared
        if body.cleared:
            _log_outcome(p, body)
    if change:
        _log(p, "Board", "; ".join(change), body.who)
    store.changed()
    return {"ok": True}


class PriorityIn(BaseModel):
    who: str
    role: str
    level: int | None = None
    reason: str = ""
    clear: bool = False                             # back to the AI + safety rules' level


@app.put("/api/patients/{pid}/priority")
def set_priority(pid: str, body: PriorityIn):
    """A doctor changes the priority (lower or raise) with a reason (14 Nov). The AI never lowers by itself; its
    decision stays underneath, unchanged, and both are logged for learning."""
    if body.role not in DOCTOR_ROLES:
        raise HTTPException(403, "Only a doctor (Doctor or Organising doctor) can change the priority")
    if body.who.strip().lower() in ("", "not recorded"):
        raise HTTPException(400, "Type your name (top right) first, so the timeline shows who changed the priority")
    p = store.get(pid)
    if p.visit is None or p.status == "working":
        raise HTTPException(409, "Wait until the AI has finished reading the notes")
    d, who = p.visit.decision, body.who.strip()
    if body.clear:
        if getattr(p, "override", {}):
            p.override = {}
            p.visit.log("Priority", f"Back to the AI + safety rules' level ({d.level})", who)
    else:
        if body.level is None or not 1 <= body.level <= 5:
            raise HTTPException(400, "Pick a level from 1 to 5")
        p.override = {"level": body.level, "reason": body.reason.strip(), "who": who, "role": body.role, "time": now(),
                      "ai_level": d.level, "ai_flagged": d.flagged, "ai_p": d.p_urgent, "ai_floor": d.floor_priority,
                      "ai_news2": d.news2,
                      "ai_picture": getattr(d, "picture", "")}
        p.visit.log("Priority", f"Level set to {body.level} ({'urgent' if body.level <= 2 else 'not urgent'}) by {who}"
                    f"{': ' + body.reason.strip() if body.reason.strip() else ''}. AI + safety rules said {d.level}"
                    f"{' (flagged urgent)' if d.flagged else ''}", who)
        store.learn("priority_override", p, doctor_level=body.level, reason=body.reason.strip(), role=body.role,
                    ai_level=d.level, ai_flagged=d.flagged, model_level=d.model_level or d.level,
                    model_flagged=d.model_flagged, p_urgent=d.p_urgent, floor_priority=d.floor_priority,
                    picture_rule=getattr(d, "picture", ""), fields={k: v for k, v in p.fields.items()
                                                                    if k != "chief_complaint"},
                    complaint=p.complaint(), who=who)
    if p.report:
        p.report = charting.set_section(p.report, charting.TRIAGE, triage_lines(p))
    store.changed()
    return {"ok": True, "level": p.shown_level()}


def _log_outcome(p, body):
    """The patient left the ED: one labelled case for learning.py (what the AI said at arrival vs what happened)."""
    d = p.visit.decision if p.visit else None
    raw = (getattr(p.visit, "ed_model", None) or p.visit.ed) if p.visit else ""     # the model's flags, for learning
    flags = re.search(r"Pathway flags: (.*)", raw or "")
    store.learn("outcome", p, stage=p.stage, diagnosis=(body.diagnosis or "").strip(), final_level=body.final_level,
                fields={k: v for k, v in p.fields.items() if k != "chief_complaint"}, complaint=p.complaint(),
                ai_level=d.level if d else None, p_urgent=d.p_urgent if d else None, flagged=d.flagged if d else None,
                floor_priority=d.floor_priority if d else None, pathway_flags=flags.group(1) if flags else "",
                # the models' own decision, apart from the picture rule, so the urgency score is judged on its own
                model_flagged=d.model_flagged if d else None, model_level=(d.model_level or d.level) if d else None,
                picture_rule=getattr(d, "picture", "") if d else "",
                doctor_level=getattr(p, "override", {}).get("level"),     # a doctor's priority, if one was set
                ai_todos=[{"text": t["text"], "done": t["done"]} for t in p.todos if t.get("ai")], who=body.who)
    if store.learning is not None and p.visit is not None:
        store.learning.save_vector(p.id, getattr(p.visit, "vector", None))
    if p.visit:
        p.visit.log("Outcome", f"final diagnosis: {body.diagnosis or 'not given'}; level judged right: "
                    f"{body.final_level or 'not given'}", body.who)


@app.get("/api/learning")
def learning_status():
    """What has been collected for learning, and how the AI is doing in real use (from the corrections)."""
    if store.learning is None:
        return {"enabled": False}
    return {"enabled": True, **store.learning.summary()}


class TodoIn(BaseModel):
    who: str
    role: str
    text: str | None = None
    done: bool | None = None
    delete: bool = False


@app.post("/api/patients/{pid}/todos")
def add_todo(pid: str, body: TodoIn):
    _editor(body.role, body.who)
    p = store.get(pid)
    if not (body.text or "").strip():
        raise HTTPException(400, "empty to-do")
    p.todos.append({"text": body.text.strip(), "done": False, "ai": False, "by": body.who})
    store.changed()
    return {"ok": True}


@app.patch("/api/patients/{pid}/todos/{i}")
def change_todo(pid: str, i: int, body: TodoIn):
    _editor(body.role, body.who)
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
    if t.get("ai"):                                 # monitoring only: ticking an AI to-do is not proof it was needed
        store.learn("todo", p, text=t["text"], by=t.get("by", "AI"),
                    action="removed" if body.delete else ("done" if t["done"] else "not done"), who=body.who)
    _log(p, "To do", what, body.who)
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
