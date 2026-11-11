"""One patient's visit: the frozen arrival card + everything the ED learns later. No model needed here.

    v = Visit(card, decision, ed="ED ASSESSMENT ...")
    v.add("bloods", "Troponin 2.1 ng/mL (high)")
    print(v.record())                  # what the chat and SBAR read

Why two stages: the urgency and level models were trained on the arrival card only, so `v.card` is never changed
and `v.decision` is never recomputed. Later results go in the record, for MedGemma to use in the chat and SBAR.

The case file (6 Nov 2026): every entry also goes on `v.timeline` with the time and who added it, so the file can
follow the patient from doctor to doctor and to the ward. `v.case_file()` is the whole file as text.
"""
import re
import time
from dataclasses import dataclass, field

from prompts import urgency_line

END = "=== END RECORD ==="
# key -> heading on the record. The first group already has a line on the arrival card.
SECTIONS = {"history": "HISTORY OF PRESENTING COMPLAINT", "past_history": "PAST HISTORY", "medications": "MEDICATIONS",
            "allergies": "ALLERGIES", "imaging": "IMAGING", "notes": "CONSULTATION NOTES",
            "bedside": "BEDSIDE CHECKS", "ecg": "ECG", "bloods": "BLOOD TESTS", "urine_swabs": "URINE AND SWABS",
            "treatment": "TREATMENT GIVEN", "disposition": "DISPOSITION"}
NOTE_SECTIONS = {"history", "past_history", "medications", "allergies"}     # usually written inside the notes
SEE_NOTES = "not filled in separately: read the CONSULTATION NOTES"


CORE_NAMES = {"rr": "RR", "sbp": "BP", "hr": "HR"}        # Caleb's rule (vitals/src/news2.py CORE)
VITAL_NAMES = {"rr": "RR", "spo2": "SpO2", "sbp": "BP", "hr": "HR", "temp_c": "temperature", "consciousness": "consciousness"}


def _and(words):
    words = list(words)
    return " and ".join([", ".join(words[:-1]), words[-1]] if len(words) > 1 else words)


def floor_facts(vitals_out):
    """What set Caleb's safety floor, read from his model's own output (his files are not changed):
    missing = the core vital signs (BP, RR, HR) not recorded, when that alone is why the floor is 1-2;
    abnormal = the vital signs scoring 2 or 3 NEWS2 points, when the floor is 1-2."""
    if vitals_out.get("floor_priority", 5) > 2:
        return {"missing": (), "abnormal": ()}
    reason = vitals_out.get("escalation_reason") or ""
    only_missing = bool(vitals_out.get("escalated")) and "could not be recorded" in reason and "unresponsive" not in reason
    missing = tuple(CORE_NAMES[k] for k in ("sbp", "rr", "hr") if k in vitals_out.get("missing", ())) if only_missing else ()
    abnormal = tuple(VITAL_NAMES[k] for k, pts in (vitals_out.get("subscores") or {}).items() if pts >= 2 and k in VITAL_NAMES)
    if "unresponsive" in reason:
        abnormal = ("unresponsive",) + tuple(a for a in abnormal if a != "consciousness")
    return {"missing": missing, "abnormal": abnormal}


@dataclass(frozen=True)
class Decision:
    """The front-door result. Frozen: later results can't change it."""
    p_urgent: float
    head_flag: bool
    floor_flag: bool
    flagged: bool
    level: int                     # ML level, made safe: at most 2 when flagged
    level_set: tuple               # conformal set of possible levels (90% coverage)
    floor_priority: int
    news2: int
    model_level: int = 0           # the level model's own answer, before the urgent flag raised it (0 = not stored)
    missing: tuple = ()            # core vital signs not recorded, when that alone set the safety floor (floor_facts)
    abnormal: tuple = ()           # vital signs scoring 2-3 NEWS2 points, when the floor is 1-2

    @property
    def needs_obs(self):
        """Urgent ONLY because core vital signs were not measured (Caleb's rule: 2 or more of RR / BP / HR missing
        = "could not be measured" = treat as high risk). Not the same as a sick patient: the screens say
        "Urgent: needs vitals" in amber and name what to measure (group test 10 Nov, A1: P05, T09)."""
        # the urgency score may flag as well (T09 "feeling unwell, no obs": 24%), but with the core vital signs
        # missing it rests on the complaint words alone, so the first thing to do is still to measure them
        return bool(self.floor_flag and self.missing)

    def floor_why(self):
        """What set the safety floor, for the record the chat and the case file read."""
        if self.missing:
            return f"{_and(self.missing)} not recorded, so the patient could not be scored: measure them"
        return f"abnormal {_and(self.abnormal)}, NEWS2 {self.news2}" if self.abnormal else ""

    def reason(self):
        """A few words for the whiteboard: why this patient is flagged ('' if not flagged)."""
        if not self.flagged:
            return ""
        if self.needs_obs:
            return (f"Needs vitals: {_and(self.missing)} not recorded"
                    + (f" (urgency score {self.p_urgent:.0%} also flags)" if self.head_flag else ""))
        vitals = f"abnormal {_and(self.abnormal)}" if self.abnormal else "vital-signs safety rule"
        score = f"urgency score {self.p_urgent:.0%}"
        text = f"{score} + {vitals}" if self.head_flag and self.floor_flag else score if self.head_flag else vitals
        return text[0].upper() + text[1:]

    def raised_by(self):
        """Why the level is more urgent than the level model said (7 Nov: the safety rule causes most over-triage,
        so the screen says so and the clinician can judge). '' when it wasn't raised."""
        own = self.model_level or self.level
        if not self.flagged or own <= self.level:
            return ""
        if self.needs_obs and not self.head_flag:
            return (f"Level model said {own}; held at {self.level} until {_and(self.missing)} "
                    f"{'is' if len(self.missing) == 1 else 'are'} measured (safety rule for missing vital signs).")
        why = ("the vital-signs safety floor" if self.floor_flag and not self.head_flag else
               "the urgency score" if self.head_flag and not self.floor_flag else "the urgency score and the vital-signs safety floor")
        return f"Level model said {own}; raised to {self.level} by {why} (set to catch 90% of urgent patients)."

    def lines(self):
        options = ", ".join(str(x) for x in self.level_set)
        out = [urgency_line(self.p_urgent, self.flagged, self.floor_flag, self.floor_why()),
               f"TRIAGE LEVEL (trained model): {self.level} (levels it can't rule out: {options})."]
        return out + ([self.raised_by()] if self.raised_by() else [])


@dataclass
class Visit:
    card: str                      # arrival card, exactly as the models saw it
    decision: Decision
    ed: str = ""                   # ED ASSESSMENT text (rules applied to the arrival data)
    explanation: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)      # key -> list of entries
    messages: list = field(default_factory=list)     # the conversation so far
    unseen: list = field(default_factory=list)       # results MedGemma hasn't been told about in the chat yet
    timeline: list = field(default_factory=list)     # the case file: {time, who, what, text}, oldest first
    vector: object = None          # MedGemma's vector of the card (real AI only), kept so learning.py can retrain on it

    def log(self, what, text, who="AI (decision support)"):
        """A case-file entry that doesn't change the record, e.g. a chat answer or a handover."""
        self.timeline.append({"time": time.strftime("%d %b %H:%M"), "who": who, "what": what, "text": text})

    def add(self, key, text, who="not recorded"):
        if key not in SECTIONS:
            raise ValueError(f"unknown section '{key}', use one of {list(SECTIONS)}")
        self.results.setdefault(key, []).append(text)
        self.unseen.append(f"{SECTIONS[key]}: {text}")
        self.log(SECTIONS[key].capitalize(), text, who)

    def case_file(self, title=""):
        """The whole file as text: the timeline, then the current record."""
        lines = [f"CASE FILE {title}".strip(), "", "TIMELINE (oldest first)"]
        lines += [f"{e['time']} | {e['who']} | {e['what']}: {e['text']}" for e in self.timeline]
        return "\n".join(lines + ["", "CURRENT RECORD", self.record()])

    def record(self):
        """Arrival card + later results + the front-door decision. Missing information says 'not recorded'.
        When there are typed notes, the card's empty history lines point to them instead of saying 'not recorded':
        MedGemma copied 'PAST HISTORY: not recorded' over a note that said 'hx DM, HTN on metformin. NKDA'
        (real-AI test, 9 Nov, REPORT_NOTES 5q #1). Only the record changes; the card the urgency model reads doesn't."""
        body = self.card.replace(END, "").rstrip()
        extra = [self.ed] if self.ed else []
        has_notes = bool(self.results.get("notes"))
        for key, heading in SECTIONS.items():
            value = " | ".join(self.results.get(key, []))
            line = re.compile(rf"^{heading}: .*$", re.M)
            if line.search(body):
                if value:                  # a lambda: the text may contain backslashes
                    body = line.sub(lambda _: f"{heading}: {value}", body)
                elif has_notes and key in NOTE_SECTIONS:
                    body = line.sub(lambda _: f"{heading}: {SEE_NOTES}", body)
            else:
                extra.append(f"{heading}: {value or 'not recorded'}")
        if self.timeline:                  # the chat can then work out time windows (stroke, chest pain)
            body = body.replace("Time: not recorded", f"Time: arrived {self.timeline[0]['time']}", 1)
        return "\n".join([body, *extra, *self.decision.lines(), END])
