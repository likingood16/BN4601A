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

    def raised_by(self):
        """Why the level is more urgent than the level model said (7 Nov: the safety rule causes most over-triage,
        so the screen says so and the clinician can judge). '' when it wasn't raised."""
        own = self.model_level or self.level
        if not self.flagged or own <= self.level:
            return ""
        why = ("the vital-signs safety floor" if self.floor_flag and not self.head_flag else
               "the urgency score" if self.head_flag and not self.floor_flag else "the urgency score and the vital-signs safety floor")
        return f"Level model said {own}; raised to {self.level} by {why} (set to catch 90% of urgent patients)."

    def lines(self):
        options = ", ".join(str(x) for x in self.level_set)
        out = [urgency_line(self.p_urgent, self.flagged, self.floor_flag),
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
        """Arrival card + later results + the front-door decision. Missing information says 'not recorded'."""
        body = self.card.replace(END, "").rstrip()
        extra = [self.ed] if self.ed else []
        for key, heading in SECTIONS.items():
            value = " | ".join(self.results.get(key, []))
            line = re.compile(rf"^{heading}: .*$", re.M)
            if line.search(body):
                if value:                  # a lambda: the text may contain backslashes
                    body = line.sub(lambda _: f"{heading}: {value}", body)
            else:
                extra.append(f"{heading}: {value or 'not recorded'}")
        return "\n".join([body, *extra, *self.decision.lines(), END])
