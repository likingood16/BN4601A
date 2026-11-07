"""Chart first, AI second (APP_DESIGN.md, 6 Nov 2026): the nurse types a messy note, this file turns it into
the pieces the app needs.

    fields, found = extract_numbers("67M chest pain since 6am, HR 112 BP 90/60 sats 94% RA, T 37.2, pain 8/10, came by ambulance")
    report = build_report(ai, visit.record(), current_report, new_notes)
    questions, todo = ask_questions(ai, report)

Numbers are pulled out with fixed rules, not by MedGemma: they feed the urgency model, so they must be predictable,
testable and shown to the nurse for checking. MedGemma (`ai` = an EDPipeline) writes the complaint line, the case
file and the questions. With ai=None (test mode) simple rules stand in, marked TEST MODE.
"""
import json
import re

from model import _messages

MARK = "[TEST MODE, not a real answer]"
HEADINGS = ["Presenting complaint", "History", "Past history, medications, allergies", "Observations",
            "Examination", "Investigations", "Imaging", "Impression (as written by clinicians)", "Plan and actions",
            "Triage decision support"]

TRIAGE = "Triage decision support"            # filled from the trained models' decision, never written by MedGemma
REPORT_SYSTEM = (
    "You are an emergency department documentation assistant. You turn clinicians' quick, messy notes into a tidy "
    "case file. Use ONLY facts written in the notes and the patient record. Never add findings, results, diagnoses or "
    "treatments that are not written there. Expand obvious abbreviations. Split the note into the right sections: "
    "never copy the whole note into one section. 'Impression' is only a diagnosis or impression a clinician wrote; "
    "otherwise write 'not recorded'. If a section has nothing, write 'not recorded'. "
    "Write markdown with exactly these headings, in this order, each as '## Heading', with short bullet points:\n"
    + "\n".join(h for h in HEADINGS if h != TRIAGE))
QUESTION_SYSTEM = (
    "You support emergency department clinicians. Read the case file. Reply with JSON only, in this form: "
    '{"questions": ["..."], "todo": ["..."]}. '
    '"questions": up to 4 questions about missing information that would change care for THIS complaint, most '
    "important first. Never ask for something the case file already has. "
    '"todo": up to 4 tasks that probably need doing or sending for THIS patient (tests, treatments, referrals), most '
    "urgent first. Each item at most 12 words. These are suggestions for clinicians to accept or ignore.")

# ---------------------------------------------------------------- numbers
RANGES = {"hr": (20, 250), "rr": (4, 70), "spo2": (50, 100), "temp_c": (30, 44), "sbp": (40, 300), "dbp": (20, 200),
          "nrs_pain": (0, 10), "age": (0, 120)}
# Between a long label and its number, case reports put a few known words: "heart rate was stable at 80",
# "blood pressure on arrival was 113/67", "pulse rate, 136". Only these words are skipped, so "respiratory distress
# for 48 hours" or "pulse oximetry 92%" can't be read as a rate. Short labels (HR, BP, T) stay tight.
FILL = (r"(?:\s*[:=,(]\s*|\s+)(?:(?:was|were|is|of|at|on|in|the|arrival|admission|presentation|stable|measured|recorded|"
        r"initially|around|approximately|about|elevated|raised|low|high|reduced|increased|decreased|to)[\s,]+){0,4}")
# Numbers end at a non-digit, not a word boundary, so units stuck on still work: "120/80mmHg", "110bpm", "98%".
# Messy-shorthand check on the train patients (7 Nov): "bp91/67mmHg", "p70", "pain 4", "37,0" and "O2 98" were missed.
END = r"(?!\d)"
PATTERNS = {             # several ways of writing each; the earliest match in the note wins
    "hr": [r"\b(?:hr|pr|pulse|p)\s*(?:of|is|was|:|=|-)?\s*(\d{2,3})" + END,
           r"\b(?:heart rate|pulse rate|pulse(?!\s*ox))" + FILL + r"(\d{2,3})" + END, r"\b(\d{2,3})\s*(?:bpm|beats)"],
    "rr": [r"\b(?:rr|resp rate|resp)\s*(?:of|is|was|:|=|-)?\s*(\d{1,2})" + END, r"\brespiratory rate" + FILL + r"(\d{1,2})" + END,
           r"\b(\d{1,2})\s*breaths"],
    "spo2": [r"\b(?:spo2|sp02|sao2|o2 sats?|sats?|saturations?|saturating|o2)\s*(?:of|is|was|:|=|-)?\s*(\d{2,3})" + END + r"\s*%?",
             r"\b(?:oxygen saturation|o2 saturation|spo2|saturation)" + FILL + r"(\d{2,3})\s*%"],
    "temp_c": [r"\b(?:temp|t)\s*(?:of|is|was|:|=|-)?\s*(\d{2,3}(?:[.,]\d)?)" + END + r"\s*°?\s*(?:c|f)?\b",
               r"\b(?:body temperature|temperature|temp)" + FILL + r"(\d{2,3}(?:[.,]\d)?)" + END
               + r"\s*°?\s*(?:c|f|celsius|fahrenheit)?\b"],
    "nrs_pain": [r"\bpain(?: score)?\s*(?:of|is|:|=)?\s*(\d{1,2})\s*/\s*10\b|\b(\d{1,2})\s*/\s*10\b",
                 r"\b(?:pain(?: score)?|nrs)\s*(?:of|is|:|=)?\s*(\d{1,2})(?!\s*(?:/|days?|d\b|hours?|(?:hrs?|h)\b(?![:=\s]*\d)|weeks?|wks?|mins?|x)|\d|[.,]\d)"],
}
BP = r"\b(?:bp|blood pressure|arterial pressure)" + r"(?:" + FILL + r")?" + r"(\d{2,3})\s*/\s*(\d{2,3})" + END
AGE_SEX = [r"(?<![\d.])\b(\d{1,3})\s*(?:yo|y/o|y\.o\.|yrs?|years?(?: old)?|-year-old)?\s*-?\s*(m|f|male|female|man|woman|boy|girl)\b",
           r"\b(m|f|male|female)\s*,?\s*(\d{1,3})\b"]
AMBULANCE = r"\b(ambulance|ems|paramedics?|995|scdf|by amb|biba|bibs)\b"
WALK_IN = r"\b(walk[- ]?in|walked in|self[- ]?present\w*|brought (?:in )?by (?:family|friend|wife|husband|son|daughter|mother|father))\b"
INJURY = r"\b(fall|fell|injur\w*|trauma|mva|rta|accident|lacerat\w*|cut|fracture\w*|assault\w*|burns?|hit by|dog bite|sprain\w*)\b"
SEX = {"m": "male", "male": "male", "man": "male", "boy": "male", "f": "female", "female": "female", "woman": "female", "girl": "female"}


def _consciousness(text):
    """AVPU word for Caleb's model. New confusion scores like 'V' in NEWS2 (3 points), so it maps to 'verbal'."""
    m = re.search(r"\bavpu\s*(?:is|:|=)?\s*([avpu])\b", text)
    if m:
        return {"a": "alert", "v": "verbal", "p": "pain", "u": "unresponsive"}[m.group(1)], m.group(0)
    m = re.search(r"\bgcs\s*(?:of|is|:|=)?\s*(\d{1,2})\b", text)
    if m and 3 <= int(m.group(1)) <= 15:
        g = int(m.group(1))
        return ("alert" if g == 15 else "verbal" if g >= 9 else "pain" if g >= 4 else "unresponsive"), m.group(0)
    for pattern, word in [(r"\b(unresponsive|unconscious)\b", "unresponsive"), (r"\bresponds? (?:only )?to pain\b", "pain"),
                          (r"\b(confused|drowsy|responds? to voice)\b", "verbal"), (r"\b(?<!not )alert\b", "alert")]:
        m = re.search(pattern, text)
        if m:
            return word, m.group(0)
    return None, None


def extract_numbers(note):
    """Messy note -> Pool 1 fields for the urgency model, plus what text each came from (shown for checking).
    Anything not found is left out ('not recorded'); numbers outside a possible range are dropped with a warning."""
    text = note.lower()
    fields, found, warnings = {}, {}, []

    def keep(key, value, source):
        lo, hi = RANGES.get(key, (None, None))
        if lo is not None and not lo <= value <= hi:
            warnings.append(f"ignored {source.strip()!r}: {key} {value:g} is outside {lo}-{hi}")
            return
        fields[key], found[key] = value, source.strip()

    for key, patterns in PATTERNS.items():
        hits = [m for m in (re.search(p, text) for p in patterns) if m]
        m = min(hits, key=lambda h: h.start()) if hits else None
        if m:
            value = float(next(g for g in m.groups() if g is not None).replace(",", "."))   # "37,5" = 37.5
            if key == "temp_c" and value > 45:          # written in Fahrenheit
                value = round((value - 32) * 5 / 9, 1)
            keep(key, value, m.group(0))
    m = re.search(BP, text)
    if m:
        keep("sbp", float(m.group(1)), m.group(0))
        keep("dbp", float(m.group(2)), m.group(0))
    hits = [(m, i) for i, pattern in enumerate(AGE_SEX) for m in [re.search(pattern, text)] if m]
    if hits:                                        # the earliest one: age and sex usually open the note
        m, i = min(hits, key=lambda h: h[0].start())
        age, sex = (m.group(1), m.group(2)) if i == 0 else (m.group(2), m.group(1))
        keep("age", float(age), m.group(0))
        fields["sex"], found["sex"] = SEX[sex], m.group(0).strip()
    word, source = _consciousness(text)
    if word:
        fields["mental"], found["mental"] = word, source
    m = re.search(AMBULANCE, text) or re.search(WALK_IN, text)
    if m:
        fields["ambulance"] = 1.0 if re.fullmatch(AMBULANCE, m.group(0)) else 0.0
        found["ambulance"] = m.group(0)
    m = re.search(INJURY, text)
    if m:                  # no injury word is not proof of no injury (e.g. "chin pain"), so it stays 'not recorded'
        fields["injured"], found["injured"] = 1.0, m.group(0)
    return fields, {"found": found, "warnings": warnings}


BODY_PARTS = ["head", "neck", "chest", "rib", "abdomen", "back", "pelvis", "hip", "shoulder", "arm", "elbow", "forearm",
              "wrist", "hand", "finger", "thumb", "leg", "thigh", "knee", "ankle", "foot", "toe", "spine", "skull", "face"]


def body_parts(note):
    """Body parts the note mentions, so the image reading can be matched to the complaint. The image reader itself
    reads the picture blind (no note): telling it what to expect can make it 'see' it (anchoring)."""
    low = note.lower()
    return [b for b in BODY_PARTS if re.search(rf"\b{b}s?\b", low)]


SENTENCE = r"(?<!\d)\.(?!\d)|\.(?=\s)|[;\n]"          # sentence breaks, but not the dot in 37.9


def first_clause(note):
    """Test-mode complaint: the first piece of the note that isn't just age/sex, numbers or how they arrived."""
    for part in re.split(SENTENCE + "|,", note):
        words = re.sub("|".join([AGE_SEX[0], AGE_SEX[1], BP, *(p for ps in PATTERNS.values() for p in ps)]), " ", part, flags=re.I).strip(" -:")
        arrival = re.search(AMBULANCE, words, re.I) or re.search(WALK_IN, words, re.I)
        if arrival and len(words.split()) <= len(arrival.group(0).split()) + 3:
            continue                                   # "brought by ambulance" is not a complaint
        words = re.sub(r"\s+", " ", words).strip()
        words = re.sub(r"^(?:an?\s+)?(?:(?:who|that)\s+)?(?:presented|presenting|presents)?\s*(?:with\s+)?", "", words, flags=re.I)
        if re.fullmatch(NOT_COMPLAINT, words, re.I):
            continue                                   # "NRS", "AVPU A", "alert", "nan" are not complaints
        if re.search(r"[a-z]{3}", words, re.I):
            return words if len(words) <= 60 else words[:60].rsplit(" ", 1)[0] + "…"
    return "not recorded"


NOT_COMPLAINT = r"(?:avpu\s*[avpu]?|gcs\s*\d*|nrs|alert|nan|none|nil|pain score|pain|walk-?in|biba|ra|on air)(?:\W+|$)+"


def complaint(ai, note):
    """The presenting complaint in a few words, for the arrival card the urgency model reads."""
    if ai is None:
        return first_clause(note)
    from answers import final_answer
    q = (f"Nurse's note:\n{note}\n\nWhat is the presenting complaint, in 8 words or fewer? Symptoms or problem only: "
         "no age, sex, arrival mode, vital signs or pain score. Reply with the words only.")
    text = (final_answer(ai._ask(_messages(None, q), "complaint")) or "").strip().strip(".").splitlines()
    words = re.sub(r"\(injury\)|\b(?:walk-?in|walked in|biba|self-presented|brought (?:in )?by \w+)\b", "",
                   text[0] if text else "", flags=re.I).strip(" ,;.-")
    return words[:80] if words else first_clause(note)


# ---------------------------------------------------------------- case file
SORT_RULES = [
    ("Impression (as written by clinicians)", r"\bimpression\b|\bimp\b|\bdx\b|diagnos|\blikely\b|suspected|\?\s?[a-z]{4,}"),
    ("Imaging", r"x-?ray|\bxr\b|\bct\b|mri|ultrasound|\bus\b|scan|image"),
    ("Investigations", r"trop|ecg|\bhb\b|wbc|crp|lactate|glucose|cbg|bloods?|\bfbc\b|u/?e|\bna\b|\bk\b|urine|swab|abg|vbg"),
    ("Plan and actions", r"\bgave\b|given|start(ed)?|\biv\b|plan|refer|admit|discharge|review|send|stat\b|neb|analgesi"),
    ("Observations", r"\bhr\b|\bbp\b|\brr\b|sats|spo2|temp|gcs|avpu|pain \d"),
    ("Past history, medications, allergies", r"\bhx\b|pmh|history of|known|\bdm\b|\bhtn\b|\bon (?!air)\w+|meds|allerg|nkda"),
    ("Examination", r"o/e|on exam|tender|chest clear|crackles|wheeze|abdo|soft|pupils|neuro"),
]


def _test_report(record, notes, images, complaint=True):
    """TEST MODE stand-in: each sentence of the notes goes under the first heading whose words it matches."""
    sections = {h: [] for h in HEADINGS}
    if complaint:
        sections["Presenting complaint"].append(first_clause(notes[0]) if notes else "not recorded")
    for note in notes:
        for piece in filter(None, (p.strip() for p in re.split(SENTENCE, note))):
            head = next((h for h, rule in SORT_RULES if re.search(rule, piece, re.I)), "History")
            sections[head].append(piece)
    sections["Imaging"] += images
    decision = [line for line in record.splitlines() if line.startswith(("URGENCY", "TRIAGE LEVEL"))]
    sections["Triage decision support"] += decision
    body = "\n\n".join(f"## {h}\n" + ("\n".join(f"- {x}" for x in xs) if xs else "- not recorded")
                       for h, xs in sections.items())
    return f"_{MARK} sorted by keyword rules, MedGemma not loaded._\n\n{body}"


def _split(report):
    """Case file markdown -> (text before the first heading, {heading: [lines]}) in order."""
    head, sections, current = [], {}, None
    for line in report.splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            sections.setdefault(current, [])
        elif current is None:
            head.append(line)
        elif line.strip():
            sections[current].append(line)
    return "\n".join(head).strip(), sections


def _empty(line):
    """A placeholder line: 'not recorded', 'none', 'Investigations not recorded.' ..."""
    t = line.strip().lstrip("-*• ").strip().rstrip(".").lower()
    return t in ("", "not recorded", "none", "none recorded", "nil", "n/a", "not done") or (t.endswith("not recorded") and len(t) < 45)


def set_section(report, heading, lines):
    """Replace one section's lines (the triage section, which comes from the trained models' decision)."""
    pre, sections = _split(report)
    sections.pop(heading, None)
    sections[heading] = [f"- {l}" for l in lines]                 # always the last section
    body = "\n\n".join(f"## {h}\n" + ("\n".join(ls) if ls else "- not recorded") for h, ls in sections.items())
    return f"{pre}\n\n{body}" if pre else body


def file_notes(new_notes, new_images=()):
    """Later notes, filed under the headings by fixed word rules (not by MedGemma): every word typed lands in the case
    file, nothing is reworded or invented. Image readings always go under Imaging."""
    sections = {h: [] for h in HEADINGS if h != TRIAGE}
    for note in new_notes:
        for piece in filter(None, (p.strip() for p in re.split(SENTENCE, note))):
            head = next((h for h, rule in SORT_RULES if re.search(rule, piece, re.I)), "History")
            sections[head].append(piece)
    sections["Imaging"] += list(new_images)
    return "\n\n".join(f"## {h}\n" + "\n".join(f"- {x}" for x in xs) for h, xs in sections.items() if xs)


def merge_into(current, addition):
    """Add the new bullets under the same headings of the current case file. Nothing already in it is changed or
    removed, so a person's edits are never lost; a 'not recorded' placeholder is replaced by the first real entry."""
    pre, cur = _split(current)
    _, add = _split(addition)
    for heading, lines in add.items():
        if heading == TRIAGE:
            continue                                   # comes from the decision (set_section), not from notes
        new = [l for l in lines if not _empty(l)]
        if not new:
            continue
        base = heading.lower().split(" (")[0]          # "Impression" matches "Impression (as written by clinicians)"
        match = next((h for h in cur if h.lower().split(" (")[0] == base), heading)
        have = cur.setdefault(match, [])
        if all(_empty(l) for l in have):
            have.clear()
        have += [l for l in new if l not in have]
    body = "\n\n".join(f"## {h}\n" + ("\n".join(ls) if ls else "- not recorded") for h, ls in cur.items())
    return f"{pre}\n\n{body}" if pre else body


def build_report(ai, record, current="", new_notes=(), notes=(), images=(), new_images=()):
    """The tidy case file. First time: written from the whole record. After that, only the NEW notes are filed under
    the headings and merged in (merge_into), so the case file is never rewritten and people's edits are kept."""
    from answers import final_answer
    if not current:
        if ai is None:
            return _test_report(record, list(notes), list(images))
        q = "PATIENT RECORD (arrival card, notes, image readings, triage decision):\n" + record + "\n\nWrite the case file."
        return final_answer(ai._ask(_messages(REPORT_SYSTEM, q), "report")) or ""
    # Real AI check (7 Nov): asking MedGemma to file only the new notes returned nothing usable, so 10 image readings
    # and a referral never reached the case file. Fixed word rules file them instead: nothing typed is ever lost.
    return merge_into(current, file_notes(new_notes, new_images))


def _test_questions(report):
    """TEST MODE stand-in: fixed checks for things that are often missing."""
    low = "\n".join(l for l in report.lower().splitlines() if not l.startswith("## "))   # headings name every topic
    checks = [(r"allerg|nkda", "Any allergies?"), (r"\bon \w+|meds|medication", "Regular medications?"),
              (r"since|onset|started|\d+ ?(h|hr|hours|days?)\b", "When did it start?"),
              (r"\bhr\b|heart rate", "Heart rate?"), (r"\bbp\b|blood pressure", "Blood pressure?")]
    questions = [f"{q} {MARK}" for pattern, q in checks if not re.search(pattern, low)]
    todo = []
    if re.search(r"chest pain|\bcp\b", low):
        todo += ["ECG within 10 min", "Troponin"]
    if re.search(INJURY, low):
        todo += ["X-ray of injured area", "Tetanus status"]
    if re.search(r"confus|drows", low):
        todo += ["Capillary glucose"]
    return questions[:5], [f"{t} (test rule)" for t in todo][:5]


ASKS_FOR = {"sbp": r"blood pressure|\bbp\b", "hr": r"heart rate|pulse|\bhr\b", "spo2": r"oxygen saturation|\bspo2\b|\bsats?\b",
            "temp_c": r"temperature|\btemp\b", "mental": r"consciousness|\bgcs\b|\bavpu\b", "rr": r"respiratory rate",
            "nrs_pain": r"pain (?:level|score|severity)", "age": r"\bage\b", "sex": r"\bsex\b|\bgender\b",
            "chief_complaint": r"presenting complaint|chief complaint"}


def drop_known(items, fields):
    """Drop questions/to-dos that only ask for values the note already has (run 2: 'What is the blood pressure?' with
    BP 193/115 recorded). An item that also asks for something missing is kept."""
    keep = []
    for item in items:
        low = item.lower()
        asked = [k for k, rule in ASKS_FOR.items() if re.search(rule, low)]
        if asked and all(fields.get(k) not in (None, "", "not recorded") for k in asked):
            continue
        keep.append(item)
    return keep


# The questions an ED clinician asks first on each pathway the ED rules flag (ed_assessment.PATHWAYS). Fixed, so
# they're always specific; MedGemma's questions come after them (7 Nov: alone, the 4B model's were mostly generic).
PATHWAY_QUESTIONS = {
    "chest pain": ["When did the pain start, and what is it like (crushing, sharp, tearing)?",
                   "Does it spread to the arm, jaw or back, with sweating or breathlessness?",
                   "Known heart disease, or risk factors (diabetes, smoking, high blood pressure)?"],
    "possible stroke": ["When was the patient last known to be well?", "What is the capillary glucose?",
                        "Is the patient on blood thinners (anticoagulants)?"],
    "breathing complaint": ["Known asthma, COPD or heart failure?", "Any chest pain, fever or leg swelling?"],
    "collapse/syncope": ["Was the collapse witnessed, and was there any warning (palpitations, chest pain)?",
                         "Any injury from the fall, especially the head?"],
    "seizure": ["Is this a first seizure, and how long did it last?", "Known epilepsy, and are medicines being taken?"],
    "possible poisoning": ["What was taken, how much, and when?", "Was it on purpose (self-harm risk)?"],
    "possible infection": ["Where might the infection be (chest, urine, skin, abdomen)?",
                           "Is the patient immunosuppressed (chemotherapy, steroids, HIV, diabetes)?"],
    "trauma": ["What was the mechanism of injury (height of fall, speed)?", "Is the patient on blood thinners?"],
    "unresponsive": ["When was the patient last seen well, and by whom?", "Any known diabetes, epilepsy or drug use?"],
}


def pathway_questions(ed_text):
    """Questions for every pathway flagged in the ED assessment, in the order the flags appear."""
    m = re.search(r"Pathway flags: (.*)", ed_text or "")
    if not m:
        return []
    out = []
    for flag in m.group(1).split("; "):
        name = flag.split("->")[0].strip().lower()
        out += PATHWAY_QUESTIONS.get(name, [])
    return out


def pathway_todos(ed_text):
    """The ED rules' pathway flags (ed_assessment.py) as to-dos, e.g. 'Possible stroke: record time last known well,
    glucose, urgent brain imaging'. Fixed rules, so every flagged pathway gets its to-do even if MedGemma gives none."""
    m = re.search(r"Pathway flags: (.*)", ed_text or "")
    if not m or m.group(1).startswith("none"):
        return []
    return [f"{name.strip().capitalize()}: {todo.strip().rstrip('.')}"
            for name, todo in (f.split("->", 1) for f in m.group(1).split("; ") if "->" in f)]


def ask_questions(ai, report, context=""):
    """-> (questions about missing information, suggested to-dos). The to-dos go on the whiteboard marked as AI.
    context: the patient line (age, sex, arrival) and the ED rules' pathway flags, which the case file doesn't repeat,
    so MedGemma stops asking for the age and asks the pathway's questions (e.g. time last known well)."""
    if ai is None:
        return _test_questions(report)
    from answers import final_answer
    report = re.sub(rf"## {TRIAGE}.*", "", report, flags=re.S)       # scores aren't what the questions are about
    raw = final_answer(ai._ask(_messages(QUESTION_SYSTEM, f"{context}\n\n{report}".strip()), "questions")) or ""
    return parse_lists(raw)


def parse_lists(raw):
    """JSON {"questions": [...], "todo": [...]}. If the answer was cut off before the JSON closed, the complete quoted
    items are still kept. Duplicates removed, at most 4 each."""
    m = re.search(r"\{.*\}", raw, re.S)
    out = {}
    try:
        out = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        pass
    for key in ("questions", "todo"):
        if not isinstance(out.get(key), list):
            part = re.search(rf'"{key}"\s*:\s*\[(.*?)(?:\]|$)', raw, re.S)
            out[key] = re.findall(r'"((?:[^"\\]|\\.)+)"', part.group(1)) if part else []

    def clean(xs):
        seen, keep = set(), []
        for x in xs:
            x = str(x).strip()
            if x and x.lower() not in seen:
                seen.add(x.lower())
                keep.append(x)
        return keep[:4]
    return clean(out["questions"]), clean(out["todo"])
