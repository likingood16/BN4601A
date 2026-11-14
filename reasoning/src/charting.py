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
    "important first. If pathway flags are given, first ask what that pathway needs to decide treatment (for a "
    "possible stroke, for example, the time last known well). Never ask for something the case file already has. "
    '"todo": up to 4 tasks that probably need doing or sending for THIS patient (tests, treatments, referrals), most '
    "urgent first. Each item at most 12 words. These are suggestions for clinicians to accept or ignore.")

# ---------------------------------------------------------------- numbers
RANGES = {"hr": (20, 250), "rr": (4, 70), "spo2": (50, 100), "temp_c": (30, 44), "sbp": (40, 280), "dbp": (20, 200),
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
BP = r"\b(?:bp|nibp|blood pressure|arterial pressure)" + r"(?:" + FILL + r")?" + r"(\d{2,3})\s*(?:/|over)\s*(\d{2,3})" + END
# Ambulance-crew values come first in many notes ("EMS found BP 70/40. On arrival BP 112/70"): the ED's own
# arrival values are used when the note marks them (group test pack check, 8 Nov 2026).
ARRIVAL = r"\b(?:on arrival|on arr|at triage|triage obs|arrival obs|in (?:the )?(?:ed|ae|a&e)|on presentation)\b"
AGE_SEX = [r"(?<![\d.])\b(\d{1,3})\s*(?:yo|y/o|y\.o\.|yrs?|years?(?: old)?|-year-old)?\s*-?\s*(m|f|male|female|man|woman|boy|girl)\b",
           r"\b(m|f|male|female)\s*,?\s*(\d{1,3})\b"]
AMBULANCE = r"\b(ambulance|ems|paramedics?|995|scdf|by amb|biba|bibs)\b"
WALK_IN = (r"\b(walk[- ]?in|walked in|self[- ]?present\w*|brought (?:in )?by (?:family|friends?|wife|husband|son|daughter|"
           r"mother|father|mum|mom|dad|parents?|guardian|carer|partner|teacher))\b")
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

    arrival = re.search(ARRIVAL, text)
    start = 0
    if arrival:                                     # from the start of the sentence that says "on arrival"
        breaks = [b.end() for b in re.finditer(r"(?<!\d)\.(?!\d)|\.(?=\s)|[;\n]", text[:arrival.start()])]
        start = breaks[-1] if breaks else 0

    def first(patterns):
        """Earliest match; when the note marks arrival values, the earliest one from there on."""
        hits = [m for m in (re.search(p, text) for p in patterns) if m]
        after = [m for m in hits if m.start() >= start] if start else []
        if start and not after:
            after = [m for m in (re.compile(p).search(text, start) for p in patterns) if m]
        best = min(after or hits, key=lambda h: h.start()) if (after or hits) else None
        early = min(hits, key=lambda h: h.start()) if hits else None
        if best is not None and early is not None and early.start() < start <= best.start() and early.group(0) != best.group(0):
            warnings.append(f"used the arrival value {best.group(0).strip()!r}, not the earlier {early.group(0).strip()!r} "
                            "(before arrival)")
        return best

    for key, patterns in PATTERNS.items():
        m = first(patterns)
        if m:
            value = float(next(g for g in m.groups() if g is not None).replace(",", "."))   # "37,5" = 37.5
            if key == "temp_c" and value > 45:          # written in Fahrenheit
                value = round((value - 32) * 5 / 9, 1)
            keep(key, value, m.group(0))
    m = first([BP])
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
    _check_bp(fields, found, warnings)
    return fields, {"found": found, "warnings": warnings}


# ---------------------------------------------------------------- numbers by AI (branch ai-reasoning, 8 Nov 2026)
NUMBERS_SYSTEM = (
    "You read emergency department triage notes. Reply with JSON only. For each item the note states, give its value "
    'and the exact words from the note it comes from, like {"hr": {"value": 112, "quote": "HR 112"}}. Items: '
    "age (years), sex (male or female), hr (heart or pulse rate; PR and P mean pulse), sbp AND dbp (blood pressure: "
    "give BOTH numbers, both quoting the same words, e.g. sbp 120 and dbp 80 from 'BP 120/80'), rr (breathing rate), "
    "spo2 (oxygen saturation %), temp (the number as written), temp_unit (C or F), nrs_pain (pain score 0-10, e.g. "
    "'3/10'), consciousness (always give it when the note says alert, AVPU or GCS: alert, confused, voice, pain or "
    "unresponsive; 'AVPU A' or GCS 15 = alert, GCS 13-14 or new confusion = confused), ambulance (true if brought by "
    "ambulance; BIBA, BIBS, EMS, SCDF, 995, paramedics all mean ambulance; false if walk-in, self-presented or brought "
    "by family), injured (true ONLY for trauma: fall, accident, cut, burn, bite, assault, sprain, fracture; pain or "
    "illness alone is not an injury). Leave out anything the note does not state. Never guess a value.")
_AVPU = {"alert": "alert", "confused": "verbal", "voice": "verbal", "verbal": "verbal", "pain": "pain",
         "unresponsive": "unresponsive"}


def _norm(s):
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def extract_numbers_ai(ai, note):
    """MedGemma reads the numbers out of the note. Same output as extract_numbers. Safety check (not reasoning): every
    value must come with the words it was read from, those words must be in the note, and a number must appear in
    its own words; otherwise it is left out with a warning, so the urgency model never gets an invented vital sign."""
    from answers import final_answer
    raw = final_answer(ai._ask(_messages(NUMBERS_SYSTEM, f"Triage note:\n{note}"), "numbers")) or ""
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        got = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        got = {}
    text, fields, found, warnings = _norm(note), {}, {}, []
    if not got:
        warnings.append("the AI could not read numbers from this note; please check them")
    # format only: the AI often gives the BP as one item ("bp": "160/100") or only the top number with a quote that
    # holds both (test, 8 Nov: diastolic missing in 48%); split the quote it gave, still checked below
    bp = got.get("bp") or got.get("blood_pressure") or got.get("sbp")
    if isinstance(bp, dict) and "dbp" not in got:
        m2 = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", str(bp.get("value", "")) + " " + str(bp.get("quote", "")))
        if m2:
            got["sbp"] = {"value": m2.group(1), "quote": bp.get("quote", "")}
            got["dbp"] = {"value": m2.group(2), "quote": bp.get("quote", "")}
    unit = got.get("temp_unit", "")
    unit = str(unit.get("value", "") if isinstance(unit, dict) else unit).upper()   # {"value": "F"} or "F"
    for key, item in got.items():
        if key == "temp_unit" or not isinstance(item, dict) or item.get("value") in (None, ""):
            continue
        value, quote = item["value"], _norm(item.get("quote", ""))
        if not quote or quote not in text:
            warnings.append(f"AI value for {key} left out: its words {item.get('quote')!r} are not in the note")
            continue
        if key in ("age", "hr", "sbp", "dbp", "rr", "spo2", "temp", "nrs_pain"):
            try:
                value = float(str(value).replace(",", "."))
            except ValueError:
                continue
            digits = re.findall(r"\d+(?:[.,]\d+)?", quote)
            if not any(abs(float(d.replace(",", ".")) - value) < 1e-6 for d in digits):
                warnings.append(f"AI value {key} {value:g} left out: not written in {item.get('quote')!r}")
                continue
            if key == "temp":
                key, value = "temp_c", round((value - 32) * 5 / 9, 1) if unit == "F" or value > 45 else value
            lo, hi = RANGES.get(key, (None, None))
            if lo is not None and not lo <= value <= hi:
                warnings.append(f"ignored {quote!r}: {key} {value:g} is outside {lo}-{hi}")
                continue
        elif key == "sex":
            value = {"m": "male", "male": "male", "f": "female", "female": "female"}.get(str(value).lower())
        elif key == "consciousness":
            key, value = "mental", _AVPU.get(str(value).lower())
        elif key in ("ambulance", "injured"):
            value = 1.0 if str(value).lower() in ("true", "1", "yes") else 0.0
            if key == "injured" and value == 0.0:
                continue            # 'no injury word' is not proof of no injury (same as the rules version)
        else:
            continue
        if value is not None:
            fields[key], found[key] = value, item.get("quote", "").strip()
    _check_bp(fields, found, warnings)
    return fields, {"found": found, "warnings": warnings}


def _check_bp(fields, found, warnings):
    """A blood pressure that can't be real is dropped whole (BP 300/20 was used and made the patient level 1, 9 Nov,
    5q #8): the diastolic must be below the systolic, at most 200 apart, and never used on its own."""
    s, d = fields.get("sbp"), fields.get("dbp")
    if s is not None and d is not None and not (d < s and s - d <= 200):
        warnings.append(f"ignored {found.get('sbp', 'the blood pressure')!r}: BP {s:g}/{d:g} is not possible")
        for k in ("sbp", "dbp"):
            fields.pop(k, None), found.pop(k, None)
    elif s is None and d is not None:
        fields.pop("dbp"), found.pop("dbp", None)


def _agree(key, a, b):
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    return abs(float(a) - float(b)) <= (0.11 if key == "temp_c" else 1e-6)


def read_numbers(ai, note):
    """Both readers on every note (fix plan 5o item 3, 8 Nov 2026). The AI alone read 89% and the same patient written
    3 ways got 3 different urgency scores, because it left things out (consciousness, diastolic BP). The rules read 100%
    of the test notes but only know set phrasings. So:
      - both found it and agree -> used;
      - only one found it -> used, marked with who read it;
      - they disagree -> the rules' value is used until a nurse checks, and both readings are shown ("check").
    ai=None (test mode): rules only."""
    rf, rinfo = extract_numbers(note)
    if ai is None:
        return rf, {**rinfo, "by": {k: "rules" for k in rf}, "check": []}
    af, ainfo = extract_numbers_ai(ai, note)
    fields, found, by, check = {}, {}, {}, []
    for key in [k for k in rf] + [k for k in af if k not in rf]:
        if key in rf and key in af:
            fields[key], found[key] = rf[key], rinfo["found"].get(key, "")
            if _agree(key, rf[key], af[key]):
                by[key] = "both"
            else:
                by[key] = "check"
                check.append({"field": key, "rules": rf[key], "ai": af[key],
                              "rules_words": rinfo["found"].get(key), "ai_words": ainfo["found"].get(key)})
        elif key in rf:
            fields[key], found[key], by[key] = rf[key], rinfo["found"].get(key, ""), "rules"
        else:
            fields[key], found[key], by[key] = af[key], ainfo["found"].get(key, ""), "AI"
    warnings = rinfo["warnings"] + [w for w in ainfo["warnings"] if w not in rinfo["warnings"]] + [
        f"please check {c['field']}: rules read {c['rules']} from {c['rules_words']!r}, AI read {c['ai']} "
        f"from {c['ai_words']!r}" for c in check]
    return fields, {"found": found, "warnings": warnings, "by": by, "check": check}


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
        # "brought by ambulance" is not a complaint, but "rt flank pain brought by family" has one (7 Nov stress test:
        # the old rule skipped any part with <= 3 other words, so short complaints were lost)
        words = re.sub(f"{AMBULANCE}|{WALK_IN}", " ", words, flags=re.I)
        words = re.sub(r"\s+", " ", words).strip(" -:,")
        words = re.sub(r"(?:\s|^)(?:via|by|brought(?: in)?(?: by)?|came(?: in)?(?: by)?|arrived(?: by)?)$", "", words,
                       flags=re.I).strip()
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
    # onset / last known well first: "known" alone means past history ("known asthmatic"), but an answer to "When was
    # the patient last known to be well?" is history (Gradio test, 9 Nov)
    ("History", r"last (?:seen|known)(?: to be)? well|\blkw\b|\bonset\b"),
    ("Impression (as written by clinicians)", r"\bimpression\b|\bimp\b|\bdx\b|diagnos|\blikely\b|suspected|\?\s?[a-z]{4,}"),
    ("Imaging", r"x-?ray|\bxr\b|\bct\b|mri|ultrasound|\bus\b|scan|image"),
    ("Investigations", r"trop|ecg|\bhb\b|wbc|crp|lactate|glucose|cbg|\bbsl\b|\bhgt\b|\binr\b|bloods?|\bfbc\b|u/?e|\bna\b|\bk\b|urine|swab|abg|vbg"),
    ("Plan and actions", r"\bgave\b|given|start(ed)?|\biv\b|plan|refer|admit|discharge|review|send|stat\b|neb|analgesi"),
    ("Observations", r"\bhr\b|\bbp\b|\brr\b|sats|spo2|temp|gcs|avpu|pain \d"),
    # "on <a medicine>" (on metformin, on warfarin, on salbutamol), not "cut finger on glass" (Gradio test, 9 Nov)
    ("Past history, medications, allergies", r"\bhx\b|pmh|history of|known|\bdm\b|\bhtn\b|meds|allerg|nkda|"
                                             r"\bon (?:insulin|aspirin|\w+(?:in|ol|ide|pril|artan|statin|pine|zole|mab|"
                                             r"done|xaban|gatran|lone|sone|mide|amine))\b"),
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
            sections[head].append(label_before_arrival(piece))
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
    return (t in ("", "not recorded", "none", "none recorded", "nil", "n/a", "not done") or (t.endswith("not recorded") and len(t) < 45)
            or t.startswith("not filled in separately"))         # the record's pointer, if MedGemma copies it


def set_section(report, heading, lines):
    """Replace one section's lines (the triage section, which comes from the trained models' decision)."""
    pre, sections = _split(report)
    sections.pop(heading, None)
    sections[heading] = [f"- {l}" for l in lines]                 # always the last section
    body = "\n\n".join(f"## {h}\n" + ("\n".join(ls) if ls else "- not recorded") for h, ls in sections.items())
    return f"{pre}\n\n{body}" if pre else body


def tidy_numbers(report):
    """Slips MedGemma makes when it copies a score out of the note: 'Pain 3/010' for 'pain 3/10' (retest 14 Nov, P05)."""
    return re.sub(r"\b(\d{1,2})/0+10\b", r"\1/10", report)


NOT_RECORDED_NAMES = {"mental": r"consciousness|level of consciousness|acvpu|avpu|gcs", "hr": r"hr|heart rate|pulse",
                      "rr": r"rr|respiratory rate|resp rate", "spo2": r"spo2|sats|oxygen saturation",
                      "sbp": r"bp|blood pressure", "dbp": r"bp|blood pressure", "temp_c": r"temp|temperature",
                      "nrs_pain": r"pain|pain score"}


def drop_not_recorded(report, keys):
    """After a later note gives a number, 'Consciousness not recorded' must not stay next to 'Consciousness alert'
    (Caleb's retest 14 Nov, T01). Removes '<name> not recorded' for those fields, in every section."""
    names = "|".join(NOT_RECORDED_NAMES[k] for k in keys if k in NOT_RECORDED_NAMES)
    if not names:
        return report
    gone = re.compile(rf"(?i)(,\s*|;\s*)?\b(?:{names})\b(?:\s*\([^)]*\))?:?\s+(?:is\s+|was\s+)?not recorded\b\.?")
    out = []
    for line in report.splitlines():
        if line.startswith("## ") or "not recorded" not in line.lower():
            out.append(line)
            continue
        new = gone.sub("", line)
        new = re.sub(r"^(\s*[-*•]\s*)[,;]\s*", r"\1", new).rstrip(" ,;")
        if new != line and not re.sub(r"[-*•\s.,;:]", "", new):
            continue                                              # the whole line was the placeholder
        out.append(new)
    return "\n".join(out)


def strip_preamble(text):
    """MedGemma sometimes opens with 'Okay, here is the SBAR handover...:' (Caleb's retest 14 Nov)."""
    return re.sub(r"(?is)^\s*(?:okay|ok|sure|certainly|of course|alright)\b[^\n]{0,160}?\b(?:here is|here's|below is)"
                  r"[^\n]*\n+", "", text, count=1).lstrip()


DONE_TESTS = {"ECG": r"\b(?:ecg|ekg|electrocardiogram)s?\b", "troponin": r"\b(?:trop|troponin|hs-?tni?|tni)s?\b",
              "chest X-ray": r"\b(?:cxr|chest x-?rays?|chest radiograph)\b", "CT head": r"\bct (?:head|brain)\b",
              "blood glucose": r"\b(?:bm|cbg|blood glucose|glucose|capillary glucose)\b", "lactate": r"\blactate\b",
              "blood gas": r"\b(?:vbg|abg|blood gas)\b"}
NOT_YET = r"(?i)\b(?:pending|awaiting|await|requested|ordered|to be done|to do|not done|not yet|tbc|will do|sent)\b"


def tests_done(notes):
    """{test name: time} for tests whose result is in a later note (not 'trop pending')."""
    done = {}
    for n in notes:
        for clause in re.split(r"[.;\n]", n["text"]):
            if re.search(NOT_YET, clause):
                continue
            for name, pat in DONE_TESTS.items():
                if re.search(pat, clause, re.I) and name not in done:
                    done[name] = n["time"]
    return done


def mark_done_tests(text, done):
    """Caleb's retest 14 Nov: the SBAR and the T01 chat still said 'obtain urgent ECG and troponin' with both already
    in the notes. MedGemma's words are kept; a sentence that asks for a done test gets a note saying when it was done
    (a repeat or serial test is a real request, so those are left alone)."""
    if not done:
        return text
    ask = re.compile(r"(?i)\b(?:obtain|order|get|perform|request|send|do|arrange|check|need|consider|recommend\w*)\b")
    # full-AI check 14 Nov: "a full ECG review" is not asking for a new ECG
    again = re.compile(r"(?i)\b(?:repeat|serial|again|second|2nd|follow-?up|trend|re-?check|recheck|another|review\w*|interpret\w*)\b")
    def fix(sentence):
        if not ask.search(sentence) or again.search(sentence):
            return sentence
        hits = [f"{name} at {t}" for name, t in done.items() if re.search(DONE_TESTS[name], sentence, re.I)]
        if not hits:
            return sentence
        end = re.search(r"[.!]?\s*$", sentence).start()
        return sentence[:end] + f" (already done: {', '.join(hits)}, see the notes)" + sentence[end:]
    # a sentence ends at . ! ? followed by a space or the end, so "e.g." doesn't split it (full-AI check 14 Nov)
    return re.sub(r"(?:[^.!?\n]|[.!?](?=[^\s]))+[.!?]?", lambda m: fix(m.group(0)), text)


def add_line(report, heading, line):
    """Add one line to a section, keeping the order of the sections (an empty section's placeholder is replaced)."""
    pre, sections = _split(report)
    if heading not in sections and TRIAGE in sections:            # a missing section goes before the triage section
        triage = sections.pop(TRIAGE)
        sections[heading], sections[TRIAGE] = [], triage
    lines = sections.get(heading, [])
    # a lone placeholder ("- not recorded") goes; real lines like "BP not recorded" stay (they say what is missing)
    lines = [] if all(_empty(l) for l in lines) and len(lines) <= 1 else lines
    sections[heading] = lines + [f"- {line}"]
    body ="\n\n".join(f"## {h}\n" + ("\n".join(ls) if ls else "- not recorded") for h, ls in sections.items())
    return f"{pre}\n\n{body}" if pre else body


def label_before_arrival(piece):
    """An ambulance crew's numbers, typed as a separate piece ("EMS found BP 70/40 HR 130"), filed next to the ED's
    own observations read like a second set of current values (group test 10 Nov, T05). Label them; the tiles
    already use the arrival values."""
    if (re.search(AMBULANCE + r"|\bpre-?hospital\b|\ben route\b", piece, re.I) and re.search(r"\d", piece)
            and not re.search(ARRIVAL, piece, re.I) and not piece.lower().startswith("before arrival")):
        return f"Before arrival (ambulance crew): {piece}"
    return piece


def file_notes(new_notes, new_images=()):
    """Later notes, filed under the headings by fixed word rules (not by MedGemma): every word typed lands in the case
    file, nothing is reworded or invented. Image readings always go under Imaging."""
    sections = {h: [] for h in HEADINGS if h != TRIAGE}
    for note in new_notes:
        for piece in filter(None, (p.strip() for p in re.split(SENTENCE, note))):
            head = next((h for h, rule in SORT_RULES if re.search(rule, piece, re.I)), "History")
            sections[head].append(label_before_arrival(piece))
    sections["Imaging"] += list(new_images)
    return "\n\n".join(f"## {h}\n" + "\n".join(f"- {x}" for x in xs) for h, xs in sections.items() if xs)


FILING_SYSTEM = (
    "You file emergency department notes into a case file. Each numbered line below is one piece of a note. Reply with "
    'JSON only, mapping each line number to one heading, like {"1": "History", "2": "Observations"}. Headings: '
    + "; ".join(h for h in HEADINGS if h not in (TRIAGE, "Imaging"))
    + ". 'Impression (as written by clinicians)' only for a diagnosis or impression a clinician wrote.")


def file_notes_ai(ai, new_notes, new_images=()):
    """Later notes filed by MedGemma: it only CHOOSES the heading for each piece; the words go in exactly as typed,
    so nothing is reworded, lost or invented (run 0, 7 Nov: asking it to rewrite the notes lost image readings and a
    referral). A piece whose heading can't be read goes under History. Image readings always go under Imaging."""
    from answers import final_answer
    pieces = [label_before_arrival(p.strip()) for note in new_notes for p in re.split(SENTENCE, note) if p.strip()]
    sections = {h: [] for h in HEADINGS if h != TRIAGE}
    # the word rules first: MedGemma filed "Trop 45 ng/L" under Presenting complaint, "ECG ST depression" under History
    # and "aspirin given" under Impression (real-AI test, 9 Nov, 5q #2); it now only places what the rules can't
    unsure = []
    for p in pieces:
        head = next((h for h, rule in SORT_RULES if re.search(rule, p, re.I)), None)
        (sections[head].append(p) if head else unsure.append(p))
    pieces = unsure
    if pieces:
        q = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(pieces))
        raw = final_answer(ai._ask(_messages(FILING_SYSTEM, q), "filing")) or ""
        m = re.search(r"\{.*\}", raw, re.S)
        try:
            where = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            where = dict(re.findall(r'"(\d+)"\s*:\s*"([^"]+)"', raw))
        for i, p in enumerate(pieces):
            want = str(where.get(str(i + 1), "History")).lower().split(" (")[0].strip()
            # exact heading first, then a short form ("Plan" -> "Plan and actions", "Past history" -> ...)
            head = (next((h for h in sections if h.lower().split(" (")[0] == want), None)
                    or next((h for h in sections if want and h.lower().startswith(want)), "History"))
            sections[head].append(p)
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


# ED shorthand -> a word MedGemma uses when it writes it out, so an expanded note still counts as covered
SHORTHAND = {"cp": "chest", "sob": "breath", "dm": "diabet", "htn": "hypertens", "nkda": "allerg", "biba": "ambulance",
             "hx": "history", "pmh": "history", "abdo": "abdom", "n": "nausea", "v": "vomit", "loc": "conscious",
             "ra": "room air", "pt": "patient", "yo": "year", "c/o": "complain", "l": "left", "r": "right",
             "rt": "right", "lt": "left", "amb": "ambulance", "meds": "medic", "sats": "spo2", "t": "temp"}
_FILLER = {"and", "the", "to", "of", "on", "in", "at", "a", "an", "with", "for", "since", "is", "was", "has", "had",
           "from", "by", "x", "or", "no", "not", "came", "pt", "patient"}


_WORD = r"[a-z/]+|\d+(?:\.\d+)?"


def _covered(piece, text, words_in):
    """Is this piece of a note already in the case file? Most of its words are there as whole words (shorthand
    written out counts; single letters only when they are known shorthand, e.g. 'L' = left)."""
    words = [w for w in re.findall(_WORD, piece.lower()) if w not in _FILLER and (len(w) > 1 or w in SHORTHAND)]
    if not words:
        return True

    def there(w):
        if w in words_in or (w in SHORTHAND and SHORTHAND[w] in text):
            return True
        if re.fullmatch(r"\d{4}", w):                        # 0600 written out as 06:00 or 6:00
            return f"{w[:2]}:{w[2:]}" in text or f"{int(w[:2])}:{w[2:]}" in text
        return len(w) >= 5 and any(b.startswith(w[:5]) for b in words_in)   # sweaty ~ sweating
    # a clock time (onset, last known well) must be there itself: it decides stroke and chest-pain care
    if any(not there(t) for t in re.findall(r"\b(?:[01]\d|2[0-3])[0-5]\d\b", piece)):
        return False
    return sum(map(there, words)) / len(words) >= 0.6


def same_meaning(a, b):
    """Two to-dos or questions that ask for the same thing: one's key words are all in the other, or they mostly
    overlap ('Chest pain: ECG within 10 minutes of arrival' and 'ECG within 10 minutes')."""
    def key(s):
        s = s.split(":", 1)[1] if re.match(r"^[\w /-]{3,30}:", s) else s      # drop a "Chest pain:" label
        return {w for w in re.findall(_WORD, s.lower()) if w not in _FILLER and w not in ("any", "what", "is", "are")}
    ka, kb = key(a), key(b)
    if not ka or not kb:
        return False
    return ka <= kb or kb <= ka or len(ka & kb) / len(ka | kb) >= 0.6


def keep_every_note(report, notes):
    """Real-AI check (9 Nov, REPORT_NOTES 5q #1): MedGemma's first case file left out past history, medicines,
    allergies and the onset. Every piece of a note it left out is now filed by the word rules, as typed, under the
    right heading, so nothing typed is lost (it may sit next to MedGemma's own wording of the same fact)."""
    if not report.strip():
        return report
    # only what is written under the headings counts (the heading "allergies" is not an allergy history)
    body = "\n".join(l for l in report.lower().splitlines() if not l.startswith("#") and not _empty(l.lstrip("-* ")))
    words_in = set(re.findall(_WORD, body))
    missing = [p.strip() for n in notes for p in re.split(SENTENCE, n) if p.strip() and not _covered(p, body, words_in)]
    return merge_into(report, file_notes(missing)) if missing else report


def build_report(ai, record, current="", new_notes=(), notes=(), images=(), new_images=()):
    """The tidy case file. First time: written from the whole record. After that, only the NEW notes are filed under
    the headings and merged in (merge_into), so the case file is never rewritten and people's edits are kept."""
    from answers import final_answer
    if not current:
        if ai is None:
            return _test_report(record, list(notes), list(images))
        q = "PATIENT RECORD (arrival card, notes, image readings, triage decision):\n" + record + "\n\nWrite the case file."
        written = final_answer(ai._ask(_messages(REPORT_SYSTEM, q), "report")) or ""
        # MedGemma copied the record's pointer "not filled in separately: read the CONSULTATION NOTES" (9 Nov)
        written = "\n".join("- not recorded" if "not filled in separately" in l.lower() else l for l in written.splitlines())
        return keep_every_note(written, notes)
    # Real AI check (7 Nov): asking MedGemma to REWRITE with the new notes lost 10 image readings and a referral. Now it
    # only picks the heading for each piece (file_notes_ai); the words go in as typed. Test mode: word rules.
    return merge_into(current, file_notes(new_notes, new_images) if ai is None else file_notes_ai(ai, new_notes, new_images))


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


# how the injury happened, as nurses write it ("fell off ladder", "fell from scooter", "cut finger on kitchen knife")
MECHANISM = (r"\bfell\b|\bfall\b|\bmv[ac]\b|\brta\b|\bhit\b|struck|assault|kicked|tripped|slipped|twisted|ladder|stairs|"
             r"bars\b|scooter|bicycle|\bbike\b|knife|glass|knock|punch|crush|collision|\bbite\b|bitten|burn")
CHEST_PAIN = r"chest (?:pain|tight\w*|discomfort|pressure)|\bcp\b|angina"

# Questions the notes' own words already answer: (what the question asks about, what in the notes answers it).
# 7 Nov stress test: "Is the patient on blood thinners?" asked with "on warfarin" in the note.
ANSWERED_BY_NOTE = [
    (r"blood thinner|anticoagul", r"warfarin|apixaban|rivaroxaban|dabigatran|edoxaban|enoxaparin|heparin|anticoag|\bdoac\b|"
                                  r"\bnoac\b|eliquis|xarelto|clexane|blood thinner"),
    (r"allerg", r"\bnkda\b|\bnka\b|allerg"),
    (r"glucose|\bbsl\b|\bcbg\b", r"(?:glucose|\bbsl\b|\bcbg\b|\bbm\b|\bhgt\b)\s*(?:of|is|:|=)?\s*\d"),
    (r"last (?:seen|known)(?: to be)? (?:well|normal)", r"last (?:seen|known)(?: to be)? (?:well|normal)|\blkw\b"),
    # real-AI test, 9 Nov (5q #6): "When did the pain start?" and "Does it spread to the arm?" with "CP since 0600,
    # sweaty, radiating to L arm" in the note
    (r"when did (?:it|the \w+) (?:start|begin)|\bonset\b|how long ago",
     r"\bsince\b|\bstarted\b|\bonset\b|\d+\s*(?:h|hrs?|hours?|mins?|minutes?|days?|wks?|weeks?)\b(?:\s*ago)?|\bx\s*\d+\s*(?:h|days?)"),
    (r"spread to the arm|radiat", r"radiat"),
    (r"risk factors|known heart disease", r"\bhx\b|\bpmh\b|\bdm\b|diabet|\bhtn\b|hypertens|smok|\bihd\b|\bcad\b|\bmi\b"),
    (r"mechanism of injury", MECHANISM),
    (r"collapse witnessed|was the collapse witnessed", r"witnessed"),
]


def drop_known(items, fields, notes=""):
    """Drop questions/to-dos that only ask for things the notes already have (run 2: 'What is the blood pressure?' with
    BP 193/115 recorded). An item that also asks for something missing is kept."""
    keep, notes = [], notes.lower()
    for item in items:
        low = item.lower()
        asked = [k for k, rule in ASKS_FOR.items() if re.search(rule, low)]
        if asked and all(fields.get(k) not in (None, "", "not recorded") for k in asked):
            continue
        topics = [ans for q, ans in ANSWERED_BY_NOTE if re.search(q, low)]
        if topics and not asked and all(re.search(ans, notes) for ans in topics):
            continue
        keep.append(item)
    return keep


SMALL_INJURY = r"\b(?:cut|lacerat\w*|graze\w*|abrasion|splinter|nick\w*)\b"
SMALL_PART = r"\b(?:finger|thumb|toe|hand|palm|knuckle|nail|fingertip)s?\b"
BIG_INJURY = (r"fell from|fall from|height|\brta\b|\bmv[ac]\b|collision|stab|deformit|can'?t (?:move|weight|bend)|"
              r"amputat|\bhead\b|numb|tendon|spurting|pulsatile")


def minor_injury(fields, notes=""):
    """A small cut to a finger or hand, numbers normal (or not taken): the trauma pathway's primary survey and an
    X-ray don't fit it (T04: "23M cut finger on kitchen knife, bleeding stopped")."""
    low = (notes or "").lower()
    if not (re.search(SMALL_INJURY, low) and re.search(SMALL_PART, low)) or re.search(BIG_INJURY, low):
        return False
    hr, sbp, spo2, mental = (fields.get(k) for k in ("hr", "sbp", "spo2", "mental"))
    return ((hr is None or hr <= 100) and (sbp is None or sbp >= 100) and (spo2 is None or spo2 >= 95)
            and mental in (None, "", "alert"))


# Pathways the model may raise from a word that isn't about them ("warfarin" -> poisoning, P11; "lightheaded" ->
# collapse, T05, full AI 12 Nov). Shown only when the notes say something about them; the model's own flags are
# still logged for learning (server._log_outcome).
NEEDS_WORDS = {
    "possible poisoning": r"overdose|\bod\b|poison|intoxic|ingest|swallow|took \d+|tablets|pills|alcohol|\betoh\b|drunk|"
                          r"\bdrugs?\b|substance|chemical|bleach|paracetamol|carbon monoxide|toxic",
    "collapse/syncope": r"collaps|syncop|faint|passed out|black(?:ed)? ?out|loss of consciousness|\bloc\b|unresponsive|"
                        r"found (?:on the )?(?:floor|down)",
}


def supported_pathways(ed_text, notes=""):
    """The ED assessment with the 'Pathway flags' line keeping only flags the notes support (NEEDS_WORDS)."""
    m = re.search(r"Pathway flags: (.*)", ed_text or "")
    if not m or m.group(1).startswith("none"):
        return ed_text
    low = (notes or "").lower()
    keep = []
    for flag in m.group(1).split("; "):
        rule = NEEDS_WORDS.get(flag.split("->")[0].strip().lower())
        if rule is None or re.search(rule, low):
            keep.append(flag)
    return ed_text[:m.start(1)] + ("; ".join(keep) if keep else "none") + ed_text[m.end(1):]


def fit_patient(items, fields, notes="", image_kinds=()):
    """Questions / to-dos that fit THIS patient (group test 10 Nov, B3-B4). After drop_known:
      - under 16: no blood-thinner question (asked for a 7- and a 9-year-old);
      - chest pain is the complaint: "Any chest pain, fever or leg swelling?" loses its chest-pain part;
      - an X-ray is already attached: no "X-ray of the injured area" (bone X-ray) or "chest X-ray" (chest X-ray);
      - a small cut to a finger or hand with normal numbers: no primary survey, and no X-ray unless glass or a
        crush could have left something in it (T04, full AI 12 Nov: "Trauma: primary survey" for a cut finger)."""
    age = fields.get("age")
    minor = minor_injury(fields, notes)
    child = age is not None and age == age and float(age) < 16
    said = f"{fields.get('chief_complaint') or ''} {notes}".lower()
    chest_pain = bool(re.search(CHEST_PAIN, said))
    kinds = set(image_kinds or ())
    out = []
    for item in drop_known(items, fields, notes):
        low = item.lower()
        if child and re.search(r"blood thinner|anticoagul", low):
            continue
        if chest_pain and re.search(r"\bany chest pain\b", low):
            item = re.sub(r"(?i)\bany chest pain(?:,| or| and)\s+", "Any ", item)
            if re.search(r"(?i)\bany chest pain\b", item):       # that was all it asked
                continue
        if minor and (re.search(r"primary survey", low) or (re.search(r"x-?ray of (?:the )?injured", low)
                                                           and not re.search(r"glass|foreign|crush", said))):
            continue
        if re.search(r"x-?ray|\bxr\b|\bcxr\b|radiograph", low) and not re.search(r"repeat|after|review|check the", low):
            of_chest = bool(re.search(r"chest|\bcxr\b", low))
            # a CT already images the injured part better than an X-ray (P11: head injury, CT head attached, still
            # got "X-ray of the injured area", full-AI re-run 11 Nov)
            if ("chest_xray" in kinds and of_chest) or (kinds & {"bone_xray", "ct_slice"} and not of_chest):
                continue
        out.append(item)
    return out


CORE_OBS = [("sbp", "blood pressure"), ("hr", "heart rate"), ("rr", "breathing rate"), ("spo2", "oxygen saturation"),
            ("temp_c", "temperature"), ("mental", "level of consciousness (ACVPU)")]


def drop_obs_questions(questions):
    """Questions that ask for one of the six observations, removed once missing_obs_question asks for them all.
    Only those six: questions about age, pain score or the complaint stay (code review 12 Nov)."""
    return [q for q in questions if not any(re.search(ASKS_FOR[k], q.lower()) for k, _ in CORE_OBS)]


def missing_obs_question(fields, at_least=3):
    """One question asking for the observations when 3 or more of the six are not recorded (10 Nov, B5: a patient
    with no observations at all was only asked "level of consciousness?"). '' when fewer are missing."""
    missing = [name for key, name in CORE_OBS if fields.get(key) in (None, "")]
    if len(missing) < at_least:
        return ""
    return f"Observations not recorded yet: what are the {', '.join(missing[:-1])} and {missing[-1]}?"


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


def pathway_questions(ed_text, most=3):
    """Fixed questions for the flagged pathways, at most `most` (10 Nov, B2: two off-topic flags filled all five
    places and pushed out MedGemma's own questions). The first flag is the strongest (pathway_heads.flags), so it
    gives all its questions first; the other flags then give their first question each while there is room."""
    m = re.search(r"Pathway flags: (.*)", ed_text or "")
    if not m:
        return []
    lists = [PATHWAY_QUESTIONS.get(flag.split("->")[0].strip().lower(), []) for flag in m.group(1).split("; ")]
    lists = [l for l in lists if l]
    out = list(lists[0]) if lists else []
    out += [l[0] for l in lists[1:]]
    return out[:most]


def pathway_todos(ed_text, notes=""):
    """The ED rules' pathway flags (ed_assessment.py) as to-dos, e.g. 'Possible stroke: record time last known well,
    glucose, urgent brain imaging'. Fixed rules, so every flagged pathway gets its to-do even if MedGemma gives none.
    notes: what was typed. The trauma to-do asks for the mechanism of injury only when the notes don't give one
    (10 Nov, B3: "mechanism of injury not recorded" on "fell off ladder")."""
    m = re.search(r"Pathway flags: (.*)", ed_text or "")
    if not m or m.group(1).startswith("none"):
        return []
    out = []
    for name, todo in (f.split("->", 1) for f in m.group(1).split("; ") if "->" in f):
        todo = todo.strip().rstrip(".")
        if name.strip().lower() == "trauma":
            todo = "primary survey" if re.search(MECHANISM, notes.lower()) else "record the mechanism of injury; primary survey"
        out.append(f"{name.strip().capitalize()}: {todo}")
    return out


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
