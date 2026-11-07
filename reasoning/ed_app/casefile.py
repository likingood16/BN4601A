"""The case file: a list of ITEMS under standard ED headings. Each item keeps the exact words it came from.

    item = {"id", "section", "text", "quote", "note", "by", "kind", "confirmed", "time"}
    kind: "ai"     MedGemma tidied it; quote = the note's words it is based on (checked by code, see verify())
          "typed"  the nurse's own words, filed by fixed rules (anything MedGemma left out, or test mode)
          "auto"   written by code from data: observations (numbers), imaging (image tools), triage (trained models)
          "person" written or edited by a person. Never changed or removed by the AI.

Why items instead of one markdown text (the old app): the AI can only ADD items, so nobody's edit is ever overwritten,
and every AI line can be traced to the note it came from. The "never invents" rule is enforced by code: an AI line
whose quote isn't in the notes, or that has a number the quote hasn't, is rejected (kept in `rejected` for review).
"""
import json
import re
import time
import uuid

import charting
from extract import NAMES, UNITS

SECTIONS = ["Presenting complaint", "History of presenting complaint", "Past medical history", "Medications",
            "Allergies", "Social history", "Observations", "Examination", "Investigations", "Imaging",
            "Impression", "Plan", "Triage decision support"]
AI_SECTIONS = [s for s in SECTIONS if s not in ("Observations", "Imaging", "Triage decision support")]
AUTO = {"Observations", "Triage decision support"}       # rebuilt by code whenever the numbers or decision change

SYSTEM = (
    "You file an emergency nurse's quick note into a case file. Reply with JSON only: a list of items "
    '[{"section": "...", "text": "...", "quote": "..."}].\n'
    "Rules:\n"
    "- quote: the exact words copied from the note that the item is based on (a short piece, copied letter for letter).\n"
    "- text: the same fact as a short clear phrase, abbreviations written out (CP -> chest pain, hx -> history of, "
    "DM -> diabetes mellitus, SOB -> shortness of breath, BIBA -> brought in by ambulance). Add NOTHING that is not "
    "in the quote: no diagnoses, no tests, no treatments, no 'assumed' values.\n"
    "- section: one of: " + ", ".join(AI_SECTIONS) + ".\n"
    "- The first item is the Presenting complaint: the main problem in 2-8 words (symptoms only, no age, sex or "
    "vital signs).\n"
    "- Skip vital sign numbers (heart rate, blood pressure, respiratory rate, oxygen saturation, temperature, pain "
    "score, AVPU/GCS): they are filed separately.\n"
    "- Impression only if the note itself states a diagnosis or impression.\n"
    "- One fact per item. Every other fact in the note gets an item.")
EXAMPLE_NOTE = "45F SOB 2/7, worse lying flat. hx asthma, on salbutamol inhaler. NKDA. HR 104 sats 93% RA. wheeze bilat."
EXAMPLE_JSON = json.dumps([
    {"section": "Presenting complaint", "text": "Shortness of breath", "quote": "SOB"},
    {"section": "History of presenting complaint", "text": "Shortness of breath for 2 days", "quote": "SOB 2/7"},
    {"section": "History of presenting complaint", "text": "Worse lying flat", "quote": "worse lying flat"},
    {"section": "Past medical history", "text": "Asthma", "quote": "hx asthma"},
    {"section": "Medications", "text": "Salbutamol inhaler", "quote": "on salbutamol inhaler"},
    {"section": "Allergies", "text": "No known drug allergies", "quote": "NKDA"},
    {"section": "Examination", "text": "Wheeze on both sides", "quote": "wheeze bilat"}])

# fixed rules for the nurse's own words (test mode, and anything MedGemma didn't file)
RULES = [
    ("Allergies", r"allerg|nkda|\bnka\b"),
    ("Medications", r"\bon (?!(?:arrival|air|exam\w*|evaluation|admission|presentation|the|a|an|and|his|her|their|"
                    r"waking|scene|route|day|room|examination)\b)[a-z]{4,}|\bmeds\b|medications?|\btablets?\b|inhaler"),
    ("Past medical history", r"\bhx\b|\bpmh\b|history of|known\b|\bk/c\b|\bbg\b|\bdm\b|\bhtn\b|\bhld\b|\bckd\b|\bihd\b|\bcopd\b"),
    ("Social history", r"smok|alcohol|\betoh\b|lives (?:alone|with)|drinks?\b|pack.?years"),
    ("Impression", r"\bimpression\b|\bimp\b|\bdx\b|\?\s?[a-z]{4,}|\bsuspect"),
    ("Investigations", r"trop|\becg\b|\bhb\b|wbc|crp|lactate|glucose|\bcbg\b|\bbloods\b|blood tests?|blood cultures?|\bfbc\b|\bu/?e\b|\bu&e\b|urine|\babg\b|\bvbg\b"),
    ("Plan", r"\bgave\b|given|start(?:ed)?\b|\biv\b|\bplan\b|refer|admit|discharge|review|\bsend\b|\bstat\b|\bneb"),
    ("Examination", r"o/e|on exam|tender|clear\b|crackles|wheez|soft\b|pupils|neuro|pale|diaphor|sweaty|clammy|swollen|bruis"),
]
IMAGING = r"x-?ray|\bxr\b|\bcxr\b|\bct\b|\bmri\b|ultrasound|\bus\b|\bpocus\b"


def now():
    return time.strftime("%H:%M")


def new_item(section, text, kind, by, quote="", note=None, confirmed=False):
    return {"id": uuid.uuid4().hex[:8], "section": section, "text": text.strip(), "quote": quote, "note": note,
            "by": by, "kind": kind, "confirmed": confirmed, "time": now()}


# ------------------------------------------------------------------ checking the AI against the note
def _norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w/%.]+", " ", s.lower())).strip(" .")


def _numbers(s):
    return set(re.findall(r"\d+(?:\.\d+)?", s))


def verify(item, note):
    """None if the item is supported by the note, else the reason it is rejected."""
    quote, text = item.get("quote", ""), item.get("text", "")
    if not quote or not text:
        return "no quote"
    if _norm(quote) not in _norm(note):
        return "quote not found in the note"
    if not _numbers(text) <= _numbers(quote):
        return "a number that is not in the quote"
    return None


def parse_items(raw):
    """MedGemma's answer -> list of dicts. Tolerates code fences and an answer cut off mid-list."""
    start = raw.find("[")
    if start < 0:
        return []
    body = raw[start:]
    for candidate in (body[:body.rfind("]") + 1], body[:body.rfind("}") + 1] + "]"):
        try:
            out = json.loads(candidate)
            return [x for x in out if isinstance(x, dict)]
        except (ValueError, TypeError):
            continue
    return [json.loads(m) for m in re.findall(r"\{[^{}]*\}", body) if _loads_ok(m)]


def _loads_ok(s):
    try:
        json.loads(s)
        return True
    except ValueError:
        return False


# ------------------------------------------------------------------ filing one note
def pieces(note):
    """The note cut into short pieces (sentences, then commas), without the dot in 37.9."""
    out = []
    for sentence in re.split(charting.SENTENCE, note):
        sentence = (sentence or "").strip(" -:")
        parts = [x.strip(" -:") for x in re.split(r",(?!\d)", sentence)]
        # nurse shorthand ("CP, sweaty, SOB") is split; a written sentence ("lightheadedness, nausea, and flushing") isn't
        if len(parts) > 1 and all(len(x.split()) <= 5 for x in parts) and not re.search(r",\s*(and|or)\b", sentence):
            out += [x for x in parts if x]
        elif sentence:
            out.append(sentence)
    return out


def _only_numbers(piece, sources):
    """True if the piece is just vital signs / age / arrival already shown as tiles (no other words)."""
    rest = piece.lower()
    for s in sources:
        if s:
            rest = rest.replace(s.lower(), " ")
    rest = re.sub(r"\b(ra|on air|room air|on arrival|arrival|pain|nrs|score|avpu|gcs|bpm|mmhg|sats?|yo|y/o|years?|old|"
                  r"male|female|m|f|walk-?in|biba|ambulance|\d+(?:\.\d+)?|°c|°f|c|f|%|/10|/)\b", " ", rest)
    return not re.search(r"[a-z]{2,}", rest)


def rule_section(piece):
    if re.search(IMAGING, piece, re.I):
        return "Investigations"
    return next((s for s, rule in RULES if re.search(rule, piece, re.I)), "History of presenting complaint")


def file_note(ai, note, note_index, who, sources):
    """One note -> (items, rejected). MedGemma tidies; code checks each line; any piece of the note that no accepted
    line covers is filed in the nurse's own words, so nothing typed is ever lost."""
    items, rejected = [], []
    if ai is not None:
        from answers import final_answer
        from model import _messages
        msgs = _messages(SYSTEM, f"Note:\n{EXAMPLE_NOTE}")
        msgs += [{"role": "assistant", "content": [{"type": "text", "text": EXAMPLE_JSON}]},
                 {"role": "user", "content": [{"type": "text", "text": f"Note:\n{note}"}]}]
        raw = final_answer(ai._ask(msgs, "report")) or ""
        for x in parse_items(raw):
            section = next((s for s in AI_SECTIONS if s.lower() == str(x.get("section", "")).strip().lower()), None)
            x = {k: str(x.get(k, "")).strip() for k in ("section", "text", "quote")}
            why = "unknown section" if section is None else verify(x, note)
            if why:
                rejected.append({**x, "why": why, "note": note_index})
                continue
            if any(i["section"] == section and i["text"].lower() == x["text"].lower() for i in items):
                continue
            items.append(new_item(section, x["text"], "ai", "AI", x["quote"], note_index))
    covered = " ".join(_norm(i["quote"]) for i in items)
    complaint_given = any(i["section"] == "Presenting complaint" for i in items)
    for piece in pieces(note):
        piece = _strip_admin(piece)
        if not piece or _norm(piece) in covered or _only_numbers(piece, sources):
            continue
        if any(_norm(i["quote"]) in _norm(piece) and len(_norm(i["quote"])) >= 0.6 * len(_norm(piece)) for i in items):
            continue
        m = re.match(r"(.+?)\s+on\s+([a-z].+)$", piece, re.I)       # "HTN on metformin" -> history + medicine
        if m and re.search(dict(RULES)["Past medical history"], m.group(1), re.I):
            items.append(new_item("Past medical history", m.group(1), "typed", who, piece, note_index))
            items.append(new_item("Medications", m.group(2), "typed", who, piece, note_index))
            continue
        items.append(new_item(rule_section(piece), piece, "typed", who, piece, note_index))
    if not complaint_given and note_index == 0:
        first = next((i for i in items if i["section"] == "History of presenting complaint"), None)
        if first is not None and len(first["text"].split()) <= 8:   # nurse shorthand: the first symptom is the complaint
            first["section"] = "Presenting complaint"
        else:
            c = charting.first_clause(note)
            if c != "not recorded":
                items.insert(0, new_item("Presenting complaint", c, "typed", who, c, note_index))
    return items, rejected


ADMIN = (r"^\s*(?:an?\s+)?\d{1,3}\s*(?:yo|y/o|y\.o\.|yrs?|-?years?(?:-old| old)?)?\s*-?\s*(?:m|f|male|female|man|woman)\b[\s,.]*"
         r"|\b(?:walk[- ]?in|walked in|self[- ]?present\w*|brought (?:in )?by (?:family|ambulance|friends?|\w+)|biba|bibs)\b")


def _strip_admin(piece):
    """'86F dyspnea walk-in' -> 'dyspnea': age/sex and arrival mode are tiles already."""
    out = re.sub(r"\s+", " ", re.sub(ADMIN, " ", piece, flags=re.I)).strip(" ,.;:-")
    if out == piece.strip(" ,.;:-"):                   # nothing stripped: keep any real words ("hx DM", "NKDA")
        return out if re.search(r"[a-z]{2,}", out, re.I) else ""
    return out if re.search(r"[a-z]{4,}", out, re.I) else ""


# ------------------------------------------------------------------ parts written by code
def fmt(k, v):
    if v is None:
        return "not recorded"
    if k == "sex":
        return v
    if k == "mental":
        return {"alert": "Alert (A)", "verbal": "Responds to voice (V)", "pain": "Responds to pain (P)",
                "unresponsive": "Unresponsive (U)"}.get(v, v)
    if k == "ambulance":
        return "Ambulance" if v else "Walk-in"
    return f"{v:g}{UNITS.get(k, '')}"


def observation_items(fields, source):
    out = []
    order = [("hr", None), ("sbp", "dbp"), ("rr", None), ("spo2", None), ("temp_c", None), ("nrs_pain", None),
             ("mental", None)]
    for k, k2 in order:
        if k not in fields:
            continue
        if k2:
            text = f"Blood pressure {fields[k]:g}/{fields.get(k2, 0):g} mmHg" if k2 in fields else f"Systolic BP {fields[k]:g} mmHg"
        else:
            text = f"{NAMES[k]} {fmt(k, fields[k])}"
        if k == "spo2":
            air = source.get(k, {}).get("air")
            text += f" on {air}" if air else " (air or oxygen not recorded)"
        out.append(new_item("Observations", text, "auto", "From the note (fixed rules)", source.get(k, {}).get("text", "")))
    for k, s in source.items():
        if s.get("prehospital"):
            out.append(new_item("Observations", f"Before arrival: {NAMES.get(k, k)} {fmt(k, s.get('value'))} "
                                "(not an arrival value)", "auto", "From the note (fixed rules)", s.get("text", "")))
    return out


def triage_items(decision):
    return [new_item("Triage decision support", line, "auto", "Trained models") for line in decision.lines()]


def replace_auto(items, section, new):
    """Swap the code-written items of one section; people's items there stay."""
    keep = [i for i in items if not (i["section"] == section and i["kind"] == "auto")]
    return keep + new


def ordered(items):
    return sorted(items, key=lambda i: SECTIONS.index(i["section"]) if i["section"] in SECTIONS else 99)


def complaint(items):
    for i in items:
        if i["section"] == "Presenting complaint" and i["text"]:
            return i["text"]
    return ""


def as_text(items, title=""):
    """The whole case file as plain text (download, chat, handover)."""
    lines = [title] if title else []
    for s in SECTIONS:
        xs = [i for i in items if i["section"] == s]
        lines += ["", s.upper()]
        lines += [f"- {i['text']}" + ("" if i["kind"] in ("person", "auto") or i["confirmed"] else " [AI, unchecked]"
                                       if i["kind"] == "ai" else "") for i in xs] or ["- not recorded"]
    return "\n".join(lines).strip()
