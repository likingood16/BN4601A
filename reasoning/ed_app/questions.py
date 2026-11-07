"""What's missing and what to do: questions (information to get) and to-dos (actions), specific to the complaint.

The old app asked MedGemma alone, and in the real-AI check it asked the same three things for every patient (past
history, medications, allergies) and turned each question into a to-do ("Obtain past medical history"). Here:
  1. Fixed ED pathways (cut-offs and sources in PATHWAYS) give the questions and actions that matter for THIS
     complaint, e.g. stroke -> time last known well, glucose, urgent CT. Each question has an `answered` rule, so it
     disappears once the notes contain the answer.
  2. Red-flag numbers add actions (low SpO2 -> oxygen; SBP <= 90 -> IV access + senior review).
  3. MedGemma may add up to 2 more questions; anything already recorded or already asked is dropped.
To-dos are actions only. Questions are things to find out. They never repeat each other.
"""
import json
import re

# (name, complaint pattern, [(question, answered-pattern)], [action])
# Sources: chest pain ECG <= 10 min (AHA/ACC 2021); stroke LKW, glucose, CT (AHA/ASA 2019); sepsis qSOFA (Sepsis-3,
# 2016); head injury anticoagulant + CT rules (NICE NG232, 2023); syncope (ESC 2018); ectopic risk (NICE NG126).
PATHWAYS = [
    ("Chest pain", r"chest (pain|tight|discomfort|pressure|heav)|\bcp\b|angina",
     [("When did the pain start, and is it constant or coming and going?", r"since|onset|start|\d+ ?(min|h|hr|hours|days?)\b|\b\d{4}\b|/7|/24"),
      ("Does the pain spread (arm, jaw, back), and is there sweating or breathlessness?", r"radiat|arm|jaw|back|sweat|diaphor|clammy|sob|breath"),
      ("Any cardiac history or risk factors (IHD, stents, diabetes, smoking)?", r"ihd|stent|mi\b|cabg|\bdm\b|diabet|smok|cardiac hx|no hx|nil pmh")],
     ["ECG within 10 minutes of arrival", "Troponin", "Continuous cardiac monitoring"]),
    ("Possible stroke", r"weak|numb|slurr|speech|aphasi|dysarthri|facial (droop|palsy)|droop|hemipleg|stroke|\bcva\b|\btia\b",
     [("When was the patient last known well (exact time)?", r"last (known|seen) well|\blkw\b|since \d|onset \d|woke up"),
      ("Is the patient on blood thinners (warfarin, DOAC)?", r"warfarin|apixaban|rivaroxaban|dabigatran|edoxaban|doac|anticoag|no blood thinner")],
     ["Capillary glucose now", "Stroke team / urgent brain CT (time-critical)", "NIHSS or FAST score"]),
    ("Breathing", r"dyspn|short(ness)? of breath|\bsob\b|breathless|wheez|asthma|copd|cough",
     [("Is the patient on home oxygen or inhalers, and how does this compare with normal?", r"home o2|home oxygen|inhaler|nebul|baseline|usual"),
      ("Any chest pain, fever or leg swelling with it?", r"chest (pain|discomfort|tight|pressure)|\bcp\b|fever|febrile|leg swell|calf|no cp")],
     ["Repeat SpO₂ and work-of-breathing check", "Chest X-ray", "Peak flow if asthma"]),
    ("Abdominal pain", r"abdo|abdominal|stomach|epigastr|\brif\b|\bliq\b|flank|vomit|diarrh",
     [("Where exactly is the pain, and any vomiting, diarrhoea or blood?", r"\brif\b|\bruq\b|\bliq\b|\bluq\b|epigastr|umbilic|vomit|diarrh|blood|melaena"),
      ("Could the patient be pregnant (if of childbearing age)?", r"pregnan|lmp|\bupt\b|hcg|postmeno")],
     ["Urine dipstick (and pregnancy test if relevant)", "Bloods: FBC, U&E, LFT, lipase", "Analgesia review"]),
    ("Head injury", r"head (injur|trauma|strike|hit)|hit (his|her|their) head|\bhi\b|fell.*head|head.*fell|\bloc\b",
     [("Any loss of consciousness, vomiting or amnesia?", r"\bloc\b|loss of consciousness|vomit|amnesi|no loc"),
      ("Is the patient on blood thinners?", r"warfarin|apixaban|rivaroxaban|dabigatran|doac|anticoag|aspirin|clopidogrel|no blood thinner")],
     ["GCS every 30 minutes", "Check CT head rules (NICE)"]),
    ("Fall / injury", r"\bfall\b|fell|trip|slip|injur|trauma|\bmva\b|\brta\b|accident|assault|sprain|fractur",
     [("How did it happen (mechanism), and why did they fall?", r"trip|slip|dizz|syncop|collapse|mechanism|hit by|from height|stairs|ladder"),
      ("Is the patient on blood thinners?", r"warfarin|apixaban|rivaroxaban|dabigatran|doac|anticoag|no blood thinner")],
     ["X-ray of the injured area", "Tetanus status", "Analgesia"]),
    ("Syncope", r"syncop|faint|collapse|passed out|blackout|loss of consciousness|lightheaded|light-headed|dizz|presyncop",
     [("What happened just before (exertion, standing, chest pain, palpitations)?", r"exert|standing|stood|chest pain|palpit|prodrom|warning"),
      ("Any injury from the fall or tongue biting / incontinence?", r"injur|tongue|incontin|no injur")],
     ["12-lead ECG", "Capillary glucose", "Lying and standing blood pressure"]),
    ("Seizure", r"seiz|fit\b|fitting|convuls|epilep",
     [("How long did the seizure last, and is the patient back to normal?", r"\d+ ?(min|sec)|lasted|postictal|back to (normal|baseline)"),
      ("Known epilepsy, and which medicines?", r"epilep|levetiracetam|valpro|keppra|carbamazep|phenytoin|first seizure")],
     ["Capillary glucose", "Airway check and recovery position"]),
    ("Poisoning", r"overdose|\bod\b|poison|ingest|intoxic|took \d+",
     [("Which substance, how much and when?", r"\d+ ?(mg|g|tabs?|tablets)|\bat \d|hours? ago|paracetamol|alcohol"),
      ("Was it deliberate (self-harm)?", r"deliberate|intentional|accidental|self.?harm|suicid")],
     ["Paracetamol and salicylate levels", "12-lead ECG", "Mental health / risk assessment"]),
    ("Possible infection", r"fever|febrile|pyrex|infect|sepsis|rigor|chills|uti|urinary|dysuria|painful urination|cellulitis",
     [("Where might the infection be from (urine, chest, skin, abdomen)?", r"urin|dysuria|cough|sputum|wound|cellul|abdo|source"),
      ("Is the patient immunosuppressed (chemo, steroids, transplant)?", r"chemo|steroid|transplant|immuno|neutropen|\baids\b|\bhiv\b")],
     ["Sepsis screen (qSOFA / lactate)", "Blood cultures before antibiotics if septic", "Urine dipstick"]),
    ("Headache", r"headache|\bha\b|migraine",
     [("Was it sudden and the worst ever (thunderclap)?", r"sudden|thunderclap|worst|gradual"),
      ("Any fever, neck stiffness, or new neurological symptoms?", r"fever|neck stiff|photophob|weak|numb|vision")],
     ["Neuro observations", "Check CT head criteria"]),
]
GENERAL = [("Any allergies?", r"allerg|nkda|\bnka\b"),
           ("Regular medicines?", r"\bon (?!(?:arrival|air|exam\w*|evaluation|admission|the|a)\b)[a-z]{4,}|\bmeds\b|medication|"
                                   r"tablets?|inhaler|insulin|nil regular")]
ASKS_FOR = {"sbp": r"blood pressure|\bbp\b", "hr": r"heart rate|pulse|\bhr\b", "spo2": r"oxygen saturation|spo2|\bsats?\b",
            "temp_c": r"temperature|\btemp\b", "mental": r"consciousness|\bgcs\b|\bavpu\b", "rr": r"respiratory rate|resp rate",
            "nrs_pain": r"pain (?:level|score|severity)", "age": r"\bage\b|how old", "sex": r"\bsex\b|gender"}

SYSTEM = (
    "You support emergency department clinicians. Read the case file and the questions already asked. Reply with "
    'JSON only: {"questions": ["...", "..."]}: at most 2 MORE questions about missing information that would change '
    "emergency care for THIS patient. Never ask about anything already in the case file or already asked. "
    "Each question at most 15 words.")


def pathways(text):
    low = text.lower()
    return [p for p in PATHWAYS if re.search(p[1], low)]


def vital_actions(fields, decision=None):
    """Red-flag numbers -> actions (NEWS2 single-parameter red scores, RCP 2017)."""
    f, out = fields, []
    if f.get("spo2") is not None and f["spo2"] <= 91:
        out.append("SpO₂ ≤ 91%: oxygen and senior review now")
    if f.get("sbp") is not None and f["sbp"] <= 90:
        out.append("SBP ≤ 90: IV access, fluids decision, senior review now")
    if f.get("hr") is not None and (f["hr"] >= 131 or f["hr"] <= 40):
        out.append(f"HR {f['hr']:g}: cardiac monitor and ECG")
    if f.get("rr") is not None and (f["rr"] >= 25 or f["rr"] <= 8):
        out.append(f"RR {f['rr']:g}: senior review of breathing")
    if f.get("mental") in ("verbal", "pain", "unresponsive"):
        out.append("Altered consciousness: capillary glucose and airway check")
    if f.get("temp_c") is not None and f["temp_c"] >= 38 and f.get("hr") is not None and f["hr"] > 90:
        out.append("Fever with fast heart rate: sepsis screen")
    if decision is not None and getattr(decision, "news2", 0) >= 7:
        out.insert(0, f"NEWS2 {decision.news2}: emergency senior review")
    return out


def _known(q, fields):
    low = q.lower()
    asked = [k for k, rule in ASKS_FOR.items() if re.search(rule, low)]
    return bool(asked) and all(fields.get(k) is not None for k in asked)


def _words(s):
    return set(re.findall(r"[a-z]{4,}", s.lower()))


def _similar(a, b):
    wa, wb = _words(a), _words(b)
    return bool(wa and wb) and len(wa & wb) / min(len(wa), len(wb)) >= 0.6


def find(ai, complaint, notes_text, case_text, fields, decision=None, dismissed=()):
    """-> (questions [{text, why, by}], todos [{text, why, by}])."""
    low = notes_text.lower()
    found = pathways(f"{complaint}\n{notes_text}")
    questions, todos = [], []
    for name, _, qs, actions in found:
        for q, answered in qs:
            if "pregnant" in q and (fields.get("sex") == "male" or (fields.get("age") or 0) > 55):
                continue
            if not re.search(answered, low) and not any(_similar(q, x["text"]) for x in questions):
                questions.append({"text": q, "why": name, "by": "ED pathway"})
        todos += [{"text": a, "why": name, "by": "ED pathway"} for a in actions]
    for q, answered in GENERAL:
        if not re.search(answered, low) and not any(_similar(q, x["text"]) for x in questions):
            questions.append({"text": q, "why": "Basic safety", "by": "ED pathway"})
    todos = [{"text": a, "why": "Vital signs", "by": "Vital-sign rules"} for a in vital_actions(fields, decision)] + todos
    if ai is not None:
        from answers import final_answer
        from model import _messages
        asked = "\n".join(f"- {q['text']}" for q in questions) or "- none"
        raw = final_answer(ai._ask(_messages(SYSTEM, f"CASE FILE:\n{case_text}\n\nALREADY ASKED:\n{asked}"),
                                   "questions")) or ""
        for q in _parse(raw)[:2]:
            if not _known(q, fields) and not any(_similar(q, x["text"]) for x in questions):
                questions.append({"text": q, "why": "Suggested by AI", "by": "AI"})
    gone = {d.lower() for d in dismissed}
    questions = [q for q in questions if q["text"].lower() not in gone][:6]
    seen, out = set(), []
    for t in todos:
        if t["text"].lower() not in seen:
            seen.add(t["text"].lower())
            out.append(t)
    return questions, out[:6]


def _parse(raw):
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        qs = json.loads(m.group(0)).get("questions", []) if m else []
    except (ValueError, AttributeError):
        qs = re.findall(r'"([^"]{8,160}\?)"', raw)
    return [str(q).strip() for q in qs if str(q).strip()]
