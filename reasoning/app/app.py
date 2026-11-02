"""Temporary Gradio app: an ED triage desk on one screen. All the medicine lives in src/pipeline.py.

    python app.py            # test mode: fake written answers, every button responds in seconds
    python app.py --real     # the real pipeline: needs a big GPU (MedGemma 1.5 + Cato's model), too much for 4 GB

Built around how EDs work: a waiting list, a quick "critical look" first, the nurse decides (the AI only suggests),
waiting patients get repeat vitals, and the first triage level stays on record (rechecks are recorded separately).
"""
import argparse
import sys
import tempfile
import time
from pathlib import Path

import gradio as gr
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402  (puts Caleb's code on the path)
import news2  # noqa: E402
from visit import SECTIONS  # noqa: E402

IMAGE_CHOICES = [("Bone X-ray (adds fracture outlines)", "bone_xray"), ("Chest X-ray", "chest_xray"),
                 ("CT, one slice", "ct_slice"), ("Ultrasound", "ultrasound"), ("MRI, one slice", "mri_slice"),
                 ("ECG picture (not tested)", "ecg"), ("Other", "other")]
RESULT_CHOICES = [("Blood tests", "bloods"), ("ECG result", "ecg"), ("Bedside checks", "bedside"),
                  ("Urine and swabs", "urine_swabs"), ("History of presenting complaint", "history"),
                  ("Past history", "past_history"), ("Medications", "medications"), ("Allergies", "allergies"),
                  ("Imaging report", "imaging"), ("Treatment given", "treatment"), ("Consultation notes", "notes"),
                  ("Disposition", "disposition")]
assert {key for _, key in RESULT_CHOICES} == set(SECTIONS)
NR = "not recorded"
BLANK = "blank = not recorded"
# our own short pick-list; anything else can be typed
COMPLAINTS = ["Chest pain", "Shortness of breath", "Abdominal pain", "Headache", "Weakness or speech difficulty",
              "Fever", "Collapse or fainting", "Seizure", "Possible overdose or poisoning", "Allergic reaction",
              "Head injury", "Limb injury", "Fall", "Wound or cut", "Back pain", "Vomiting or diarrhoea"]
# made up for the demo, not real patients: (minutes waiting, form values in the order of FORM)
DEMO = [(6, ["67", "male", "walk-in", "Chest pain", "Crushing, 1 hour, sweating", "112", "94", "60", "24", "91", "36.8", "alert", "8", "no"]),
        (25, ["45", "female", "walk-in", "Fever", "3 days, cough", "96", "118", "74", "20", "96", "38.4", "alert", "3", "no"]),
        (50, ["24", "female", "walk-in", "Limb injury", "Twisted right ankle playing football", "78", "122", "76", "14", "99", "36.6", "alert", "4", "yes"])]
QUICK = {"Main concerns": "What are your main concerns for this patient, and how urgent is this?",
         "Explain to the patient": "Please explain what is going on to the patient in simple, non-medical words.",
         "What is missing?": "What important information is still missing, and what would you ask for next?"}
# Canadian CTAS (see REPORT_NOTES 5f): minutes to see a doctor, and to recheck someone waiting. We found no public
# Singapore targets, so these are shown as Canadian.
SEE_WITHIN = {1: 0, 2: 15, 3: 30, 4: 60, 5: 120}
RECHECK_EVERY = {1: 0, 2: 15, 3: 30, 4: 60, 5: 120}
ROUGH_PACS = {1: "P1", 2: "P2", 3: "P2/P3", 4: "P3", 5: "P4"}    # our own conversion, not validated
VITAL_NAMES = {"hr": "HR", "sbp": "SBP", "dbp": "DBP", "rr": "RR", "spo2": "SpO2", "temp_c": "Temp", "nrs_pain": "Pain"}
LIMITS = {"Age": (0, 120), "Heart rate": (0, 300), "Systolic BP": (0, 300), "Diastolic BP": (0, 200),
          "Breathing rate": (0, 80), "Oxygen %": (0, 100), "Temperature C": (25, 45), "Pain 0-10": (0, 10)}
START = "Pick a patient from the waiting list, or triage a new one."
WARNING = ("**Decision support only: the nurse and doctor decide.** Research prototype, not for clinical use. "
           "A negative AI result doesn't rule anything out.")
CSS = """
.strip {background:#1f2937;color:#fff;padding:10px 14px;border-radius:8px}
.strip b {color:#fff}
.chip {display:inline-block;padding:2px 10px;border-radius:12px;font-weight:700;color:#fff;margin-right:8px}
.l1, .l2 {background:#b42318} .l3 {background:#b54708} .l4, .l5 {background:#1d6f42}
.late {color:#fda29b;font-weight:700}
.ai-urgent {background:#b42318;color:#fff;padding:10px 14px;border-radius:8px}
.ai-calm {background:#1d6f42;color:#fff;padding:10px 14px;border-radius:8px}
.ai-urgent b, .ai-calm b {color:#fff}
"""
ed = None          # the pipeline, one per server


def number(name, text, problems):
    """Blank -> None (not recorded). Not a number or outside the limits -> a message for the user."""
    text = str(text if text is not None else "").strip()
    if not text:
        return None
    low, high = LIMITS[name]
    try:
        value = float(text)
    except ValueError:
        problems.append(f"{name}: '{text}' is not a number")
        return None
    if not low <= value <= high:
        problems.append(f"{name}: {text} is outside {low}-{high}")
        return None
    return value


def read_vitals(hr, sbp, dbp, rr, spo2, temp, pain, problems):
    names = ("Heart rate", "Systolic BP", "Diastolic BP", "Breathing rate", "Oxygen %", "Temperature C", "Pain 0-10")
    return dict(zip(("hr", "sbp", "dbp", "rr", "spo2", "temp_c", "nrs_pain"),
                    (number(n, x, problems) for n, x in zip(names, (hr, sbp, dbp, rr, spo2, temp, pain)))))


def read_form(form, problems, default_complaint=""):
    age, sex, arrival, pick, details, hr, sbp, dbp, rr, spo2, temp, mental, pain, injured = form
    complaint = ": ".join(x.strip() for x in (pick or "", details or "") if x.strip()) or default_complaint
    return {"age": number("Age", age, problems), "sex": None if sex == NR else sex, "chief_complaint": complaint,
            "ambulance": {"ambulance": 1.0, "walk-in": 0.0}.get(arrival), "injured": {"yes": 1.0, "no": 0.0}.get(injured),
            "mental": None if mental == NR else mental,
            **read_vitals(hr, sbp, dbp, rr, spo2, temp, pain, problems)}


def level_of(entry):
    return entry["nurse_level"] or entry["visit"].decision.level


def current(desk):
    return desk["patients"].get(desk["current"])


def add_patient(desk, form, patient, waited=0, nurse_level=None, note=None):
    visit = ed.triage(patient, explain=False)
    pid = f"p{desk['next']}"
    desk["next"] += 1
    desk["patients"][pid] = {"form": list(form), "patient": patient, "visit": visit, "arrived": time.time() - waited * 60,
                             "nurse_level": nurse_level, "nurse_note": note, "last_check": time.time() - waited * 60,
                             "news2": visit.decision.news2, "chat": [], "overlay": None, "image_text": "", "sbar": "", "sbar_file": None}
    if note:
        visit.add("notes", note)
    return pid


def queue_update(desk):
    icon = {1: "🔴", 2: "🔴", 3: "🟡", 4: "🟢", 5: "🟢"}
    order = sorted(desk["patients"].items(), key=lambda kv: (level_of(kv[1]), kv[1]["arrived"]))
    choices = []
    for pid, e in order:
        p = e["patient"]
        who = " ".join(x for x in (f"{p['age']:.0f}" if p["age"] is not None else "", (p["sex"] or "")[:1].upper()) if x)
        level = level_of(e)
        mark = "" if e["nurse_level"] else " (unconfirmed)"
        choices.append((f"{icon[level]} L{level} | {who} {p['chief_complaint'][:44]} | {int((time.time() - e['arrived']) // 60)} min{mark}", pid))
    return gr.update(choices=choices, value=desk["current"])


def header_html(e):
    if e is None:
        return ""
    p, level = e["patient"], level_of(e)
    waited = int((time.time() - e["arrived"]) // 60)
    target = SEE_WITHIN[level]
    if target == 0:
        see = "<span class='late'>See immediately</span>"
    else:
        left = target - waited
        see = f"See within {target} min: " + (f"{left} min left" if left >= 0 else f"<span class='late'>{-left} min overdue</span>")
    every, since = RECHECK_EVERY[level], int((time.time() - e["last_check"]) // 60)
    if every == 0:
        check = "Monitor continuously"
    else:
        check = f"Recheck every {every} min: " + (f"last {since} min ago" if since <= every else f"<span class='late'>due, last {since} min ago</span>")
    who = " ".join(x for x in (f"{p['age']:.0f} yrs" if p["age"] is not None else "age not recorded", p["sex"] or "") if x)
    state = "nurse confirmed" if e["nurse_level"] else "AI suggestion, not yet confirmed"
    return (f"<div class='strip'><span class='chip l{level}'>LEVEL {level}</span> <b>{who} | {p['chief_complaint']}</b><br>"
            f"{state} | rough PACS {ROUGH_PACS[level]} (our own conversion, untested) | waited {waited} min<br>"
            f"{see} | {check} <small>(Canadian CTAS targets)</small></div>")


def ai_html(e):
    if e is None:
        return ""
    d = e["visit"].decision
    why = " (safety floor from vital signs)" if d.floor_flag and d.flagged else ""
    verdict = f"<b>FLAGGED URGENT{why}</b>" if d.flagged else "<b>Not flagged as urgent</b>"
    options = ", ".join(str(x) for x in d.level_set)
    return (f"<div class='{'ai-urgent' if d.flagged else 'ai-calm'}'>AI suggests: {verdict}<br>Urgency score {d.p_urgent:.0%} | "
            f"Level {d.level} (levels it can't rule out: {options}) | NEWS2 {d.news2}</div>")


def explanation_text(e):
    if e is None:
        return ""
    x = e["visit"].explanation
    if not x:
        return "*No explanation written yet. Press \"Explain this decision\".*"
    bullets = lambda key: "\n".join(f"- {i}" for i in x.get(key) or []) or "- none given"
    return (f"**Why:** {x.get('reasoning', 'not given')}\n\n**Department:** {x.get('department', 'not given')}\n\n"
            f"**Red flags**\n{bullets('red_flags')}\n\n**Next steps**\n{bullets('next_steps')}\n\n"
            f"**Missing information**\n{bullets('missing_information')}")


def view(desk):
    """Everything on screen that depends on who is selected. Order = OUT in build()."""
    e = current(desk)
    if e is None:
        return desk, queue_update(desk), "", "", "", "", "", [], None, "", "", None
    return (desk, queue_update(desk), header_html(e), ai_html(e), e["nurse_note"] or "*Not yet confirmed by a nurse.*",
            explanation_text(e), e["visit"].record(), e["chat"], e["overlay"], e["image_text"], e["sbar"], e["sbar_file"])


def load_demo(desk):
    for waited, form in DEMO:
        problems = []
        add_patient(desk, form, read_form(form, problems), waited)
    return view(desk)


def triage(desk, *form):
    """Stage 1. The decision shows at once, then the explanation fills in."""
    if current(desk):
        gr.Warning("This patient is already triaged, and the first level stays on record. Save repeat vitals below, or press + New patient.")
        yield view(desk)
        return
    problems = []
    patient = read_form(form, problems)
    if not patient["chief_complaint"]:
        problems.append("please pick or type the presenting complaint")
    if problems:
        gr.Warning("Please fix: " + "; ".join(problems))
        yield view(desk)
        return
    pid = add_patient(desk, form, patient)
    desk["current"] = pid
    yield view(desk)
    entry = desk["patients"][pid]
    ed.explain(entry["visit"])
    yield view(desk)


def critical(desk, *form):
    """The critical look: no typing needed. Whatever is on the form comes along if the patient is new."""
    if current(desk):
        e = current(desk)
        e["nurse_level"], e["nurse_note"] = 1, "Nurse critical look: set to level 1, resuscitation."
        e["visit"].add("notes", e["nurse_note"])
        return view(desk)
    problems = []
    patient = read_form(form, problems, default_complaint="Critically ill on arrival (nurse critical look)")
    if problems:
        gr.Warning("Please fix: " + "; ".join(problems))
        return view(desk)
    desk["current"] = add_patient(desk, form, patient, nurse_level=1, note="Nurse critical look: set to level 1, resuscitation.")
    return view(desk)


def accept(desk):
    e = current(desk)
    if e is None:
        gr.Warning(START)
        return view(desk)
    if e["nurse_level"] and e["nurse_level"] < e["visit"].decision.level:
        gr.Warning(f"This would lower the patient from level {e['nurse_level']}. Use Change level, with a reason.")
        return view(desk)
    e["nurse_level"] = e["visit"].decision.level
    e["nurse_note"] = f"Nurse accepted the AI's level {e['nurse_level']}."
    e["visit"].add("notes", e["nurse_note"])
    return view(desk)


def change_level(desk, level, reason):
    e = current(desk)
    if e is None:
        gr.Warning(START)
        return (*view(desk), gr.update(), gr.update())
    if not level or not (reason or "").strip():
        gr.Warning("Pick a level and give a reason.")
        return (*view(desk), gr.update(), gr.update())
    e["nurse_level"] = int(level)
    e["nurse_note"] = f"Nurse set level {e['nurse_level']} (AI suggested {e['visit'].decision.level}). Reason: {reason.strip()}"
    e["visit"].add("notes", e["nurse_note"])
    return (*view(desk), None, "")


def recheck(desk, hr, sbp, dbp, rr, spo2, temp, mental, pain):
    e = current(desk)
    if e is None:
        gr.Warning(START)
        return view(desk)
    problems = []
    v = read_vitals(hr, sbp, dbp, rr, spo2, temp, pain, problems)
    if problems:
        gr.Warning("Please fix: " + "; ".join(problems))
        return view(desk)
    mental = None if mental == NR else mental
    if all(x is None for x in v.values()) and mental is None:
        gr.Warning("Enter at least one repeat observation.")
        return view(desk)
    score = news2.score(rr=v["rr"], spo2=v["spo2"], sbp=v["sbp"], hr=v["hr"], temp_c=v["temp_c"], consciousness=mental)["news2"]
    seen = ", ".join(f"{VITAL_NAMES[k]} {x:g}" for k, x in v.items() if x is not None)
    seen = ", ".join(x for x in (seen, f"consciousness {mental}" if mental else "") if x)
    e["visit"].add("bedside", f"Repeat vitals at {time.strftime('%H:%M')}: {seen}. NEWS2 {score}.")
    if score > e["news2"]:
        gr.Warning(f"NEWS2 went up from {e['news2']} to {score}. Consider changing the level. (The score only counts what was entered.)")
    e["news2"], e["last_check"] = score, time.time()
    return view(desk)


def explain(desk):
    e = current(desk)
    if e is None:
        gr.Warning(START)
        yield view(desk)
        return
    ed.explain(e["visit"])
    yield view(desk)


def select(desk, pid):
    desk["current"] = pid
    e = current(desk)
    return (*view(desk), *(e["form"] if e else blank_form()), None, "")


def blank_form():
    return ["", NR, NR, None, "", "", "", "", "", "", "", NR, "", NR]


def new_patient(desk):
    desk["current"] = None
    return (*view(desk), *blank_form(), None, "")


def remove(desk):
    if current(desk) is None:
        gr.Warning(START)
        return (*view(desk), *[gr.update()] * 16)
    del desk["patients"][desk["current"]]
    desk["current"] = None
    return (*view(desk), *blank_form(), None, "")


def add_result(desk, key, text):
    e = current(desk)
    if e is None:
        gr.Warning(START)
        return (*view(desk), gr.update())
    if not (text or "").strip():
        gr.Warning("Please type the result first.")
        return (*view(desk), gr.update())
    e["visit"].add(key, text.strip())
    gr.Info("Added to the record.")
    return (*view(desk), "")


def read_image(desk, path, image_type):
    e = current(desk)
    if e is None:
        gr.Warning(START)
        return view(desk)
    if not path:
        gr.Warning("Please upload an image first.")
        return view(desk)
    out = ed.read_image(e["visit"], path, image_type)
    found = out["detector"]
    text = f"**AI description, not a radiologist report:** {out['text'] or 'no description produced'}"
    if found is not None:
        text += (f"\n\n**Fracture detector:** {found['n']} possible fracture(s) marked, top confidence {found['top_conf']:.0%}."
                 if found["n"] else "\n\n**Fracture detector:** nothing marked. This does not rule out a fracture.")
    e["overlay"] = np.ascontiguousarray(found["overlay"]) if found is not None else path
    e["image_text"] = text
    return view(desk)


def send(desk, question):
    e = current(desk)
    if e is None:
        gr.Warning(START)
        yield (*view(desk), gr.update())
        return
    question = (question or "").strip()
    if not question:
        gr.Warning("Please type a question.")
        yield (*view(desk), gr.update())
        return
    e["chat"] = e["chat"] + [{"role": "user", "content": question}]
    yield (*view(desk), "")
    answer = ed.chat(e["visit"], question)
    e["chat"] = e["chat"] + [{"role": "assistant", "content": answer}]
    yield (*view(desk), "")


def make_sbar(desk):
    e = current(desk)
    if e is None:
        gr.Warning(START)
        return view(desk)
    text = ed.sbar(e["visit"])
    path = Path(tempfile.mkdtemp()) / "sbar_handover.txt"
    path.write_text(text, encoding="utf8")
    e["sbar"], e["sbar_file"] = text, str(path)
    e["chat"] = e["chat"] + [{"role": "user", "content": "Write the SBAR handover."}, {"role": "assistant", "content": text}]
    return view(desk)


def build(test_mode):
    with gr.Blocks(title="ED Triage Desk") as demo:
        gr.Markdown("# ED Triage Desk")
        if test_mode:
            gr.Markdown("### TEST MODE: the urgency score and all written answers are fake. Only for checking buttons.")
        gr.Markdown(WARNING)
        desk = gr.State({"patients": {}, "current": None, "next": 1})
        with gr.Row():
            with gr.Column(scale=1, min_width=260):
                gr.Markdown("### Waiting list")
                queue = gr.Radio([], label="Most urgent first", type="value")
                new_btn = gr.Button("+ New patient", variant="primary")
                demo_btn = gr.Button("Load made-up patients")
                remove_btn = gr.Button("Seen by doctor: remove")
            with gr.Column(scale=3):
                header = gr.HTML()
                critical_btn = gr.Button("🚨 Critically ill: level 1 now", variant="stop")
                ai = gr.HTML()
                with gr.Row():
                    accept_btn = gr.Button("Accept AI level", variant="primary")
                    level = gr.Dropdown(["1", "2", "3", "4", "5"], label="Change level to", value=None)
                    reason = gr.Textbox(label="Reason (required)")
                    change_btn = gr.Button("Change level")
                nurse = gr.Markdown()
                with gr.Accordion("Patient details and vitals", open=True):
                    with gr.Row():
                        age = gr.Textbox(label="Age", placeholder=BLANK)
                        sex = gr.Radio(["female", "male", NR], label="Sex", value=NR)
                        arrival = gr.Radio(["ambulance", "walk-in", NR], label="Arrival", value=NR)
                    with gr.Row():
                        pick = gr.Dropdown(COMPLAINTS, label="Presenting complaint (pick or type)", value=None, allow_custom_value=True)
                        details = gr.Textbox(label="Details")
                    with gr.Row():
                        hr = gr.Textbox(label="Heart rate", placeholder=BLANK)
                        sbp = gr.Textbox(label="Systolic BP", placeholder=BLANK)
                        dbp = gr.Textbox(label="Diastolic BP", placeholder=BLANK)
                        rr = gr.Textbox(label="Breathing rate", placeholder=BLANK)
                    with gr.Row():
                        spo2 = gr.Textbox(label="Oxygen %", placeholder=BLANK)
                        temp = gr.Textbox(label="Temperature C", placeholder=BLANK)
                        pain = gr.Textbox(label="Pain 0-10", placeholder=BLANK)
                    with gr.Row():
                        mental = gr.Dropdown(["alert", "confused", "verbal", "pain", "unresponsive", NR], label="Consciousness", value=NR)
                        injured = gr.Radio(["yes", "no", NR], label="Injury", value=NR)
                    with gr.Row():
                        triage_btn = gr.Button("Run triage", variant="primary")
                        recheck_btn = gr.Button("Save repeat vitals (keeps the first level)")
                with gr.Tabs():
                    with gr.Tab("Why"):
                        explain_btn = gr.Button("Explain this decision")
                        explanation = gr.Markdown()
                    with gr.Tab("Results & images"):
                        with gr.Row():
                            with gr.Column():
                                section = gr.Dropdown(RESULT_CHOICES, label="Type of result", value="bloods")
                                result = gr.Textbox(label="Result", lines=2, placeholder="e.g. Troponin 2.1 ng/mL (high)")
                                add_btn = gr.Button("Add to record")
                                image = gr.Image(label="Upload one image", type="filepath", height=220)
                                image_type = gr.Dropdown(IMAGE_CHOICES, label="Type of image", value="bone_xray")
                                read_btn = gr.Button("Read image")
                            with gr.Column():
                                overlay = gr.Image(label="Image, with any fracture outlines", interactive=False, height=320)
                                image_text = gr.Markdown()
                    with gr.Tab("Chat"):
                        chatbot = gr.Chatbot(label="Consultation", height=360)
                        question = gr.Textbox(label="Your question", placeholder="New results are passed on automatically")
                        send_btn = gr.Button("Send", variant="primary")
                        with gr.Row():
                            quick = {name: gr.Button(name) for name in QUICK}
                    with gr.Tab("SBAR"):
                        sbar_btn = gr.Button("Write SBAR from everything so far", variant="primary")
                        sbar_text = gr.Markdown()
                        sbar_file = gr.File(label="Download the handover", interactive=False)
                    with gr.Tab("Full record"):
                        record = gr.Textbox(label="Patient record (the arrival card never changes)", lines=18, interactive=False)

        FORM = [age, sex, arrival, pick, details, hr, sbp, dbp, rr, spo2, temp, mental, pain, injured]
        OUT = [desk, queue, header, ai, nurse, explanation, record, chatbot, overlay, image_text, sbar_text, sbar_file]
        model = {"concurrency_id": "model", "concurrency_limit": 1}      # one GPU: one answer at a time
        demo_btn.click(load_demo, desk, OUT, api_name="load_demo", **model)
        triage_btn.click(triage, [desk, *FORM], OUT, api_name="triage", **model)
        critical_btn.click(critical, [desk, *FORM], OUT, api_name="critical", **model)
        accept_btn.click(accept, desk, OUT, api_name="accept")
        change_btn.click(change_level, [desk, level, reason], [*OUT, level, reason], api_name="change_level")
        recheck_btn.click(recheck, [desk, hr, sbp, dbp, rr, spo2, temp, mental, pain], OUT, api_name="recheck")
        explain_btn.click(explain, desk, OUT, api_name="explain", **model)
        queue.input(select, [desk, queue], [*OUT, *FORM, level, reason], api_name="select")
        new_btn.click(new_patient, desk, [*OUT, *FORM, level, reason], api_name="new_patient")
        remove_btn.click(remove, desk, [*OUT, *FORM, level, reason], api_name="remove")
        add_btn.click(add_result, [desk, section, result], [*OUT, result], api_name="add_result")
        read_btn.click(read_image, [desk, image, image_type], OUT, api_name="read_image", **model)
        send_btn.click(send, [desk, question], [*OUT, question], api_name="send", **model)
        question.submit(send, [desk, question], [*OUT, question], api_name=False, **model)
        for name, button in quick.items():
            button.click(send, [desk, gr.State(QUICK[name])], [*OUT, question], api_name=False, **model)
        sbar_btn.click(make_sbar, desk, OUT, api_name="sbar", **model)
        # keep the waiting times and countdowns moving
        gr.Timer(30).tick(lambda d: (queue_update(d), header_html(current(d))), desk, [queue, header], api_name=False)
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    if a.real:
        from pipeline import EDPipeline
        ed = EDPipeline()
    else:
        from test_backend import TestPipeline
        ed = TestPipeline()
    build(not a.real).queue().launch(server_port=a.port, css=CSS)
