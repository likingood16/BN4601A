"""Every button of the app, pressed through the back end the screens use, each checked for doing its job.

    python test_app.py            # starts its own test-mode app on port 8002 (empty board, temporary save file)

Each check prints PASS or FAIL. One FAIL = the button doesn't do what it says. At the end the app is restarted to
check that the board and case files survive a restart. The buttons are listed in APP_DESIGN.md (section 11).
"""
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "reasoning" / "src"))
import charting  # noqa: E402  (same_meaning, for the to-do check)

HERE = Path(__file__).resolve().parent
URL = "http://127.0.0.1:8002"
# a FracAtlas TEST fracture (adult, IMG0003458, CC BY 4.0) that detector v3 marks at the adult cut-off
# (52 of the 63 test fractures are marked); copied here so a fresh clone can run this test
IMAGE = HERE / "test_images" / "bone_xray_fracture_fracatlas_IMG0003458.jpg"
CHEST = HERE / "test_images" / "chest_xray_multicare.jpg"
CT = HERE / "test_images" / "ct_slice_multicare.jpg"
# picture rule (14 Nov): MultiCaRe figures (CC BY 4.0) from the test pack (P11 epidural bleed; P07 resolved pneumothorax,
# which the chest models mark amber "possible" only)
CT_BLEED = HERE / "test_images" / "ct_head_bleed_multicare.jpg"
CXR_AMBER = HERE / "test_images" / "chest_xray_pneumothorax_resolved_multicare.jpg"
CALM = "30M fell off bike, hit head, now alert GCS 15. HR 78 BP 124/78 RR 14 sats 99% RA T 36.8 pain 2/10. walk-in"
NURSE = {"who": "Nurse Tan", "role": "Triage nurse"}
DOCTOR = {"who": "Dr Wong", "role": "Doctor"}                 # can look, can't change the board
LEAD = {"who": "Dr Lim", "role": "Organising doctor"}
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"   ({detail})" if detail and not ok else ""), flush=True)


def start(save):
    proc = subprocess.Popen([sys.executable, str(HERE.parent / "server.py"), "--test", "--port", "8002", "--case-files", save],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        try:
            requests.get(f"{URL}/api/meta", timeout=1)
            return proc
        except requests.RequestException:
            time.sleep(1)
    raise SystemExit("the app didn't start")


def get(path):
    return requests.get(URL + path, timeout=30)


def wait(pid):
    for _ in range(240):
        p = get(f"/api/patients/{pid}").json()
        if p["status"] != "working":
            return p
        time.sleep(0.5)
    raise SystemExit(f"{pid} never finished")


def new(text, files=None):
    return requests.post(URL + "/api/patients", data={"text": text, "who": NURSE["who"]}, files=files, timeout=60)


def note(pid, text, files=None, who="Nurse Tan"):
    return requests.post(f"{URL}/api/patients/{pid}/notes", data={"text": text, "who": who}, files=files, timeout=60)


def board(pid, body):
    return requests.patch(f"{URL}/api/patients/{pid}/board", json=body, timeout=30)


def main():
    save = str(Path(tempfile.mkdtemp()) / "board.pkl")
    proc = start(save)
    try:
        run()
        n = len(get("/api/board").json()["patients"]) + len(get("/api/board?cleared=true").json()["patients"])
        proc.terminate(); proc.wait(10)
        proc = start(save)                                    # restart: is everything still there?
        b = get("/api/board").json()
        left = get("/api/board?cleared=true").json()
        check("restart: patients and case files are kept", len(b["patients"]) + len(left["patients"]) == n
              and any(p["report"] for p in [get(f"/api/patients/{x['id']}").json() for x in b["patients"]]))
    finally:
        proc.terminate()
    print(f"\n{sum(results)} PASS, {len(results) - sum(results)} FAIL")
    sys.exit(0 if all(results) else 1)


def run():
    meta = get("/api/meta").json()
    check("top bar: roles and stages are served", "Organising doctor" in meta["roles"] and "Going to surgery" in meta["stages"])

    # ---------------- charting screen
    check("Submit with an empty note is refused", new("   ").status_code == 400)
    r = new("67M chest pain since 0600, sweaty, radiating L arm. HR 118 BP 88/60 sats 93% RA RR 24. hx DM on metformin. "
            "came by ambulance. pain 8/10")
    check("Submit creates a patient", r.status_code == 200 and "id" in r.json())
    a = r.json()["id"]
    check("Submit: the AI starts at once (status working)", get(f"/api/patients/{a}").json()["status"] in ("working", "ready"))
    p = wait(a)
    check("Submit: AI finishes without error", p["status"] == "ready", p.get("error"))
    f = p["fields"]
    check("numbers pulled out of the note", (f.get("hr"), f.get("sbp"), f.get("dbp"), f.get("spo2"), f.get("rr"), f.get("age"))
          == (118, 88, 60, 93, 24, 67), f)
    check("each number shows where it came from", p["found"]["found"].get("hr") == "hr 118")
    check("urgency + level on the header", p["decision"] and p["decision"]["flagged"] and p["level"] in (1, 2))
    check("case file written, with the standard headings", "## Presenting complaint" in p["report"] and "## Plan and actions" in p["report"])
    check("questions about what's missing", isinstance(p["questions"], list) and len(p["questions"]) >= 1)
    check("AI to-dos marked as AI", p["todos"] and all(t["ai"] for t in p["todos"]))
    check("the patient appears on the whiteboard", any(x["id"] == a for x in get("/api/board").json()["patients"]))

    with open(IMAGE, "rb") as fh:
        r = new("45M fell off ladder, L wrist pain 7/10, HR 96 BP 130/80 sats 99, walk-in, NKDA",
                files=[("files", ("wrist.jpg", fh, "image/jpeg"))])
    b = r.json()["id"]
    q = wait(b)
    img = q["notes"][0]["images"][0] if q["notes"][0]["images"] else {}
    check("📎 image upload: saved and read by the AI", img.get("kind") == "bone_xray" and img.get("text"))
    check("image: detector marks + automatic outline saved", img.get("overlay") and img.get("outline"))
    check("Files tab: the pictures can be opened", all(get(f"/api/images/{img[k]}").status_code == 200 for k in ("file", "overlay", "outline")))
    check("Files tab: other paths are refused", get("/api/images/..%2Fserver.py").status_code == 404)
    check("image: sorted by the trained sorter, fracture model with the adult cut-off",
          img.get("sorted_by") == "image sorter" and "adult cut-off" in img.get("text", "") and
          any(f["name"] == "Fracture" and f["flag"] for f in img.get("findings", [])))
    with open(CHEST, "rb") as fh, open(CT, "rb") as fh2:
        r = note(b, "CXR and CT head done", files=[("files", ("cxr.jpg", fh, "image/jpeg")), ("files", ("ct.jpg", fh2, "image/jpeg"))])
    q = wait(b)
    imgs = {i["name"]: i for n in q["notes"] for i in n["images"]}
    cx, ct = imgs.get("cxr.jpg", {}), imgs.get("ct.jpg", {})
    check("chest X-ray: sorted as chest, VinDr detector findings listed",
          cx.get("kind") == "chest_xray" and any(f["name"] == "Pneumothorax" for f in cx.get("findings", [])))
    check("head CT (note says CT head): bleed model ran",
          ct.get("kind") == "ct_slice" and any(f["name"] == "Bleeding inside the skull" for f in ct.get("findings", [])))
    r = note(b, "video", files=[("files", ("clip.mp4", b"0000", "video/mp4"))])
    check("video upload is refused with a reason", r.status_code == 400 and "only pictures" in r.json()["detail"])

    c = new("81 yo woman confused since morning, GCS 13, temp 38.6, BP 150/90 pulse 96, brought by family").json()["id"]
    wait(c)

    # ---------------- fixes from the group test (reasoning/FEEDBACK_10NOV.md, 10 Nov)
    import io
    import numpy as np
    from PIL import Image

    def white(name):                                          # share of pure-white pixels in a stored picture
        pic = np.asarray(Image.open(io.BytesIO(get(f"/api/images/{name}").content)).convert("RGB"))
        return float((pic == 255).all(-1).mean())
    deep = io.BytesIO()                                       # the fracture X-ray as a 16-bit PNG (mode I;16), like real X-rays
    Image.fromarray(np.asarray(Image.open(IMAGE).convert("L")).astype(np.uint16) * 257).save(deep, "PNG")
    k = wait(new("45M fell off ladder, L ankle swollen. pain 7/10. HR 96 BP 130/80 RR 16 sats 99. walk-in",
                 files=[("files", ("ankle16.png", deep.getvalue(), "image/png"))]).json()["id"])
    ki = k["notes"][0]["images"][0]
    check("C1: a 16-bit X-ray is stored as a normal picture, not a white one", white(ki["file"]) < 0.2, white(ki["file"]))
    check("C1: the AI marks are drawn on the X-ray, not on white", ki.get("overlay") and white(ki["overlay"]) < 0.2)
    texts = [t["text"].lower() for t in k["todos"]]
    check("B3: no 'mechanism of injury not recorded' when the note says how (fell off ladder)",
          not any("mechanism" in t for t in texts) and not any("mechanism" in q.lower() for q in k["questions"]), texts)
    check("B4: no 'X-ray of the injured area' to-do when the X-ray is attached", not any("x-ray" in t for t in texts), texts)
    check("a sick patient stays red: flagged, with the reason, not 'needs vitals'",
          p["decision"]["flagged"] and not p["decision"]["needs_obs"] and "abnormal" in p["reason"].lower(), p["reason"])
    kid_id = new("7yo girl fell from scooter, L wrist sore, moving it. pain 3/10. HR 96. brought by dad").json()["id"]
    kid = wait(kid_id)
    d = kid["decision"]
    check("A1: missing BP and RR -> 'needs vitals' (still level 1-2), naming what to measure",
          d["flagged"] and d["needs_obs"] and d["missing"] == ["BP", "RR"] and kid["level"] <= 2
          and kid["reason"] == "Needs vitals: BP and RR not recorded", (d, kid["reason"]))
    check("A1: the case file's triage section says why", "BP and RR not recorded" in kid["report"]
          and "until BP and RR are measured" in kid["report"])
    row = next(x for x in get("/api/board").json()["patients"] if x["id"] == kid_id)
    check("A1: the whiteboard row carries the reason", row["needs_obs"] and "BP and RR" in row["reason"])
    r = requests.post(f"{URL}/api/patients/{kid_id}/chat", json={"text": "Why is she urgent?", "who": "Dr Lim"})
    check("A1: the text the chat reads says why", "BP and RR not recorded" in r.json().get("answer", ""))
    check("A2: under 16 is labelled 'models built on adults'", kid["child"] and "under 16" in kid["report"].lower()
          and not p["child"])
    check("B4: no blood-thinner question for a child", not any("blood thinner" in q.lower() for q in kid["questions"]),
          kid["questions"])
    check("B5: 3+ observations missing -> the first question asks for them",
          kid["questions"] and kid["questions"][0].startswith("Observations not recorded yet")
          and "blood pressure" in kid["questions"][0] and not k["questions"][0].startswith("Observations"), kid["questions"])
    r = requests.put(f"{URL}/api/patients/{kid_id}/fields", json={"fields": {**kid["fields"], "sbp": 100, "dbp": 60, "rr": 22},
                                                                   "who": "Nurse Tan"})
    kid2 = get(f"/api/patients/{kid_id}").json()
    check("A1: once BP and RR are entered the 'needs vitals' flag goes", r.status_code == 200
          and not kid2["decision"]["needs_obs"] and "not recorded, so" not in kid2["report"], kid2["reason"])
    # code review 12 Nov
    kept = charting.drop_obs_questions(["What is the blood pressure?", "What is the respiratory rate?",
                                        "What is the pain score?", "What is the patient's age?"])
    check("B5: asking for all observations drops only the observation questions, not pain or age",
          kept == ["What is the pain score?", "What is the patient's age?"], kept)
    ct = charting.fit_patient(["X-ray of the injured area", "Check tetanus status"], {"age": 78}, "fall, CT head", ["ct_slice"])
    check("B4: no 'X-ray of the injured area' to-do when a CT is attached", ct == ["Check tetanus status"], ct)
    ed = "  Pathway flags: possible poisoning -> find out the substance; collapse/syncope -> ECG; trauma -> primary survey"
    shown = charting.supported_pathways(ed, "78F fall at home, hit head, on warfarin, vomited twice")
    check("pathways: no poisoning or collapse when the notes don't mention them (P11 'warfarin')",
          "poisoning" not in shown and "collapse" not in shown and "trauma" in shown, shown)
    check("pathways: poisoning kept when the note says what was taken",
          "poisoning" in charting.supported_pathways(ed, "took 20 paracetamol tablets"))
    small = charting.fit_patient(["Trauma: primary survey", "X-ray of the injured area", "Check tetanus status"],
                                 {"hr": 76, "sbp": 124, "spo2": 99, "mental": "alert"}, "23M cut finger on kitchen knife")
    check("small cut finger, normal numbers: no primary survey or X-ray (T04)", small == ["Check tetanus status"], small)
    filed = charting.file_notes(["EMS found BP 70/40 HR 130. On arrival BP 112/70 HR 92"])
    check("ambulance numbers in the case file are labelled 'before arrival' (T05)",
          "Before arrival (ambulance crew): EMS found BP 70/40" in filed and "- On arrival BP 112/70" in filed, filed)
    p11 = wait(new("78F fall at home, hit head, on warfarin, vomited twice, GCS 14. HR 92 BP 168/92 sats 97%. BIBA").json()["id"])
    check("whole app: a head injury on warfarin gets no poisoning to-do or question",
          not any("poison" in t["text"].lower() for t in p11["todos"])
          and not any("what was taken" in q.lower() for q in p11["questions"]), [t["text"] for t in p11["todos"]])
    from image_checklist import disagreements
    frac = [{"name": "Fracture", "flag": True}]
    check("C5: 'no displacement of the fracture' is not read as 'no fracture'",
          disagreements(frac, "There is no displacement of the fracture.") == ""
          and disagreements(frac, "No evidence of fracture.") != "")

    # ---------------- add information / answer a question
    r = note(a, "ECG sinus tachycardia, no ST changes. gave aspirin 300mg")
    check("Add (add-information bar) is accepted", r.status_code == 200)
    r2 = note(a, "trop sent, repeat in 3h")                  # straight away, while the AI may still be busy
    check("Add while the AI is busy is accepted (an ED doesn't wait)", r2.status_code == 200)
    p = wait(a)
    check("Add: the case file is rebuilt with BOTH new notes", "aspirin 300" in p["report"] and "trop sent" in p["report"]
          and len(p["notes"]) == 3 and p["status"] == "ready")
    question = (p["questions"] or ["Any allergies?"])[0].split(" [TEST")[0]
    note(a, f"{question} → NKDA")
    p = wait(a)
    check("answering a question adds it to the notes", any("NKDA" in n["text"] for n in p["notes"]))
    check("Rebuilt case file keeps the earlier notes", "aspirin 300" in p["report"])
    # retest 14 Nov (Caleb, T01): an answered number reached the case file but not the tiles or the score
    note(a, "What is the level of consciousness (AVPU/GCS)? → alert, GCS 15. T 37.9. HR 140")
    p = wait(a)
    f = p["fields"]
    check("answer: missing numbers are read into the tiles (consciousness, temperature)",
          f.get("mental") == "alert" and f.get("temp_c") == 37.9, f)
    check("answer: a number already on a tile is not changed, the nurse is asked to check it",
          f.get("hr") == 118 and any(w.startswith("please check HR") for w in p["found"].get("warnings", [])),
          p["found"].get("warnings"))
    check("answer: re-scored and logged; the case file says what was read",
          any("Numbers from a later note" in e["text"] for e in p["timeline"])
          and re.search(r"From a later note: [^\n]*Consciousness alert", p["report"]) is not None,
          ([e["text"] for e in p["timeline"]][-4:], p["report"].split("## Observations")[-1][:300]))
    check("case-file slips tidied: 'Pain 3/010' -> 'Pain 3/10' (retest 14 Nov, P05)",
          charting.tidy_numbers("Pain 3/010, pain 8/10") == "Pain 3/10, pain 8/10")
    # Caleb's 2nd retest 14 Nov (13:27): three points
    check("case file: 'Consciousness not recorded' goes once a later note says alert (Caleb 2nd retest)",
          not re.search(r"(?i)consciousness:? not recorded", p["report"]) and charting.drop_not_recorded(
              "## Observations\n- HR 112, Consciousness not recorded\n- Consciousness: not recorded\n- From a later note: "
              "Consciousness alert", {"mental": "alert"}) == "## Observations\n- HR 112\n- From a later note: Consciousness alert",
          p["report"].split("## Observations")[-1][:300])
    check("SBAR: 'Okay, here is the SBAR…' opening is removed (Caleb 2nd retest)",
          charting.strip_preamble("Okay, here is the SBAR handover:\n\n**Situation**\nx") == "**Situation**\nx"
          and charting.strip_preamble("**Situation**\nOkay overall") == "**Situation**\nOkay overall")
    done = charting.tests_done([{"time": "11:05", "text": "trop 85. ECG ST depression V4-6"}, {"time": "11:10", "text": "CXR pending"}])
    marked = charting.mark_done_tests("Obtain urgent ECG and troponin. Repeat troponin at 3 h. Order a chest X-ray.", done)
    check("asking for a test already done is marked 'already done' (Caleb 2nd retest: 'obtain urgent ECG and troponin')",
          done == {"troponin": "11:05", "ECG": "11:05"}
          and "troponin (already done: troponin at 11:05, ECG at 11:05, see the notes)." in marked
          and "Repeat troponin at 3 h." in marked and "Order a chest X-ray." in marked, (done, marked))

    # ---------------- case file window
    fixes_before = get("/api/learning").json()["number_corrections"]
    r = requests.put(f"{URL}/api/patients/{a}/fields", json={"fields": {**p["fields"], "hr": 150, "spo2": 85}, "who": "Nurse Tan"})
    p2 = get(f"/api/patients/{a}").json()
    check("Fix numbers: saved and re-scored", r.status_code == 200 and p2["fields"]["hr"] == 150 and p2["found"]["checked_by"] == "Nurse Tan")
    check("Fix numbers: earlier notes and timeline kept", len(p2["notes"]) == len(p["notes"]) and any("corrected" in e["text"] for e in p2["timeline"]))
    obs = p2["report"].split("## Observations")[1].split("\n## ")[0]
    check("Fix numbers: the case file's Observations says what was corrected (13 Nov)",
          "Corrected by Nurse Tan: HR 150 (was " in obs and "SpO2 % 85 (was " in obs
          and p2["report"].rstrip().split("\n## ")[-1].startswith(charting.TRIAGE), obs[-200:])
    learn = get("/api/learning").json()
    check("learning: the number fix is logged (what the AI read vs the correction)", learn["enabled"]
          and learn["number_corrections"] == fixes_before + 1 and {"hr", "spo2"} <= set(learn["numbers_corrected_by_field"]))
    edited = p2["report"].replace("## Presenting complaint\n", "## Presenting complaint\n- Crushing central chest pain\n", 1)
    r = requests.put(f"{URL}/api/patients/{a}/report", json={"text": edited, "who": "Dr Lim"})
    p3 = get(f"/api/patients/{a}").json()
    check("✎ Edit → Save: the case file keeps the edit and who made it", r.status_code == 200 and p3["report"] == edited and p3["report_by"] == "Dr Lim")
    check("an edited complaint shows on the header and whiteboard", p3["complaint"] == "Crushing central chest pain"
          and next(x for x in get("/api/board").json()["patients"] if x["id"] == a)["complaint"] == "Crushing central chest pain")
    r = requests.post(f"{URL}/api/patients/{a}/chat", json={"text": "What should we rule out?", "who": "Dr Lim"})
    p4 = get(f"/api/patients/{a}").json()
    check("Chat → Ask: answered and kept", r.status_code == 200 and r.json()["answer"] and len(p4["chat"]) == 2)
    check("Chat reads the edited case file", "Crushing central chest pain" in r.json()["answer"])
    text = get(f"/api/patients/{a}/case_file").text
    check("⬇ Download the whole case file: timeline + case file + notes", "TIMELINE" in text and "Crushing central chest pain" in text
          and "NOTES AS TYPED" in text and "aspirin 300" in text)
    note(a, "surgical team aware")                          # a later note must not undo the person's edit
    p5 = wait(a)
    check("a later note is added without losing the person's edit", "Crushing central chest pain" in p5["report"]
          and "surgical team aware" in p5["report"] and p5["report_by"] == "Dr Lim")
    # Core task 3: SBAR handover from everything gathered (later notes, answers, the chat)
    check("SBAR: nothing to download before one is written", get(f"/api/patients/{a}/sbar.txt").status_code == 404)
    r = requests.post(f"{URL}/api/patients/{a}/sbar", json={"who": "Dr Lim"})
    s = r.json() if r.status_code == 200 else {}
    check("SBAR handover: four headings, written for the person who asked",
          all(h in s.get("text", "") for h in ("Situation", "Background", "Assessment", "Recommendation"))
          and s.get("who") == "Dr Lim", r.text[:200])
    check("SBAR uses the later notes and the chat, not just the first note",
          "aspirin 300" in s.get("text", "") and "surgical team aware" in s.get("text", "")
          and "1 chat turn" in s.get("text", ""), s.get("text"))
    check("SBAR: later notes and chat questions copied word for word under the AI's text (none can go missing)",
          "Added after arrival (copied from the notes" in s.get("text", "")
          and "Asked in the chat" in s.get("text", "") and "What should we rule out?" in s.get("text", ""), s.get("text"))
    p6 = get(f"/api/patients/{a}").json()
    check("SBAR: kept on the patient and logged on the timeline", p6["sbar"].get("text") == s.get("text")
          and any(e["what"] == "Handover" and "Dr Lim" in e["text"] for e in p6["timeline"]))
    text = get(f"/api/patients/{a}/sbar.txt").text
    check("⬇ Download the SBAR: says it is an AI draft to check", "AI draft" in text and "Situation" in text)

    # ---------------- picture rule (14 Nov): vitals + words + pictures all feed the priority, pictures only move UP
    def with_picture(text, path, name):
        with open(path, "rb") as fh:
            return wait(new(text, files=[("files", (name, fh, "image/jpeg"))]).json()["id"])
    ct = with_picture(CALM, CT_BLEED, "ct_head.jpg")
    d = ct["decision"]
    check("picture rule: a red CT bleed moves a patient with normal vitals to urgent, level 1-2, with the reason",
          d["flagged"] and d["level"] <= 2 and d["picture"] == "CT shows bleeding inside the skull"
          and "moved up: ct shows bleeding inside the skull (picture model)" in ct["reason"].lower(),
          (d, ct["reason"], [i.get("findings") for n in ct["notes"] for i in n["images"]]))
    check("picture rule: on the case file's triage section and the timeline",
          "picture rule: CT shows bleeding inside the skull" in ct["report"]
          and any("Moved up by the picture rule" in e["text"] for e in ct["timeline"]))
    row = next(x for x in get("/api/board").json()["patients"] if x["id"] == ct["id"])
    check("picture rule: the whiteboard row says why", "CT shows bleeding inside the skull" in row["reason"], row["reason"])
    r = requests.put(f"{URL}/api/patients/{ct['id']}/fields", json={"fields": {**ct["fields"], "hr": 76}, "who": "Dr Lim"})
    ct2 = get(f"/api/patients/{ct['id']}").json()
    note(ct["id"], "neuro obs unchanged, GCS 15")
    ct3 = wait(ct["id"])
    check("picture rule: kept after Fix numbers and after a later note",
          r.status_code == 200 and ct2["decision"]["picture"] and ct2["flagged"]
          and ct3["decision"]["picture"] and ct3["decision"]["level"] <= 2
          and sum("Moved up by the picture rule" in e["text"] for e in ct3["timeline"]) == 1,
          (ct2["decision"], ct3["decision"]))
    frac = with_picture(CALM.replace("hit head", "hurt R ankle"), IMAGE, "ankle.jpg")
    check("picture rule: a red fracture does NOT move a patient up",
          any(f["name"] == "Fracture" and f["flag"] for n in frac["notes"] for i in n["images"] for f in i.get("findings") or [])
          and not frac["decision"]["picture"] and "picture" not in frac["reason"],
          [i.get("findings") for n in frac["notes"] for i in n["images"]])
    amber = with_picture("28M review, chest pain settled. HR 80 RR 16 BP 122/76 sats 98% RA. walk-in", CXR_AMBER, "cxr.jpg")
    finds = [f for n in amber["notes"] for i in n["images"] for f in i.get("findings") or []]
    check("picture rule: an amber 'possible pneumothorax' does NOT move a patient up",
          any(f["name"] == "Pneumothorax" and f.get("possible") and not f["flag"] for f in finds)
          and not amber["decision"]["picture"], finds)

    # ---------------- a doctor changes the priority (14 Nov); the AI never lowers by itself
    cp = wait(new("67M CP since 0600, sweaty, radiating to L arm. HR 112 BP 90/60 RR 22 sats 94% RA pain 8/10. walk-in")
              .json()["id"])
    order = lambda: [x["id"] for x in get("/api/board").json()["patients"]]
    levels = {x["id"]: x["level"] for x in get("/api/board").json()["patients"]}
    check("priority: T01-like patient starts urgent at level 2 (AI + safety rules)", cp["flagged"] and cp["level"] <= 2,
          cp["decision"])
    ai_level = cp["level"]
    before = order()
    put = lambda body: requests.put(f"{URL}/api/patients/{cp['id']}/priority", json=body, timeout=30)
    r = put({"who": "Nurse Tan", "role": "Triage nurse", "level": 3, "reason": "settled"})
    check("priority: a nurse can't change it (doctors only)", r.status_code == 403, r.text)
    r = put({"who": "Dr Wong", "role": "Doctor", "level": 3})
    check("priority: works without a reason", r.status_code == 200 and get(f"/api/patients/{cp['id']}").json()["reason"]
          == f"Level 3, set by Dr Wong (AI + safety rules said {ai_level})", r.text)
    learned = get("/api/learning").json().get("priority_changed_by_doctor", 0)
    r = put({"who": "Dr Wong", "role": "Doctor", "level": 3, "reason": "Chest pain settled, ECG and troponin normal"})
    p = get(f"/api/patients/{cp['id']}").json()
    check("priority: a doctor lowers 2 -> 3 with a reason: level 3, not urgent, the AI's own level kept underneath",
          r.status_code == 200 and p["level"] == 3 and not p["flagged"] and p["decision"]["level"] == ai_level
          and p["reason"] == f"Level 3, set by Dr Wong: Chest pain settled, ECG and troponin normal (AI + safety rules "
                             f"said {ai_level})", (r.text, p["reason"]))
    after = order()
    check("priority: the whiteboard sorts by the doctor's level",
          all(after.index(cp["id"]) > after.index(i) for i, l in levels.items() if l and l <= 2 and i in after
              and i != cp["id"]) and after != before, (before, after))
    check("priority: logged on the timeline, the case file and for learning",
          any(e["what"] == "Priority" and "Level set to 3" in e["text"] and "Dr Wong" in e["text"] for e in p["timeline"])
          and "PRIORITY SET BY A DOCTOR: Level 3, set by Dr Wong" in p["report"]
          and get("/api/learning").json().get("priority_changed_by_doctor", 0) == learned + 1)
    note(cp["id"], "now responds to pain only, GCS 9. T 39.4")      # sicker: the consciousness tile was empty
    p = wait(cp["id"])
    check("priority: a later note that makes the AI + rules more worried keeps the doctor's level and warns",
          p["level"] == 3 and p["override"]["level"] == 3 and p["review"].startswith("New information since Dr Wong's "
          "change: AI and rules now say level") and p["review"].endswith("please review")
          and any(e["what"] == "Priority" and "please review" in e["text"] for e in p["timeline"])
          and "PLEASE REVIEW" in p["report"], (p["fields"].get("mental"), p["decision"], p["review"]))
    r = put({"who": "Dr Lim", "role": "Organising doctor", "clear": True})
    p = get(f"/api/patients/{cp['id']}").json()
    check("priority: 'Back to the AI's level' undoes it", r.status_code == 200 and p["override"] is None
          and p["level"] == p["decision"]["level"] and not p["review"])
    r = requests.post(f"{URL}/api/patients/{a}/rerun")
    check("Re-run AI works", r.status_code == 200 and wait(a)["status"] == "ready")
    tl = get(f"/api/patients/{a}").json()["timeline"]
    check("Timeline tab: every step is logged with who and when", {"Triage", "Case file", "Chat"} <= {e["what"] for e in tl}
          and all(e["time"] and e["who"] for e in tl))

    # ---------------- whiteboard
    check("viewer can't change the stage", board(b, {**DOCTOR, "stage": "Being seen"}).status_code == 403)
    check("viewer can't add a to-do", requests.post(f"{URL}/api/patients/{b}/todos", json={**DOCTOR, "text": "x"}).status_code == 403)
    check("viewer can't clear a patient", board(b, {**DOCTOR, "cleared": True}).status_code == 403)
    check("unknown stage refused", board(b, {**LEAD, "stage": "Lunch"}).status_code == 400)
    r = board(b, {**LEAD, "stage": "Being seen", "doctor": "Dr Lim"})
    x = next(x for x in get("/api/board").json()["patients"] if x["id"] == b)
    check("Stage and Doctor change on the board", r.status_code == 200 and x["stage"] == "Being seen" and x["doctor"] == "Dr Lim")
    check("board change logged in the case file", any(e["what"] == "Board" for e in get(f"/api/patients/{b}").json()["timeline"]))
    n0 = len(x["todos"])
    first_is_ai = bool(x["todos"]) and x["todos"][0]["ai"]
    requests.post(f"{URL}/api/patients/{b}/todos", json={**LEAD, "text": "Repeat X-ray after reduction"})
    x = next(x for x in get("/api/board").json()["patients"] if x["id"] == b)
    check("+ to-do adds a person's to-do (not marked AI)", len(x["todos"]) == n0 + 1 and not x["todos"][-1]["ai"])
    requests.patch(f"{URL}/api/patients/{b}/todos/0", json={**LEAD, "done": True})
    check("tick a to-do", next(x for x in get("/api/board").json()["patients"] if x["id"] == b)["todos"][0]["done"])
    requests.patch(f"{URL}/api/patients/{b}/todos/0", json={**LEAD, "done": False})
    check("untick a to-do", not next(x for x in get("/api/board").json()["patients"] if x["id"] == b)["todos"][0]["done"])
    requests.patch(f"{URL}/api/patients/{b}/todos/0", json={**LEAD, "delete": True})
    check("× removes a to-do", len(next(x for x in get("/api/board").json()["patients"] if x["id"] == b)["todos"]) == n0)
    ai_todos = get("/api/learning").json()["ai_todos"]
    check("learning: ticking / removing an AI to-do is logged (monitoring only)",
          not first_is_ai or sum(v["removed"] for v in ai_todos.values()) >= 1)
    check("Clear is refused before the patient has somewhere to go", board(b, {**LEAD, "cleared": True}).status_code == 400)
    for stage in ("Going to surgery", "Transfer to another department", "Admitted to ward", "Discharged home"):
        r = board(c, {**LEAD, "stage": stage})
        check(f"stage '{stage}' can be set", r.status_code == 200)
    rows = get("/api/board").json()["patients"]
    check("leaving patients sink to the bottom of the board", rows[-1]["id"] == c and rows[-1]["leaving"])
    check("Clear: a level outside 1-5 is refused", board(c, {**LEAD, "cleared": True, "final_level": 9}).status_code == 400)
    r = board(c, {**LEAD, "cleared": True, "diagnosis": "Acute coronary syndrome", "final_level": 2})
    learn = get("/api/learning").json()
    check("learning: Clear with a final diagnosis + level logs an outcome case", learn["patients_with_outcome"] == 1
          and learn["with_diagnosis_or_level"] == 1 and any(e["what"] == "Outcome" for e in get(f"/api/patients/{c}").json()["timeline"]))
    check("learning: test mode has no MedGemma vector, so nothing is used for training", learn["usable_for_learning"] == 0)
    on, left = get("/api/board").json(), get("/api/board?cleared=true").json()
    check("✓ Clear from board: gone from the board", r.status_code == 200 and all(x["id"] != c for x in on["patients"]))
    check("Left the ED list shows them, case file kept", any(x["id"] == c for x in left["patients"]) and on["n_cleared"] == 1
          and get(f"/api/patients/{c}").json()["report"])
    board(c, {**LEAD, "cleared": False})
    check("Put back on the board", any(x["id"] == c for x in get("/api/board").json()["patients"]))
    board(c, {**LEAD, "cleared": True})
    lv = [x["level"] for x in get("/api/board").json()["patients"] if not x["leaving"]]
    check("board order: most urgent first", lv == sorted(lv))
    rows = [x for x in get("/api/board").json()["patients"] if x["level"]]
    check("three groups: Urgent (flagged or level 1-2) / Can wait (3) / Minor (4-5) on every board row (14 Nov)",
          all(x["group"] == ("Urgent" if x["flagged"] or x["level"] <= 2 else "Can wait" if x["level"] == 3 else "Minor")
              for x in rows) and len({x["group"] for x in rows}) >= 2,
          [(x["level"], x["flagged"], x.get("group")) for x in rows])
    d = new("30M cut finger, HR 80").json()["id"]
    wait(d)
    check("remove a patient: viewer refused", requests.delete(f"{URL}/api/patients/{d}?role=Doctor").status_code == 403)
    check("remove a patient: organising doctor allowed", requests.delete(f"{URL}/api/patients/{d}?role=Organising%20doctor").status_code == 200
          and get(f"/api/patients/{d}").status_code == 404)
    check("the screens are served", get("/").status_code == 200 and "root" in get("/").text)
    # fixes from the real-AI test (REPORT_NOTES 5q, 9 Nov)
    e = new("45M SOB, cough. HR 400 BP 300/20 sats 150% temp 45 RR 2. alert").json()["id"]
    r = board(e, {"who": "", "role": "Triage nurse", "doctor": "Dr Anon"})
    check("board edit without a name refused (5q #9)", r.status_code == 400 and "name" in r.json().get("detail", ""))
    board(e, {**LEAD, "doctor": "Dr Early"})                  # while the AI may still be working on it
    wait(e)
    pe = get(f"/api/patients/{e}").json()
    check("impossible BP 300/20 dropped, each warning once (5q #8)", "sbp" not in pe["fields"]
          and len(pe["found"]["warnings"]) == len(set(pe["found"]["warnings"])))
    check("board change before the AI finished is on the timeline (5q #9)",
          any(t["what"] == "Board" and "Dr Early" in t["text"] for t in pe["timeline"]))
    check("no two to-dos say the same thing (5q #3)",
          not any(charting.same_meaning(a["text"], b["text"]) for i, a in enumerate(pe["todos"]) for b in pe["todos"][i + 1:]))


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")       # the app started by the test
    sys.stdout.reconfigure(encoding="utf-8")                  # this test's own output (📎 crashed when saved to a file)
    main()
