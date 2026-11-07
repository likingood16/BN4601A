"""Every button of the app, pressed through the back end the screens use, each checked for doing its job.

    python test_app.py            # starts its own test-mode app on port 8002 (empty board, temporary save file)

Each check prints PASS or FAIL. One FAIL = the button doesn't do what it says. At the end the app is restarted to
check that the board and case files survive a restart. The buttons are listed in APP_DESIGN.md (section 11).
"""
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
URL = "http://127.0.0.1:8002"
IMAGE = HERE.parents[1] / "vision" / "detector" / "data" / "yolo" / "images" / "test" / "IMG0003297.jpg"
NURSE = {"who": "Nurse Tan", "role": "Triage nurse"}
DOCTOR = {"who": "Dr Wong", "role": "Doctor"}                 # can look, can't change the board
LEAD = {"who": "Dr Lim", "role": "Organising doctor"}
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"   ({detail})" if detail and not ok else ""), flush=True)


def start(save):
    proc = subprocess.Popen([sys.executable, str(HERE / "server.py"), "--test", "--port", "8002", "--case-files", save],
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
        proc.terminate(); proc.wait(10)
        proc = start(save)                                    # restart: is everything still there?
        b = get("/api/board").json()
        left = get("/api/board?cleared=true").json()
        check("restart: patients and case files are kept", len(b["patients"]) + len(left["patients"]) == 3
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
    r = note(b, "video", files=[("files", ("clip.mp4", b"0000", "video/mp4"))])
    check("video upload is refused with a reason", r.status_code == 400 and "only pictures" in r.json()["detail"])

    c = new("81 yo woman confused since morning, GCS 13, temp 38.6, BP 150/90 pulse 96, brought by family").json()["id"]
    wait(c)

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

    # ---------------- case file window
    r = requests.put(f"{URL}/api/patients/{a}/fields", json={"fields": {**p["fields"], "hr": 150, "spo2": 85}, "who": "Nurse Tan"})
    p2 = get(f"/api/patients/{a}").json()
    check("Fix numbers: saved and re-scored", r.status_code == 200 and p2["fields"]["hr"] == 150 and p2["found"]["checked_by"] == "Nurse Tan")
    check("Fix numbers: earlier notes and timeline kept", len(p2["notes"]) == len(p["notes"]) and any("corrected" in e["text"] for e in p2["timeline"]))
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
    requests.post(f"{URL}/api/patients/{b}/todos", json={**LEAD, "text": "Repeat X-ray after reduction"})
    x = next(x for x in get("/api/board").json()["patients"] if x["id"] == b)
    check("+ to-do adds a person's to-do (not marked AI)", len(x["todos"]) == n0 + 1 and not x["todos"][-1]["ai"])
    requests.patch(f"{URL}/api/patients/{b}/todos/0", json={**LEAD, "done": True})
    check("tick a to-do", next(x for x in get("/api/board").json()["patients"] if x["id"] == b)["todos"][0]["done"])
    requests.patch(f"{URL}/api/patients/{b}/todos/0", json={**LEAD, "done": False})
    check("untick a to-do", not next(x for x in get("/api/board").json()["patients"] if x["id"] == b)["todos"][0]["done"])
    requests.patch(f"{URL}/api/patients/{b}/todos/0", json={**LEAD, "delete": True})
    check("× removes a to-do", len(next(x for x in get("/api/board").json()["patients"] if x["id"] == b)["todos"]) == n0)
    check("Clear is refused before the patient has somewhere to go", board(b, {**LEAD, "cleared": True}).status_code == 400)
    for stage in ("Going to surgery", "Transfer to another department", "Admitted to ward", "Discharged home"):
        r = board(c, {**LEAD, "stage": stage})
        check(f"stage '{stage}' can be set", r.status_code == 200)
    rows = get("/api/board").json()["patients"]
    check("leaving patients sink to the bottom of the board", rows[-1]["id"] == c and rows[-1]["leaving"])
    r = board(c, {**LEAD, "cleared": True})
    on, left = get("/api/board").json(), get("/api/board?cleared=true").json()
    check("✓ Clear from board: gone from the board", r.status_code == 200 and all(x["id"] != c for x in on["patients"]))
    check("Left the ED list shows them, case file kept", any(x["id"] == c for x in left["patients"]) and on["n_cleared"] == 1
          and get(f"/api/patients/{c}").json()["report"])
    board(c, {**LEAD, "cleared": False})
    check("Put back on the board", any(x["id"] == c for x in get("/api/board").json()["patients"]))
    board(c, {**LEAD, "cleared": True})
    lv = [x["level"] for x in get("/api/board").json()["patients"] if not x["leaving"]]
    check("board order: most urgent first", lv == sorted(lv))
    d = new("30M cut finger, HR 80").json()["id"]
    wait(d)
    check("remove a patient: viewer refused", requests.delete(f"{URL}/api/patients/{d}?role=Doctor").status_code == 403)
    check("remove a patient: organising doctor allowed", requests.delete(f"{URL}/api/patients/{d}?role=Organising%20doctor").status_code == 200
          and get(f"/api/patients/{d}").status_code == 404)
    check("the screens are served", get("/").status_code == 200 and "root" in get("/").text)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    main()
