"""End-to-end check of the app against PRODUCT.md, through the HTTP API (test mode or real AI).

    python server.py --test --db /tmp/t.sqlite        # in one window
    python test_app.py --url http://127.0.0.1:8000    # in another; prints PASS/FAIL per check and a summary
    python test_app.py --url ... --real               # on vanda: also stricter checks that need MedGemma

Each check is one acceptance criterion. Exit code 1 if any check fails.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTES = HERE.parent / "web" / "test_notes"
IMAGES = HERE.parent / "web" / "test_images"
RESULTS = []


def req(method, url, body=None, form=None, files=()):
    headers, data = {}, None
    if form is not None:
        b = uuid.uuid4().hex
        parts = []
        for k, v in form.items():
            parts.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
        for path in files:
            parts.append(f'--{b}\r\nContent-Disposition: form-data; name="files"; filename="{Path(path).name}"\r\n'
                         f"Content-Type: application/octet-stream\r\n\r\n".encode() + Path(path).read_bytes() + b"\r\n")
        data = b"".join(parts) + f"--{b}--\r\n".encode()
        headers["Content-Type"] = f"multipart/form-data; boundary={b}"
    elif body is not None:
        data, headers["Content-Type"] = json.dumps(body).encode(), "application/json"
    r = urllib.request.Request(A.url + url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(r, timeout=600) as resp:
            raw = resp.read().decode()
            return resp.status, (json.loads(raw) if raw.startswith(("{", "[")) else raw)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)


def wait(pid, timeout=900):
    t = time.time()
    while time.time() - t < timeout:
        _, p = req("GET", f"/api/patients/{pid}")
        if p["status"] != "working":
            return p, time.time() - t
        time.sleep(0.5)
    raise TimeoutError(pid)


def new(text, who="Nurse Test", files=()):
    s, out = req("POST", "/api/patients", form={"text": text, "who": who}, files=files)
    assert s == 200, out
    return out["id"]


def items(p, section=None):
    return [i for i in p["items"] if section is None or i["section"] == section]


def main():
    nurse = {"who": "Nurse Test", "role": "Triage nurse"}
    viewer = {"who": "Dr View", "role": "Doctor"}

    # ---- 1. charting: numbers, prehospital, case file, decision
    s, prev = req("POST", "/api/extract", {"text": "67M CP. EMS found HR 140. On arrival HR 96 BP 150/90 sats 97% RA"})
    check("Live tiles: arrival HR used, EMS HR ignored", prev["fields"].get("hr") == 96)
    s, prev = req("POST", "/api/extract", {"text": "61F dizzy. EMS: BP 80/40. now feels better"})
    check("Live tiles: EMS-only value flagged as before arrival, not used",
          "sbp" not in prev["fields"] and prev["source"].get("sbp", {}).get("prehospital"), str(prev["warnings"]))

    note = "67M CP since 0600, sweaty, radiating to L arm. HR 112 BP 90/60 RR 22 sats 94% RA T 37.2 pain 8/10. hx DM, HTN on metformin. NKDA. BIBA"
    t0 = time.time()
    pid = new(note)
    p, secs = wait(pid)
    check("Submit -> case file ready", p["status"] == "ready", f"{secs:.1f} s")
    if A.real:
        check("Case file ready within 45 s (rubric: Good)", secs <= 45, f"{secs:.1f} s")
    f = p["fields"]
    check("Numbers right (HR, BP, RR, SpO2, temp, pain, age, sex, arrival)",
          (f["hr"], f["sbp"], f["dbp"], f["rr"], f["spo2"], f["temp_c"], f["nrs_pain"], f["age"], f["sex"], f["ambulance"])
          == (112, 90, 60, 22, 94, 37.2, 8, 67, "male", 1))
    obs = " | ".join(i["text"] for i in items(p, "Observations"))
    check("Observations written by code, room air recorded, nothing assumed", "on room air" in obs and "assumed" not in obs, obs)
    check("Triage decision present and flagged (SBP 90 -> safety floor)", p["decision"] and p["decision"]["flagged"])
    tri = [i["text"] for i in items(p, "Triage decision support")]
    check("Triage section is the models' decision word for word", tri and tri[0].startswith("URGENCY DECISION"))
    all_text = " ".join(i["text"] for i in p["items"]).lower()
    check("Every fact typed is in the case file (DM, metformin, NKDA, sweaty, radiating)",
          all(w in all_text for w in ("dm" if not A.real else "diabet", "metformin", "nkda" if not A.real else "allerg",
                                      "sweat", "radiat")), all_text[:200])
    ai_items = [i for i in p["items"] if i["kind"] == "ai"]
    norm = lambda s: " ".join("".join(c if c.isalnum() or c in "/%." else " " for c in s.lower()).split())
    check("Every AI line has a quote that is in the note", all(norm(i["quote"]) in norm(note) for i in ai_items),
          f"{len(ai_items)} AI lines, {len(p['rejected'])} rejected")
    check("To-dos are actions for this complaint (ECG for chest pain)", any("ECG" in t["text"] for t in p["todos"]),
          "; ".join(t["text"] for t in p["todos"]))
    check("To-dos never repeat questions ('Obtain ...')", not any(t["text"].lower().startswith("obtain") for t in p["todos"]))
    check("To-dos are only things to send/do, no advice (analgesia, monitoring, senior review)",
          not any(w in t["text"].lower() for t in p["todos"] for w in ("analges", "monitor", "senior review", "gcs every", "score")))
    check("Red-flag vitals shown as alerts, not to-dos (SBP 90)", any("SBP" in a for a in p["alerts"]), "; ".join(p["alerts"]))
    check("No question asks for what the note has (BP, HR, allergies)",
          not any(w in q["text"].lower() for q in p["questions"] for w in ("blood pressure", "heart rate", "allerg")),
          "; ".join(q["text"] for q in p["questions"]))

    # ---- 2. questions: pathway, answer, dismiss
    pid2 = new("72F rt sided weakness and slurred speech. BP 182/100 HR 88 RR 18 sats 97% RA. walk-in")
    p2, _ = wait(pid2)
    qs = [q["text"] for q in p2["questions"]]
    check("Stroke -> asks time last known well", any("last known well" in q for q in qs), "; ".join(qs))
    check("Stroke -> to-do glucose + brain CT", any("glucose" in t["text"].lower() for t in p2["todos"])
          and any("CT" in t["text"] for t in p2["todos"]))
    lkw = next(q for q in qs if "last known well" in q)
    s, _ = req("POST", f"/api/patients/{pid2}/notes", form={"text": "Last known well at 0830 today", "who": "Nurse Test", "answers": lkw})
    p2, _ = wait(pid2)
    check("Answering a question removes it and files the answer",
          lkw not in [q["text"] for q in p2["questions"]] and "0830" in " ".join(i["text"] for i in p2["items"]))
    other = p2["questions"][0]["text"] if p2["questions"] else None
    if other:
        req("POST", f"/api/patients/{pid2}/questions/dismiss", {"who": "Dr X", "text": other})
        req("POST", f"/api/patients/{pid2}/notes", form={"text": "pupils equal and reactive", "who": "Dr X"})
        p2, _ = wait(pid2)
        check("A dismissed question doesn't come back", other not in [q["text"] for q in p2["questions"]])

    pid_neg = new("30M fell off bike onto L wrist, swollen. no head injury, denies chest pain. HR 80 BP 125/80. walk-in")
    pn, _ = wait(pid_neg)
    check("Negated findings don't trigger pathways ('no head injury' -> no CT head)",
          not any("CT head" in t["text"] or "ECG" in t["text"] for t in pn["todos"]), "; ".join(t["text"] for t in pn["todos"]))

    # ---- 3. edits are never overwritten; notes accepted while the AI is busy
    target = items(p, "Presenting complaint")[0]
    req("PATCH", f"/api/patients/{pid}/items/{target['id']}", {"who": "Dr Lim", "text": "Central chest pain since 06:00"})
    pid3 = new("45M abdo pain RIF since yesterday, vomited x2. HR 98 BP 128/80 T 38.1. walk-in")
    s1, r1 = req("POST", f"/api/patients/{pid3}/notes", form={"text": "Bloods sent. WBC 15", "who": "Nurse Test"})
    check("A note added while the AI is busy is accepted", s1 == 200, json.dumps(r1))
    req("POST", f"/api/patients/{pid}/notes", form={"text": "Troponin 45 ng/L. ECG: ST depression V4-V6. Plan: aspirin, cardiology review",
                                                    "who": "Dr Lim"})
    p, _ = wait(pid)
    p3, _ = wait(pid3)
    pc = [i for i in items(p, "Presenting complaint") if i["id"] == target["id"]]
    check("A person's edit survives new notes", pc and pc[0]["text"] == "Central chest pain since 06:00" and pc[0]["kind"] == "person")
    check("New results go under the right headings",
          any("troponin" in i["text"].lower() for i in items(p, "Investigations")) and
          any("aspirin" in i["text"].lower() or "cardiology" in i["text"].lower() for i in items(p, "Plan")),
          "; ".join(f"{i['section']}: {i['text']}" for i in p["items"] if i["note"] == 1))
    check("The busy-time note reached the case file", "wbc" in " ".join(i["text"] for i in p3["items"]).lower())
    check("Header and board show the edited complaint", req("GET", "/api/board")[1]["patients"] and
          any(r["complaint"] == "Central chest pain since 06:00" for r in req("GET", "/api/board")[1]["patients"]))
    ai_line = next((i for i in p3["items"] if i["kind"] in ("ai", "typed")), None)
    req("PATCH", f"/api/patients/{pid3}/items/{ai_line['id']}", {"who": "Dr Lim", "confirmed": True})
    p3, _ = wait(pid3)
    check("A line can be checked by a person", next(i for i in p3["items"] if i["id"] == ai_line["id"])["confirmed"])
    req("POST", f"/api/patients/{pid3}/items", {"who": "Dr Lim", "section": "Impression", "text": "?Appendicitis"})
    p3, _ = wait(pid3)
    check("A person can add a line under any heading", any(i["text"] == "?Appendicitis" for i in items(p3, "Impression")))

    # ---- 4. correcting numbers re-scores
    before = p3["decision"]["p_urgent"]
    s, _ = req("PUT", f"/api/patients/{pid3}/fields", {"who": "Nurse Test", "fields": {**p3["fields"], "sbp": 82, "hr": 128}})
    p3, _ = wait(pid3)
    check("Correcting numbers re-scores and flags", s == 200 and p3["decision"]["flagged"] and p3["decision"]["p_urgent"] != before)
    check("Observations rewritten from corrected numbers, no duplicates",
          sum("Blood pressure" in i["text"] for i in items(p3, "Observations")) == 1
          and any("82/" in i["text"] for i in items(p3, "Observations")))

    # ---- 5. whiteboard: order, roles, stages, clear
    b = req("GET", "/api/board")[1]["patients"]
    levels = [r["level"] for r in b if not r["leaving"]]
    check("Board: most urgent first", levels == sorted(levels), str(levels))
    s, _ = req("PATCH", f"/api/patients/{pid}/board", {**viewer, "stage": "Being seen"})
    check("Board: a doctor (viewer) can't change it (403)", s == 403)
    s, _ = req("POST", f"/api/patients/{pid}/todos", {**viewer, "text": "x"})
    check("To-dos: a viewer can't add (403)", s == 403)
    s, _ = req("PATCH", f"/api/patients/{pid}/board", {**nurse, "stage": "Being seen", "doctor": "Dr Lim"})
    check("Board: triage nurse sets stage + doctor", s == 200)
    seen1 = next(r for r in req("GET", "/api/board")[1]["patients"] if r["id"] == pid)["seen_at"]
    time.sleep(1.1)
    req("PATCH", f"/api/patients/{pid}/board", {**nurse, "stage": "Results pending"})
    seen2 = next(r for r in req("GET", "/api/board")[1]["patients"] if r["id"] == pid)["seen_at"]
    check("Waiting clock stops when seen and stays stopped at later stages", seen1 is not None and seen1 == seen2)
    s2 = next(r for r in req("GET", "/api/board")[1]["patients"] if r["id"] == pid2)
    check("Patients still waiting keep the clock running", s2["seen_at"] is None)
    s, _ = req("PATCH", f"/api/patients/{pid}/board", {**nurse, "cleared": True})
    check("Board: can't clear a patient who hasn't left", s == 400)
    t = p["todos"][0]
    req("PATCH", f"/api/patients/{pid}/todos/{t['id']}", {**nurse, "done": True})
    req("PATCH", f"/api/patients/{pid}/board", {"who": "Dr Org", "role": "Organising doctor", "stage": "Admitted to ward"})
    s, _ = req("PATCH", f"/api/patients/{pid}/board", {"who": "Dr Org", "role": "Organising doctor", "cleared": True})
    b = req("GET", "/api/board")[1]
    left = req("GET", "/api/board?cleared=true")[1]
    check("Board: cleared patient leaves the board, case file kept under 'Left the ED'",
          s == 200 and pid not in [r["id"] for r in b["patients"]] and pid in [r["id"] for r in left["patients"]])
    _, p = req("GET", f"/api/patients/{pid}")
    check("To-do tick is kept with who did it", next(x for x in p["todos"] if x["id"] == t["id"])["done_by"] == "Nurse Test")
    check("Timeline records who did what", any(e["who"] == "Dr Org" for e in p["timeline"]))
    removed = p["todos"][1]["text"]
    req("PATCH", f"/api/patients/{pid}/todos/{p['todos'][1]['id']}", {**nurse, "delete": True})
    req("POST", f"/api/patients/{pid}/notes", form={"text": "repeat obs HR 90", "who": "Nurse Test"})
    p, _ = wait(pid)
    check("A removed AI to-do isn't suggested again", removed not in [x["text"] for x in p["todos"]] and
          sum(x["text"] == p["todos"][0]["text"] for x in p["todos"]) == 1)
    check("Later obs update the tiles and offer a re-score", p["fields"]["hr"] == 90 and p["rescore"])

    # ---- 6. chat, export, pictures
    s, out = req("POST", f"/api/patients/{pid}/chat", {"who": "Dr Lim", "text": "What is the presenting complaint?"})
    check("Chat answers", s == 200 and out["answer"])
    if A.real:
        check("Chat reads the latest case file (knows the edited complaint / troponin)",
              any(w in out["answer"].lower() for w in ("chest", "troponin")), out["answer"][:200])
    s, text = req("GET", f"/api/patients/{pid}/export")
    check("Case file downloads as text with all headings", s == 200 and "PRESENTING COMPLAINT" in text and "TIMELINE" in text)
    xray = IMAGES / "bone_xray_fracture_fracatlas.jpg"
    if xray.exists():
        pid4 = new("30M fell off bike onto L wrist, swollen and tender. pain 7/10. walk-in", files=[xray])
        p4, secs = wait(pid4)
        img = p4["notes"][0]["images"][0]
        check("Picture read and put under Imaging", img["kind"] and items(p4, "Imaging"), f"{img['kind']}, {secs:.1f} s")
        check("Bone X-ray: detector marks shown (detect -> segment)", img.get("overlay") or "nothing marked" in img.get("tools", ""),
              img.get("tools", ""))
        # the person says what it is at upload -> that type's check runs, no AI sorting
        s, out = req("POST", "/api/patients", form={"text": "50F fall, chest wall pain. walk-in", "who": "Nurse Test",
                                                    "kinds": json.dumps(["chest_xray"])}, files=[xray])
        p5, _ = wait(out["id"])
        img5 = p5["notes"][0]["images"][0]
        check("Picture type chosen at upload is used (not AI-sorted)", img5["kind"] == "chest_xray" and not img5["sorted_by_ai"])
        s, _ = req("POST", f"/api/patients/{pid4}/images/{img['file']}/read", {"who": "Dr Lim", "kind": "chest_xray"})
        p4, _ = wait(pid4)
        img = p4["notes"][0]["images"][0]
        readings = [i for i in items(p4, "Imaging") if i.get("image") == img["file"]]
        check("Changing the picture type reads it again, one reading kept", s == 200 and img["kind"] == "chest_xray"
              and len(readings) == 1 and "Chest X-ray" in readings[0]["text"])
        s, _ = req("POST", f"/api/patients/{pid4}/images/{img['file']}/outline", {"who": "Dr Lim", "box": [40, 40, 200, 220]})
        p4, _ = wait(pid4)
        img = p4["notes"][0]["images"][0]
        check("Outline an area: a dragged box gives an outline picture", s == 200 and img.get("outline"), img.get("tools", ""))
        s, _ = req("POST", f"/api/patients/{pid4}/images/{img['file']}/outline", {"who": "Dr Lim", "box": [1, 1, 2, 2]})
        check("Outline refuses a box that is too small", s == 400)
        r = next(i for i in items(p4, "Imaging") if i.get("image") == img["file"])
        req("PATCH", f"/api/patients/{pid4}/items/{r['id']}", {"who": "Dr Lim", "confirmed": True})
        p4, _ = wait(pid4)
        check("Picture reading can be marked correct", next(i for i in p4["items"] if i["id"] == r["id"])["confirmed"])
    s, _ = req("POST", "/api/patients", form={"text": "x", "who": "N"}, files=[HERE / "server.py"])
    check("Non-picture files are refused", s == 400)
    s, _ = req("POST", "/api/patients", form={"text": "x", "who": ""})
    check("Unsigned notes are refused", s in (400, 422))

    # ---- 7. the 10 test notes (speed + no errors + urgent caught)
    key = {}
    for line in (NOTES / "answer_key.csv").read_text().splitlines()[1:]:
        cols = line.split(",")
        key[cols[0]] = cols[3]
    ids = {f.name: new(f.read_text(encoding="utf-8")) for f in sorted(NOTES.glob("patient_*.txt"))}
    t = time.time()
    done = {n: wait(i)[0] for n, i in ids.items()}
    errors = [n for n, d in done.items() if d["status"] != "ready"]
    check("10 test notes: no AI errors", not errors, str(errors))
    urgent = [n for n, k in key.items() if k == "yes"]
    check("10 test notes: every truly urgent patient flagged", all(done[n]["decision"]["flagged"] for n in urgent),
          ", ".join(f"{n}: {done[n]['decision']['flagged']}" for n in urgent))
    check("10 test notes: each has a complaint", all(d["complaint"] for d in done.values()),
          "; ".join(f"{n[:10]}={d['complaint']!r}" for n, d in done.items()))
    print(f"10 notes processed in {time.time() - t:.0f} s")

    n_ok = sum(ok for _, ok, _ in RESULTS)
    print(f"\n{n_ok}/{len(RESULTS)} checks passed")
    out = HERE / "data" / ("test_results_real.json" if A.real else "test_results.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps([{"check": n, "pass": ok, "detail": d} for n, ok, d in RESULTS], indent=1))
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--real", action="store_true")
    A = ap.parse_args()
    sys.exit(main())
