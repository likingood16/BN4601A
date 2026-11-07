"""Real-AI check of the app: every kind of input through the running app, every output saved for a person to read.

    python real_ai_check.py --url http://127.0.0.1:8010        # the vanda app, through connect_app.ps1's tunnel

1. Text: the 10 notes in test_notes/ (held-out KTAS / NHAMCS patients + 2 case reports) as new patients. Board level
   and flag are compared with answer_key.csv.
2. Images: one patient, then one note per picture in test_images/ (bone X-ray with fracture, chest X-ray normal and
   pneumonia, CT, MRI, ultrasound, ECG, other). The AI's image type is compared with the file name.
3. Case file behaviour with the real AI: a person's edit, then a new note: the edit must still be there.
4. Chat: one question.
Saves outputs/metrics/real_ai_check.json and outputs/real_ai_check.md (all case files, questions, image readings).
"""
import argparse
import csv
import json
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "outputs"
WHO = "Real-AI check"


def wait(url, pid, limit=600):
    t = time.time()
    while time.time() - t < limit:
        p = requests.get(f"{url}/api/patients/{pid}", timeout=30).json()
        if p["status"] != "working":
            return p, time.time() - t
        time.sleep(2)
    raise TimeoutError(pid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8010")
    ap.add_argument("--allow-test", action="store_true", help="run against a test-mode app (to check this script)")
    a = ap.parse_args()
    url = a.url.rstrip("/")
    meta = requests.get(f"{url}/api/meta", timeout=30).json()
    if meta["test"] and not a.allow_test:
        raise SystemExit("this app is in TEST MODE: point --url at the real-AI app")
    res = {"text": [], "images": [], "checks": {}}

    # 1. text notes
    key = {r["file"]: r for r in csv.DictReader(open(HERE / "test_notes" / "answer_key.csv", encoding="utf8"))}
    for f in sorted((HERE / "test_notes").glob("patient_*.txt")):
        note = f.read_text(encoding="utf8").strip()
        pid = requests.post(f"{url}/api/patients", data={"text": note, "who": WHO}, timeout=60).json()["id"]
        p, secs = wait(url, pid)
        k = key[f.name]
        row = {"file": f.name, "pid": pid, "seconds": round(secs, 1), "status": p["status"], "error": p["error"],
               "true_level": k["true_triage_level"], "truly_urgent": k["truly_urgent_level_1_2"],
               "board_level": p["level"], "flagged": p["flagged"], "complaint": p["complaint"], "fields": p["fields"],
               "questions": p["questions"], "todos": [t["text"] for t in p["todos"]], "report": p["report"], "note": note}
        res["text"].append(row)
        print(f"{f.name}: {p['status']} {secs:.0f}s level {p['level']} (true {k['true_triage_level']}) flagged {p['flagged']}", flush=True)

    # 2. images: one patient per picture, each with a note that fits it (as a nurse would send it)
    notes = {"bone_xray_fracture_fracatlas": "45M fell off ladder, L wrist pain 7/10, HR 96 BP 130/80 sats 99, walk-in, NKDA",
             "bone_xray_multicare": "30F fell on outstretched hand, R forearm pain 8/10, HR 88 BP 120/75 sats 99",
             "chest_xray_label0": "8M cough 3 days, T 37.8, RR 24, sats 97%, alert",
             "chest_xray_label1": "6F fever and cough 4 days, T 39.1, RR 32, sats 92%, drowsy",
             "chest_xray_multicare": "60M SOB 1 week, RR 26 sats 91% BP 130/80 HR 104",
             "ct_slice_multicare": "40F jaw pain and swelling 1 week, HR 80 BP 125/80 T 37.4",
             "ecg_multicare": "58M palpitations since 2h, HR 140 BP 110/70 sats 97%",
             "mri_slice_multicare": "35M headache 2 weeks, BP 130/85 HR 75, alert",
             "other_multicare": "62M weight loss and fever 2 months, T 38.2 HR 98 BP 118/70",
             "ultrasound_multicare": "55F SOB, HR 115 BP 95/60 RR 24 sats 93%"}
    for img in sorted((HERE / "test_images").glob("*.jpg")):
        with open(img, "rb") as fh:
            r = requests.post(f"{url}/api/patients", data={"text": notes.get(img.stem, "patient with an image"), "who": WHO},
                              files=[("files", (img.name, fh, "image/jpeg"))], timeout=120)
        pid = r.json()["id"]
        p, secs = wait(url, pid)
        got = p["notes"][0]["images"][0] if p["notes"][0]["images"] else {}
        true_kind = img.stem.split("_multicare")[0].split("_label")[0].split("_fracture")[0]
        imaging = p["report"].split("## Imaging")[1].split("##")[0].strip() if "## Imaging" in p["report"] else ""
        res["images"].append({"file": img.name, "pid": pid, "true_type": true_kind, "ai_type": got.get("kind"),
                              "seconds": round(secs, 1), "reading": got.get("text"), "marks": bool(got.get("overlay")),
                              "outline": bool(got.get("outline")), "case_file_imaging": imaging, "level": p["level"],
                              "questions": p["questions"], "todos": [t["text"] for t in p["todos"]]})
        print(f"{img.name}: AI says {got.get('kind')} ({secs:.0f}s), marks {bool(got.get('overlay'))}, "
              f"outline {bool(got.get('outline'))}, in case file {bool(imaging) and 'not recorded' not in imaging.lower()}", flush=True)
    pid = next(i["pid"] for i in res["images"] if i["file"].startswith("bone_xray_fracture"))
    res["image_patient"] = pid

    # 3. a person's edit survives a later note
    p = requests.get(f"{url}/api/patients/{pid}", timeout=30).json()
    edited = p["report"] + "\n\n## Clinician note\n- Distal radius fracture confirmed on review (Dr Check)"
    requests.put(f"{url}/api/patients/{pid}/report", json={"text": edited, "who": "Dr Check"}, timeout=30)
    requests.post(f"{url}/api/patients/{pid}/notes", data={"text": "backslab applied, ortho referral sent", "who": WHO}, timeout=60)
    p, _ = wait(url, pid)
    res["checks"]["edit_kept_after_new_note"] = "Distal radius fracture confirmed on review (Dr Check)" in p["report"]
    res["checks"]["new_note_added"] = "ortho" in p["report"].lower() or "backslab" in p["report"].lower()

    # 4. chat
    t = time.time()
    ans = requests.post(f"{url}/api/patients/{pid}/chat", json={"text": "What should we do next for this patient?", "who": "Dr Check"},
                        timeout=300).json()["answer"]
    res["chat"] = {"seconds": round(time.time() - t, 1), "answer": ans}

    # 5. the same patient written three ways: the complaint line and the urgency score should not change
    ways = ["86F dyspnea walk-in. bp 124/60 rr 24 AVPU A pain 0/10 PR 77 temp 36.4C",
            "86 yo woman, walked in, short of breath. BP 124/60, RR 24, alert, pain 0/10, HR 77, T 36.4",
            "F 86 SOB. self-presented. HR 77 BP 124/60 RR 24 temp 36.4 C, pain 0/10, AVPU A"]
    same = []
    for w in ways:
        q = requests.post(f"{url}/api/patients", data={"text": w, "who": WHO}, timeout=60).json()["id"]
        d, _ = wait(url, q)
        same.append({"note": w, "complaint": d["fields"].get("chief_complaint"), "p_urgent": round(d["decision"]["p_urgent"], 3),
                     "level": d["level"], "flagged": d["flagged"]})
    res["same_patient_three_ways"] = same
    res["checks"]["same_level_three_ways"] = len({s["level"] for s in same}) == 1
    res["checks"]["same_flag_three_ways"] = len({s["flagged"] for s in same}) == 1
    res["final_image_patient_report"] = p["report"]

    # summary
    lab = [r for r in res["text"] if r["true_level"].isdigit() and r["board_level"]]
    res["summary"] = {
        "text_patients": len(res["text"]), "errors": sum(r["status"] != "ready" for r in res["text"]),
        "median_seconds_per_patient": sorted(r["seconds"] for r in res["text"])[len(res["text"]) // 2],
        "urgent_flagged": f"{sum(r['flagged'] for r in lab if r['truly_urgent'] == 'yes')}/{sum(r['truly_urgent'] == 'yes' for r in lab)}",
        "level_exact": f"{sum(int(r['true_level']) == r['board_level'] for r in lab)}/{len(lab)}",
        "level_within_1": f"{sum(abs(int(r['true_level']) - r['board_level']) <= 1 for r in lab)}/{len(lab)}",
        "image_type_right": f"{sum(i['true_type'] == i['ai_type'] for i in res['images'])}/{len(res['images'])}",
        "image_reading_in_case_file": f"{sum(bool(i['case_file_imaging']) and 'not recorded' not in i['case_file_imaging'].lower() for i in res['images'])}/{len(res['images'])}",
        "patients_with_questions": f"{sum(bool(r['questions']) for r in res['text'] + res['images'])}/{len(res['text']) + len(res['images'])}",
        **res["checks"],
    }
    (OUT / "metrics" / "real_ai_check.json").write_text(json.dumps(res, indent=1, default=str))
    md = ["# Real-AI check of the app", "", "```", json.dumps(res["summary"], indent=1), "```", ""]
    for r in res["text"]:
        md += [f"## {r['file']}: board level {r['board_level']} (true {r['true_level']}), flagged {r['flagged']}, {r['seconds']} s",
               f"**Note:** {r['note']}", "", f"**Complaint:** {r['complaint']}", f"**Questions:** {r['questions']}",
               f"**To-dos:** {r['todos']}", "", r["report"], ""]
    md += ["## Images (one patient each)", ""]
    for i in res["images"]:
        md += [f"### {i['file']} (true {i['true_type']}): AI type {i['ai_type']}, level {i['level']}, marks {i['marks']}, "
               f"outline {i['outline']}, {i['seconds']} s", f"**AI reading:** {i['reading']}", "",
               f"**Case file, Imaging section:** {i['case_file_imaging']}", f"**Questions:** {i['questions']}",
               f"**To-dos:** {i['todos']}", ""]
    md += ["", "## Image patient: final case file (after a person's edit + a new note)", "", res["final_image_patient_report"],
           "", f"## Chat ({res['chat']['seconds']} s)", "", res["chat"]["answer"],
           "", "## Same patient, written three ways", ""] + [
        f"- complaint **{s['complaint']}**, urgency {s['p_urgent']}, level {s['level']}, flagged {s['flagged']}  ← {s['note']}"
        for s in res["same_patient_three_ways"]]
    (OUT / "real_ai_check.md").write_text("\n".join(md), encoding="utf8")
    print(json.dumps(res["summary"], indent=1))


if __name__ == "__main__":
    main()
