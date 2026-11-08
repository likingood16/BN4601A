"""The group test pack (8 Nov 2026): test cases + pictures + what the app should show, checked against the app itself.

    python server.py --test --port 8010 --case-files ""     # 1. a test-mode app with an empty board
    python make_test_pack.py --url http://127.0.0.1:8010    # 2. writes test_pack/ED_app_test_pack(.zip)

Every case is sent to the running app the same way the screens send it, and what the app showed is written into
expected_results.csv. So the guide and the answer key describe what the app really does (they are not hoped-for
results). Cases whose check fails are printed and the script exits with an error, so a changed app can't silently
disagree with the guide.

Pictures are all CC BY 4.0 (FracAtlas, GRAZPEDWRI-DX, MultiCaRe, Kermany pediatric pneumonia); none from VinDr or
RSNA (non-commercial, they stay on vanda). Notes are written for testing or come from held-out test patients
(KTAS, NHAMCS, MultiCaRe case reports).
"""
import argparse
import csv
import shutil
import time
import zipfile
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PICS_IN = HERE / "test_pack_pictures"          # the CC BY pictures picked on vanda (pictures.csv says which)
OUT = HERE / "test_pack" / "ED_app_test_pack"
FRACATLAS = REPO / "vision" / "detector" / "data" / "yolo" / "images" / "test"
WHO = "Test nurse"


def field(name):
    return lambda p: p["fields"].get(name)


def flagged(p):
    return bool(p["decision"] and p["decision"]["flagged"])


def finding(p, name, key="flag"):
    for n in p["notes"]:
        for i in n["images"]:
            for f in i.get("findings", []):
                if f["name"] == name:
                    return bool(f.get(key))
    return None


def img0(p):
    return p["notes"][0]["images"][0] if p["notes"][0]["images"] else {}


# id, title, note, pictures, what to look at, check(p) -> True if the app did what the guide says
CASES = [
    ("T01", "Chest pain, low blood pressure",
     "67M CP since 0600, sweaty, radiating to L arm. HR 112 BP 90/60 RR 22 sats 94% RA T 37.2 pain 8/10. hx DM, HTN on metformin. NKDA. BIBA",
     [], "Flagged URGENT; BP tile red; chest-pain to-dos (ECG, troponin)",
     lambda p: flagged(p) and p["fields"].get("sbp") == 90 and any("ECG" in t["text"] for t in p["todos"])),
    ("T02", "Possible stroke",
     "72F rt sided weakness and slurred speech since 0830. BP 182/100 HR 88 RR 18 sats 97% RA. walk-in",
     [], "Stroke pathway: asks when she was last known to be well, glucose, blood thinners; stroke to-do",
     lambda p: any("last known" in q.lower() for q in p["questions"]) and any("stroke" in t["text"].lower() for t in p["todos"])),
    ("T03", "Fever, fast heart rate, confused",
     "45F fever and rigors 2 days, dysuria. T 38.9 HR 118 BP 104/62 RR 24 sats 95% RA. confused. walk-in",
     [], "Flagged URGENT; consciousness tile shows C/V; sepsis to-dos",
     lambda p: flagged(p) and p["fields"].get("mental") == "verbal"),
    ("T04", "Small finger cut (should NOT be urgent)",
     "23M cut finger on kitchen knife, bleeding stopped. HR 76 BP 124/78 RR 14 sats 99% T 36.6 pain 3/10 alert walk-in",
     [], "Not flagged; level 3-5",
     lambda p: not flagged(p) and p["level"] >= 3),
    ("T05", "Ambulance numbers, then arrival numbers",
     "61F lightheaded and flushing. EMS found BP 70/40 HR 130. On arrival BP 112/70 HR 92 sats 98% RA. alert. hx pulmonary hypertension on tadalafil",
     [], "Tiles show the ARRIVAL values BP 112/70, HR 92 (not 70/40, 130), with a note saying so",
     lambda p: p["fields"].get("sbp") == 112 and p["fields"].get("hr") == 92),
    ("T06", "Fahrenheit and GCS",
     "80 yo female, confused since this morning, GCS 13, temp 101.3F, pulse 104, BP 150/90, RR 22, sats 95% on air, brought by family",
     [], "Temp shown as 38.5 C; consciousness C/V (GCS 13)",
     lambda p: abs((p["fields"].get("temp_c") or 0) - 38.5) < 0.11 and p["fields"].get("mental") == "verbal"),
    ("T07", "Unusual writing: 'over', decimal comma",
     "M 52 headache. BP 130 over 85, HR 70, T 37,5, SpO2 98, pain 6/10, walk-in, AVPU A",
     [], "BP 130/85 and temp 37.5 read correctly",
     lambda p: p["fields"].get("sbp") == 130 and p["fields"].get("dbp") == 85 and p["fields"].get("temp_c") == 37.5),
    ("T08", "Impossible number (typo)",
     "30F palpitations. HR 400 BP 120/80 RR 16 sats 99% T 36.8 alert walk-in",
     [], "HR 400 is ignored with a red warning; HR tile says not recorded",
     lambda p: p["fields"].get("hr") is None and any("400" in w for w in p["found"].get("warnings", []))),
    ("T09", "No numbers at all",
     "pt c/o feeling unwell, no obs done yet",
     [], "Every tile 'not recorded'; the AI asks for the missing observations",
     lambda p: not any(p["fields"].get(k) for k in ("hr", "sbp", "rr", "spo2"))),
    ("T10a", "Same patient, way 1",
     "58 yo man, chest tightness, came by ambulance. HR 104, BP 100/60, RR 24, SpO2 93% RA, T 37.0, pain 7/10, alert",
     [], "Same numbers, level and flag as T10b and T10c", None),
    ("T10b", "Same patient, way 2 (shorthand)",
     "58M chest tightness BIBA. hr104 bp100/60mmHg rr=24 O2 93 temp 37,0 pain 7 AVPU A",
     [], "Same as T10a", None),
    ("T10c", "Same patient, way 3 (capitals, other order)",
     "CHEST TIGHTNESS, 58 YO MALE. BIBA. BP 100/60 HR 104 RR 24 SATS 93% RA T 37.0 PAIN 7/10 ALERT",
     [], "Same as T10a", None),
    # pictures (the trained image models are real in test mode too)
    ("P01", "Adult ankle X-ray with a fracture", "45M fell off ladder onto L leg, ankle swollen, can't weight bear. pain 7/10. HR 96 BP 130/80 sats 99%. walk-in",
     ["adult_ankle_fracture.jpg"], "Sorted as bone X-ray; fracture marked (adult cut-off); outline picture",
     lambda p: img0(p).get("kind") == "bone_xray" and finding(p, "Fracture") is True),
    ("P02", "Adult leg X-ray with a fracture the model MISSES", "38F tripped down stairs, R lower leg pain 6/10, limping. HR 84 BP 122/76. walk-in",
     ["adult_leg_fracture_missed.jpg"], "Nothing marked although there is a fracture: the detector misses about 1 in 6 adult fractures",
     lambda p: img0(p).get("kind") == "bone_xray" and finding(p, "Fracture") is False),
    ("P03", "Adult leg X-ray, no fracture", "50M kicked in the lower leg playing football, can walk. old leg operation (wires). pain 4/10. HR 78 BP 128/82. walk-in",
     ["adult_leg_no_fracture.jpg"], "Record what the app shows: about 1 in 4 normal adult X-rays get a mark", None),
    ("P04", "Child's wrist with a fracture", "9yo boy fell off monkey bars onto R wrist, swollen. pain 8/10. HR 110 RR 22 sats 99% T 36.9. alert. brought by mum",
     ["child_wrist_fracture.png"], "Fracture marked, using the CHILD cut-off (age 9)",
     lambda p: "child cut-off" in img0(p).get("text", "")),
    ("P05", "Child's wrist, no fracture", "7yo girl fell from scooter, L wrist sore, moving it. pain 3/10. HR 96. brought by dad",
     ["child_wrist_normal.png"], "Child cut-off; record whether anything is marked",
     lambda p: "child cut-off" in img0(p).get("text", "")),
    ("P06", "Chest X-ray: large pneumothorax", "28M sudden sharp L chest pain and breathless while playing football. HR 112 RR 26 sats 92% RA BP 118/74. walk-in",
     ["cxr_pneumothorax.jpg"], "Pneumothorax red (box) or amber 'possible' (second model)",
     lambda p: img0(p).get("kind") == "chest_xray" and (finding(p, "Pneumothorax") or finding(p, "Pneumothorax", "possible"))),
    ("P07", "Chest X-ray after the pneumothorax resolved", "28M review, chest pain settled. HR 80 RR 16 sats 98% RA. walk-in",
     ["cxr_pneumothorax_resolved.jpg"], "Record what the app shows (should be no red pneumothorax)", None),
    ("P08", "Chest X-ray: big pleural effusion", "66F breathless for a week, worse lying flat. HR 102 RR 26 sats 91% RA BP 138/84 T 37.4. walk-in",
     ["cxr_effusion.jpg"], "Pleural effusion red",
     lambda p: finding(p, "Pleural effusion") is True),
    ("P09", "Child's chest X-ray: pneumonia", "4yo girl fever and cough 3 days, fast breathing. T 39.1 HR 150 RR 44 sats 93% RA. brought by mum",
     ["child_cxr_pneumonia.jpg"], "Pneumonia-type shadowing red", lambda p: finding(p, "Pneumonia-type shadowing") is True),
    ("P10", "Child's chest X-ray: normal", "5yo boy cough 2 days, playing. T 37.6 HR 110 RR 24 sats 98%. brought by mum",
     ["child_cxr_normal.jpg"], "Record what the app shows", None),
    ("P11", "Head CT: epidural bleed", "78F fall at home, hit head, on warfarin, vomited twice, GCS 14. HR 92 BP 168/92 sats 97%. BIBA. CT head done",
     ["ct_head_bleed_epidural.jpg"], "Head-CT bleed check runs (the note says head)",
     lambda p: finding(p, "Bleeding inside the skull") is not None),
    ("P12", "Head CT: subdural bleed (contrast scan)", "70M confused after a fall last week, headache. GCS 14. HR 84 BP 150/88. brought by family. CT head",
     ["ct_head_bleed_subdural.jpg"], "Bleed check runs; record the result",
     lambda p: finding(p, "Bleeding inside the skull") is not None),
    ("P13", "Head CT: normal", "34F headache after minor knock to the head, GCS 15. HR 72 BP 118/70. walk-in. CT head",
     ["ct_head_normal.jpg"], "Bleed check runs; record the result",
     lambda p: finding(p, "Bleeding inside the skull") is not None),
    ("P14", "CT of the chest (not the head)", "60M breathless, leg swelling. HR 98 RR 24 sats 93% BP 140/90. walk-in. CT chest done",
     ["ct_chest.jpg"], "Sorted as CT; the bleed check does NOT run (the note doesn't say head)",
     lambda p: img0(p).get("kind") == "ct_slice" and finding(p, "Bleeding inside the skull") is None),
    ("P15", "MRI picture", "55F back pain 2 weeks. HR 76 BP 130/80. walk-in", ["mri_spine.jpg"],
     "Sorted as MRI; no trained detector, only the AI checklist", lambda p: img0(p).get("kind") == "mri_slice"),
    ("P16", "Ultrasound picture", "40M chest pain, known heart problem. HR 88 BP 126/78. walk-in", ["ultrasound_heart.jpg"],
     "Sorted as ultrasound", lambda p: img0(p).get("kind") == "ultrasound"),
    ("P17", "ECG picture", "70M palpitations. HR 130 irregular BP 110/70. BIBA", ["ecg.jpg"],
     "Sorted as ECG", lambda p: img0(p).get("kind") == "ecg"),
    ("P18", "Not a scan (a photo)", "48F abdominal mass, sent from clinic. HR 80 BP 124/76. walk-in", ["photo_not_a_scan.jpg"],
     "Sorted as other", lambda p: img0(p).get("kind") == "other"),
]
PICTURES = {"adult_ankle_fracture.jpg": FRACATLAS / "IMG0003458.jpg", "adult_leg_fracture_missed.jpg": FRACATLAS / "IMG0003297.jpg",
            "adult_leg_no_fracture.jpg": FRACATLAS / "IMG0000007.jpg",       # all three are lower legs (9 Nov check)
            "child_wrist_fracture.png": "child_wrist_fracture_1.png", "child_wrist_normal.png": "child_wrist_normal_1.png",
            "cxr_pneumothorax.jpg": "cxr_pneumothorax_1.jpg", "cxr_pneumothorax_resolved.jpg": "cxr_pneumothorax_resolved.jpg",
            "cxr_effusion.jpg": "cxr_effusion_1.jpg", "child_cxr_pneumonia.jpg": "child_cxr_pneumonia_1.jpg",
            "child_cxr_normal.jpg": "child_cxr_normal_1.jpg", "ct_head_bleed_epidural.jpg": "ct_head_bleed_2.jpg",
            "ct_head_bleed_subdural.jpg": "ct_head_bleed_1.jpg", "ct_head_normal.jpg": "ct_head_no_bleed_1.jpg",
            "ct_chest.jpg": "ct_body_1.jpg", "mri_spine.jpg": "mri_1.jpg", "ultrasound_heart.jpg": "ultrasound_1.jpg",
            "ecg.jpg": "ecg_1.jpg", "photo_not_a_scan.jpg": "photo_other_1.jpg"}
SOURCES = {"adult_": "FracAtlas (Abedeen et al. 2023, CC BY 4.0), test split",
           "child_wrist": "GRAZPEDWRI-DX (Nagy et al. 2022, CC BY 4.0), test split",
           "child_cxr": "Kermany et al. 2018 pediatric chest X-rays (CC BY 4.0), test split",
           "": "MultiCaRe case-report figures (Nievas Offidani & Delrieux 2024, CC BY 4.0)"}


def wait(url, pid, limit=180):
    for _ in range(limit * 2):
        p = requests.get(f"{url}/api/patients/{pid}", timeout=30).json()
        if p["status"] != "working" and not p.get("pending"):
            return p
        time.sleep(0.5)
    raise SystemExit(f"patient {pid} did not finish")


def summary(p):
    d = p["decision"] or {}
    f = p["fields"]
    vit = ", ".join(f"{k} {f[k]:g}" if isinstance(f[k], float) else f"{k} {f[k]}" for k in
                    ("age", "sex", "hr", "sbp", "dbp", "rr", "spo2", "temp_c", "nrs_pain", "mental", "ambulance") if k in f)
    out = {"level": p["level"], "flagged_urgent": "yes" if d.get("flagged") else "no", "numbers": vit,
           "number_warnings": " | ".join(p["found"].get("warnings", [])),
           "questions": " | ".join(p["questions"]), "todos": " | ".join(t["text"] for t in p["todos"])}
    imgs = [i for n in p["notes"] for i in n["images"]]
    if imgs:
        i = imgs[0]
        marks = [("RED " if f["flag"] else "amber possible ") + f["name"] for f in i.get("findings", [])
                 if f.get("flag") or f.get("possible")]
        out.update({"picture_sorted_as": i.get("kind"), "picture_findings": "; ".join(marks) or "nothing flagged",
                    "picture_text": i.get("text", "")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8010")
    ap.add_argument("--pics", default=str(PICS_IN))
    a = ap.parse_args()
    url, pics = a.url.rstrip("/"), Path(a.pics)
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "cases").mkdir(parents=True)
    (OUT / "pictures").mkdir()
    for name, src in PICTURES.items():
        shutil.copy(src if isinstance(src, Path) else pics / src, OUT / "pictures" / name)
    # real held-out patients (true triage level known), from make_test_notes.py
    real = list(csv.DictReader(open(HERE / "test_notes" / "answer_key.csv", encoding="utf8")))
    rows, failed = [], []
    for cid, title, note, pictures, expect, check in CASES:
        (OUT / "cases" / f"{cid}.txt").write_text(note + "\n", encoding="utf8")
        files = [("files", (n, open(OUT / "pictures" / n, "rb"), "image/png" if n.endswith(".png") else "image/jpeg"))
                 for n in pictures]
        r = requests.post(f"{url}/api/patients", data={"text": note, "who": WHO}, files=files or None, timeout=120)
        r.raise_for_status()
        p = wait(url, r.json()["id"])
        ok = None if check is None else bool(check(p))
        if ok is False:
            failed.append(cid)
        rows.append({"case": cid, "title": title, "file": f"cases/{cid}.txt", "pictures": " ".join(f"pictures/{n}" for n in pictures),
                     "what_the_guide_says": expect, "check": {True: "PASS", False: "FAIL", None: "record"}[ok], **summary(p)})
        print(f"{cid:5} {rows[-1]['check']:6} level {p['level']} flagged {rows[-1]['flagged_urgent']:3} "
              f"{rows[-1].get('picture_sorted_as', '')} {rows[-1].get('picture_findings', '')}", flush=True)
    same = [r for r in rows if r["case"].startswith("T10")]
    consistent = len({(r["numbers"], r["level"], r["flagged_urgent"]) for r in same}) == 1
    for r in same:
        r["check"] = "PASS" if consistent else "FAIL"
    if not consistent:
        failed.append("T10")
    for i, k in enumerate(real, 1):
        cid = f"R{i:02d}"
        note = (HERE / "test_notes" / k["file"]).read_text(encoding="utf8").strip()
        (OUT / "cases" / f"{cid}.txt").write_text(note + "\n", encoding="utf8")
        r = requests.post(f"{url}/api/patients", data={"text": note, "who": WHO}, timeout=120)
        p = wait(url, r.json()["id"])
        rows.append({"case": cid, "title": f"Real held-out patient ({k['source']})", "file": f"cases/{cid}.txt", "pictures": "",
                     "what_the_guide_says": f"True triage level {k['true_triage_level']} (urgent: {k['truly_urgent_level_1_2']}); "
                                            "compare with the app's level", "check": "record", **summary(p)})
        print(f"{cid:5} level {p['level']} vs true {k['true_triage_level']}", flush=True)
    cols = ["case", "title", "file", "pictures", "what_the_guide_says", "check", "level", "flagged_urgent", "numbers",
            "number_warnings", "questions", "todos", "picture_sorted_as", "picture_findings", "picture_text"]
    with open(OUT / "expected_results.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(OUT / "feedback_form.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["your_name", "case_or_step", "what_you_did", "what_the_guide_says", "what_happened", "same_as_guide (yes/no)",
                    "how_bad (1 cosmetic - 5 unsafe)", "screenshot_file", "time_taken_s", "comments"])
        for r in rows:
            w.writerow(["", r["case"], "", r["what_the_guide_says"], "", "", "", "", "", ""])
    (OUT / "ATTRIBUTION.txt").write_text(
        "Pictures (all CC BY 4.0; used for testing only, not for diagnosis):\n"
        + "\n".join(f"  {n}: {next(v for k, v in SOURCES.items() if n.startswith(k))}" for n in PICTURES)
        + "\n\nNotes T01-T10 and P01-P18 were written for this test. Notes R01-R10 are held-out test patients from KTAS "
          "(Moon et al. 2019, Kaggle), NHAMCS 2022 (CDC, public domain) and MultiCaRe case reports (CC BY 4.0).\n"
          "These are not real patients of ours. This app is a student research prototype, not a medical device.\n",
        encoding="utf8")
    shutil.copy(HERE / "test_pack_README.txt", OUT / "README.txt")
    guide = HERE / "test_pack" / "ED_app_user_guide.pdf"
    if guide.exists():
        shutil.copy(guide, OUT / guide.name)
    zpath = HERE / "test_pack" / "ED_app_test_pack.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(OUT.parent))
    print(f"\n{len(rows)} cases, failed checks: {failed or 'none'}\nwrote {zpath} ({zpath.stat().st_size / 1e6:.1f} MB)")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
