"""Test notes you can load into the app: one .txt per patient + an answer key with the true triage level.

    python make_test_notes.py          # writes web/test_notes/ (10 notes + answer_key.csv)

Patients come from the held-out test splits (KTAS, NHAMCS) written as messy nurse notes, plus 2 real case-report
openings (MultiCaRe). Load one with "📄 Open a text file" on the charting screen, or copy-paste it.
"""
import csv
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1] / "reasoning" / "src"), str(HERE)]
from stress_test import make_patients                                    # noqa: E402

OUT = HERE.parent / "tests" / "test_notes"


def main():
    OUT.mkdir(exist_ok=True)
    patients = make_patients(4, 4, 2)
    random.Random(1).shuffle(patients)
    rows = []
    for i, p in enumerate(patients, 1):
        name = f"patient_{i:02d}_{p['source']}.txt"
        (OUT / name).write_text(p["note"] + "\n", encoding="utf8")
        t = p["truth"] or {}
        rows.append({"file": name, "source": p["source"], "true_triage_level": t.get("level", "unknown (case report)"),
                     "truly_urgent_level_1_2": {1: "yes", 0: "no"}.get(t.get("urgent"), "unknown")})
    with open(OUT / "answer_key.csv", "w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} notes + answer_key.csv to {OUT}")


if __name__ == "__main__":
    main()
