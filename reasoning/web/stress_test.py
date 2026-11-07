"""Stress test: many made-up ED patients from our datasets, sent through the app, then checked against the whiteboard.

    python stress_test.py --url http://127.0.0.1:8001            # a running app (test mode): 450 patients, 8 at once
    python stress_test.py --url http://127.0.0.1:8000 --ktas 6 --nhamcs 6 --multicare 3 --tag demo
                                                                  # put a small demo set on your own board
    python stress_test.py --direct --real                         # HPC GPU: the real MedGemma, same app code, no browser

Patients (all held out from training):
  KTAS test (Korean ED, 190) and NHAMCS test (US EDs, sample of 200): written up as messy nurse notes from their real
  values (charting_check.note_for), with the true triage level known.
  MultiCaRe ED case reports (sample of 60): the opening of a real published case, as a long narrative "note" with no
  true level. These check that long, real text doesn't break anything.
What is measured:
  - load: errors, time from Submit to case file ready (includes waiting behind other patients), board response time
  - numbers: vital signs pulled out correctly / wrong / invented, per source
  - the whiteboard against the truth: urgent patients (level 1-2) flagged on the board (sensitivity), non-urgent left
    unflagged (specificity), under-triage; level exact / within 1 (real AI only: test mode has no level model);
    every patient on the board, most urgent first
  - real AI only: numbers in the case file that appear nowhere in the notes (a sign of invented facts), questions given
Saves outputs/metrics/stress_<tag>.json. Direct mode also saves the board (stress_board.pkl) so it can be opened in the
app on the laptop: python server.py --test --case-files <that file>.
"""
import argparse
import json
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "src"), str(HERE.parent / "app"), str(HERE)]

from charting_check import note_for, same                                 # noqa: E402
from config import OUTPUTS                                                 # noqa: E402
from data import load_multicare, load_pool1                                # noqa: E402

VITALS = ["hr", "sbp", "dbp", "rr", "spo2", "temp_c", "nrs_pain"]
SEED = 42


def make_patients(n_ktas, n_nhamcs, n_multicare):
    rng = random.Random(SEED)
    out = []
    pool = load_pool1(split="test")
    for source, n in (("ktas", n_ktas), ("nhamcs", n_nhamcs)):
        rows = pool[pool["source"] == source]
        rows = rows.sample(min(n, len(rows)), random_state=SEED) if n < len(rows) else rows
        for _, r in rows.iterrows():
            out.append({"source": source, "id": str(r["id"]), "note": note_for(r, rng),
                        "truth": {**{v: r[v] for v in VITALS + ["age", "sex"]}, "level": int(r["triage_level"]),
                                  "urgent": int(r["urgent"])}})
    mc = load_multicare().sample(n_multicare, random_state=SEED) if n_multicare else []
    for _, r in (mc.iterrows() if n_multicare else []):
        text = " ".join(re.split(r"(?<=\.)\s+", r["case_text"])[:6])[:900]      # the first few sentences
        out.append({"source": "multicare", "id": r["case_id"], "note": text, "truth": None})
    rng.shuffle(out)                                                       # arrivals mixed, like a real ED
    return out


# ------------------------------------------------------------------ two ways to run the same patients
class HttpRun:
    """Through a running app, many nurses at once."""

    def __init__(self, url, workers):
        self.url, self.workers = url.rstrip("/"), workers

    def run(self, patients):
        def submit(p):
            t = time.time()
            r = requests.post(f"{self.url}/api/patients", data={"text": p["note"], "who": f"Stress nurse {p['source']}"},
                              timeout=60)
            r.raise_for_status()
            p["pid"], p["t_submit"], p["submit_s"] = r.json()["id"], t, time.time() - t

        start = time.time()
        with ThreadPoolExecutor(self.workers) as ex:
            list(ex.map(lambda p: self._safe(submit, p), patients))
        waiting = {p["pid"]: p for p in patients if "pid" in p}
        while waiting:
            for pid in list(waiting):
                d = requests.get(f"{self.url}/api/patients/{pid}", timeout=30).json()
                if d["status"] != "working":
                    p = waiting.pop(pid)
                    p["result"], p["ready_s"] = d, time.time() - p["t_submit"]
            time.sleep(0.5)
        total = time.time() - start
        board_times = []
        for _ in range(20):
            t = time.time()
            board = requests.get(f"{self.url}/api/board", timeout=30).json()
            board_times.append(time.time() - t)
        return total, board, board_times

    @staticmethod
    def _safe(fn, p):
        try:
            fn(p)
        except Exception as e:                        # counted as an error, the run goes on
            p["error"] = repr(e)


class DirectRun:
    """No browser: the app's own Store and AI job, one patient after another (the GPU runs one job at a time anyway)."""

    def __init__(self, real, path):
        import server
        if real:
            from pipeline import EDPipeline
            ai = EDPipeline()
        else:
            from test_backend import TestPipeline
            ai = TestPipeline()
        server.store = server.Store(ai, path)
        self.server, self.store = server, server.store

    def run(self, patients):
        start = time.time()
        for i, p in enumerate(patients):
            t = time.time()
            pid = f"s{i:04d}"
            q = self.server.Patient(id=pid, bed=f"P{i + 1}", created=time.time())
            q.notes.append({"time": self.server.now(), "who": f"Stress nurse {p['source']}", "text": p["note"], "images": []})
            q.pending = [0]
            self.store.patients[pid] = q
            self.store._job(q)
            p["pid"], p["submit_s"], p["ready_s"] = pid, 0.0, time.time() - t
            p["result"] = {"version": 0, **q.full()}
            p["times"] = dict(getattr(self.store.ai, "times", {}))
            print(f"{i + 1}/{len(patients)} {p['source']} {q.status} {p['ready_s']:.1f}s", flush=True)
        total = time.time() - start
        board_times = []
        for _ in range(20):
            t = time.time()
            board = self.server.board()
            board_times.append(time.time() - t)
        return total, board, board_times


# ------------------------------------------------------------------ scoring
def numbers_in(text):
    return {float(x) for x in re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])", text or "")}


def score(patients, total, board, board_times, real):
    done = [p for p in patients if "result" in p]
    errors = [p for p in patients if "error" in p] + [p for p in done if p["result"]["status"] == "error"]
    res = {"n_patients": len(patients), "by_source": {s: sum(p["source"] == s for p in patients) for s in ("ktas", "nhamcs", "multicare")},
           "errors": len(errors), "error_examples": [p.get("error") or p["result"]["error"] for p in errors[:5]],
           "total_minutes": round(total / 60, 2),
           "ready_seconds_median": round(float(np.median([p["ready_s"] for p in done])), 2) if done else None,
           "ready_seconds_p90": round(float(np.percentile([p["ready_s"] for p in done], 90)), 2) if done else None,
           "board_ms_median": round(1000 * float(np.median(board_times)), 1),
           "board_rows": len(board["patients"])}

    # numbers, per source
    for src in ("ktas", "nhamcs"):
        c = w = m = inv = n = 0
        for p in done:
            if p["source"] != src:
                continue
            got, truth = p["result"]["fields"], p["truth"]
            for v in VITALS:
                t = truth[v]
                if v in ("sbp", "dbp") and (truth["sbp"] != truth["sbp"] or truth["dbp"] != truth["dbp"]):
                    t = float("nan")
                if t == t:
                    n += 1
                    if v not in got:
                        m += 1
                    elif same(v, t, got[v]):
                        c += 1
                    else:
                        w += 1
                elif v in got:
                    inv += 1
        res[f"numbers_{src}"] = {"written": n, "correct": round(c / max(n, 1), 4), "wrong": w, "missed": m, "invented": inv}
    mc = [p for p in done if p["source"] == "multicare"]
    if mc:
        res["multicare_with_3plus_vitals_found"] = round(np.mean([sum(v in p["result"]["fields"] for v in VITALS) >= 3 for p in mc]), 3)
        res["multicare_with_complaint"] = round(np.mean([bool(p["result"]["complaint"]) for p in mc]), 3)

    # the whiteboard against the truth
    on_board = {r["id"]: r for r in board["patients"]}
    labelled = [p for p in done if p["truth"] and p["result"]["decision"]]
    res["all_patients_on_board"] = all(p["pid"] in on_board for p in done)
    levels = [r["level"] if r["level"] is not None else 9 for r in board["patients"] if not r.get("leaving")]
    res["board_most_urgent_first"] = levels == sorted(levels)
    for src in ("ktas", "nhamcs", "all"):
        rows = [p for p in labelled if src in ("all", p["source"])]
        if not rows:
            continue
        y = np.array([p["truth"]["urgent"] for p in rows], bool)
        flag = np.array([on_board[p["pid"]]["flagged"] if p["pid"] in on_board else False for p in rows], bool)
        out = {"n": len(rows), "n_urgent": int(y.sum()),
               "sensitivity": round(float(flag[y].mean()), 3) if y.any() else None,
               "specificity": round(float((~flag[~y]).mean()), 3) if (~y).any() else None,
               "under_triage": round(float((~flag[y]).mean()), 3) if y.any() else None,
               "flagged_share": round(float(flag.mean()), 3)}
        if real:
            lv = np.array([on_board[p["pid"]]["level"] for p in rows])
            tl = np.array([p["truth"]["level"] for p in rows])
            out.update({"level_exact": round(float((lv == tl).mean()), 3),
                        "level_within_1": round(float((abs(lv - tl) <= 1).mean()), 3)})
        res[f"board_vs_truth_{src}"] = out

    if real:                                              # case file: numbers not found in what was typed
        bad, total_n = 0, 0
        for p in done:
            source_numbers = numbers_in(p["note"]) | numbers_in(json.dumps(p["result"].get("decision")))
            report = re.sub(r"## Triage decision support.*", "", p["result"]["report"], flags=re.S)
            found = numbers_in(report)
            total_n += len(found)
            bad += len({x for x in found if x not in source_numbers and x not in (1, 2, 3, 4, 5, 10)})
        res["case_file_numbers_not_in_notes"] = round(bad / max(total_n, 1), 4)
        res["questions_per_patient_median"] = float(np.median([len(p["result"]["questions"]) for p in done]))
        steps = {}
        for p in done:
            for k, v in p.get("times", {}).items():
                steps.setdefault(k, []).append(v)
        res["seconds_per_step_median"] = {k: round(float(np.median(v)), 2) for k, v in steps.items()}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8001")
    ap.add_argument("--direct", action="store_true", help="no browser: the app's code called directly")
    ap.add_argument("--real", action="store_true", help="with --direct: real MedGemma (GPU)")
    ap.add_argument("--real-metrics", action="store_true", help="http mode against the real-AI app: also score levels and case files")
    ap.add_argument("--ktas", type=int, default=190)
    ap.add_argument("--nhamcs", type=int, default=200)
    ap.add_argument("--multicare", type=int, default=60)
    ap.add_argument("--workers", type=int, default=8, help="nurses submitting at the same time (http mode)")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    tag = a.tag or ("real" if a.real else "direct" if a.direct else "http")
    patients = make_patients(a.ktas, a.nhamcs, a.multicare)
    print(f"{len(patients)} patients", flush=True)
    if a.direct:
        path = OUTPUTS / "metrics" / f"stress_board_{tag}.pkl"
        runner = DirectRun(a.real, path)
    else:
        runner = HttpRun(a.url, a.workers)
    total, board, board_times = runner.run(patients)
    res = score(patients, total, board, board_times, a.real or a.real_metrics)
    res["mode"] = tag
    out = OUTPUTS / "metrics" / f"stress_{tag}.json"
    out.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
    print("saved", out)


if __name__ == "__main__":
    main()
