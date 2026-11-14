"""Fill the whiteboard for the live demo (16 Nov 2026), so the slow full-AI work happens before you talk.

    python tools\\demo_seed.py                          # app already running: python server.py (port 8000)
    python tools\\demo_seed.py --url http://127.0.0.1:8060

Sends 6 background patients from the test pack, one after another, and waits for each (about 2 min each on the
laptop's full AI): T03 sepsis, T04 small cut, T06 confused elderly, R05 real patient, P06 pneumothorax X-ray, P11
head-CT bleed. T01 (chest pain) and P01 (ankle X-ray) are NOT sent: you submit those live (docs/DEMO_SCRIPT.md).
Prints each patient's level and urgent flag. Skips a case that is already on the board (safe to run twice).
"""
import argparse
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
PACK = HERE.parent / "build" / "ED_app_test_pack"
WHO = "Nurse Tan"
SEED = [("T03", []), ("T04", []), ("T06", []), ("R05", []),
        ("P06", ["cxr_pneumothorax.jpg"]), ("P11", ["ct_head_bleed_epidural.jpg"])]


def note_of(case):
    return (PACK / "cases" / f"{case}.txt").read_text(encoding="utf8").strip()


def already_there(url):
    """Notes already on the board (so a second run doesn't add the same patients again)."""
    have = set()
    for b in requests.get(f"{url}/api/board", timeout=30).json()["patients"]:
        p = requests.get(f"{url}/api/patients/{b['id']}", timeout=30).json()
        if p.get("notes"):
            have.add(p["notes"][0]["text"].strip())
    return have


def wait(url, pid, limit=1800):
    start = time.time()
    while time.time() - start < limit:
        p = requests.get(f"{url}/api/patients/{pid}", timeout=30).json()
        if p["status"] != "working" and not p.get("pending"):
            return p, time.time() - start
        time.sleep(3)
    sys.exit(f"patient {pid} did not finish in {limit} s: is the full AI running?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    a = ap.parse_args()
    url = a.url.rstrip("/")
    try:
        meta = requests.get(f"{url}/api/meta", timeout=10).json()
    except requests.RequestException:
        sys.exit(f"The app isn't answering at {url}. Start it first: python server.py")
    if meta.get("test"):
        print("WARNING: the app is in TEST MODE. For the demo start it without --test.")
    have = already_there(url)
    for case, pics in SEED:
        note = note_of(case)
        if note in have:
            print(f"{case}: already on the board, skipped")
            continue
        files = [("files", (n, open(PACK / "pictures" / n, "rb"), "image/jpeg")) for n in pics]
        r = requests.post(f"{url}/api/patients", data={"text": note, "who": WHO}, files=files or None, timeout=120)
        for _, (_, fh, _) in files:
            fh.close()
        r.raise_for_status()
        print(f"{case}: sent, the AI is working...", flush=True)
        p, secs = wait(url, r.json()["id"])
        flag = "URGENT" if p.get("flagged") else "not urgent"
        print(f"{case}: level {p.get('level')}, {flag}  ({secs:.0f} s)", flush=True)
    print("\nWhiteboard ready. Now follow docs/DEMO_SCRIPT.md section D: submit T01 and P01, then the slides.")


if __name__ == "__main__":
    main()
