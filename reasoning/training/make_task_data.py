"""Option 2: training data in EXACTLY the app's tasks, from TRAINING patients only (KTAS + NHAMCS train split).

    python make_task_data.py --mode full --triage 2500 --chats 800        # HPC, base MedGemma as the teacher
    python make_task_data.py --mode 4bit --triage 2 --chats 1              # quick laptop check

Plain MedGemma is the "teacher": it writes the answers, we keep only good ones, and the adapter learns from them
(distillation). Thinking is removed, so the adapter learns to answer directly.
1. Triage: card + urgency decision (out-of-fold score, see head_scores.py) -> JSON. The teacher is told the
   TARGET level, so every example carries the right level:
     flagged     -> true level, but at most 2 (never lower a flagged patient; safe over-triage)
     not flagged -> true level (1-2 when the score missed an urgent patient = learn to raise on red flags)
   Urgent patients are oversampled to half the examples, so "always level 3" stops working.
2. Consultations: the same 4 turns as the test, but with DIFFERENT new facts from the test's, so the test still
   measures memory on facts never seen in training. Kept only if turns 2 and 4 use the fact, the report has all
   four SBAR headings, and turns 1-2 hedge at least once.
Saves data/processed/task_{train,val}.jsonl (95/5 split): {"messages": [...], "source": "task_triage"|"task_chat"}.
"""
import sys as _sys; from pathlib import Path as _Path  # noqa: E401,E702  (reasoning/src etc. importable)
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1] / _d) for _d in ("src", "training", "checks")]
import argparse
import json
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from answers import HEDGES, final_answer, generate, generate_chats, has_sbar_headings, parse_triage, cut_off
from config import PROCESSED, SEED
from model import load
from prompts import CHAT_SYSTEM, TRIAGE_SYSTEM, TURNS, card_with_urgency

TRAIN_FACTS = [("The patient takes aspirin and clopidogrel after a coronary stent.", r"aspirin|clopidogrel|antiplatelet|stent"),
               ("The patient is on chemotherapy for breast cancer, last cycle 8 days ago.", r"chemo|neutropen|cancer"),
               ("The patient had abdominal surgery two weeks ago.", r"surgery|post-?op"),
               ("The patient drinks heavily, about a bottle of spirits a day.", r"alcohol|drink|withdrawal"),
               ("The patient returned from a long-haul flight yesterday.", r"flight|travel|thrombo|embol|DVT|PE\b"),
               ("The patient has an allergy to iodinated contrast.", r"contrast|iodin"),
               ("The patient lives alone and has fallen twice this week.", r"alone|fall|fell"),
               ("The patient takes lithium for bipolar disorder.", r"lithium")]


def target_level(true_level, flagged):
    return min(int(true_level), 2) if flagged else int(true_level)


def teacher_prompt(card, level):
    return (card + f"\n\n(For teaching: the confirmed correct triage level for this patient is {level}. "
            f"Answer with triage_level {level} and explain why.)")


def triage_examples(model, processor, pool, n, batch):
    urgent, other = pool[pool["urgent"] == 1], pool[pool["urgent"] == 0]
    pick = pd.concat([urgent.sample(min(len(urgent), n // 2), random_state=SEED),
                      other.sample(n - min(len(urgent), n // 2), random_state=SEED)])
    cards = [card_with_urgency(c, p, f, ff) for c, p, f, ff in
             zip(pick["card"], pick["p_urgent"], pick["flagged"], pick["floor_flag"])]
    levels = [target_level(t, f) for t, f in zip(pick["triage_level"], pick["flagged"])]
    raws = generate(model, processor, TRIAGE_SYSTEM, [teacher_prompt(c, l) for c, l in zip(cards, levels)], 2048, batch)
    out, dropped = [], 0
    for card, level, raw in zip(cards, levels, raws):
        p = parse_triage(final_answer(raw))
        leaks_hint = bool(p) and re.search(r"for teaching|confirmed (correct )?(triage )?level", json.dumps(p), re.I)
        if not p or not p["valid_json"] or p["triage_level"] != level or cut_off(raw) or leaks_hint:
            dropped += 1
            continue
        answer = {k: p.get(k) for k in ("triage_level", "department", "reasoning", "red_flags", "next_steps",
                                         "missing_information")}
        out.append({"source": "task_triage", "messages": [
            {"role": "system", "content": TRIAGE_SYSTEM}, {"role": "user", "content": card},
            {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)}]})
    print(f"triage: kept {len(out)} of {len(cards)} (dropped {dropped})", flush=True)
    return out


def chat_examples(model, processor, pool, n, batch):
    pick = pool.sample(n, random_state=SEED + 1)
    cards = [card_with_urgency(c, p, f, ff) for c, p, f, ff in
             zip(pick["card"], pick["p_urgent"], pick["flagged"], pick["floor_flag"])]
    facts = [TRAIN_FACTS[i % len(TRAIN_FACTS)] for i in range(n)]
    chats = [[{"role": "system", "content": [{"type": "text", "text": CHAT_SYSTEM}]}] for _ in cards]
    answers, bad = [[] for _ in cards], [False] * n
    for t, turn in enumerate(TURNS):
        for i, c in enumerate(chats):
            text = turn.format(fact=facts[i][0])
            c.append({"role": "user", "content": [{"type": "text", "text": cards[i] + "\n\n" + text if t == 0 else text}]})
        for i, r in enumerate(generate_chats(model, processor, chats, 2048 if t == 3 else 1536, batch)):
            bad[i] |= cut_off(r) or not final_answer(r)
            chats[i].append({"role": "assistant", "content": [{"type": "text", "text": final_answer(r)}]})
            answers[i].append(final_answer(r))
    out = []
    for c, f, a, b in zip(chats, facts, answers, bad):
        good = (not b and re.search(f[1], a[1], re.I) and re.search(f[1], a[3], re.I) and has_sbar_headings(a[3])
                and re.search(HEDGES, a[0] + " " + a[1], re.I))
        if good:
            out.append({"source": "task_chat", "messages": [{"role": m["role"], "content": m["content"][0]["text"]}
                                                            for m in c]})
    print(f"chats: kept {len(out)} of {n}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default=None, choices=[None, "4bit", "full"])
    ap.add_argument("--triage", type=int, default=2500)
    ap.add_argument("--chats", type=int, default=800)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()

    cards = pd.read_parquet(PROCESSED / "cards_pool1.parquet").merge(
        pd.read_parquet(PROCESSED / "head_scores.parquet"), on="id")
    pool = cards[cards["split"] == "train"]                 # TRAINING patients only: no test leakage
    model, processor = load(mode=a.mode)
    rows = triage_examples(model, processor, pool, a.triage, a.batch) + chat_examples(model, processor, pool, a.chats, a.batch)
    tr, va = train_test_split(rows, test_size=0.05, random_state=SEED,
                              stratify=[r["source"] for r in rows] if len(rows) >= 40 else None)
    for name, part in (("train", tr), ("val", va)):
        with open(PROCESSED / f"task_{name}.jsonl", "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps({**r, "scope": "emergency"}, ensure_ascii=False) + "\n")
        print(name, len(part), {s: sum(r["source"] == s for r in part) for s in ("task_triage", "task_chat")})


if __name__ == "__main__":
    main()
