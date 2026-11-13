"""Reworded patient cards: the same patient, the complaint written the way a person types it (13 Nov 2026, GPU).

    python reword_cards.py --batch 64          # vanda A40: cards_pool1 + cards_children -> cards_reworded.parquet
    python reword_cards.py --limit 8 --batch 4 # quick check

Why (feedback A3/B1): in the app MedGemma writes the complaint from the nurse's note, so the same patient's card can say
"CP radiating L arm, sweaty" instead of NHAMCS's coded "Chest pain and related symptoms". The urgency score moved 10-20
points with that wording, and the pathway heads flagged stroke for some headaches. The heads were trained only on the
coded wording. Here MedGemma (4-bit, as in the app, no thinking) rewrites ONLY the complaint, twice per patient:
  r1 = a triage nurse's shorthand (abbreviations), r2 = plain words as the patient or family said it.
Everything else on the card (numbers, vitals line, "(injury)") stays the same, so the label stays true.
Saved every 1,000 cards (resumable): processed/cards_reworded.parquet, ids "<id>__r1" / "<id>__r2" plus "orig_id".
"""
import sys as _sys; from pathlib import Path as _Path  # noqa: E401,E702  (reasoning/src etc. importable)
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1] / _d) for _d in ("src", "training", "checks")]
import argparse
import re
import time

import pandas as pd
import torch

from config import PROCESSED
from model import _messages, load

STYLES = {
    # r1 (13 Nov, first try) is not used: MedGemma mostly echoed "**Reason for visit:** <same words>", and 654 copied the
    # example abbreviations ("Fever, SOB, CP, abd pain ...") onto patients without those problems. r3 replaces it, with
    # worked examples given as earlier turns of the conversation instead of a list in the instruction.
    "r1": "Rewrite this emergency department reason for visit the way a busy triage nurse types it in a note: short, "
          "with common abbreviations (e.g. SOB, CP, abd pain, N&V, LOC, ?#). Same meaning, add nothing new.",
    "r2": "Rewrite this emergency department reason for visit in plain everyday words, the way the patient or their "
          "family would say it at the desk. Same meaning, add nothing new.",
    "r3": "You turn a coded emergency department reason for visit into the few words a busy triage nurse would type. "
          "Use common abbreviations. Keep every problem listed, add nothing, no explanation. Answer with the words only.",
}
EXAMPLES = {   # (coded, nurse) pairs shown before the real one, for r3 only; none is a test patient's wording
    "r3": [("Chest pain; Shortness of breath", "CP + SOB"),
           ("Abdominal pain, cramps, spasms, NOS; Nausea; Vomiting", "abdo pain, N+V"),
           ("Injury, other and unspecified, of ankle; Fall", "fall, ankle injury"),
           ("Headache, pain in head", "headache")],
}
CHATTER = re.compile(r"^(?:okay|ok|sure|here|i can|certainly|of course|this|the reason|note)\b", re.I)
LINE = re.compile(r"^PRESENTING COMPLAINT: (.*?)( \(injury\))?$", re.M)
OUT = PROCESSED / "cards_reworded.parquet"


def complaint_of(card):
    m = LINE.search(card)
    return (m.group(1), m.group(2) or "") if m else (None, "")


def clean(text):
    """First real line of the answer: markdown, "Reason for visit:"-type labels and chatty lines removed."""
    for line in text.strip().splitlines():
        line = re.sub(r"\*\*|__|^[-*>#\s]+", "", line).strip()
        line = re.sub(r"^(?:reason for visit|rewritten|answer|nurse(?:'s)? note|patient|triage note)\s*:\s*", "", line,
                      flags=re.I).strip().strip('"').strip()
        if line and not CHATTER.match(line):
            return line[:160]
    return ""


def bad(before, after):
    """Rows not to train on: empty, unchanged, chatty, or the r1 example list copied in."""
    a = (after or "").strip()
    return (len(a) < 3 or a.lower().strip(" .") == (before or "").lower().strip(" .") or bool(CHATTER.match(a))
            or "**" in a or bool(re.search(r"reason for visit|SOB, CP|N&V, LOC|\?#", a, re.I)))


def conversation(style, system, complaint):
    msgs = _messages(system, f"Reason for visit: {EXAMPLES[style][0][0]}" if style in EXAMPLES else f"Reason for visit: {complaint}")
    if style in EXAMPLES:
        msgs = msgs[:-1]
        for coded, nurse in EXAMPLES[style]:
            msgs += [{"role": "user", "content": [{"type": "text", "text": f"Reason for visit: {coded}"}]},
                     {"role": "assistant", "content": [{"type": "text", "text": nurse}]}]
        msgs.append({"role": "user", "content": [{"type": "text", "text": f"Reason for visit: {complaint}"}]})
    return msgs


@torch.no_grad()
def generate(model, processor, prompts, max_new_tokens=40):
    """prompts: list of (style, system, complaint)."""
    tok = processor.tokenizer
    tok.padding_side = "left"
    texts = [processor.apply_chat_template(conversation(*p), tokenize=False, add_generation_prompt=True) for p in prompts]
    inputs = tok(texts, padding=True, add_special_tokens=False, return_tensors="pt").to(model.device)
    no_think = [tok.convert_tokens_to_ids("<unused94>")]
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, suppress_tokens=no_think)
    return [clean(tok.decode(o[inputs["input_ids"].shape[1]:], skip_special_tokens=True)) for o in out]


def main(batch, limit, styles):
    cards = pd.concat([pd.read_parquet(PROCESSED / "cards_pool1.parquet"),
                       pd.read_parquet(PROCESSED / "cards_children.parquet")], ignore_index=True)
    if limit:
        cards = cards.groupby("source").head(limit)
    done = pd.read_parquet(OUT) if OUT.exists() and not limit else pd.DataFrame(columns=["id"])
    have = set(done["id"])
    jobs = []
    for _, r in cards.iterrows():
        text, injury = complaint_of(r["card"])
        if not text or text == "not recorded":
            continue
        for k in styles:
            system = STYLES[k]
            if f"{r['id']}__{k}" not in have:
                jobs.append((r, k, system, text, injury))
    print(f"{len(cards)} cards, {len(have)} reworded already, {len(jobs)} to go", flush=True)
    model, processor = load(mode="4bit")
    rows, t = [], time.time()
    for start in range(0, len(jobs), batch):
        part = jobs[start:start + batch]
        new = generate(model, processor, [(k, s, txt) for _, k, s, txt, _ in part])
        for (r, k, _, txt, injury), words in zip(part, new):
            if bad(txt, words):
                continue
            card = LINE.sub(lambda m: f"PRESENTING COMPLAINT: {words}{injury}", r["card"], count=1)
            rows.append({**{c: r[c] for c in r.index if c != "card"}, "id": f"{r['id']}__{k}", "orig_id": r["id"],
                         "style": k, "complaint_before": txt, "complaint_after": words, "card": card})
        n = start + len(part)
        if len(rows) >= 1000 or n == len(jobs):
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
            done.to_parquet(OUT if not limit else PROCESSED / "cards_reworded_check.parquet", index=False)
            rows = []
            print(f"  {n}/{len(jobs)}  ({(time.time() - t) / n:.3f} s each)", flush=True)
    print(done[["style", "complaint_before", "complaint_after"]].tail(12).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="per source, quick check (saved separately)")
    ap.add_argument("--styles", default="r3", help="comma-separated, e.g. r2,r3 (r1 = first try, not used)")
    a = ap.parse_args()
    main(a.batch, a.limit, a.styles.split(","))
