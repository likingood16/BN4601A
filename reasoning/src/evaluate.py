"""Compare base MedGemma with LoRA checkpoints on held-out data (works on laptop 4-bit or full GPU).

    python evaluate.py mixed/stage1_8448 mixed/stage1_22448          # base + these adapters
    python evaluate.py --per-source 30 --medqa mixed/latest

Measures, for every model, on the SAME examples:
  * validation loss (Pool 2 val, assistant tokens only, per source and overall) - lower = better
  * MedQA accuracy on the 200 held-out questions, with --medqa: medqa_acc_direct = letter as the first answer token,
    medqa_acc_suffix = after 'The best option is'; medqa_acc = the better of the two (fair to base AND LoRA:
    base explains first, LoRA was trained to answer with the letter straight away)
Saves outputs/metrics/lora_eval.json (merged with earlier results).
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel

from config import LORA_DIR, OUTPUTS, SEED
from data import load_bench, load_pool2
from model import load

MODEL_TURN, END = "<start_of_turn>model\n", "<end_of_turn>"
LETTERS = ["A", "B", "C", "D"]


def encode(processor, messages, max_len):
    tok = processor.tokenizer
    msgs = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in messages]
    text = processor.apply_chat_template(msgs, tokenize=False)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True, truncation=True, max_length=max_len)
    spans, start = [], 0                               # character spans of the model's answers
    while (i := text.find(MODEL_TURN, start)) != -1:
        a = i + len(MODEL_TURN)
        b = text.find(END, a)
        b = len(text) if b == -1 else b + len(END)
        spans.append((a, b))
        start = b
    labels = [t if any(a <= s < b for a, b in spans) else -100 for t, (s, _) in zip(enc["input_ids"], enc["offset_mapping"])]
    return enc["input_ids"], labels


@torch.no_grad()
def val_loss(model, data):
    """data: list of (source, scope, ids, labels).
    Returns ({source: loss, 'scope:emergency'/'scope:general': loss, 'all': loss}, per-example [loss sum, tokens])."""
    tot, n, per_example = {}, {}, []
    for src, scope, ids, labels in data:
        lab = torch.tensor(labels[1:], device=model.device)
        keep = (lab != -100).nonzero().squeeze(1)
        if len(keep) == 0:
            per_example.append((0.0, 0))
            continue
        logits = model(input_ids=torch.tensor([ids], device=model.device), logits_to_keep=keep).logits[0].float()
        loss = F.cross_entropy(logits, lab[keep], reduction="sum").item()
        per_example.append((loss, len(keep)))
        for k in (src, f"scope:{scope}", "all"):         # scope = emergency / general: what the 3 arms differ on
            tot[k], n[k] = tot.get(k, 0) + loss, n.get(k, 0) + len(keep)
    return {k: round(tot[k] / n[k], 4) for k in tot}, np.array(per_example)


def paired_ci(groups, per_new, per_old, n=2000):
    """Resample the SAME examples for both models: 95% CI of (new loss - old loss), overall and per group."""
    rng, out = np.random.default_rng(SEED), {}
    for g in ["all", *sorted(set(groups))]:
        idx = np.arange(len(groups)) if g == "all" else np.flatnonzero(np.array(groups) == g)
        diffs = []
        for _ in range(n):
            i = rng.choice(idx, len(idx))
            diffs.append(per_new[i, 0].sum() / per_new[i, 1].sum() - per_old[i, 0].sum() / per_old[i, 1].sum())
        point = per_new[idx, 0].sum() / per_new[idx, 1].sum() - per_old[idx, 0].sum() / per_old[idx, 1].sum()
        out[g] = {"diff": round(float(point), 4), "ci95": [round(float(np.percentile(diffs, q)), 4) for q in (2.5, 97.5)]}
    return out


def paired_acc_ci(right_new, right_old, n=2000):
    """MedQA: 95% CI of the accuracy difference on the same questions."""
    rng, d = np.random.default_rng(SEED), []
    for _ in range(n):
        i = rng.integers(0, len(right_new), len(right_new))
        d.append(right_new[i].mean() - right_old[i].mean())
    return {"diff": round(float(right_new.mean() - right_old.mean()), 4),
            "ci95": [round(float(np.percentile(d, q)), 4) for q in (2.5, 97.5)]}


@torch.no_grad()
def medqa(model, processor, bench):
    """Two letter-probability scores on the same questions:
    direct = P('A'..'D') as the FIRST answer token (the format the question asks for and LoRA was trained on)
    suffix = P(' A'..' D') after appending 'The best option is' (older method; penalises trained adapters)"""
    tok = processor.tokenizer
    ids_direct = [tok.encode(l, add_special_tokens=False)[0] for l in LETTERS]
    ids_suffix = [tok.encode(" " + l, add_special_tokens=False)[-1] for l in LETTERS]
    right = {"direct": [], "suffix": []}                  # per question: 1 = right, 0 = wrong
    for _, q in bench.iterrows():
        msgs = [{"role": "user", "content": [{"type": "text", "text": q["prompt"]}]}]
        prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for name, text, ids in [("direct", prompt, ids_direct), ("suffix", prompt + "The best option is", ids_suffix)]:
            enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
            logits = model(**enc, logits_to_keep=1).logits[0, -1]
            right[name].append(int(LETTERS[int(torch.argmax(logits[ids]))] == q["answer"]))
    return {k: np.array(v) for k, v in right.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("adapters", nargs="*", help="adapters under models/lora/, e.g. mixed/stage1_22448")
    ap.add_argument("--per-source", type=int, default=30)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--medqa", action="store_true")
    ap.add_argument("--mode", default=None, choices=[None, "4bit", "full"])
    ap.add_argument("--split", default="val", choices=["val", "test"], help="test = the untouched Pool 2 test split")
    a = ap.parse_args()

    base, processor = load(mode=a.mode)
    rows = load_pool2(a.split).groupby("source").sample(a.per_source, random_state=SEED)
    data = [(r["source"], r["scope"], *encode(processor, r["messages"], a.max_len)) for _, r in rows.iterrows()]
    groups, scopes = list(rows["source"]), [f"scope:{s}" for s in rows["scope"]]
    bench = load_bench() if a.medqa else None

    out_path = OUTPUTS / "metrics" / ("lora_eval.json" if a.split == "val" else f"lora_eval_{a.split}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    model, per, right = base, {}, {}
    for name in ["base", *a.adapters]:
        t = time.time()
        if name != "base":
            path = str(LORA_DIR / name / "adapter")
            if not isinstance(model, PeftModel):
                model = PeftModel.from_pretrained(base, path, adapter_name=name)
            else:
                model.load_adapter(path, adapter_name=name)
            model.set_adapter(name)
        losses, per[name] = val_loss(model, data)
        r = {"val_loss": losses, "per_source": a.per_source, "split": a.split}
        if name != "base":     # how sure is the gain? same examples, resampled 2,000 times
            r["vs_base"] = paired_ci(groups, per[name], per["base"]) | paired_ci(scopes, per[name], per["base"])
        if bench is not None:
            m = medqa(model, processor, bench)
            # base answers best after 'The best option is', LoRA best as the first token: score each in its better format
            best = max(m, key=lambda k: m[k].mean())
            right[name] = m[best]
            r["medqa_acc"], r["medqa_format"] = float(m[best].mean()), best
            r["medqa_acc_direct"], r["medqa_acc_suffix"] = float(m["direct"].mean()), float(m["suffix"].mean())
            if name != "base":
                r["medqa_vs_base"] = paired_acc_ci(right[name], right["base"])
        results[name] = r
        print(f"{name}: {r}  ({time.time() - t:.0f}s)", flush=True)
    out_path.write_text(json.dumps(results, indent=1))
    print("saved", out_path)


if __name__ == "__main__":
    main()
