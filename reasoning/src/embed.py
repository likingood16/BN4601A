"""Turn every patient card into one MedGemma vector (the input of the score heads).

    python embed.py --table cards_pool1 --name base                 # all Pool 1 cards, base MedGemma
    python embed.py --table cards_pool1 --name lora_mixed --lora mixed/latest
    python embed.py --table cards_pool1 --name test --limit 20      # quick laptop check
    python embed.py --table cards_pool1 --name base --batch 16      # HPC (A40, 48 GB): 16 cards at once

Vector = average of MedGemma's last hidden layer over all tokens of the prompt (system + card),
one number list per patient (2,560 numbers). Saved as outputs/embeddings/<table>__<name>.npz with ids.
Resumable: re-running skips patients already done (saved every 500).
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch

from config import OUTPUTS, PROCESSED
from model import _messages, load

SYSTEM = "You are an emergency department triage assistant. Read this patient record."


@torch.no_grad()
def embed_one(model, processor, card):
    inputs = processor.apply_chat_template(_messages(SYSTEM, card), add_generation_prompt=False, tokenize=True,
                                           return_dict=True, return_tensors="pt").to(model.device)
    out = model(**inputs, output_hidden_states=True, logits_to_keep=1)   # logits for 1 token only: saves memory
    h = out.hidden_states[-1][0].float()                                  # (tokens, 2560)
    return h.mean(dim=0).cpu().numpy()


@torch.no_grad()
def embed_batch(model, processor, cards):
    """Several cards at once (much faster on a big GPU). Gives the same vectors as embed_one:
    shorter cards are padded at the END, and padding is left out of the average."""
    tok = processor.tokenizer
    tok.padding_side = "right"
    texts = [processor.apply_chat_template(_messages(SYSTEM, c), tokenize=False, add_generation_prompt=False)
             for c in cards]
    inputs = tok(texts, padding=True, add_special_tokens=False, return_tensors="pt").to(model.device)
    out = model(**inputs, output_hidden_states=True, logits_to_keep=1)
    h = out.hidden_states[-1].float()                                     # (cards, tokens, 2560)
    mask = inputs["attention_mask"].unsqueeze(-1).float()                 # 1 = real token, 0 = padding
    return ((h * mask).sum(dim=1) / mask.sum(dim=1)).cpu().numpy()


def run(table, name, lora=None, limit=None, mode=None, batch=1):
    cards = pd.read_parquet(PROCESSED / f"{table}.parquet")
    id_col = "id" if "id" in cards.columns else "case_id"
    if limit:
        cards = cards.head(limit)
    path = OUTPUTS / "embeddings" / f"{table}__{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = dict(np.load(path, allow_pickle=True)) if path.exists() else {"ids": np.array([]), "X": np.zeros((0, 0))}
    ids, X = list(done["ids"]), list(done["X"])
    todo = cards[~cards[id_col].astype(str).isin(set(map(str, ids)))]
    print(f"{table}/{name}: {len(ids)} done, {len(todo)} to go", flush=True)
    if len(todo) == 0:
        return path
    model, processor = load(mode=mode, lora=lora)
    t, done_now, last_save = time.time(), 0, 0
    for start in range(0, len(todo), batch):
        part = todo.iloc[start:start + batch]
        if batch == 1:
            X.append(embed_one(model, processor, part.iloc[0]["card"]))
        else:
            X.extend(embed_batch(model, processor, list(part["card"])))
        ids.extend(part[id_col].astype(str))
        done_now += len(part)
        if done_now - last_save >= 500 or done_now == len(todo):
            np.savez(path, ids=np.array(ids), X=np.stack(X))
            last_save = done_now
            print(f"  {len(ids)}/{len(cards)}  ({(time.time() - t) / done_now:.2f} s per patient)", flush=True)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="cards_pool1")
    ap.add_argument("--name", default="base")
    ap.add_argument("--lora", default=None, help="adapter under models/lora/, e.g. mixed/latest")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--mode", default=None, choices=[None, "4bit", "full"])
    ap.add_argument("--batch", type=int, default=1, help="cards per step: 1 on the laptop, 16 on the HPC A40")
    a = ap.parse_args()
    print("saved:", run(a.table, a.name, a.lora, a.limit, a.mode, a.batch))
