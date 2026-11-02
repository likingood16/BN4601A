"""Load the tables notebook 02 made. Building them stays in notebook 02 (seed 42, reproducible).

    from data import load_pool1, load_pool2, load_bench, load_multicare
"""
import pandas as pd

from config import PROCESSED


def load_pool1(split=None, source=None):
    """Pool 1 (score heads): KTAS + NHAMCS patients, one row each, triage-desk information only."""
    df = pd.read_parquet(PROCESSED / "pool1.parquet")
    if split:
        df = df[df["split"] == split]
    if source:
        df = df[df["source"] == source]
    return df.reset_index(drop=True)


def load_pool2(split, scope=None):
    """Pool 2 (LoRA): chat examples {id, source, scope, messages}. split = train / val / test."""
    df = pd.read_json(PROCESSED / f"pool2_{split}.jsonl", lines=True)
    if scope:
        df = df[df["scope"] == scope]
    return df.reset_index(drop=True)


def load_bench():
    """MedQA benchmark: 200 held-out exam questions {id, prompt, answer (A-D), answer_text}."""
    return pd.read_json(PROCESSED / "medqa_bench200.jsonl", lines=True)


def load_multicare():
    """MultiCaRe adult ED case reports with vitals in the text: full-pipeline test + demo cases."""
    return pd.read_parquet(PROCESSED / "multicare_ed_test.parquet")
