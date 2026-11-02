"""Answer quality: let each model WRITE answers, then score them. Every answer is saved for reading.

    python answers.py --mode full mixed/stage12 emergency/stage3 two_stage/stage2     # HPC
    python answers.py --limit 3 mixed/stage12                                          # quick laptop check

Test A, triage decisions (scored automatically against the true triage level):
  KTAS test (all 190) + NHAMCS test (300, random, seed 42). The model reads the patient card and must answer
  in JSON: {"triage_level": 1-5, "reasoning": ..., "red_flags": [...]}.
  Scores: valid JSON, exact level, level within 1, urgent (level 1-2) sensitivity / specificity,
  under-triage (truly urgent but called 3-5).
Test B, SBAR handover reports (for reading + simple checks):
  20 MultiCaRe ED case reports. Checks: all four SBAR headings present, length.
Test C, a 4-turn consultation (Core Tasks 2 + 3 of the brief), 40 KTAS test patients:
  1 card + "main concerns?"  2 a NEW fact + "does this change your assessment?"  3 "explain to the patient simply"
  4 "write the SBAR including everything we discussed".
  Checks: follow-up and report use the new fact, 4 headings, reading ease of turn 3 vs turn 1, hedging,
  definitive-diagnosis phrases, words per turn.
Saves outputs/answers/answers.json (every answer) and outputs/metrics/answer_quality.json (scores).
"""
import argparse
import json
import re
import time

import numpy as np
import pandas as pd
import torch
from peft import PeftModel

from config import LORA_DIR, OUTPUTS, PROCESSED, SEED
from model import _messages, load
from prompts import CHAT_SYSTEM, SBAR_SYSTEM, TRIAGE_SYSTEM, TURNS, card_with_urgency

# test A: the model picks the level ALONE (kept to show why the app does not let it)
TRIAGE_UNAIDED = (
    "You are an emergency department triage assistant. Read the patient record and choose the triage level on a "
    "5-level scale: 1 = resuscitation (immediate), 2 = emergent, 3 = urgent, 4 = less urgent, 5 = non-urgent. "
    'Answer ONLY with JSON: {"triage_level": <1-5>, "reasoning": "<2-3 sentences>", "red_flags": ["..."]}')
SBAR_HEADINGS = ["situation", "background", "assessment", "recommendation"]
TRIAGE_FIELDS = ["department", "reasoning", "red_flags", "next_steps", "missing_information"]


@torch.no_grad()
def generate_chats(model, processor, chats, max_new_tokens, batch=8):
    """Greedy next answer for many conversations (lists of messages), several at a time.
    Padding goes on the LEFT so every answer starts right after its own prompt."""
    tok = processor.tokenizer
    tok.padding_side = "left"
    answers = []
    for i in range(0, len(chats), batch):
        texts = [processor.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in chats[i:i + batch]]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        answers += [a.strip() for a in tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
        if (i // batch) % 10 == 9 or i + batch >= len(chats):            # live progress for the HPC log
            print(f"    {time.strftime('%H:%M')} answered {len(answers)}/{len(chats)}", flush=True)
    return answers


# MedGemma 1.5 may "think" first: <unused94>thought ... <unused95> then the real answer
THINK_START, THINK_END = "<unused94>", "<unused95>"


def final_answer(raw):
    """The part a user would see: after the thinking block. Empty if the model never finished thinking."""
    if THINK_END in raw:
        return raw.split(THINK_END)[-1].strip()
    return "" if THINK_START in raw else raw.strip()


def cut_off(raw):
    return THINK_START in raw and THINK_END not in raw


def has_sbar_headings(text):
    plain = re.sub(r"[*#_]", "", text.lower())          # '**S**ituation' -> 'situation'
    return all(h in plain for h in SBAR_HEADINGS)


def generate(model, processor, system, prompts, max_new_tokens, batch=8):
    """One-turn version: the same system prompt + one user message each."""
    return generate_chats(model, processor, [_messages(system, p) for p in prompts], max_new_tokens, batch)


# ---------- test C: a 4-turn consultation per patient (Core Tasks 2 + 3) ----------
# a new fact for turn 2 (rotates over patients) + the words that show the model used it
FACTS = [("The family has just told us the patient takes warfarin for atrial fibrillation.", r"warfarin|anticoag"),
         ("The patient now tells us about a severe penicillin allergy.", r"penicillin"),
         ("The patient had a kidney transplant last year and takes immunosuppressants.", r"transplant|immunosupp"),
         ("The patient has type 1 diabetes and uses insulin.", r"insulin|diabet")]
HEDGES = r"\b(may|might|could|possibl\w*|likely|unlikely|suggest\w*|consider|cannot rule out|rule out|differential)\b"
DEFINITIVE = r"\b(the diagnosis is|is diagnosed with|definitely|certainly has|confirms? (that|a|the))\b"


def reading_ease(text):
    """Flesch reading ease: higher = easier (60-70 plain English, < 30 very hard)."""
    words = re.findall(r"[A-Za-z]+", text)
    if not words:
        return None
    sentences = max(1, len(re.findall(r"[.!?]+", text)))
    syllables = sum(max(1, len(re.findall(r"[aeiouy]+", w.lower())) - w.lower().endswith("e")) for w in words)
    return 206.835 - 1.015 * len(words) / sentences - 84.6 * syllables / len(words)


def run_chats(model, processor, cards, batch):
    """Returns facts, final answers per turn, raw answers per turn (with any thinking)."""
    chats = [[{"role": "system", "content": [{"type": "text", "text": CHAT_SYSTEM}]}] for _ in cards]
    facts = [FACTS[i % len(FACTS)] for i in range(len(cards))]
    answers, raws = [[] for _ in cards], [[] for _ in cards]
    for t, turn in enumerate(TURNS):
        for i, c in enumerate(chats):
            text = turn.format(fact=facts[i][0])
            if t == 0:
                text = cards[i] + "\n\n" + text
            c.append({"role": "user", "content": [{"type": "text", "text": text}]})
        replies = generate_chats(model, processor, chats, 2048 if t == 3 else 1536, batch)
        for i, r in enumerate(replies):
            # only the visible answer goes back into the conversation, as in the app
            chats[i].append({"role": "assistant", "content": [{"type": "text", "text": final_answer(r)}]})
            answers[i].append(final_answer(r))
            raws[i].append(r)
    return facts, answers, raws


def score_chats(facts, answers, raws):
    uses = lambda turn: np.mean([bool(re.search(f[1], a[turn], re.I)) for f, a in zip(facts, answers)])
    ease = lambda turn: np.nanmean([reading_ease(a[turn]) or np.nan for a in answers])
    clinical = [a[t] for a in answers for t in (0, 1)]
    return {"n": len(answers),
            "cut_off_while_thinking": round(float(np.mean([cut_off(r) for rs in raws for r in rs])), 4),
            "followup_uses_new_fact": round(float(uses(1)), 4),
            "report_uses_new_fact": round(float(uses(3)), 4),
            "report_all_4_headings": round(float(np.mean([has_sbar_headings(a[3]) for a in answers])), 4),
            "reading_ease_clinical_turn1": round(float(ease(0)), 1),
            "reading_ease_patient_turn3": round(float(ease(2)), 1),
            "hedging_rate": round(float(np.mean([bool(re.search(HEDGES, x, re.I)) for x in clinical])), 4),
            "definitive_phrase_rate": round(float(np.mean([bool(re.search(DEFINITIVE, x, re.I)) for a in answers for x in a])), 4),
            "mean_words_per_turn": [round(float(np.mean([len(a[t].split()) for a in answers])), 1) for t in range(len(TURNS))]}


def parse_triage(text):
    """JSON answer -> dict, or None. Tolerates ```json fences and text around the JSON."""
    start = text.find("{")
    if start != -1:
        body = text[start:text.rfind("}") + 1] if "}" in text[start:] else text[start:]
        # strict JSON first; then one repair (a missing closing brace), counted as "repaired", not valid
        for candidate, valid in ((body, True), (body.rstrip().rstrip(",") + "}", False)):
            try:
                d = json.loads(candidate)
                level = int(d.get("triage_level"))
                if 1 <= level <= 5:
                    return {**d, "triage_level": level, "valid_json": valid, "repaired": not valid}
            except (ValueError, TypeError, AttributeError):
                pass
    m = re.search(r"triage[_ ]level\D{0,5}([1-5])", text, re.I)      # broken JSON but a readable level
    return {"triage_level": int(m.group(1)), "valid_json": False} if m else None


def score_triage(cases, parsed):
    true = cases["triage_level"].to_numpy()
    got = np.array([p["triage_level"] if p else np.nan for p in parsed], dtype=float)
    ok = ~np.isnan(got)
    urgent_true, urgent_got = true <= 2, got <= 2
    tp, fn = (urgent_true & urgent_got & ok).sum(), (urgent_true & ~(urgent_got & ok)).sum()
    tn, fp = (~urgent_true & ~urgent_got & ok).sum(), (~urgent_true & urgent_got & ok).sum()
    div = lambda a, b: round(float(a / b), 4) if b else None
    return {"n": int(len(true)), "answered": div(ok.sum(), len(true)),
            "valid_json": div(sum(bool(p and p["valid_json"]) for p in parsed), len(true)),
            "exact_level": div((got[ok] == true[ok]).sum(), len(true)),
            "within_1_level": div((np.abs(got[ok] - true[ok]) <= 1).sum(), len(true)),
            "urgent_sensitivity": div(tp, tp + fn), "urgent_specificity": div(tn, tn + fp),
            "under_triage_of_urgent": div(fn, tp + fn),
            "mean_level_error": round(float(np.mean(got[ok] - true[ok])), 3) if ok.any() else None}


def score_guided(cases, parsed):
    """Test A2 (the app's design): level given the urgency decision. Extra checks: never lowers a flagged patient
    (flagged -> 1-2), how often it raises a non-flagged patient, and are all JSON fields filled?"""
    s = score_triage(cases, parsed)
    flagged = cases["flagged"].to_numpy()
    urgent_out = np.array([bool(p) and p["triage_level"] <= 2 for p in parsed])
    s["never_lowers_flagged"] = round(float(urgent_out[flagged].mean()), 4) if flagged.any() else None
    s["raises_not_flagged"] = round(float(urgent_out[~flagged].mean()), 4) if (~flagged).any() else None
    s["all_fields_filled"] = round(float(np.mean([bool(p) and all(p.get(k) not in (None, "", []) for k in TRIAGE_FIELDS)
                                                  for p in parsed])), 4)
    return s


def score_sbar(raws):
    texts = [final_answer(r) for r in raws]
    return {"n": len(texts), "cut_off_while_thinking": round(float(np.mean([cut_off(r) for r in raws])), 4),
            "all_4_headings": round(float(np.mean([has_sbar_headings(t) for t in texts])), 4),
            "mean_words": round(float(np.mean([len(t.split()) for t in texts])), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("adapters", nargs="*", help="adapters under models/lora/, e.g. mixed/stage12")
    ap.add_argument("--mode", default=None, choices=[None, "4bit", "full"])
    ap.add_argument("--nhamcs", type=int, default=300)
    ap.add_argument("--sbar", type=int, default=20)
    ap.add_argument("--chat", type=int, default=40, help="KTAS test patients for the 4-turn consultation test")
    ap.add_argument("--limit", type=int, default=None, help="only the first N cases of each test (quick check)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--tag", default="", help="name for this run's files, e.g. 'stages' -> answers_stages.json")
    ap.add_argument("--skip-base", action="store_true", help="only the adapters (base already tested with this code)")
    a = ap.parse_args()

    cards = pd.read_parquet(PROCESSED / "cards_pool1.parquet").merge(
        pd.read_parquet(PROCESSED / "head_scores.parquet"), on="id")          # made by head_scores.py
    test = cards[cards["split"] == "test"]
    cases = pd.concat([test[test["source"] == "ktas"],
                       test[test["source"] == "nhamcs"].sample(a.nhamcs, random_state=SEED)]).reset_index(drop=True)
    sbar_cases = pd.read_parquet(PROCESSED / "cards_multicare.parquet").sample(a.sbar, random_state=SEED)
    chat_cases = test[test["source"] == "ktas"].sample(a.chat, random_state=SEED)
    if a.limit:
        cases = cases.groupby("source").head(a.limit).reset_index(drop=True)
        sbar_cases, chat_cases = sbar_cases.head(a.limit), chat_cases.head(a.limit)

    base, processor = load(mode=a.mode)
    model, saved, scores = base, {"triage_cases": cases[["id", "source", "triage_level"]].to_dict("records"),
                                  "sbar_cases": sbar_cases[["case_id"]].to_dict("records")}, {}
    # baseline: the urgency decision alone (score + floor), no language model, on the same patients
    as_levels = [{"triage_level": 2 if f else 3, "valid_json": True} for f in cases["flagged"]]
    scores["score_and_floor_only"] = score_triage(cases, as_levels)
    for name in ([] if a.skip_base else ["base"]) + a.adapters:
        t = time.time()
        if name != "base":
            path = str(LORA_DIR / name / "adapter")
            if not isinstance(model, PeftModel):
                model = PeftModel.from_pretrained(base, path, adapter_name=name)
            else:
                model.load_adapter(path, adapter_name=name)
            model.set_adapter(name)
        # big token budgets: plain MedGemma 1.5 often thinks for several hundred tokens before answering
        raw = generate(model, processor, TRIAGE_UNAIDED, list(cases["card"]), 2048, a.batch)
        parsed = [parse_triage(final_answer(r)) for r in raw]
        guided_cards = [card_with_urgency(c, p, f, ff) for c, p, f, ff in
                        zip(cases["card"], cases["p_urgent"], cases["flagged"], cases["floor_flag"])]
        raw_g = generate(model, processor, TRIAGE_SYSTEM, guided_cards, 2048, a.batch)
        parsed_g = [parse_triage(final_answer(r)) for r in raw_g]
        sbar = generate(model, processor, SBAR_SYSTEM, list(sbar_cases["card"]), 2048, a.batch)
        scores[name] = {src: score_triage(cases[m], [parsed[i] for i in np.flatnonzero(m)])
                        for src, m in (("ktas", cases["source"] == "ktas"), ("nhamcs", cases["source"] == "nhamcs"),
                                       ("all", np.ones(len(cases), bool)))}
        scores[name]["triage_cut_off_while_thinking"] = round(float(np.mean([cut_off(r) for r in raw])), 4)
        scores[name]["guided"] = score_guided(cases, parsed_g)
        scores[name]["sbar"] = score_sbar(sbar)
        chat_cards = [card_with_urgency(c, p, f, ff) for c, p, f, ff in zip(
            chat_cases["card"], chat_cases["p_urgent"], chat_cases["flagged"], chat_cases["floor_flag"])]   # as in the app
        facts, chats, chat_raws = run_chats(model, processor, chat_cards, a.batch)
        scores[name]["chat"] = score_chats(facts, chats, chat_raws)
        saved[name] = {"triage": [{"id": i, "answer": final_answer(r), "raw": r, "parsed": p}
                                  for i, r, p in zip(cases["id"], raw, parsed)],
                       "guided_triage": [{"id": i, "answer": final_answer(r), "parsed": p}
                                         for i, r, p in zip(cases["id"], raw_g, parsed_g)],
                       "sbar": [{"case_id": c, "report": final_answer(s), "raw": s}
                                for c, s in zip(sbar_cases["case_id"], sbar)],
                       "chat": [{"id": i, "new_fact": f[0],
                                 "turns": [{"user": u.format(fact=f[0]), "assistant": x, "raw": r}
                                           for u, x, r in zip(TURNS, ans, rs)]}
                                for i, f, ans, rs in zip(chat_cases["id"], facts, chats, chat_raws)]}
        print(f"{name} ({time.time() - t:.0f}s): {json.dumps(scores[name]['all'])}\n  guided {scores[name]['guided']}"
              f"\n  SBAR {scores[name]['sbar']}"
              f"\n  chat {scores[name]['chat']}", flush=True)

    (OUTPUTS / "answers").mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "metrics").mkdir(parents=True, exist_ok=True)
    tag = f"_{a.tag}" if a.tag else ""
    answers_path, scores_path = OUTPUTS / "answers" / f"answers{tag}.json", OUTPUTS / "metrics" / f"answer_quality{tag}.json"
    answers_path.write_text(json.dumps(saved, indent=1, ensure_ascii=False), encoding="utf-8")
    scores_path.write_text(json.dumps(scores, indent=1))
    print("saved", answers_path, "and", scores_path)


if __name__ == "__main__":
    main()
