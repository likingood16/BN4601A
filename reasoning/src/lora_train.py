"""Staged LoRA training of MedGemma on Pool 2 (same method as notebook 05, as a script for HPC/Colab/Kaggle).

    python lora_train.py --arm mixed --stages 99 --max-hours 46          # NUS HPC: everything in one job (48 h limit)
    python lora_train.py --arm emergency --stages 3
    python lora_train.py --arm two_stage --stages 2 --two-stage-size 30000

Stages: the training examples are shuffled once (seed 42) and cut into chunks of --chunk-size; each stage
trains on one chunk, then measures validation loss + MedQA and saves. Everything needed to continue is in
--ckpt-dir (adapter/ + progress.json), so a stopped run continues where it left off.
Arms: mixed = all training examples | emergency = emergency-tagged only |
      two_stage = general first, then emergency (with --two-stage-size: N general then N emergency) |
      task = the app's own tasks (make_task_data.py) + --replay old Pool 2 examples so earlier skills are not
             forgotten, --epochs passes, started from another adapter's weights with --init-from:
    python lora_train.py --arm task --init-from ../models/lora/mixed/stage12 --epochs 2 --max-seq-len 2048
"""
import argparse
import gc
import json
import os
import re
import shutil
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--arm", default="mixed", choices=["mixed", "emergency", "two_stage", "task"])
ap.add_argument("--epochs", type=int, default=1, help="task arm: passes over the (small) task data")
ap.add_argument("--replay", type=int, default=2000, help="task arm: old Pool 2 examples mixed in")
ap.add_argument("--chunk-size", type=int, default=30_000)
ap.add_argument("--stages", type=int, default=2, help="max stages to run in this job")
ap.add_argument("--max-hours", type=float, default=None, help="save + stop cleanly after this long")
ap.add_argument("--two-stage-size", type=int, default=None)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--batch", type=int, default=2)
ap.add_argument("--grad-acc", type=int, default=8)
ap.add_argument("--max-seq-len", type=int, default=1024)
ap.add_argument("--save-every-min", type=float, default=30)
ap.add_argument("--val-examples", type=int, default=600)
ap.add_argument("--ckpt-dir", default=None, help="default: models/lora/<arm>/run")
ap.add_argument("--init-from", default=None, help="start from an existing adapter folder (e.g. the Colab one)")
args = ap.parse_args()

START = time.time()
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import LORA_BASE_MODEL, LORA_DIR, PROCESSED, SEED   # noqa: E402
from data import load_bench, load_pool2                      # noqa: E402

from unsloth import FastModel, is_bf16_supported              # noqa: E402  (import before transformers)
from unsloth.chat_templates import get_chat_template, train_on_responses_only  # noqa: E402
import pandas as pd                                          # noqa: E402
import torch                                                 # noqa: E402
from datasets import Dataset                                 # noqa: E402
from transformers import TrainerCallback                     # noqa: E402
from trl import SFTConfig, SFTTrainer                        # noqa: E402

CKPT = Path(args.ckpt_dir or LORA_DIR / args.arm / "run")
CKPT.mkdir(parents=True, exist_ok=True)
log = lambda *a: print(time.strftime("%H:%M"), *a, flush=True)

# ---------- progress (continue if a checkpoint exists) ----------
progress = {"arm": args.arm, "chunk": 0, "rows_done": 0, "history": [], "base_medqa_acc": None, "lr": args.lr}
if (CKPT / "progress.json").exists():
    progress = json.loads((CKPT / "progress.json").read_text())
elif args.init_from:                                            # e.g. continue the Colab run
    shutil.copytree(Path(args.init_from) / "adapter", CKPT / "adapter", dirs_exist_ok=True)
    src = Path(args.init_from) / "progress.json"
    if src.exists() and args.arm != "task":                     # the task arm starts its own stage count
        progress = json.loads(src.read_text())
progress["lr"] = args.lr
log("progress:", {k: v for k, v in progress.items() if k != "history"})

# ---------- stage plan ----------
train = load_pool2("train")
if args.arm == "emergency":
    train = train[train["scope"] == "emergency"]
order = train.sample(frac=1, random_state=SEED)
if args.arm == "task":
    task = pd.read_json(PROCESSED / "task_train.jsonl", lines=True)
    train = pd.concat([task, train.sample(args.replay, random_state=SEED)])
    order = pd.concat([train.sample(frac=1, random_state=SEED + e) for e in range(args.epochs)])
if args.arm == "two_stage":
    gen, emer = order[order["scope"] == "general"], order[order["scope"] == "emergency"]
    if args.two_stage_size:
        gen, emer = gen.head(args.two_stage_size), emer.head(args.two_stage_size)
    order = pd.concat([gen, emer])
starts = list(range(0, len(order), args.chunk_size))
chunk_sizes = [min(args.chunk_size, len(order) - s) for s in starts]
needed = set(range(progress["chunk"], min(progress["chunk"] + args.stages, len(starts))))
chunks = {i: order.iloc[s:s + args.chunk_size] for i, s in enumerate(starts) if i in needed}
val = load_pool2("val").groupby("source").sample(args.val_examples // 4, random_state=SEED)
if args.arm == "task":                                          # task val examples + a few old ones
    val = pd.concat([pd.read_json(PROCESSED / "task_val.jsonl", lines=True),
                     val.groupby("source").head(25)])
bench = load_bench()
del train, order
gc.collect()
log(f"{args.arm}: {len(starts)} stages; this job: {sorted(needed)}")

# ---------- model ----------
model, tokenizer = FastModel.from_pretrained(LORA_BASE_MODEL, max_seq_length=args.max_seq_len, load_in_4bit=True)
model = FastModel.get_peft_model(model, finetune_vision_layers=False, finetune_language_layers=True,
                                 finetune_attention_modules=True, finetune_mlp_modules=True,
                                 r=16, lora_alpha=16, lora_dropout=0, bias="none", random_state=SEED)
tokenizer = get_chat_template(tokenizer, "gemma-3")
if (CKPT / "adapter" / "adapter_model.safetensors").exists():
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    set_peft_model_state_dict(model, load_file(str(CKPT / "adapter" / "adapter_model.safetensors")))
    log("continued from checkpoint")

tok = getattr(tokenizer, "tokenizer", tokenizer)
LETTERS = ["A", "B", "C", "D"]
LETTER_IDS = [tok.encode(" " + l, add_special_tokens=False)[-1] for l in LETTERS]   # ' A' after 'The best option is'
DIRECT_IDS = [tok.encode(l, add_special_tokens=False)[0] for l in LETTERS]          # 'A' as the first answer token


def to_text(messages):
    turns = []
    for m in messages:      # Gemma needs user/assistant to alternate: join two turns in a row from the same role
        if turns and turns[-1]["role"] == m["role"]:
            turns[-1]["content"] += "\n\n" + m["content"]
        else:
            turns.append({"role": m["role"], "content": m["content"]})
    return tokenizer.apply_chat_template(turns, tokenize=False).removeprefix("<bos>")


def as_dataset(df):
    return Dataset.from_dict({"text": [to_text(m) for m in df["messages"]]})


def medqa_accuracy():
    """Letter probability, two ways: (suffix) end the prompt with 'The best option is' and compare P(' A'..' D');
    (direct) compare P('A'..'D') as the first answer token, the format the MedQA training answers use."""
    FastModel.for_inference(model)
    right = {"suffix": 0, "direct": 0}
    for _, q in bench.iterrows():
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": q["prompt"]}], tokenize=False,
                                               add_generation_prompt=True)
        for name, text, ids in [("suffix", prompt + "The best option is", LETTER_IDS), ("direct", prompt, DIRECT_IDS)]:
            enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
            with torch.no_grad():
                logits = model(**enc).logits[0, -1]
            right[name] += LETTERS[int(torch.argmax(logits[ids]))] == q["answer"]
    FastModel.for_training(model)
    return {k: v / len(bench) for k, v in right.items()}


def save(note):
    tmp = CKPT / "adapter_tmp"
    model.save_pretrained(str(tmp))
    shutil.rmtree(CKPT / "adapter", ignore_errors=True)
    os.replace(tmp, CKPT / "adapter")
    progress["last_note"] = note
    (CKPT / "progress_tmp.json").write_text(json.dumps(progress, indent=1))
    os.replace(CKPT / "progress_tmp.json", CKPT / "progress.json")
    log("saved:", note)


over_time = lambda: args.max_hours is not None and time.time() - START > args.max_hours * 3600


class Saver(TrainerCallback):
    def __init__(self, rows_at_start):
        self.rows_at_start, self.last, self.stopped = rows_at_start, time.time(), False

    def on_step_end(self, a, state, control, **kw):
        if time.time() - self.last > args.save_every_min * 60 or over_time():
            progress["rows_done"] = self.rows_at_start + state.global_step * args.batch * args.grad_acc
            save(f"stage {progress['chunk'] + 1}: {progress['rows_done']:,} examples")
            self.last = time.time()
            if over_time():
                self.stopped, control.should_training_stop = True, True


if progress.get("medqa_method") != "letter_prob" or "base_medqa_direct" not in progress:
    with model.disable_adapter():
        m = medqa_accuracy()
    progress["base_medqa_acc"], progress["base_medqa_direct"] = m["suffix"], m["direct"]
    progress["medqa_method"] = "letter_prob"
    log("base MedQA (suffix, direct):", m["suffix"], m["direct"])

val_ds = as_dataset(val)
for _ in range(args.stages):
    if progress["chunk"] >= len(starts):
        log("all stages done")
        break
    stage = progress["chunk"] + 1
    chunk = chunks[progress["chunk"]].iloc[progress["rows_done"]:]
    log(f"=== stage {stage}/{len(starts)}: {len(chunk):,} examples ===")
    t0 = time.time()
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=as_dataset(chunk), eval_dataset=val_ds,
        args=SFTConfig(dataset_text_field="text", max_length=args.max_seq_len, num_train_epochs=1,
                       per_device_train_batch_size=args.batch, gradient_accumulation_steps=args.grad_acc,
                       per_device_eval_batch_size=4, learning_rate=args.lr,
                       lr_scheduler_type="constant_with_warmup", warmup_steps=20, optim="adamw_8bit",
                       weight_decay=0.01, seed=SEED, fp16=not is_bf16_supported(), bf16=is_bf16_supported(),
                       logging_steps=25, save_strategy="no", report_to="none",
                       output_dir=str(CKPT / "trainer_tmp")))
    trainer = train_on_responses_only(trainer, instruction_part="<start_of_turn>user\n",
                                      response_part="<start_of_turn>model\n")
    saver = Saver(progress["rows_done"])
    trainer.add_callback(saver)
    trainer.train()
    if saver.stopped:
        log("time limit: saved mid-stage, stopping (re-submit to continue)")
        break
    val_loss = trainer.evaluate()["eval_loss"]
    m = medqa_accuracy()
    progress["history"].append({"stage": stage, "examples_seen": sum(chunk_sizes[:stage]), "val_loss": round(val_loss, 4),
                                "medqa_acc": m["suffix"], "medqa_direct": m["direct"],
                                "minutes": round((time.time() - t0) / 60), "lr": args.lr})
    progress["chunk"], progress["rows_done"] = progress["chunk"] + 1, 0
    save(f"stage {stage} done: val_loss {val_loss:.3f}, MedQA {m['suffix']:.1%} (direct {m['direct']:.1%})")
    # keep a copy of every finished stage, so the learning curve can be re-checked later
    shutil.copytree(CKPT / "adapter", CKPT.parent / f"stage{stage}" / "adapter", dirs_exist_ok=True)
    shutil.copy(CKPT / "progress.json", CKPT.parent / f"stage{stage}" / "progress.json")
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
log("history:", json.dumps(progress["history"], indent=1))
