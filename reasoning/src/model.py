"""Load MedGemma (+ optional LoRA) and ask it questions. Same code on laptop, Colab and HPC.

    from model import load, ask
    model, processor = load()                         # picks 4-bit on small GPUs, full size on big ones
    model, processor = load(lora="mixed/latest")      # with a trained LoRA adapter from models/lora/
    print(ask(model, processor, "You are an ED triage assistant.", card_text))
"""
import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM, BitsAndBytesConfig

from config import BASE_MODEL, LORA_DIR


def pick_mode():
    """'4bit' if the GPU has < 12 GB (laptop RTX 3050), else 'full'."""
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU found")
    gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    return "4bit" if gb < 12 else "full"


def compute_dtype():
    """bfloat16 where the GPU supports it natively (RTX 30xx, A100, H100), else float16 (T4, V100)."""
    major, _ = torch.cuda.get_device_capability(0)
    return torch.bfloat16 if major >= 8 else torch.float16


def load(mode=None, lora=None, model_id=BASE_MODEL):
    """Return (model, processor). lora: path under models/lora/ (e.g. 'mixed/latest') or None."""
    mode = mode or pick_mode()
    processor = AutoProcessor.from_pretrained(model_id)
    if mode == "4bit":
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=compute_dtype())
        # everything on GPU 0: device_map="auto" wrongly offloads part of it to the CPU on a 4 GB GPU
        model = AutoModelForMultimodalLM.from_pretrained(model_id, quantization_config=bnb, device_map={"": 0})
    else:
        # Gemma 3 overflows in float16, so full-size runs use bfloat16 if available, otherwise float32
        dtype = torch.bfloat16 if compute_dtype() == torch.bfloat16 else torch.float32
        model = AutoModelForMultimodalLM.from_pretrained(model_id, dtype=dtype, device_map={"": 0})
    if lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(LORA_DIR / lora / "adapter"))
    model.eval()
    return model, processor


def _messages(system, user):
    msgs = [{"role": "system", "content": [{"type": "text", "text": system}]}] if system else []
    return msgs + [{"role": "user", "content": [{"type": "text", "text": user}]}]


@torch.no_grad()
def ask_messages(model, processor, messages, max_new_tokens=512):
    """Next answer for a whole conversation (may include images), deterministic, prompt cut off."""
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                           return_dict=True, return_tensors="pt")
    inputs = inputs.to(model.device, dtype=compute_dtype())          # dtype only touches the image numbers
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)


def ask(model, processor, system, user, max_new_tokens=512):
    """One answer to one question."""
    return ask_messages(model, processor, _messages(system, user), max_new_tokens)
