"""The whole Model 3 flow for one patient, in order. The app calls only this file.

    from pipeline import EDPipeline
    ed = EDPipeline()
    visit = ed.triage({"age": 67, "sex": "male", "chief_complaint": "chest pain", "hr": 110, "sbp": 95, ...})
    print(visit.decision, visit.explanation)               # stage 1: the front door
    visit.add("bloods", "Troponin 2.1 ng/mL (high)")       # stage 2: results typed in during the visit
    ed.read_image(visit, "xray.jpg", "bone_xray")          # stage 2: any image
    print(ed.chat(visit, "What are your main concerns?"))
    print(ed.sbar(visit))

Patient fields = Pool 1 column names; anything unknown can be left out. Stage 1 needs the 4-bit model, because the
head and level model were trained on vectors from it (`base4bit`).

One MedGemma 1.5 sits in memory, with adapters switched per step (6 Nov 2026):
  vector for the score  -> no adapter (the heads were trained on the plain model's vectors)
  explanation/chat/SBAR -> EXPLAINER, the task adapter (rubric, REPORT_NOTES 5h: same safety as base, better levels,
                           shorter SBAR, never cut off); explainer=None gives plain MedGemma
  image                 -> Cato's adapter (image_reader.IMAGE_ADAPTER)
No thinking block and capped answers (CAPS): an answer takes seconds on a big GPU instead of minutes.
Urgency cut-off: catches 90% of urgent KTAS validation patients (was 80%); more false alarms, fewer misses.
"""
import os
import threading
import time
from contextlib import contextmanager

import joblib
import numpy as np
import torch

from answers import cut_off, final_answer, parse_triage
from cards import assess, make_card
from config import HEADS_DIR, LORA_DIR
from ed_assessment import ed_assessment
from embed import embed_one
import image_models
from image_reader import IMAGE_ADAPTER, IMAGE_TYPES, describe_image, imaging_line
from image_checklist import LOCATABLE, chest_anatomy, checklist, checklist_line, disagreements, locate
from image_tools import outline, sort_image
from model import _messages, ask_messages, compute_dtype, load
from prompts import TRIAGE_SYSTEM, TURNS, VISIT_CHAT_SYSTEM, card_with_urgency
from visit import Decision, Visit, floor_facts

NO_ANSWER = "[No answer: the model ran out of room. Please ask again, more briefly.]"
EXPLAINER = "task/stage1"
QUESTIONER = "questions/stage1"        # make_question_data.py + lora_train.py --task-file questions (8 Nov); questions step only
NUMBER_READER = "numbers/stage1"       # make_number_data.py + lora_train.py --task-file numbers (8 Nov); numbers step only
URGENT_SENSITIVITY = 0.90
CAPS = {"explain": 768, "chat": 512, "sbar": 768, "report": 900, "questions": 600, "complaint": 24,
        "numbers": 600, "filing": 300}                                               # tokens per answer


class EDPipeline:
    def __init__(self, vitals_model=None, emb="base4bit", explainer=EXPLAINER, sensitivity=URGENT_SENSITIVITY,
                 think=False):
        from peft import PeftModel
        if vitals_model is None:
            from vitals_model import VitalsModel
            vitals_model = VitalsModel()
        self.vm = vitals_model
        # 4-bit: the heads were trained on the 4-bit model's vectors. BN4601A_LOAD_MODE=full is a fallback for a GPU
        # where bitsandbytes doesn't load (scores then come from slightly different vectors)
        base, self.processor = load(mode=os.environ.get("BN4601A_LOAD_MODE", "4bit"))
        self.model = PeftModel.from_pretrained(base, str(IMAGE_ADAPTER), adapter_name="image").eval()
        self.text_adapter = None
        if explainer:
            self.model.load_adapter(str(LORA_DIR / explainer / "adapter"), adapter_name="text")
            self.text_adapter = "text"
        self.step_adapter = {}
        if explainer and (LORA_DIR / QUESTIONER / "adapter").exists():
            self.model.load_adapter(str(LORA_DIR / QUESTIONER / "adapter"), adapter_name="questions")
            self.step_adapter["questions"] = "questions"
        if explainer and (LORA_DIR / NUMBER_READER / "adapter").exists():
            self.model.load_adapter(str(LORA_DIR / NUMBER_READER / "adapter"), adapter_name="numbers")
            self.step_adapter["numbers"] = "numbers"
        # adapters in the compute dtype (bfloat16 on RTX 30xx/A40) instead of float32: half the memory, and 2-3x faster
        # on the 4 GB laptop GPU (same calls timed: number reader 504 s -> 246 s, picture question 76 s -> 23 s, 9 Nov)
        dtype = compute_dtype()
        for name, param in self.model.named_parameters():
            if "lora_" in name and param.dtype == torch.float32:
                param.data = param.data.to(dtype)
        # one model, many screens: each AI step takes the model on its own (not a whole patient's run), so a nurse's
        # re-score or a chat question waits seconds, not for another patient to finish (5q #4: it waited 21-34 min)
        self.gpu = threading.RLock()
        self.think = think
        self.times = {}                                  # seconds of the last call of each step, for the trial run
        urgent = joblib.load(HEADS_DIR / f"urgent__{emb}__ktas_nhamcs.joblib")
        level = joblib.load(HEADS_DIR / f"level__{emb}.joblib")
        thresholds = urgent.get("thresholds", {0.8: urgent["threshold"]})
        if sensitivity not in thresholds:
            raise ValueError(f"no cut-off saved for {sensitivity}: run heads.py --add-threshold {sensitivity}")
        self.head, self.threshold = urgent["head"], thresholds[sensitivity]
        # 13 Nov 2026 (REPORT_NOTES 5u): under-16s get a children's urgency score (NHAMCS 2022 children; test AUROC 0.79
        # vs 0.70 for the adults' head, same catch, fewer false alarms). Optional file: without it, the adults' head.
        child = HEADS_DIR / f"urgent__{emb}__children.joblib"
        self.child_head, self.child_threshold = None, None
        if child.exists():
            c = joblib.load(child)
            self.child_head, self.child_threshold = c["head"], c.get("thresholds", {}).get(sensitivity, c["threshold"])
        self.level_model, self.cutoff = level["model"], level["conformal_cutoff"]
        # ED pathway flags by a trained gate + heads on the same vector + complaint words (pathway_heads.PathwayFlagger),
        # not word rules (branch rules-reasoning)
        import pathway_heads  # noqa: F401  (the saved flagger's class)
        self.pathway_flagger = joblib.load(HEADS_DIR / f"pathways__{emb}.joblib")

    def pathways(self, vector, complaint):
        """Vector + complaint -> ED pathway flags: the gate must say "needs a pathway" and that pathway's head agree."""
        return self.pathway_flagger.flags(vector, complaint or "")

    @contextmanager
    def _adapter(self, name, step):
        """Run one step with one adapter (None = plain MedGemma) and time it. Holds the model for just this step."""
        with self.gpu:
            start = time.time()
            if name is None:
                with self.model.disable_adapter():
                    yield
            else:
                self.model.set_adapter(name)
                yield
            self.times[step] = round(time.time() - start, 2)

    def _ask(self, messages, step):
        with self._adapter(self.step_adapter.get(step, self.text_adapter), step):
            return ask_messages(self.model, self.processor, messages, CAPS[step], self.think)

    def uses_child_score(self, age):
        return self.child_head is not None and age is not None and age == age and float(age) < image_models.CHILD_AGE

    def decide(self, vector, vitals_out, age=None):
        """Vector + Caleb's result -> urgency and level. Same rules as head_scores.py and level_model.py.
        Under 16 (and the children's head installed): the children's urgency score and its own cut-off."""
        x = np.asarray(vector).reshape(1, -1)
        head, cut = (self.child_head, self.child_threshold) if self.uses_child_score(age) else (self.head, self.threshold)
        p = float(head.predict_proba(x)[0, 1])
        head_flag, floor_flag = p >= cut, vitals_out["floor_priority"] <= 2
        flagged = bool(head_flag or floor_flag)
        probs = self.level_model.predict_proba(x)[0]                 # columns = levels 1..5
        level = int(probs.argmax()) + 1
        return Decision(p_urgent=p, head_flag=bool(head_flag), floor_flag=bool(floor_flag), flagged=flagged,
                        level=min(level, 2) if flagged else level,
                        level_set=tuple(int(i) + 1 for i in np.flatnonzero(probs >= 1 - self.cutoff)),
                        floor_priority=vitals_out["floor_priority"], news2=vitals_out["news2"], model_level=level,
                        level_probs=tuple(round(float(x), 4) for x in probs),
                        **floor_facts(vitals_out))

    def triage(self, patient, explain=True):
        """Stage 1, the front door: Caleb's vitals -> arrival card -> vector -> urgency + level -> explanation."""
        out = assess(self.vm, patient)
        card = make_card(patient, out)
        with self._adapter(None, "decide"):
            vector = embed_one(self.model, self.processor, card)
            decision = self.decide(vector, out, patient.get("age"))
        visit = Visit(card=card, decision=decision, ed=ed_assessment(patient, flags=self.pathways(vector, patient.get("chief_complaint"))),
                      vector=np.asarray(vector, dtype=np.float32).ravel())
        if explain:
            self.explain(visit)
        return visit

    def explain(self, visit):
        """MedGemma explains the decision. Separate step: the decision is quick, the writing can take minutes."""
        d = visit.decision
        guided = card_with_urgency(visit.card, d.p_urgent, d.flagged, d.floor_flag)
        raw = self._ask(_messages(TRIAGE_SYSTEM, guided), "explain")
        # the decision stands even if MedGemma's explanation can't be read
        visit.explanation = parse_triage(final_answer(raw)) or {"valid_json": False, "reasoning": NO_ANSWER}
        visit.explanation["cut_off"] = cut_off(raw)

    def read_image(self, visit, path, image_type=None, who="not recorded", age=None, note=""):
        """Stage 2: any image. image_type=None: the trained sorter picks the kind (MedGemma only if the sorter isn't on
        this computer); the clinician can still correct it. Then (8 Nov 2026, fix plan 5o item 1):
          the trained model for that kind (image_models.py): fracture v3 with an age cut-off, the VinDr chest detector
          (+ CheXpert for pneumothorax), the head-CT bleed model;
          MedGemma: a short description (Cato's adapter on X-rays) and the ED checklist for that kind;
          for MRI and ultrasound, which have no trained detector, MedGemma's box + MedSAM outline (experimental).
        Everything goes on the record, so the chat and SBAR can use it."""
        sorted_as = None
        if image_type is None and image_models.SORTER.exists():
            image_type, _ = image_models.sort_type(path)
            sorted_as = "image sorter"
        elif image_type is None:
            with self._adapter(None, "sort"):
                image_type, sorted_as = sort_image(self.model, self.processor, path, self.think)
        start = time.time()
        models = image_models.read_picture(path, image_type, age, note)
        anatomy = chest_anatomy(path) if image_type == "chest_xray" else None
        self.times["image_models"] = round(time.time() - start, 2)
        # Cato's adapter only for X-rays: on CT/MRI/ultrasound/ECG it wrote figure-caption text (real-AI check, 7 Nov)
        xray = image_type in ("bone_xray", "chest_xray")
        with self._adapter("image" if xray else None, "image"):
            text = describe_image(self.model, self.processor, path, image_type, think=self.think)
        with self._adapter(None, "checklist"):
            found_list = checklist(self.model, self.processor, path, image_type)
            located = []
            if image_type in ("mri_slice", "ultrasound"):        # no trained detector for these
                for item in [k for k, v in found_list.items() if v["seen"] == "yes"][:2]:
                    box = locate(self.model, self.processor, path, item)
                    if box:
                        located.append({"finding": item, "box": box, "experimental": True})
        for loc in located:
            seg = outline(path, loc["box"], fracture=False)
            loc.update({"area_pct": seg["area_pct"], "overlay": seg["overlay"]})
        disagree = disagreements(models.get("findings"), text, found_list)
        line = (imaging_line(image_type, text) + (" " + models["line"] if models["line"] else "")
                + checklist_line(image_type, found_list, anatomy and anatomy["ctr"], age) + disagree)
        if located:
            line += " Outlined: " + ", ".join(f"{l['finding']} (experimental)" for l in located) + "."
        if sorted_as is not None:
            line = f"(image type sorted by {sorted_as if sorted_as == 'image sorter' else 'AI'}: {IMAGE_TYPES[image_type][0]}) " + line
        visit.add(IMAGE_TYPES[image_type][1], line, who)
        return {"kind": image_type, "sorted_by_ai": sorted_as is not None, "text": text, "models": models,
                "checklist": found_list, "located": located, "anatomy": anatomy, "line": line, "disagree": disagree}

    def outline(self, visit, path, box, who="not recorded", fracture=True):
        """MedSAM outline inside a box (the clinician's or a detector's). fracture=True: the MedSAM tuned on fracture
        outlines, when it's on this computer. Only the size goes on the record; the picture is shown."""
        with self.gpu:                                   # MedSAM shares the GPU with MedGemma
            found = outline(path, box, fracture=fracture)
        visit.log("Imaging", f"Area outlined by AI inside a box drawn by {who}: {found['area_pct']:.1f}% of the picture.")
        return found

    def _turn(self, visit, question, step):
        if not visit.messages:                           # first turn: MedGemma reads the whole record
            visit.messages = [{"role": "system", "content": [{"type": "text", "text": VISIT_CHAT_SYSTEM}]}]
            question = f"{visit.record()}\n\n{question}"
        elif visit.unseen:                               # later turns: only what's new since the last answer
            question = "New results added to the record:\n" + "\n".join(visit.unseen) + f"\n\n{question}"
        visit.unseen = []
        visit.messages.append({"role": "user", "content": [{"type": "text", "text": question}]})
        answer = final_answer(self._ask(visit.messages, step)) or NO_ANSWER
        visit.messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
        return answer

    def chat(self, visit, question):
        """Core Task 2: one turn. Earlier turns stay in the conversation, so it remembers them."""
        return self._turn(visit, question, "chat")

    def sbar(self, visit):
        """Core Task 3: the handover, written from the whole conversation plus all results so far."""
        return self._turn(visit, TURNS[3], "sbar")
