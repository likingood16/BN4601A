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
"""
import joblib
import numpy as np

from answers import cut_off, final_answer, parse_triage
from cards import assess, make_card
from config import HEADS_DIR
from ed_assessment import ed_assessment
from embed import embed_one
from image_reader import IMAGE_TYPES, describe_image, detect_fracture, imaging_line, load_image_model
from model import _messages, ask_messages, load
from prompts import TRIAGE_SYSTEM, TURNS, VISIT_CHAT_SYSTEM, card_with_urgency
from visit import Decision, Visit

NO_ANSWER = "[No answer: the model ran out of room while thinking. Please ask again, more briefly.]"


class EDPipeline:
    def __init__(self, vitals_model=None, emb="base4bit"):
        if vitals_model is None:
            from vitals_model import VitalsModel
            vitals_model = VitalsModel()
        self.vm = vitals_model
        self.model, self.processor = load(mode="4bit")
        urgent = joblib.load(HEADS_DIR / f"urgent__{emb}__ktas_nhamcs.joblib")
        level = joblib.load(HEADS_DIR / f"level__{emb}.joblib")
        self.head, self.threshold = urgent["head"], urgent["threshold"]
        self.level_model, self.cutoff = level["model"], level["conformal_cutoff"]
        self.image_model = None                          # loaded the first time an image arrives

    def decide(self, vector, vitals_out):
        """Vector + Caleb's result -> urgency and level. Same rules as head_scores.py and level_model.py."""
        x = np.asarray(vector).reshape(1, -1)
        p = float(self.head.predict_proba(x)[0, 1])
        head_flag, floor_flag = p >= self.threshold, vitals_out["floor_priority"] <= 2
        flagged = bool(head_flag or floor_flag)
        probs = self.level_model.predict_proba(x)[0]                 # columns = levels 1..5
        level = int(probs.argmax()) + 1
        return Decision(p_urgent=p, head_flag=bool(head_flag), floor_flag=bool(floor_flag), flagged=flagged,
                        level=min(level, 2) if flagged else level,
                        level_set=tuple(int(i) + 1 for i in np.flatnonzero(probs >= 1 - self.cutoff)),
                        floor_priority=vitals_out["floor_priority"], news2=vitals_out["news2"])

    def triage(self, patient, explain=True):
        """Stage 1, the front door: Caleb's vitals -> arrival card -> vector -> urgency + level -> explanation."""
        out = assess(self.vm, patient)
        card = make_card(patient, out)
        decision = self.decide(embed_one(self.model, self.processor, card), out)
        visit = Visit(card=card, decision=decision, ed=ed_assessment(patient))
        if explain:
            self.explain(visit)
        return visit

    def explain(self, visit):
        """MedGemma explains the decision. Separate step: the decision is quick, the writing can take minutes."""
        d = visit.decision
        guided = card_with_urgency(visit.card, d.p_urgent, d.flagged, d.floor_flag)
        raw = ask_messages(self.model, self.processor, _messages(TRIAGE_SYSTEM, guided), 2048)
        # the decision stands even if MedGemma's explanation can't be read
        visit.explanation = parse_triage(final_answer(raw)) or {"valid_json": False, "reasoning": NO_ANSWER}
        visit.explanation["cut_off"] = cut_off(raw)

    def read_image(self, visit, path, image_type):
        """Stage 2: any image. Its reading goes on the record, so the chat and SBAR can use it."""
        if self.image_model is None:
            self.image_model = load_image_model()
        text = describe_image(*self.image_model, path, image_type)
        found = detect_fracture(path) if image_type == "bone_xray" else None
        visit.add(IMAGE_TYPES[image_type][1], imaging_line(image_type, text, found))
        return {"text": text, "detector": found}

    def _turn(self, visit, question, max_new_tokens):
        if not visit.messages:                           # first turn: MedGemma reads the whole record
            visit.messages = [{"role": "system", "content": [{"type": "text", "text": VISIT_CHAT_SYSTEM}]}]
            question = f"{visit.record()}\n\n{question}"
        elif visit.unseen:                               # later turns: only what's new since the last answer
            question = "New results added to the record:\n" + "\n".join(visit.unseen) + f"\n\n{question}"
        visit.unseen = []
        visit.messages.append({"role": "user", "content": [{"type": "text", "text": question}]})
        answer = final_answer(ask_messages(self.model, self.processor, visit.messages, max_new_tokens)) or NO_ANSWER
        visit.messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
        return answer

    def chat(self, visit, question):
        """Core Task 2: one turn. Earlier turns stay in the conversation, so it remembers them."""
        return self._turn(visit, question, 1536)

    def sbar(self, visit):
        """Core Task 3: the handover, written from the whole conversation plus all results so far."""
        return self._turn(visit, TURNS[3], 2048)
