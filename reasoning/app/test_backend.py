"""Test mode: the same methods as pipeline.EDPipeline with MedGemma's writing replaced by labelled fake text, so every
button can be tested in seconds. Real here: Caleb's vitals model and safety floor, the patient record, the ED
assessment and the fracture detector (CPU). Fake: the urgency score (it's Caleb's vitals score, not the MedGemma head),
the level, and every written answer.
"""
from cards import assess, make_card
from ed_assessment import ed_assessment
from visit import Decision, Visit

MARK = "[TEST MODE, not a real answer]"


class TestPipeline:
    def __init__(self):
        from vitals_model import VitalsModel
        self.vm = VitalsModel()

    def triage(self, patient, explain=True):
        out = assess(self.vm, patient)
        floor_flag = out["floor_priority"] <= 2
        flagged = bool(out["urgent_flag"] or floor_flag)
        decision = Decision(p_urgent=out["p_urgent"], head_flag=out["urgent_flag"], floor_flag=floor_flag, flagged=flagged,
                            level=2 if flagged else 3, level_set=(1, 2) if flagged else (3, 4),
                            floor_priority=out["floor_priority"], news2=out["news2"])
        visit = Visit(card=make_card(patient, out), decision=decision, ed=ed_assessment(patient))
        if explain:
            self.explain(visit)
        return visit

    def explain(self, visit):
        visit.explanation = {"triage_level": visit.decision.level, "valid_json": True, "department": "Emergency medicine",
                             "reasoning": f"{MARK} MedGemma's explanation of the decision appears here.",
                             "red_flags": [f"{MARK} red flag"], "next_steps": [f"{MARK} next step"],
                             "missing_information": [f"{MARK} missing information"]}

    def read_image(self, visit, path, image_type):
        from image_reader import IMAGE_TYPES, detect_fracture, imaging_line      # slow import, only when needed
        text = f"{MARK} The image description appears here."
        found = detect_fracture(path) if image_type == "bone_xray" else None
        visit.add(IMAGE_TYPES[image_type][1], imaging_line(image_type, text, found))
        return {"text": text, "detector": found}

    def _turn(self, visit, question):
        new = f" New results it was told about: {'; '.join(visit.unseen)}." if visit.unseen else ""
        visit.unseen = []
        answer = f"{MARK} You asked: \"{question}\". Earlier turns remembered: {len(visit.messages) // 2}.{new}"
        visit.messages += [{"role": "user", "content": [{"type": "text", "text": question}]},
                           {"role": "assistant", "content": [{"type": "text", "text": answer}]}]
        return answer

    def chat(self, visit, question):
        return self._turn(visit, question)

    def sbar(self, visit):
        turns = len(visit.messages) // 2
        self._turn(visit, "Write the SBAR handover")
        results = "; ".join(f"{k}: {' | '.join(v)}" for k, v in visit.results.items()) or "not recorded"
        d = visit.decision
        return (f"{MARK}\n\n**Situation**\nUrgency {d.p_urgent:.0%}, level {d.level}.\n\n"
                f"**Background**\nResults so far: {results}\n\n"
                f"**Assessment**\nBased on {turns} chat turn(s).\n\n**Recommendation**\nClinician review.")
