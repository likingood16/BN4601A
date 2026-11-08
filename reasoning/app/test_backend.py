"""Test mode: the same methods as pipeline.EDPipeline with MedGemma's writing replaced by labelled fake text, so every
button can be tested in seconds. Real here: Caleb's vitals model and safety floor, the patient record, the ED
assessment and the trained image models (sorter, fracture, chest, head-CT bleed; CPU) and MedSAM if it is on this
computer. Fake: the urgency score (it's Caleb's vitals score, not the MedGemma head),
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
                            floor_priority=out["floor_priority"], news2=out["news2"], model_level=3)
        visit = Visit(card=make_card(patient, out), decision=decision, ed=ed_assessment(patient))
        if explain:
            self.explain(visit)
        return visit

    def explain(self, visit):
        visit.explanation = {"triage_level": visit.decision.level, "valid_json": True, "department": "Emergency medicine",
                             "reasoning": f"{MARK} MedGemma's explanation of the decision appears here.",
                             "red_flags": [f"{MARK} red flag"], "next_steps": [f"{MARK} next step"],
                             "missing_information": [f"{MARK} missing information"]}

    def read_image(self, visit, path, image_type=None, who="not recorded", age=None, note=""):
        """Real: the trained sorter and image models (image_models.py, CPU). Fake: MedGemma's description and checklist."""
        import image_models
        from image_reader import IMAGE_TYPES, imaging_line      # slow import, only when needed
        sorted_by_ai = image_type is None
        models = image_models.read_picture(path, image_type, age, note)
        image_type = models["kind"] or "bone_xray"           # no sorter on this computer: say bone X-ray
        text = f"{MARK} The image description appears here."
        line = imaging_line(image_type, text) + (" " + models["line"] if models["line"] else "")
        if sorted_by_ai:
            line = f"(image type sorted by {models['sorted_by'] or 'test mode'}: {IMAGE_TYPES[image_type][0]}) " + line
        found_list = {}                 # MedGemma's ED checklist only appears with the real AI
        visit.add(IMAGE_TYPES[image_type][1], line, who)
        return {"kind": image_type, "sorted_by_ai": sorted_by_ai, "text": text, "models": models,
                "checklist": found_list, "located": [], "anatomy": None, "line": line}

    @staticmethod
    def _fake_outline(path, box):
        import numpy as np
        from PIL import Image
        from image_tools import overlay
        w, h = Image.open(path).size
        mask = np.zeros((h, w), np.float32)
        x0, y0, x1, y1 = (int(v) for v in box)
        mask[y0:y1, x0:x1] = 1
        return {"area_pct": float(mask.mean() * 100), "overlay": overlay(path, mask, (0, 160, 255))}

    def outline(self, visit, path, box, who="not recorded", fracture=True):
        """Real MedSAM when it is already on this computer (never downloaded here); otherwise the box is shaded."""
        try:
            import os
            from image_tools import outline
            os.environ["BN4601A_NO_DOWNLOAD"] = "1"
            found = outline(path, box, fracture=fracture)
            visit.log("Imaging", f"Area outlined by MedSAM inside a box drawn by {who}: {found['area_pct']:.1f}% of the picture.")
            return found
        except Exception:
            pass
        import numpy as np
        from PIL import Image
        from image_tools import overlay
        w, h = Image.open(path).size
        mask = np.zeros((h, w), np.float32)
        x0, y0, x1, y1 = (int(v) for v in box)
        mask[min(y0, y1):max(y0, y1), min(x0, x1):max(x0, x1)] = 1
        visit.log("Imaging", f"{MARK} Area outlined inside a box drawn by {who}: {mask.mean() * 100:.1f}% of the picture.")
        return {"mask": mask, "area_pct": float(mask.mean() * 100), "overlay": overlay(path, mask, (0, 160, 255))}

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
