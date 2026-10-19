# Model 3: Reasoning (MedGemma)

Triage reasoning, chat and SBAR report with `google/medgemma-1.5-4b-it`, plus a learned urgency score
(logistic regression on MedGemma embeddings, KTAS train) with Caleb's safety floor.

- `notebooks/`: working notebooks (local kernel `bn4601a`, or Colab via the VS Code Colab extension)
- `src/`: finished functions; `api.py` exposes `triage()`, `chat()`, `report()`
- `models/`: saved score head
- `outputs/`: run outputs (gitignored)
