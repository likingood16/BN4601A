# Notebooks

Notebooks are for learning and checking step by step. The finished code lives in `../src/`.

| Notebook | What it does | Status |
|---|---|---|
| `00_playground.ipynb` | practice cells | practice only |
| `01_load_model.ipynb` | load MedGemma (4-bit on the laptop) and ask it a question | done (Task 1) |
| `02_datasets.ipynb` | download, clean and split all datasets into Pool 1 + Pool 2 | done; makes `data/processed/` |
| `05_lora.ipynb` | LoRA training on Google Colab | **old**: replaced by `training/lora_train.py` on the NUS HPC |
| `05_lora_kaggle_emergency.ipynb` | LoRA emergency arm on Kaggle | **old**: never run; replaced by the HPC |

There are no notebooks 03 and 04. Cards, vectors and heads were written straight into `src/`
(`cards.py`, `embed.py`, `heads.py`), because they run for hours and work better as scripts.
The old Colab notebooks are kept because the first 22,448 examples of the mixed arm were trained with them.
