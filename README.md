# BN4601A: ED triage app

Student research prototype for emergency-department triage decision support. **Not a medical device.**
dataset: https://huggingface.co/datasets/OpenMed/multicare-cases

## Branches

| Branch | Whose | What it holds |
|---|---|---|
| `main` | group | the shared base |
| `cato` | Cato | Model 1: image model (MedGemma LoRA, ED-VLM); open pull request into `main` |
| `caleb` | Caleb | Model 2: vitals model and the MultiCaRe A&E filter (already contained in `ed-app-group-test`) |
| `ed-app-group-test` | Alexander | **the latest app**: Model 3 reasoning, the full pipeline, the React app, the Gradio version, image models, tests and guides |
| `reasoning` | Alexander | an older version of the app (2 Nov), fully contained in `ed-app-group-test` |

## Run the app

Model files are not in git (size, and some were trained on data that can't be shared publicly): get
`ED_app_full_ai_models.zip` from the group chat and put it next to the code.

```
cd app
pip install -r requirements.txt              # Python 3.11
python install_models.py                     # finds ED_app_full_ai_models.zip and puts every model in place
python server.py                             # React app, full AI (NVIDIA GPU):  http://127.0.0.1:8000
python server.py --test                      # any computer: MedGemma's writing faked, everything else real
cd gradio && python app.py                   # Gradio app (same back end):     http://127.0.0.1:7860
```
Guides: `app/build/ED_app_user_guide.pdf` and `ED_app_gradio_guide.pdf` (built by `app/tools/make_guide.py` and
`make_gradio_guide.py`). Results, methods and references: `reasoning/REPORT_NOTES.md`. New here: `START_HERE.md`.

## Layout

```
app/                 THE APP (what the group tests and demos)
  server.py            back end (patients, AI job, whiteboard); frontend/ = React screens (dist/ is built)
  gradio/              the same app with Gradio screens (Hugging Face ZeroGPU Space)
  install_models.py    puts the model files in place (zip, split parts or the Hugging Face model repo)
  tests/               test_app.py (68 checks), stress_test.py, real_ai_check.py, test notes and images
  tools/               builders: user guides, test pack, Telegram files, the Space bundle
  build/               what the tools make (guides, zips, Space files): not in git
reasoning/           Model 3 (Alexander)
  src/                 what the app runs: pipeline.py, charting, heads, image steps, chat, test_backend (test mode)
  training/            LoRA training, training-data makers, head / level / fusion experiments, evaluation
  checks/              accuracy checks of single steps (numbers, pathways, questions, image tools, pipeline)
  hpc/                 NUS vanda job scripts;  notebooks/  experiments;  models/ data/ outputs/  not in git
  REPORT_NOTES.md      all results, methods and references for the report
vitals/              Model 2 (Caleb): vitals -> NEWS2, p_urgent, safety floor
vision/              image models: picture sorter, fracture, chest X-ray, head-CT bleed; cato/ = Model 1 (Cato)
multicare_filter/    data prep: picks MultiCaRe ED cases (image + case text + vitals)
old/                 the first Gradio mock-up (kept for reference, not used)
```

Pipeline: note -> numbers (rules + MedGemma) -> Model 2 vitals note + safety floor -> patient card -> urgency + level
(heads on MedGemma's vector) -> pictures (sorter -> trained model for that kind) -> case file, questions, to-dos (MedGemma)
-> whiteboard.
