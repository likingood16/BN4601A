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

On your own computer, no GPU, test mode (MedGemma's writing replaced; the trained number, urgency, level and image
models are real). Model files are not in git (size, and some were trained on data that can't be shared publicly): get
them from the group chat.

```
cd reasoning/web
pip install -r requirements_app.txt          # Python 3.11
python install_models.py                     # finds ED_app_models.zip.001 ... and puts the models in place
python server.py --test                      # React app:  http://127.0.0.1:8000
cd ../gradio_app && python app.py --test     # Gradio app: http://127.0.0.1:7860
```
Guides: `ED_app_user_guide.pdf` and `ED_app_gradio_guide.pdf` (built by `reasoning/web/make_guide.py` and
`make_gradio_guide.py`). Results, methods and references: `reasoning/REPORT_NOTES.md`.

## Layout

One folder per part of the app. Each has its own `notebooks/` (experiments, numbered), `src/` (functions the
others import), `models/` (trained files) and a README. Data and outputs stay out of git.

```
multicare_filter/   data prep: picks MultiCaRe ED cases (image + case text + vitals) for testing and the demo
vitals/             Model 2: vitals -> NEWS2, p_urgent, safety floor, note     (VitalsModel().assess())
vision/             image models: picture sorter, fracture detector, chest X-ray detector, head-CT bleed check
reasoning/          Model 3: numbers, urgency + level, pathways, case file, questions, chat (src/pipeline.py)
reasoning/web/      the app: server.py (back end) + frontend/ (React screens), test pack and guide builders
reasoning/gradio_app/  the same app with Gradio screens, for a Hugging Face ZeroGPU Space
```

Pipeline: note -> numbers (rules + MedGemma) -> Model 2 vitals note + safety floor -> patient card -> urgency + level
(heads on MedGemma's vector) -> pictures (sorter -> trained model for that kind) -> case file, questions, to-dos (MedGemma)
-> whiteboard.
