---
title: ED triage app
emoji: 🚑
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 6.27.0
python_version: "3.12"
app_file: app/gradio/app.py
pinned: false
suggested_hardware: zero-a10g
preload_from_hub:
  - unsloth/medgemma-1.5-4b-it
short_description: BN4601A student prototype - ED triage decision support
---

# ED triage app (BN4601A, Model 3), Gradio version

Student research prototype. **Not a medical device. Never enter real patients' details.**

A triage nurse types the arrival note the way they normally would and can add X-rays or scans. The app pulls the
numbers out of the note, scores urgency and a triage level with the trained models, sorts each picture and runs the
trained model for that kind, writes a tidy case file, asks about what's missing, suggests to-dos, and puts the
patient on a shared whiteboard. The same back end as the React app (`app/server.py`).

## How it is set up

- The Space holds the **code only** (made by `app/tools/make_space_bundle.py` from the GitHub branch `main`).
- MedGemma comes from `unsloth/medgemma-1.5-4b-it` (the same weights as `google/medgemma-1.5-4b-it`, not gated).
- The trained parts (urgency, level and pathway heads; case-file, question and number adapters; the image adapter;
  the picture sorter, fracture, chest X-ray and head-CT models) are in the group's model repo
  `likingood/bn4601a-ed-models`, which is not public: some were trained on data that can't be shared (VinDr-CXR,
  RSNA). The Space downloads them when it starts, with its **`HF_TOKEN` secret** (Settings -> Variables and secrets:
  a "Read" token of the repo's owner). Without it the Space starts in test mode and says why at the top.
- Hardware: **ZeroGPU** (free: the GPU is lent for each call). Optional variable `BN4601A_TEST=1` = test mode.

## How it uses the GPU

ZeroGPU gives a GPU only inside functions marked `@spaces.GPU`, in a separate process. One patient's whole AI run is
one GPU call (asks for 90 s + 40 s per picture); re-scoring after "Fix numbers" (30 s) and each chat answer (60 s) are
calls of their own. Each visitor has a daily ZeroGPU allowance (2 min logged out, 5 min with a free Hugging Face
account, 40 min PRO), so log in to test more than a couple of patients.

## Run it on a computer instead

```
cd app/gradio
python app.py --test        # any computer: MedGemma's writing faked
python app.py               # full AI on an NVIDIA GPU (after python ../install_models.py --hub)
```
Then open http://127.0.0.1:7860.
