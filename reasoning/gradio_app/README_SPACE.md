---
title: ED triage app
emoji: 🚑
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 6.27.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: BN4601A student prototype - ED triage decision support
---

# ED triage app (BN4601A, Model 3), Gradio version

Student research prototype. **Not a medical device. Never enter real patients' details.**

A triage nurse types the arrival note the way they normally would and can add X-rays or scans. The app pulls the
numbers out of the note, scores urgency and a triage level with the trained models, sorts each picture and runs the
trained model for that kind, writes a tidy case file, asks about what's missing, suggests to-dos, and puts the
patient on a shared whiteboard. The same back end as the React app (`reasoning/web/server.py`).

## Setting up the Space (once)

1. Create a new Space: SDK **Gradio**, visibility **Private** (some models were trained on data that can't be shared
   publicly: VinDr-CXR and RSNA).
2. Upload everything in this folder (drag the files into "Files" -> "Add file" -> "Upload files", or
   `huggingface-cli upload <you>/<space> . .` from this folder).
3. Settings -> **Hardware: ZeroGPU** (free for accounts older than 30 days with a verified email; PRO gives more
   daily GPU time).
4. Settings -> **Secrets**: `HF_TOKEN` = a token of an account that has accepted MedGemma's terms on
   https://huggingface.co/google/medgemma-1.5-4b-it (the model is downloaded when the Space starts).
5. Optional: Settings -> Variables -> `BN4601A_TEST` = `1` runs test mode (MedGemma's writing faked, no GPU needed).

## How it uses the GPU

ZeroGPU gives a GPU only inside functions marked `@spaces.GPU`, in a separate process. One patient's whole AI run is
one GPU call (up to 4 minutes); re-scoring after "Fix numbers" and each chat answer are calls of their own. Each
viewer has a daily ZeroGPU quota (5 min free, 40 min PRO), so about 5-10 patients a day per free viewer.

## Not tested on ZeroGPU yet (9 Nov 2026)

The app was tested on a laptop in test mode (and with `BN4601A_FAKE_ZERO=1`, which imitates ZeroGPU's
copy-in/copy-out). Loading MedGemma in 4-bit (bitsandbytes) at start-up under ZeroGPU has not been tried: if the Space
fails to start, look at the logs for bitsandbytes/CUDA errors.

## Run it on a computer instead

```
pip install -r requirements.txt gradio==6.27.0
python app.py --test        # no GPU: MedGemma's writing faked
python app.py               # real AI (needs a GPU; slow on 4 GB)
```
Then open http://127.0.0.1:7860.
