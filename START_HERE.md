# Start here

| I want to ... | Go to |
|---|---|
| run the app | `app/` : `python server.py` (full AI) or `python server.py --test`; Gradio: `app/gradio/app.py` |
| change what a screen shows | `app/frontend/src/` (React), then `npm run build` in `app/frontend` |
| change the back end (patients, whiteboard, AI job) | `app/server.py` |
| change how the AI reads notes, scores, writes the case file | `reasoning/src/` (start at `pipeline.py`) |
| run the app's checks | `python app/tests/test_app.py` (68 checks, test mode) |
| rebuild the guides / test pack / Telegram files / Space | `app/tools/` (outputs land in `app/build/`) |
| train or evaluate a model | `reasoning/training/`, `reasoning/checks/`, vanda jobs in `reasoning/hpc/jobs/` |
| find a result for the report | `reasoning/REPORT_NOTES.md` |
| Caleb's vitals model / Cato's image model / image detectors | `vitals/`, `vision/cato/`, `vision/` |

Not code: `app/build/`, `reasoning/models/`, `reasoning/data/`, `reasoning/outputs/` (all kept out of git).
Old, not used: `old/first_mockup/`. `reasoning/ed_app/` is a separate prototype from another session.
