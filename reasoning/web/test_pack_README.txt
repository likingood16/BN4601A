ED TRIAGE APP - GROUP TEST PACK (BN4601A, Model 3)
====================================================

Please read ED_app_user_guide.pdf first (about 10 minutes). It shows every screen and button.

What is in here
  ED_app_user_guide.pdf   how to install the app on your computer (section 1) and use it, step by step, with pictures
  ED_app_gradio_guide.pdf the Gradio version of the same app (same install), for the Hugging Face demo
  cases/                  one note per test patient (.txt). Open one with "Open a text file" or copy-paste it.
                            T01-T10  written to test triage and number reading (T10a/b/c = same patient 3 ways)
                            P01-P18  a note + a picture (pictures/ folder) to test the image models
                            R01-R10  real held-out test patients; their true triage level is in expected_results.csv
  pictures/               X-rays, CTs and other pictures for the P cases (all CC BY 4.0, see ATTRIBUTION.txt)
  expected_results.csv    what the app showed for every case when we ran it (test mode, 8 Nov 2026)
  feedback_form.csv       please fill this in and send it back (one row per case or step you tried)

What we would like you to do (about 30-45 minutes)
  1. Part A of the guide: walk through the app once with case T01.
  2. Run the cases in cases/. For each one, compare what you see with expected_results.csv.
  3. Try the workflow steps in Part C of the guide (add information, fix numbers, whiteboard, clear with outcome).
  4. Try to break it (Part D of the guide): odd notes, many patients at once, two people on the whiteboard.
  5. Fill in feedback_form.csv: what you did, what happened, whether it matched, how bad (1 cosmetic - 5 unsafe).
     Screenshots help: name them by case, e.g. P06_1.png.

Full AI (the final product)
  - On your own computer with an NVIDIA GPU: ED_app_full_ai_models.zip from the chat + section 1b of the user guide.
  - Online on a Hugging Face GPU (Gradio screens, nothing to install): the link comes in the chat.

Important
  - Test mode: the urgency score, numbers, pathway rules, image models and whiteboard are real. MedGemma's writing
    (case-file wording, chat answers, image descriptions) is replaced by text marked [TEST MODE].
  - This is a student research prototype. It is not a medical device and must not be used on real patients.
