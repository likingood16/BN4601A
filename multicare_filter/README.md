# MultiCaRe filter (A&E triage project)

Selects the images and case reports from [OpenMed/multicare-images](https://huggingface.co/datasets/OpenMed/multicare-images)
(joined to OpenMed/multicare-cases on `patient_id` = `case_id`) that fit an A&E triage app. Independent of `vitals/`.

## Run it on Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com) → **File → Upload notebook** → choose
   `MultiCaRe_filter_colab.ipynb` (it contains its own copy of the filter code; nothing else needs uploading).
2. **Runtime → Run all.** A CPU runtime is enough. The first run downloads the two datasets from Hugging Face
   (about 3.3 GB, a few minutes).
3. At the end the results download as `multicare_filter_output.zip`. Colab deletes files when the session ends; set
   `SAVE_TO_DRIVE = True` in the last cell to keep a copy in Google Drive.

To change a rule, edit `CONFIG` in step 1 of the notebook and run again. Locally, the same code runs from
`notebooks/01_filter_multicare.ipynb` and uses the Hugging Face cache if the data is already downloaded.

## Rules (all in `CONFIG`, `src/multicare_filter.py`)

| Rule | Default |
|---|---|
| Licence | CC0, CC BY, CC BY-SA, CC BY-NC, CC BY-NC-SA (no ND, no unlicensed) |
| Modality | X-ray, CT, ultrasound; clinical photos only if the caption describes an acute problem and isn't a theatre/specimen photo |
| Caption | ≥ 20 characters |
| Timepoint | drop post-operative, follow-up and healed images |
| Patient | age ≥ 16 |
| Setting | case text says the patient came to an emergency department |
| Vitals | ≥ 1 measured vital sign mentioned in the case text |

Options: ECG images, MRI, excluding echocardiograms, stricter vitals (≥ 3), CC BY only. The notebook shows the counts for each.

## Result (default rules)

**2,340 images from 1,015 cases.** CT 1,339 · X-ray 513 · ultrasound 396 (207 echocardiograms) · acute photos 92.
Department guess per case: Respiratory 260 · Cardiology 219 · Surgery 193 · Neurology 143 · Orthopaedics 45 · unclear 155.

## Files

```
src/multicare_filter.py            rules, loading (from the local HF cache), department guess
notebooks/01_filter_multicare.ipynb  funnel, breakdowns, options, example images (outputs saved)
output/selected_images.csv         one row per image: file_id, licence, modality, region, caption, case_id
output/selected_cases.csv          one row per case: age, gender, department guess, vitals mentioned, case text
output/department_hand_check.csv   40 random cases to label by hand (fill `correct_department`)
output/funnel.csv
```

`output/` is not committed (it contains case text); re-run the notebook to regenerate it. Load the actual images from
the Hugging Face dataset by `file_id`.

## Caveats

- **Keyword rules:** the selection is noisy. Spot-check images before fine-tuning.
- **Department labels:** the department is a per-case keyword guess, not a label. Hand-check it before using it as ground truth.
- **Vital signs:** "vitals mentioned" only means the text contains them; the numbers still have to be extracted.
