# BN4601A
dataset: https://huggingface.co/datasets/OpenMed/multicare-cases

## Layout

One folder per part of the app. Each has its own `notebooks/` (experiments, numbered), `src/` (functions the
others import), `models/` (trained files) and a README. Data and outputs stay out of git.

```
multicare_filter/   data prep: picks MultiCaRe ED cases (image + case text + vitals) for testing and the demo
vitals/             Model 2: vitals -> NEWS2, p_urgent, safety floor, note     (VitalsModel().assess())
reasoning/          Model 3: patient card -> triage, chat, SBAR report          (triage(), chat(), report())
```

Pipeline: image -> Model 1 findings, vitals -> Model 2 note + floor -> patient card -> Model 3 -> `enforce_floor()` ->
chat -> report.
