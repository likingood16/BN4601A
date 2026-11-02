# reasoning/models: trained models (laptop copies, not in git, too big)

## LoRA adapters: `lora/<arm>/`

| Folder | What it is |
|---|---|
| `lora/mixed/stage12/` | **final mixed adapter** (all 347,779 examples): the best arm |
| `lora/emergency/stage3/` | final emergency-only adapter (73,583 examples) |
| `lora/two_stage/stage2/` | final two-stage adapter (30k general, then 30k emergency) |
| `lora/<arm>/stageN/` | the adapter after stage N, kept so the learning curve can be re-checked |
| `lora/<arm>/run/` | the working copy used while training (resume point) |
| `lora/mixed/stage1_8448/`, `stage1_22448/`, `latest/` | early Colab snapshots (before the HPC) |

Each folder has:
- `adapter/`: the adapter itself (`adapter_model.safetensors`, ~120 MB, + `adapter_config.json`)
- `progress.json`: stages done, val loss and MedQA after each stage

Trained on: `unsloth/medgemma-1.5-4b-it-unsloth-bnb-4bit` (MedGemma 1.5 4B, 4-bit).
Used with: `model.load(lora="mixed/stage12")`.

**Getting them:** training ran on the NUS HPC. Copy the adapters to the laptop with:

    powershell -ExecutionPolicy Bypass -File reasoning\hpc\sync_down.ps1

## Urgency heads: `heads/`

`urgent__<vectors>__<trained on>.joblib`, made by `src/heads.py`. Example:
`urgent__base4bit__ktas_nhamcs.joblib` = head on laptop 4-bit MedGemma vectors, trained on KTAS + NHAMCS.
Each file holds the head and its alarm threshold. The test results are in
`outputs/metrics/<same name>.json`.
