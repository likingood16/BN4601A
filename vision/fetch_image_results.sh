#!/bin/bash
# Copy the image models' result files (small JSON, no images or patient data) from vanda into vision/results/,
# where reasoning/src/rubric.py reads them. Run from the repo root on the laptop.
set -e
R=/scratch/e0959158/bn4601a/BN4601A/vision
mkdir -p vision/results
for pair in runs/chest_vindr/test_metrics.json:chest_vindr.json runs/chest_vindr/test_metrics_first.json:chest_vindr_first.json \
            runs/fracture_v2/test_metrics.json:fracture_v2.json runs/fracture_v2/test_metrics_first.json:fracture_v2_first.json \
            ct_head/runs/ich/test_metrics.json:ich.json image_type/runs/image_type/test_metrics.json:image_type.json \
            runs/external_tests.json:external_tests.json; do
  scp -q -o BatchMode=yes "nus-vanda:$R/${pair%%:*}" "vision/results/${pair##*:}" 2>/dev/null && echo "got ${pair##*:}" || echo "not yet: ${pair%%:*}"
done
