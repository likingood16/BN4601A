#!/bin/bash
# Run on the HPC LOGIN NODE in the background, after the user's kaggle.json is in $ROOT/home/.kaggle:
#   nohup bash reasoning/hpc/download_kaggle.sh > reasoning/outputs/logs/download_kaggle.log 2>&1 &
# Kaggle data for the image models (7 Nov 2026). Terms: non-commercial / academic use, not shared with anyone who
# hasn't accepted them, so these stay on vanda and out of git.
#   1. VinBigData chest X-ray boxes: competition train.csv (15,000 X-rays, 14 findings, 3 radiologists each) and the
#      same X-rays resized to 1024 px PNG (xhlulu/vinbigdata-chest-xray-resized-png-1024x1024, 7.9 GB).
#   2. RSNA head-CT bleeds: JPG slices with the competition labels (jhoward/rsna-hemorrhage-jpg, 4.1 GB).
set -e
cd "$(dirname "$0")/../.."
source reasoning/hpc/hpc.conf
export HOME=$ROOT/home KAGGLE_CONFIG_DIR=$ROOT/home/.kaggle
DATA=$ROOT/image_data
mkdir -p "$DATA/vinbigdata" "$DATA/rsna_ich"
"$VENV/bin/python" -u - <<EOF
import time
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
t = time.time()
api.competition_download_file("vinbigdata-chest-xray-abnormalities-detection", "train.csv", path="$DATA/vinbigdata", quiet=True)
print("VinBigData train.csv", flush=True)
api.dataset_download_files("xhlulu/vinbigdata-chest-xray-resized-png-1024x1024", path="$DATA/vinbigdata", unzip=True, quiet=True)
print(f"VinBigData PNGs ({time.time() - t:.0f} s)", flush=True)
api.dataset_download_files("jhoward/rsna-hemorrhage-jpg", path="$DATA/rsna_ich", unzip=True, quiet=True)
print(f"RSNA head-CT JPGs ({time.time() - t:.0f} s)", flush=True)
EOF
cd "$DATA/vinbigdata" && [ -f train.csv.zip ] && unzip -o -q train.csv.zip
du -sh "$DATA/vinbigdata" "$DATA/rsna_ich"
find "$DATA/vinbigdata" -maxdepth 2 | head
find "$DATA/rsna_ich" -maxdepth 3 -type d | head
echo "kaggle downloads done $(date +%H:%M)"
