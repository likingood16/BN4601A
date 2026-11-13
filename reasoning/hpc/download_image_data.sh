#!/bin/bash
# Run on the HPC LOGIN NODE (internet), in the background:
#   nohup bash reasoning/hpc/download_image_data.sh > reasoning/outputs/logs/download_image_data.log 2>&1 &
# Open datasets for the image models (7 Nov 2026; Kaggle datasets come separately with download_kaggle.sh):
#   1. GRAZPEDWRI-DX (Nagy et al. 2022, Sci Data; figshare 14825193; CC BY 4.0): 20,327 paediatric wrist X-rays,
#      6,091 patients, YOLO boxes for fractures and other findings. 16.2 GB of zips.
#   2. MultiCaRe images (OpenMed/multicare-images, CC BY): every picture with its own type labels, for the
#      image-type classifier. About 3.2 GB.
set -e
cd "$(dirname "$0")/../.."
source reasoning/hpc/hpc.conf
export HOME=$ROOT/home HF_HOME
DATA=$ROOT/image_data
mkdir -p "$DATA/grazpedwri"
cd "$DATA/grazpedwri"
for id_name in "35026432 dataset.csv" "34268819 folder_structure.zip" "34268828 images_part1.zip" \
               "34268849 images_part2.zip" "34268864 images_part3.zip" "34268891 images_part4.zip"; do
  set -- $id_name
  if [ ! -s "$2" ]; then
    echo "$(date +%H:%M) downloading $2"
    wget -q -c -O "$2" "https://ndownloader.figshare.com/files/$1"
  fi
done
ls -la
for z in folder_structure.zip images_part*.zip; do
  [ -f "${z%.zip}.done" ] || { echo "$(date +%H:%M) unzipping $z"; unzip -q -o "$z" -d unpacked && touch "${z%.zip}.done"; }
done
find unpacked -name "*.png" | wc -l
cd - > /dev/null
echo "$(date +%H:%M) MultiCaRe images"
"$VENV/bin/python" -u - <<'EOF'
from datasets import load_dataset
ds = load_dataset("OpenMed/multicare-images", split="train")
print("MultiCaRe images:", len(ds), "columns:", ds.column_names)
EOF
echo "$(date +%H:%M) all open downloads done"
