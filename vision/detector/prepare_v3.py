"""Fracture data v3 (8 Nov 2026): the v2 data (prepare_combined.py) with every ADULT training X-ray (FracAtlas, fa_*)
repeated 4 times, so adults weigh about as much as the 15k children's wrists (GRAZ) in training.
Why: v2 caught 84% of children's wrist fractures but 22% of adult ones at the shared cut-off.
Validation and test are the same files as v2 (symlinked folders), so v2 and v3 are compared on identical X-rays.

    python prepare_v3.py        # login node, after prepare_combined.py
"""
import os
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
V2 = HERE / "data" / "combined"
V3 = HERE / "data" / "combined_v3"
REPEAT = 4


def main():
    for kind in ("images", "labels"):
        (V3 / kind / "train").mkdir(parents=True, exist_ok=True)
        for split in ("val", "test"):
            link = V3 / kind / split
            if not link.exists():
                os.symlink(V2 / kind / split, link)
    n_adult = 0
    for img in sorted((V2 / "images" / "train").iterdir()):
        lab = V2 / "labels" / "train" / f"{img.stem}.txt"
        copies = REPEAT if img.name.startswith("fa_") else 1
        n_adult += img.name.startswith("fa_")
        for k in range(copies):
            stem = img.stem if k == 0 else f"{img.stem}__dup{k}"
            dst = V3 / "images" / "train" / f"{stem}{img.suffix}"
            if not dst.exists():
                os.symlink(os.path.realpath(img), dst)
            (V3 / "labels" / "train" / f"{stem}.txt").write_text(lab.read_text() if lab.exists() else "")
    data = yaml.safe_load((V2 / "combined.yaml").read_text())
    data["path"] = str(V3)
    (V3 / "combined_v3.yaml").write_text(yaml.safe_dump(data))
    n = len(list((V3 / "images" / "train").iterdir()))
    print(f"adult training X-rays: {n_adult} x {REPEAT}; training images now {n}", flush=True)


if __name__ == "__main__":
    main()
