"""Everything the Gradio app needs, in one zip that unpacks into a Hugging Face Space (9 Nov 2026).

    python make_space_bundle.py     # -> reasoning/web/test_pack/ED_app_gradio_space.zip (gitignored)

In the zip (Space layout: app.py at the top):
    app.py, README.md (the Space settings), requirements.txt
    reasoning/src, reasoning/web/server.py, reasoning/app/test_backend.py      the code (as committed)
    vitals/src, vitals/models                                                  Caleb's vitals model
    vision/results, vision/*.py ...                                            image-model code and cut-offs
    reasoning/models/heads/*.joblib, reasoning/models/lora/<task, questions, numbers>, vision/cato/...  trained parts
    vision/.../*.pt and home/.torchxrayvision/...                              image models (app.py puts CheXpert in place)
Not in it: MedGemma itself (the Space downloads it with its HF_TOKEN), data, notebooks, outputs.
Keep the zip and the Space private: some models were trained on VinDr-CXR and RSNA data.
"""
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = REPO / "reasoning" / "web" / "test_pack" / "ED_app_gradio_space.zip"
sys.path.insert(0, str(REPO / "reasoning" / "src"))
import image_models as im                                      # noqa: E402  (the image-model paths)
from config import HEADS_DIR, LORA_DIR                         # noqa: E402
from image_reader import IMAGE_ADAPTER                         # noqa: E402
from pipeline import EXPLAINER, NUMBER_READER, QUESTIONER         # noqa: E402

EMB = "base4bit"                                               # the vectors the heads were trained on (EDPipeline default)

CODE = ["reasoning/src", "reasoning/web/server.py", "reasoning/app/test_backend.py", "vitals/src", "vitals/models",
        "vision"]
SKIP = ("/notebooks/", "/data/", "/outputs/", "/__pycache__/", ".ipynb")


def tracked(prefixes):
    """Files as committed in git under these paths (so the Space runs what is on GitHub)."""
    out = subprocess.run(["git", "ls-files", *prefixes], cwd=REPO, capture_output=True, text=True, check=True).stdout
    return [f for f in out.splitlines() if not any(s in f"/{f}" for s in SKIP)]


def model_files():
    heads = [HEADS_DIR / f"urgent__{EMB}__ktas_nhamcs.joblib", HEADS_DIR / f"level__{EMB}.joblib",
             HEADS_DIR / f"pathways__{EMB}.joblib"]
    adapters = [p for name in (EXPLAINER, QUESTIONER, NUMBER_READER) for p in (LORA_DIR / name / "adapter").glob("*")]
    image = [im.SORTER, im.FRACTURE, im.CHEST, im.ICH, im.ICH_GREY] + list(Path(IMAGE_ADAPTER).glob("*"))
    return heads + adapters + image


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(HERE / "app.py", "app.py")
        z.write(HERE / "README_SPACE.md", "README.md")
        z.write(HERE / "requirements.txt", "requirements.txt")
        for f in tracked(CODE):
            z.write(REPO / f, f)
        for f in model_files():
            if f.is_file():
                z.write(f, f.relative_to(REPO).as_posix())
                total += f.stat().st_size
        z.write(im.CHEX_FILE, "home/" + im.CHEX_FILE.relative_to(Path.home()).as_posix())
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.0f} MB; model files {total / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
