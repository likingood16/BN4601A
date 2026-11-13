"""Uploaded pictures as 8-bit, in one place (group test 10 Nov, FEEDBACK_10NOV C1).

    from image_io import save_8bit
    save_8bit("case_files/images/ab12_wrist.png")     # rewrites the file only if it holds more than 8 bits a pixel

Real X-rays are often 16-bit (PIL mode I;16). `Image.open(path).convert("RGB")` does not rescale them: every value
above 255 becomes 255, so 95-98% of the picture turned white, and the overlays, MedGemma's description and the
checklist all worked on a white picture. Both apps call save_8bit right after saving an upload, so every reader
(the detectors, MedSAM, MedGemma, the browser) gets the same correct picture.
"""
import numpy as np
from PIL import Image

DEEP_MODES = ("I", "F")            # I, I;16, I;16B, I;16L, F: one channel with more than 8 bits


def to_8bit(img):
    """A PIL picture with more than 8 bits a pixel -> 8-bit grey, stretched between its own darkest and brightest
    pixel. Any other picture is returned as it is."""
    if not img.mode.startswith(DEEP_MODES):
        return img
    a = np.asarray(img).astype(np.float64)
    lo, hi = float(a.min()), float(a.max())
    a = (a - lo) / (hi - lo) * 255 if hi > lo else np.zeros_like(a)
    return Image.fromarray(a.round().astype(np.uint8), "L")


def save_8bit(path):
    """Rewrite the file as 8-bit if it is a deep (e.g. 16-bit) picture. True if it was rewritten. A file PIL cannot
    open (e.g. DICOM) is left alone: the image step reports that itself."""
    try:
        with Image.open(path) as img:
            if not img.mode.startswith(DEEP_MODES):
                return False
            small = to_8bit(img)
    except Exception:
        return False
    small.save(path)
    return True
