"""Zero-asset pixel-art Le Chonk: an orange tabby cat drawn procedurally.

No image files in the repo -- the cat is a tiny character grid rendered
with PIL, so the whole project stays a clean pip-install with no assets.
"""

from __future__ import annotations

from PIL import Image

CAT_PALETTE = {
    "o": (168, 88, 32),    # dark outline
    "G": (244, 157, 55),   # orange body
    "g": (224, 131, 39),   # darker stripes
    "B": (43, 43, 43),     # eyes
    "P": (232, 106, 146),  # pink nose
}

CAT_PIXELS = [
    "..oo....oo..",
    ".oGGo..oGGo.",
    ".oGGGooGGGo.",
    ".oGGGGGGGGo.",
    "oGGGGGGGGGGo",
    "oGGBGGGGBGGo",
    "oGGGGGGGGGGo",
    ".oGGGPPGGGo.",
    ".oGGGGGGGGo.",
    "oGGGGGGGGGGo",
    "ogGGgGGgGGgo",
    "oooooooooooo",
]

_CAT_CACHE: dict[int, Image.Image] = {}


def cat_image(scale: int = 4) -> Image.Image:
    """Render the pixel cat as an RGBA image with a transparent background."""
    if scale in _CAT_CACHE:
        return _CAT_CACHE[scale]
    w = max(len(row) for row in CAT_PIXELS)
    h = len(CAT_PIXELS)
    img = Image.new("RGBA", (w * scale, h * scale), (0, 0, 0, 0))
    px = img.load()
    for y, row in enumerate(CAT_PIXELS):
        for x, ch in enumerate(row):
            color = CAT_PALETTE.get(ch)
            if color is None:
                continue
            block = color + (255,)
            for dy in range(scale):
                for dx in range(scale):
                    px[x * scale + dx, y * scale + dy] = block
    _CAT_CACHE[scale] = img
    return img
