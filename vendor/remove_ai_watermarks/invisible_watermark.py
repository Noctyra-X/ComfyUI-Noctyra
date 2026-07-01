"""Detect open invisible watermarks embedded by the ``invisible-watermark``
(imwatermark) library -- used by Stable Diffusion, SDXL, and FLUX.

Unlike SynthID (proprietary, no local decoder), these are DWT-DCT watermarks
with a PUBLIC decoder and no secret key, so a fresh, un-re-encoded output can be
identified locally. The known fixed patterns were verified against upstream
source:

- **Stable Diffusion XL** -- diffusers ``StableDiffusionXLWatermarker``
  ``WATERMARK_MESSAGE`` (48-bit).
- **FLUX.2** -- ``black-forest-labs/flux2`` ``src/flux2/watermark.py`` (48-bit).
- **Stable Diffusion 1.x / 2.x** -- the library's default ``"StableDiffusionV1"``
  string (136-bit).

The watermark is fragile: it does NOT survive JPEG re-encoding or resizing
(verified -- gone after JPEG q90), so detection works only on pristine PNG
originals. Absence is never proof. Requires the optional ``invisible-watermark``
package (extra: ``detect``); ``detect_invisible_watermark`` returns None when it
is not installed.
"""

# imwatermark ships no type stubs (like cv2); its decoder returns are Unknown.
# Relax the untyped-library diagnostics for this thin wrapper module only.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)

# Known 48-bit ``bits`` watermarks (dwtDct, no key), name -> message integer.
_BITS_48: dict[str, int] = {
    "Stable Diffusion XL": 0b101100111110110010010000011110111011000110011110,
    "FLUX.2 (Black Forest Labs)": 0b001010101111111010000111100111001111010100101110,
}
# The invisible-watermark default string watermark (SD 1.x / 2.x).
_SD1_STRING = b"StableDiffusionV1"

# Decoded bits/bytes never match a 48-bit pattern by chance: random decode lands
# near 24/48, an exact embed at 48/48 (measured). 44 (<=4 bit errors) is a safe
# floor that tolerates light perturbation without risking a false positive.
_MATCH_48 = 44
_MATCH_SD1_FRAC = 0.92  # fraction of the 136 string bits that must match


def is_available() -> bool:
    """True if the optional imwatermark decoder is installed."""
    import importlib.util

    return importlib.util.find_spec("imwatermark") is not None


def _bits_match(value: int, ref: int, width: int = 48) -> int:
    """Number of matching bits between two ``width``-bit integers."""
    return width - bin(value ^ ref).count("1")


def _bytes_match_frac(a: bytes, b: bytes) -> float:
    """Fraction of matching bits between two equal-length byte strings."""
    if len(a) != len(b) or not a:
        return 0.0
    diff = sum(bin(x ^ y).count("1") for x, y in zip(a, b, strict=True))
    return 1.0 - diff / (8 * len(b))


def detect_invisible_watermark(image_path: Path) -> str | None:
    """Return the embedding scheme name if a known open watermark is decoded.

    Returns e.g. ``"Stable Diffusion XL"`` / ``"FLUX.2 (Black Forest Labs)"`` /
    ``"Stable Diffusion 1.x / 2.x"``, or None if none matches, the decoder is
    unavailable, or the image can't be read. Meaningful only on pristine
    (un-re-encoded) images.
    """
    if not is_available():
        return None
    from imwatermark import WatermarkDecoder

    from remove_ai_watermarks import image_io

    img = image_io.imread(image_path)
    if img is None:
        return None

    # 48-bit fixed-message watermarks (SDXL, FLUX.2).
    try:
        bits = WatermarkDecoder("bits", 48).decode(img, "dwtDct")
        value = 0
        for bit in bits:
            value = (value << 1) | (1 if bit else 0)
        for name, ref in _BITS_48.items():
            if _bits_match(value, ref) >= _MATCH_48:
                return name
    except Exception as exc:  # decode can fail on tiny images
        log.debug("48-bit watermark decode failed for %s: %s", image_path, exc)

    # 136-bit default string watermark (SD 1.x / 2.x).
    try:
        raw = cast("bytes", WatermarkDecoder("bytes", 8 * len(_SD1_STRING)).decode(img, "dwtDct"))
        if _bytes_match_frac(raw, _SD1_STRING) >= _MATCH_SD1_FRAC:
            return "Stable Diffusion 1.x / 2.x"
    except Exception as exc:
        log.debug("string watermark decode failed for %s: %s", image_path, exc)

    return None
