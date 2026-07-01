"""Detect Adobe TrustMark invisible watermarks.

TrustMark (github.com/adobe/trustmark, MIT) is the open, keyless image watermark
behind Adobe "Durable Content Credentials": when a C2PA manifest is stripped, a
TrustMark soft binding can still re-link the asset to its manifest in a
repository. Unlike SynthID it has a PUBLIC decoder with no secret key, so a
TrustMark-stamped image can be identified locally. Adobe's shipping products use
Variant P (the ``com.adobe.trustmark.P`` soft-binding ``alg``); this wrapper
loads that model.

Optional dependency (extra: ``trustmark``); the model weights download on first
use. ``detect_trustmark`` returns None when the package is absent. This detects
provenance (Adobe Content Credentials), NOT AI generation as such -- TrustMark
also marks human-authored content -- so callers should treat it as a watermark
signal, not proof of AI origin.
"""

# trustmark ships no type stubs; relax untyped-library diagnostics for this thin
# wrapper module only.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingImports=false

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)

# Adobe ships Variant P in production (com.adobe.trustmark.P).
_MODEL_TYPE = "P"
# Lazily constructed singleton -- model load + first-use download is expensive.
_tm: Any = None


def is_available() -> bool:
    """True if the optional ``trustmark`` package is installed."""
    import importlib.util

    return importlib.util.find_spec("trustmark") is not None


def _decoder() -> Any:
    global _tm
    if _tm is None:
        from trustmark import TrustMark

        _tm = TrustMark(verbose=False, model_type=_MODEL_TYPE)
    return _tm


def detect_trustmark(image_path: Path) -> str | None:
    """Return a TrustMark scheme note if a TrustMark watermark is decoded, else None.

    Returns e.g. ``"Adobe TrustMark (variant P, schema 0)"`` when the decoder
    reports the watermark present, or None if it is absent, the optional
    ``trustmark`` package is not installed, or the image cannot be read/decoded.
    """
    if not is_available():
        return None
    try:
        from PIL import Image

        with Image.open(image_path) as img:
            cover = img.convert("RGB")
        _wm_secret, wm_present, wm_schema = _decoder().decode(cover)
    except Exception as exc:  # model download / decode failure / unreadable image
        log.debug("TrustMark decode failed for %s: %s", image_path, exc)
        return None
    return f"Adobe TrustMark (variant {_MODEL_TYPE}, schema {wm_schema})" if wm_present else None
