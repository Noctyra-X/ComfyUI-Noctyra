"""Text-region protection for diffusion-based watermark removal.

SDXL img2img (the ``invisible`` pipeline) regenerates every pixel, so small text
and CJK glyphs get deformed at the strengths that defeat SynthID (issue #21).
This module detects text regions and builds a per-pixel "change map" for
Differential Diffusion: the background is regenerated normally while text
regions are largely preserved, so glyphs survive the watermark-removal pass.

Detection uses only OpenCV's DNN module (no torch): the PP-OCRv3 text detector
is a ~2.4 MB ONNX model (Apache-2.0, from opencv_zoo) that is CJK-native and
returns rotated quadrilaterals. The model is downloaded and cached on first use;
it is never bundled in this repo.

Change-map polarity (verified empirically against the differential pipeline):
white (1.0) = PRESERVE the original pixels, black (0.0) = MAXIMUM change. So the
map is black everywhere except the text polygons, which are painted toward
white. ``preserve`` stays below a hard 1.0 freeze by default: SynthID is
designed to survive cropping, so totally freezing text pixels would leave the
watermark intact there. A high-but-partial preserve still scrubs lightly.
"""

# cv2 ships no type stubs; mirror the pragma used by the other cv2-using modules.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportCallIssue=false, reportArgumentType=false, reportReturnType=false

from __future__ import annotations

import logging
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# PP-OCRv3 Chinese text detector (DB head), opencv_zoo, Apache-2.0.
_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/text_detection_ppocr/text_detection_cn_ppocrv3_2023may.onnx"
)
_MODEL_FILENAME = "text_detection_cn_ppocrv3_2023may.onnx"

# DB detector input: long side scaled to this, rounded to a multiple of 32.
_DET_INPUT_LONG_SIDE = 736
# ImageNet mean (x255) and 1/255 scale -- the normalization PP-OCRv3 expects.
_DET_MEAN = (0.485 * 255, 0.456 * 255, 0.406 * 255)
_DET_SCALE = 1 / 255.0


def is_available() -> bool:
    """True when OpenCV's DNN text-detection model is importable."""
    try:
        import cv2

        return hasattr(cv2.dnn, "TextDetectionModel_DB")
    except ImportError:
        return False


def _cache_dir() -> Path:
    """Local cache directory for the detector model (created on demand)."""
    cache = Path.home() / ".cache" / "remove-ai-watermarks"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _model_path() -> Path:
    """Return the cached detector path, downloading it on first use."""
    target = _cache_dir() / _MODEL_FILENAME
    if target.exists() and target.stat().st_size > 0:
        return target
    logger.info("Downloading PP-OCRv3 text detector (~2.4 MB) to %s", target)
    # Download to a temp file in the same dir, then atomically rename so a
    # partial download never leaves a corrupt model cached.
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), suffix=".onnx.part")
    tmp_path = Path(tmp_name)
    try:
        os.close(fd)
        with urllib.request.urlopen(_MODEL_URL) as resp:  # noqa: S310 (trusted GitHub URL)
            tmp_path.write_bytes(resp.read())
        tmp_path.replace(target)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return target


def build_change_map(
    boxes: list[NDArray[Any]],
    height: int,
    width: int,
    preserve: float = 0.9,
    feather: int = 15,
) -> NDArray[Any]:
    """Build a Differential-Diffusion change map from text polygons.

    Args:
        boxes: Text-region polygons as arrays of (x, y) vertices.
        height: Output map height in pixels.
        width: Output map width in pixels.
        preserve: Map value painted inside text polygons (0..1). White (1.0)
            fully preserves the original pixels; the default 0.9 preserves
            strongly while still letting a light scrub through.
        feather: Gaussian-blur kernel size for soft polygon edges (forced odd).

    Returns:
        Float32 HxW array in [0, 1]: ~0 in the background (full change),
        ``preserve`` inside text regions, blended at the edges.
    """
    import cv2
    import numpy as np

    change_map = np.zeros((height, width), np.float32)
    if boxes:
        polys = [np.asarray(b, np.int32) for b in boxes]
        cv2.fillPoly(change_map, polys, float(preserve))
    if feather > 0:
        if feather % 2 == 0:
            feather += 1
        change_map = cv2.GaussianBlur(change_map, (feather, feather), 0)
        # GaussianBlur can overshoot the painted value by a float epsilon; keep
        # the contract that the map stays a valid [0, 1] change map.
        np.clip(change_map, 0.0, 1.0, out=change_map)
    return change_map


class TextProtector:
    """Detect text regions with PP-OCRv3 for diffusion change-map protection."""

    def __init__(
        self,
        binary_threshold: float = 0.3,
        polygon_threshold: float = 0.5,
        max_candidates: int = 200,
        unclip_ratio: float = 2.0,
    ) -> None:
        import cv2

        self._detector = cv2.dnn.TextDetectionModel_DB(str(_model_path()))
        self._detector.setBinaryThreshold(binary_threshold)
        self._detector.setPolygonThreshold(polygon_threshold)
        self._detector.setMaxCandidates(max_candidates)
        self._detector.setUnclipRatio(unclip_ratio)

    def detect_text_boxes(self, bgr_image: NDArray[Any]) -> list[NDArray[Any]]:
        """Detect text regions, returning a list of rotated quad polygons.

        Args:
            bgr_image: Image as an HxWx3 BGR uint8 array (OpenCV convention).

        Returns:
            One array of four (x, y) vertices per detected text region.
        """
        height, width = bgr_image.shape[:2]
        scale = _DET_INPUT_LONG_SIDE / max(height, width)
        in_w = max((round(width * scale) // 32) * 32, 32)
        in_h = max((round(height * scale) // 32) * 32, 32)
        self._detector.setInputParams(
            scale=_DET_SCALE,
            size=(in_w, in_h),
            mean=_DET_MEAN,
            swapRB=True,
        )
        boxes, _confidences = self._detector.detect(bgr_image)
        return list(boxes)
