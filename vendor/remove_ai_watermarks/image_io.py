"""Unicode-safe cv2 image IO (issue #17).

``cv2.imread`` / ``cv2.imwrite`` pass the path to the platform C runtime, which
on Windows uses the narrow (ANSI) code-page API and therefore fails on paths
containing non-ASCII characters (Chinese, Cyrillic, ...). The symptom is a
``can't open/read file`` warning and a ``None`` decode even though the file
exists.

These wrappers route through numpy buffers instead: ``np.fromfile`` /
``ndarray.tofile`` open the path in Python (full Unicode), and
``cv2.imdecode`` / ``cv2.imencode`` do the codec work. The decoded/encoded
bytes are byte-for-byte identical to ``imread`` / ``imwrite``. On macOS/Linux
cv2 already accepts UTF-8 paths, so the wrappers are behavior-neutral there.

cv2/numpy are imported lazily inside the functions so importing this module
stays cheap in a bare environment (matching the rest of the package).
"""

# cv2 ships no type stubs; mirror the pragma used by the other cv2-using modules.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy.typing import NDArray


def imread(path: str | Path, flags: int | None = None) -> NDArray[Any] | None:
    """Unicode-safe ``cv2.imread``.

    ``flags`` defaults to ``cv2.IMREAD_COLOR`` (same as ``cv2.imread``). Returns
    ``None`` when the file is missing or cannot be decoded, matching
    ``cv2.imread`` semantics so existing ``if img is None`` checks keep working.
    """
    import cv2
    import numpy as np

    if flags is None:
        flags = cv2.IMREAD_COLOR
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite(path: str | Path, img: NDArray[Any]) -> bool:
    """Unicode-safe ``cv2.imwrite``.

    The output format is taken from the path extension (e.g. ``.png``), exactly
    like ``cv2.imwrite``. Returns ``True`` on success, ``False`` if the codec
    rejects the image.
    """
    import cv2

    ext = Path(path).suffix or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(str(path))
    return True
