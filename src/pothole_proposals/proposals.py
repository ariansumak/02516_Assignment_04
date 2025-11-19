from __future__ import annotations

import logging
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image

LOGGER = logging.getLogger(__name__)

try:
    import cv2

    _HAVE_XIMGPROC = hasattr(cv2, "ximgproc")
except ImportError:  # pragma: no cover - OpenCV may be missing in the sandbox
    cv2 = None  # type: ignore
    _HAVE_XIMGPROC = False

try:
    import selectivesearch
except ImportError:  # pragma: no cover - optional dependency
    selectivesearch = None  # type: ignore


Box = Tuple[int, int, int, int]


def _prepare_image(image_path) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def generate_selective_search_proposals(
    image_path,
    *,
    mode: str = "fast",
    max_proposals: int = 2000,
    min_size: int = 20,
) -> List[Box]:
    """
    Run Selective Search on an image. The function will use OpenCV's implementation when
    available and automatically fall back to the pure Python package otherwise.
    """

    image = _prepare_image(image_path)
    if _HAVE_XIMGPROC:
        return _generate_with_opencv(image, mode, max_proposals, min_size)

    if selectivesearch is None:
        raise RuntimeError(
            "Neither OpenCV (with ximgproc) nor the 'selectivesearch' package is available"
        )

    return _generate_with_python(image, max_proposals, min_size)


def _generate_with_opencv(
    image: np.ndarray,
    mode: str,
    max_proposals: int,
    min_size: int,
) -> List[Box]:
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ss.setBaseImage(bgr)
    if mode == "quality":
        ss.switchToSelectiveSearchQuality()
    else:
        ss.switchToSelectiveSearchFast()

    rects = ss.process()
    boxes: List[Box] = []
    for (x, y, w, h) in rects:
        if w * h < min_size * min_size:
            continue
        boxes.append((int(x), int(y), int(x + w), int(y + h)))
        if len(boxes) >= max_proposals:
            break
    return boxes


def _generate_with_python(
    image: np.ndarray,
    max_proposals: int,
    min_size: int,
) -> List[Box]:
    # selectivesearch expects floats in [0, 1]
    img_float = image.astype(np.float32) / 255.0
    _, regions = selectivesearch.selective_search(
        img_float, scale=150, sigma=0.8, min_size=min_size
    )
    seen = set()
    boxes: List[Box] = []
    for region in regions:
        rect = region["rect"]
        if rect in seen:
            continue
        seen.add(rect)
        x, y, w, h = rect
        boxes.append((int(x), int(y), int(x + w), int(y + h)))
        if len(boxes) >= max_proposals:
            break
    return boxes
