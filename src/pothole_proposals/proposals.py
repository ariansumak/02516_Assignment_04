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

    if selectivesearch is not None:
        return _generate_with_python(image, max_proposals, min_size)

    LOGGER.warning(
        "Selective Search dependencies missing – using sliding-window fallback proposals"
    )
    return _generate_with_sliding_windows(image, max_proposals, min_size)


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


def _generate_with_sliding_windows(
    image: np.ndarray,
    max_proposals: int,
    min_size: int,
) -> List[Box]:
    """
    Dependency-free fallback that enumerates sliding-window boxes at multiple scales.

    While much simpler than Selective Search, this still provides deterministic coverage
    so the rest of the pipeline (evaluation, labelling) can run without extra packages.
    """

    height, width = image.shape[:2]
    if height == 0 or width == 0:
        return []

    min_size = max(8, int(min_size))
    short_edge = min(height, width)

    # Build a list of window sizes anchored to the dataset resolution.
    base_sizes = [min_size]
    while base_sizes[-1] < short_edge:
        base_sizes.append(int(base_sizes[-1] * 1.6))

    aspect_ratios = [
        (1.0, 1.0),
        (1.5, 1.0),
        (1.0, 1.5),
        (2.0, 1.0),
        (1.0, 2.0),
    ]

    boxes: List[Box] = []
    for size in base_sizes:
        for ratio_w, ratio_h in aspect_ratios:
            win_w = int(size * ratio_w)
            win_h = int(size * ratio_h)
            if win_w < min_size or win_h < min_size:
                continue
            if win_w > width or win_h > height:
                continue

            stride_x = max(4, win_w // 4)
            stride_y = max(4, win_h // 4)

            for y in range(0, height - win_h + 1, stride_y):
                for x in range(0, width - win_w + 1, stride_x):
                    boxes.append((x, y, x + win_w, y + win_h))
                    if len(boxes) >= max_proposals:
                        return boxes

    return boxes
