import math
import numpy as np

from .models import YOLO_MODELS
from .utils import find_filled_cells, split_img_4x4


# =========================
# Basic helpers
# =========================


def _run_model_for_target(image, target_num):
    if target_num == 1001:
        return YOLO_MODELS["crosswalk"](image, conf=0.4), 0

    return YOLO_MODELS["yolo"](image, conf=0.4), target_num


def _find_target_boxes(result, target_num):
    return [
        i for i, box in enumerate(result)
        if box["class_id"] == target_num
    ]

# =========================
# 4x4 captcha
# =========================

def detect_cells_4x4(image, target_num):
    if target_num < 1000:
        return _detect_4x4_with_seg(image, target_num)

    return _detect_4x4_with_boxes(image, target_num)


# =========================
# Segmentation based
# =========================

def _detect_4x4_with_seg(image, target_num):
    result = YOLO_MODELS["yolo-seg"](image, conf=0.4)
    target_boxes = _find_target_boxes(result, target_num)

    masks = [
        res["mask"] for res in result
    ]

    cells = []

    for idx in target_boxes:
        mask = masks[idx]
        splited_mask = split_img_4x4(mask)
        for i, section in enumerate(splited_mask):
            if np.any(section == 1):
                cells.append(i+1)


    return sorted(set(cells))


# =========================
# Box based
# =========================

def _detect_4x4_with_boxes(image, target_num):
    result, target_num = _run_model_for_target(image, target_num)
    boxes = [
        res["bbox"] for res in result
    ]

    target_boxes = _find_target_boxes(result, target_num)
    cells = []

    for idx in target_boxes:
        cells.extend(_box_to_4x4_cells(image, boxes[idx]))

    return sorted(set(cells))


def _box_to_4x4_cells(image, box):
    x1, y1, x4, y4 = map(int, box[:4])

    corners = [
        (x1, y1),
        (x4, y1),
        (x1, y4),
        (x4, y4),
    ]

    cell_size = image.shape[0] / 4
    max_x = max(p[0] for p in corners)
    max_y = max(p[1] for p in corners)

    cells = []

    for x, y in corners:
        row = math.floor(y / cell_size) + 1
        col = math.floor(x / cell_size) + 1

        if math.isclose(y % cell_size, 0) and math.isclose(y, max_y):
            row -= 1
        if math.isclose(x % cell_size, 0) and math.isclose(x, max_x):
            col -= 1

        row = max(1, min(4, row))
        col = max(1, min(4, col))

        cells.append((row - 1) * 4 + col)

    return find_filled_cells(cells)
