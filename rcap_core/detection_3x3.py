from .models import YOLO_MODELS

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
# 3x3 captcha
# =========================

def detect_cells_3x3(image, target_num):
    result, target_num = _run_model_for_target(image, target_num)

    target_boxes = _find_target_boxes(result, target_num)
    boxes = [
        res["bbox"] for res in result
    ]

    cells = set()

    for idx in target_boxes:
        box = boxes[idx]
        xc = (box[0] + box[2]) / 2
        yc = (box[1] + box[3]) / 2

        col = int(xc / (image.shape[1] / 3))
        row = int(yc / (image.shape[0] / 3))

        cells.add(row * 3 + col + 1)

    return list(cells)
