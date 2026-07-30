import numpy as np
from PIL import Image


def find_filled_cells(corners):
    """Find all cells filled by the given corner numbers in a 4x4 grid.

    It takes corner numbers (like [1,4,13,16]) and returns all cells inside the rectangle they form.
    Simple: find min/max rows/cols, then fill the square.
    """

    # Get row and col for each corner (0-based)
    positions = [((v - 1) // 4, (v - 1) % 4) for v in corners]
    rows = [pos[0] for pos in positions]
    cols = [pos[1] for pos in positions]

    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)

    filled = []
    for r in range(min_row, max_row + 1):
        for c in range(min_col, max_col + 1):
            cell_num = (r * 4) + c + 1
            filled.append(cell_num)

    return sorted(set(filled))  # Remove duplicates if any


def prepare_input(img, size):
    orig_w, orig_h = img.size

    r = min(size / orig_w, size / orig_h)

    new_w = int(round(orig_w * r))
    new_h = int(round(orig_h * r))

    img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    pad_w = size - new_w
    pad_h = size - new_h

    left = int(round(pad_w / 2 - 0.1))
    right = int(round(pad_w / 2 + 0.1))
    top = int(round(pad_h / 2 - 0.1))
    bottom = int(round(pad_h / 2 + 0.1))

    canvas = np.full(
        (size, size, 3),
        114,
        dtype=np.uint8,
    )

    canvas[
        top:top + new_h,
        left:left + new_w
    ] = np.asarray(img)

    inp = canvas.astype(np.float32) / 255.0
    inp = inp.transpose(2, 0, 1)[None]

    return inp, r, (left, top), (orig_w, orig_h)


def post_process(outp, conf_thres=0.7, iou_thres=0.5):
    preds = np.squeeze(outp[0]).T

    # Remove low-conf preds
    scores = np.max(preds[:, 4:], axis=1)
    keep = scores > conf_thres

    # get boxes, scores and class_ids
    preds = preds[keep, :]
    boxes = preds[:, :4]
    boxes = xywh2xyxy(boxes)

    scores = np.max(preds[:, 4:], axis=1)
    
    class_ids = np.argmax(preds[:, 4:], axis=1)

    # do multiclass nms
    indices = multiclass_nms(boxes, scores, class_ids, iou_thres=iou_thres)
    
    return boxes[indices], scores[indices], class_ids[indices]


def post_process_e2e(outp, conf_thres=0.5):
    """Post-process an end-to-end / NMS-free ONNX output (e.g. YOLO26 default export).

    Output shape is (1, num_dets, 6) -> [x1, y1, x2, y2, confidence, class_id],
    already decoded to xyxy (in the padded/resized input's pixel space) and
    already NMS'd by the model itself. No argmax over classes, no NMS needed here -
    just a confidence filter.
    """
    preds = np.squeeze(outp[0])  # (num_dets, 6)

    scores = preds[:, 4]
    keep = scores > conf_thres
    preds = preds[keep]

    boxes = preds[:, :4]
    scores = preds[:, 4]
    class_ids = preds[:, 5].astype(int)

    return boxes, scores, class_ids


def _split_seg_outputs(outp):
    """Segmentation exports return two ONNX outputs, in no guaranteed order:
    a 3-D detections tensor and a 4-D prototype-mask tensor. Identify by ndim.
    """
    det_out, proto_out = None, None
    for o in outp:
        if o.ndim == 4:
            proto_out = o
        else:
            det_out = o
    if det_out is None or proto_out is None:
        raise ValueError(
            f"Expected 2 outputs (detections + prototypes) for a segmentation model, "
            f"got shapes {[o.shape for o in outp]}. Is this actually a -seg ONNX export?"
        )
    return det_out, proto_out


def post_process_seg(outp, conf_thres=0.7, iou_thres=0.5):
    """Traditional (non end-to-end) YOLO segmentation output, e.g. YOLOv8-seg,
    or YOLO26-seg exported with end2end=False.

    outp: [detections (1, 4+nc+nm, num_anchors), prototypes (1, nm, mh, mw)]
    nc (num classes) is inferred from nm (read off the prototype tensor) so this
    works regardless of how many classes the model has.
    """
    det_out, proto_out = _split_seg_outputs(outp)
    proto = np.squeeze(proto_out)  # (nm, mh, mw)
    nm = proto.shape[0]

    preds = np.squeeze(det_out).T  # (num_anchors, 4+nc+nm)
    nc = preds.shape[1] - 4 - nm

    cls_part = preds[:, 4:4 + nc]
    mask_part = preds[:, 4 + nc:4 + nc + nm]

    scores = np.max(cls_part, axis=1)
    keep = scores > conf_thres

    boxes = xywh2xyxy(preds[keep, :4])
    scores = scores[keep]
    class_ids = np.argmax(cls_part[keep], axis=1)
    mask_coeffs = mask_part[keep]

    indices = multiclass_nms(boxes, scores, class_ids, iou_thres=iou_thres)

    return boxes[indices], scores[indices], class_ids[indices], mask_coeffs[indices], proto


def post_process_seg_e2e(outp, conf_thres=0.5):
    """End-to-end / NMS-free YOLO26-seg output (the default -seg export mode).

    outp: [detections (1, num_dets, 6+nm), prototypes (1, nm, mh, mw)]
    Each detection row is [x1, y1, x2, y2, confidence, class_id, *mask_coeffs],
    already decoded and NMS'd - just a confidence filter needed here.
    """
    det_out, proto_out = _split_seg_outputs(outp)
    proto = np.squeeze(proto_out)  # (nm, mh, mw)

    preds = np.squeeze(det_out)  # (num_dets, 6+nm)
    scores = preds[:, 4]
    keep = scores > conf_thres
    preds = preds[keep]

    boxes = preds[:, :4]
    scores = preds[:, 4]
    class_ids = preds[:, 5].astype(int)
    mask_coeffs = preds[:, 6:]

    return boxes, scores, class_ids, mask_coeffs, proto


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def process_masks(mask_coeffs, proto, boxes, ratio, pad, orig_size, input_size=640):
    """Turn per-detection mask coefficients + the shared prototype tensor into
    full-resolution masks, one per detection, cropped to that detection's box.

    Returned as uint8 arrays with values 0/1 (matching ultralytics' own
    `results.masks.data` convention), not True/False booleans.

    `boxes` must already be scaled to the ORIGINAL image (i.e. passed through
    scale_boxes() first) - same original image these masks will be sized to.
    """
    if len(mask_coeffs) == 0:
        return []

    nm, mh, mw = proto.shape
    masks = sigmoid(mask_coeffs @ proto.reshape(nm, -1)).reshape(-1, mh, mw)

    ow, oh = orig_size
    left, top = pad
    new_w = int(round(ow * ratio))
    new_h = int(round(oh * ratio))

    out_masks = []
    for i, m in enumerate(masks):
        # prototypes are at a lower resolution than the model input; upsample first
        m_full = np.asarray(
            Image.fromarray((m * 255).astype(np.uint8)).resize((input_size, input_size), Image.BILINEAR)
        )
        # undo the letterbox padding, then resize to the original image size
        m_crop = m_full[top:top + new_h, left:left + new_w]
        m_resized = np.asarray(Image.fromarray(m_crop).resize((ow, oh), Image.BILINEAR))

        x1, y1, x2, y2 = [int(round(v)) for v in boxes[i]]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, ow), min(y2, oh)

        mask = np.zeros((oh, ow), dtype=np.uint8)
        mask[y1:y2, x1:x2] = (m_resized[y1:y2, x1:x2] > 127).astype(np.uint8)
        out_masks.append(mask)

    return out_masks


def parse_detections_w_masks(boxes, scores, class_ids, masks):
    detections = []
    for box, score, class_id, mask in zip(boxes, scores, class_ids, masks):
        detections.append({
            'bbox': [int(b) for b in box],
            'score': float(round(score, 3)),
            'class_id': int(class_id),
            'mask': mask,  # uint8 array (values 0/1), shape == original image (H, W)
        })
    return detections


def xywh2xyxy(boxes):
    new_boxes = np.copy(boxes)
    new_boxes[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
    new_boxes[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
    new_boxes[..., 2] = boxes[..., 0] + boxes[..., 2] / 2
    new_boxes[..., 3] = boxes[..., 1] + boxes[..., 3] / 2
    
    return new_boxes


def multiclass_nms(boxes, scores, class_ids, iou_thres=0.5):
    unique_ids = np.unique(class_ids)

    keep_boxes = []
    for class_id in unique_ids:
        class_indices = np.where(class_ids == class_id)[0]
        class_boxes = boxes[class_indices,:]
        class_scores = scores[class_indices]

        class_keep_boxes = nms(class_boxes, class_scores, iou_thres=iou_thres)
        keep_boxes.extend(class_indices[class_keep_boxes])

    return keep_boxes


def nms(boxes, scores, iou_thres=0.5):

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_thres)[0]
        order = order[inds + 1]

    return keep 


def scale_boxes(boxes, ratio, pad, orig_size):
    left, top = pad
    ow, oh = orig_size

    boxes = boxes.copy()

    boxes[:, [0, 2]] -= left
    boxes[:, [1, 3]] -= top

    boxes /= ratio

    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, ow)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, oh)

    return boxes


def parse_detections(boxes, scores, class_ids):
    detections = []
    for box, score, class_id in zip(boxes, scores, class_ids):
        detections.append({
            'bbox': [int(b) for b in box],
            'score': float(round(score, 3)),
            'class_id': int(class_id)
        })
    return detections


def split_img_4x4(img):
    def box_and_gap(size, n):
        box = size // n
        while (size - box * n) % (n - 1) != 0:
            box -= 1
        gap = (size - box * n) // (n - 1)
        return box, gap

    h, w = img.shape
    n = 4
    bh, gap_h = box_and_gap(h, n)
    bw, gap_w = box_and_gap(w, n)

    cells = []
    for i in range(n):
        for j in range(n):
            y = i * (bh + gap_h)
            x = j * (bw + gap_w)
            cells.append(img[y:y+bh, x:x+bw])

    return cells