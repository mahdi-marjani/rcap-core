import onnxruntime
import numpy as np
import utils


class _YOLOBase:
    """Shared ONNX Runtime session setup, common to all YOLO ONNX variants."""

    def __init__(self, model_path, device="cpu", half=False):
        if device == 'cpu':
            providers = ['CPUExecutionProvider']
        elif device == 'gpu':
            providers = ['CUDAExecutionProvider']
        else:
            assert False, f'Device {device} is not available.'

        self.session = onnxruntime.InferenceSession(model_path, providers=providers)
        self.half = half

    def inference(self, input_tensor):
        if self.half:  # convert to float16
            input_tensor = np.float16(input_tensor)

        input_name = self.session.get_inputs()[0].name
        # Fetch ALL outputs, not just the first: detection-only models have one
        # output, but segmentation models have a second (prototype mask) output
        # that would otherwise be silently dropped.
        output_names = [o.name for o in self.session.get_outputs()]
        outputs = self.session.run(
            output_names,
            {input_name: input_tensor},
        )

        if self.half:  # convert back to float32
            outputs = [np.float32(o) for o in outputs]
        return outputs


class YOLOv8(_YOLOBase):
    """Traditional (non end-to-end) YOLO ONNX exports.

    Output shape: (1, 4+nc, num_anchors) e.g. (1, 84, 8400) for 80-class COCO.
    Needs per-class argmax over the class scores + NMS to get final detections.
    Covers YOLOv8/v11-style exports, and YOLO26 exported with end2end=False.
    """

    def __call__(self, img, size=640, conf=0.25, iou_thres=0.7):
        # prepare input
        inp, ratio, pad, orig_size = utils.prepare_input(img, size)

        # Perform inference on the image
        outp = self.inference(inp)

        # post-process (argmax over classes + multiclass NMS)
        boxes, scores, class_ids = utils.post_process(outp, conf_thres=conf, iou_thres=iou_thres)

        # resize boxes back to original image
        boxes = utils.scale_boxes(boxes, ratio, pad, orig_size)

        # parse detections
        detections = utils.parse_detections(boxes, scores, class_ids)

        return detections


class YOLOv8Seg(_YOLOBase):
    """Traditional (non end-to-end) YOLO segmentation ONNX exports, e.g. YOLOv8-seg,
    or YOLO26-seg exported with end2end=False.

    Two ONNX outputs: detections (1, 4+nc+nm, num_anchors) and prototypes (1, nm, mh, mw).
    Each detection dict includes a 'mask' key: a boolean array the size of the
    original image, cropped to that detection's own box.
    """

    def __call__(self, img, size=640, conf=0.25, iou_thres=0.7):
        inp, ratio, pad, orig_size = utils.prepare_input(img, size)
        outp = self.inference(inp)

        boxes, scores, class_ids, mask_coeffs, proto = utils.post_process_seg(
            outp, conf_thres=conf, iou_thres=iou_thres
        )

        boxes = utils.scale_boxes(boxes, ratio, pad, orig_size)
        masks = utils.process_masks(mask_coeffs, proto, boxes, ratio, pad, orig_size, input_size=size)

        return utils.parse_detections_w_masks(boxes, scores, class_ids, masks)


class YOLOv26Seg(_YOLOBase):
    """End-to-end / NMS-free YOLO26-seg ONNX exports (the default -seg export mode).

    Two ONNX outputs: detections (1, num_dets, 6+nm) and prototypes (1, nm, mh, mw).
    Each detection dict includes a 'mask' key: a boolean array the size of the
    original image, cropped to that detection's own box.
    """

    def __call__(self, img, size=640, conf=0.25):
        inp, ratio, pad, orig_size = utils.prepare_input(img, size)
        outp = self.inference(inp)

        boxes, scores, class_ids, mask_coeffs, proto = utils.post_process_seg_e2e(
            outp, conf_thres=conf
        )

        boxes = utils.scale_boxes(boxes, ratio, pad, orig_size)
        masks = utils.process_masks(mask_coeffs, proto, boxes, ratio, pad, orig_size, input_size=size)

        return utils.parse_detections_w_masks(boxes, scores, class_ids, masks)


class YOLOv26(_YOLOBase):
    """End-to-end / NMS-free YOLO26 ONNX exports (the default export mode for YOLO26).

    Output shape: (1, num_dets, 6) -> [x1, y1, x2, y2, confidence, class_id],
    already decoded to xyxy and already NMS'd by the model itself. No per-class
    argmax and no NMS needed here - just a confidence filter.
    """

    def __call__(self, img, size=640, conf=0.25):
        # prepare input
        inp, ratio, pad, orig_size = utils.prepare_input(img, size)

        # Perform inference on the image
        outp = self.inference(inp)

        # post-process (confidence filter only, already NMS'd)
        boxes, scores, class_ids = utils.post_process_e2e(outp, conf_thres=conf)

        # resize boxes back to original image
        boxes = utils.scale_boxes(boxes, ratio, pad, orig_size)

        # parse detections
        detections = utils.parse_detections(boxes, scores, class_ids)

        return detections