"""
detect.py
Runs YOLO11n (ONNX) person detection on a frame via ONNX Runtime.

Handles the standard Ultralytics ONNX export output shape (1, 84, 8400):
  - 4 box values (cx, cy, w, h) in the 640x640 model input space
  - 80 class confidence scores (COCO)
We keep only class 0 ("person"), threshold, then apply NMS.
"""

import cv2
import numpy as np
import onnxruntime as ort

import config


class Detector:
    def __init__(self, model_path=None):
        model_path = model_path or config.MODEL_PATH
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = config.INTRA_OP_THREADS
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]

        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = config.MODEL_INPUT

    # ---------- preprocessing ----------
    def _letterbox(self, frame):
        h, w = frame.shape[:2]
        scale = self.input_size / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        top = (self.input_size - nh) // 2
        left = (self.input_size - nw) // 2
        canvas[top:top + nh, left:left + nw] = resized

        return canvas, scale, left, top

    def _preprocess(self, frame):
        canvas, scale, pad_left, pad_top = self._letterbox(frame)
        img = canvas[:, :, ::-1].astype(np.float32) / 255.0   # BGR -> RGB, normalize
        img = img.transpose(2, 0, 1)[None, ...]                # HWC -> NCHW
        return np.ascontiguousarray(img), scale, pad_left, pad_top

    # ---------- inference ----------
    def detect(self, frame):
        """
        Returns a list of [x1, y1, x2, y2, confidence] in ORIGINAL frame coordinates,
        person-class only, after confidence filtering and NMS.
        """
        img, scale, pad_left, pad_top = self._preprocess(frame)
        outputs = self.session.run(None, {self.input_name: img})[0]  # (1, 84, 8400)

        preds = outputs[0].transpose(1, 0)   # -> (8400, 84)
        boxes_xywh = preds[:, :4]
        class_scores = preds[:, 4:]

        class_ids = np.argmax(class_scores, axis=1)
        confidences = np.max(class_scores, axis=1)

        person_mask = (class_ids == config.PERSON_CLASS_ID) & (
            confidences >= config.CONFIDENCE_THRESHOLD
        )
        boxes_xywh = boxes_xywh[person_mask]
        confidences = confidences[person_mask]

        if len(boxes_xywh) == 0:
            return []

        cx, cy, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        x1 = (x1 - pad_left) / scale
        y1 = (y1 - pad_top) / scale
        x2 = (x2 - pad_left) / scale
        y2 = (y2 - pad_top) / scale

        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        nms_boxes = boxes_xyxy.copy()
        nms_boxes[:, 2] -= nms_boxes[:, 0]
        nms_boxes[:, 3] -= nms_boxes[:, 1]

        keep = cv2.dnn.NMSBoxes(
            nms_boxes.tolist(),
            confidences.tolist(),
            config.CONFIDENCE_THRESHOLD,
            config.NMS_IOU_THRESHOLD,
        )
        if len(keep) == 0:
            return []
        keep = np.array(keep).flatten()

        detections = []
        fh, fw = frame.shape[:2]
        for i in keep:
            bx1, by1, bx2, by2 = boxes_xyxy[i]
            bx1 = max(0, min(fw - 1, bx1))
            by1 = max(0, min(fh - 1, by1))
            bx2 = max(0, min(fw - 1, bx2))
            by2 = max(0, min(fh - 1, by2))
            detections.append([float(bx1), float(by1), float(bx2), float(by2), float(confidences[i])])

        return detections
