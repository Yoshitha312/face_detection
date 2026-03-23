"""
core/face_detector.py
--------------------------------------------------------------
YOLOv8 face detector wrapper.

• Uses yolov8n-face.pt (face-tuned model) by default.
• Falls back to yolov8n.pt (general) if the face model is unavailable.
• Returns a list of (x1, y1, x2, y2, confidence) tuples per frame.
"""

import logging
import os
from typing import List, Tuple

import numpy as np

logger = logging.getLogger("face_tracker.detector")

Detection = Tuple[int, int, int, int, float]


class FaceDetector:
    def __init__(self,
                 model_path: str  = "yolov8n-face.pt",
                 confidence: float = 0.50,
                 iou_threshold: float = 0.45,
                 min_face_size: int   = 30):
        self.confidence    = confidence
        self.iou_threshold = iou_threshold
        self.min_face_size = min_face_size
        self.model         = self._load(model_path)
        logger.info(
            f"[Detector] Ready  conf={confidence}  iou={iou_threshold}"
            f"  min_px={min_face_size}"
        )

    def _load(self, model_path: str):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise RuntimeError("Install ultralytics:  pip install ultralytics")

        if os.path.exists(model_path):
            logger.info(f"[Detector] Loading local: {model_path}")
            return YOLO(model_path)

        for name in ["yolov8n-face.pt", "yolov8n.pt"]:
            try:
                logger.info(f"[Detector] Downloading {name} …")
                return YOLO(name)
            except Exception as e:
                logger.warning(f"[Detector] {name} failed: {e}")

        raise RuntimeError("Could not load any YOLO model.")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Return list of (x1, y1, x2, y2, conf) for every face found."""
        if frame is None or frame.size == 0:
            return []

        results = self.model(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            verbose=False
        )

        dets: List[Detection] = []
        h, w = frame.shape[:2]

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if (x2 - x1) < self.min_face_size or (y2 - y1) < self.min_face_size:
                    continue
                dets.append((x1, y1, x2, y2, conf))

        return dets
