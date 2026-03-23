"""
core/face_recognizer.py
--------------------------------------------------------------
InsightFace ArcFace embedding and cosine-similarity matcher.

Why ArcFace over face_recognition?
  • 99.83 % LFW accuracy vs ~98 %
  • ONNX-runtime: fast on CPU and GPU
  • Robust to lighting, pose, age variation
"""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("face_tracker.recognizer")


class FaceRecognizer:
    def __init__(self,
                 model_name: str   = "buffalo_l",
                 similarity_threshold: float = 0.45,
                 device: str       = "cpu"):
        self.threshold = similarity_threshold
        self.app       = self._load(model_name, device)
        logger.info(
            f"[Recognizer] Ready  model={model_name}"
            f"  threshold={similarity_threshold}  device={device}"
        )

    def _load(self, model_name: str, device: str):
        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            raise RuntimeError(
                "Install insightface:  pip install insightface onnxruntime"
            )
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda" else ["CPUExecutionProvider"]
        )
        ctx_id = 0 if device == "cuda" else -1
        app = FaceAnalysis(name=model_name, providers=providers)
        app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        return app

    # ── embedding extraction ─────────────────────────────────────────────
    def get_embedding(self, frame: np.ndarray,
                      bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Return L2-normalised 512-dim ArcFace embedding for the face at bbox,
        or None if extraction fails.
        """
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        emb = self._from_crop(crop)
        if emb is None:
            emb = self._from_full(frame, bbox)
        return emb

    def _from_crop(self, crop: np.ndarray) -> Optional[np.ndarray]:
        try:
            h, w = crop.shape[:2]
            if h < 112 or w < 112:
                scale = max(112 / h, 112 / w)
                crop  = cv2.resize(crop, (int(w * scale), int(h * scale)))
            faces = self.app.get(crop)
            if faces:
                best = max(faces, key=lambda f: f.det_score)
                return self._norm(best.embedding)
        except Exception as e:
            logger.debug(f"[Recognizer] crop embed err: {e}")
        return None

    def _from_full(self, frame: np.ndarray,
                   bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        try:
            faces = self.app.get(frame)
            if not faces:
                return None
            x1, y1, x2, y2 = bbox
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            best = min(faces, key=lambda f: (
                ((f.bbox[0] + f.bbox[2]) / 2 - cx) ** 2 +
                ((f.bbox[1] + f.bbox[3]) / 2 - cy) ** 2
            ))
            return self._norm(best.embedding)
        except Exception as e:
            logger.debug(f"[Recognizer] full-frame embed err: {e}")
        return None

    # ── cosine matching ───────────────────────────────────────────────────
    def find_match(self, query: np.ndarray,
                   registered: List[Dict]) -> Tuple[Optional[str], float]:
        """
        Cosine-similarity scan against all registered embeddings.
        Returns (face_id, score) or (None, best_score) if below threshold.
        """
        if query is None or not registered:
            return None, 0.0

        best_id, best_score = None, -1.0
        for entry in registered:
            score = float(np.dot(query, entry["embedding"]))
            if score > best_score:
                best_score = score
                best_id    = entry["face_id"]

        return (best_id, best_score) if best_score >= self.threshold else (None, best_score)

    @staticmethod
    def _norm(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / n if n > 0 else v
