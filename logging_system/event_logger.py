"""
logging_system/event_logger.py
--------------------------------------------------------------
Handles:
  1. setup_logging() – configures Python root logger to write
     both to a file (events.log) and coloured console output.
  2. EventLogger class:
       • Saves cropped face images to
           logs/entries|exits|registered/YYYY-MM-DD/<id>_<ts>.jpg
       • Appends structured JSON records to logs/events.jsonl
       • Emits human-readable lines via Python logger → events.log
"""

import json
import logging
import os
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("face_tracker.events")


# ── logging setup ─────────────────────────────────────────────────────────────
def setup_logging(log_file: str, level: int = logging.INFO):
    """Configure root logger: file handler + optional colour console."""
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

    fmt      = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"
    plain    = logging.Formatter(fmt, datefmt=date_fmt)

    file_h = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_h.setFormatter(plain)
    file_h.setLevel(level)

    try:
        import colorlog
        cfmt  = ("%(log_color)s%(asctime)s | %(levelname)-8s%(reset)s"
                 " | %(cyan)s%(name)s%(reset)s | %(message)s")
        con_h = colorlog.StreamHandler()
        con_h.setFormatter(colorlog.ColoredFormatter(cfmt, datefmt=date_fmt))
    except ImportError:
        con_h = logging.StreamHandler()
        con_h.setFormatter(plain)
    con_h.setLevel(level)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(file_h)
    root.addHandler(con_h)

    logging.getLogger("face_tracker").info(f"Logging → {log_file}")


# ── EventLogger ───────────────────────────────────────────────────────────────
class EventLogger:
    """Saves images and writes structured JSONL event records."""

    def __init__(self, base_log_dir: str = "logs", image_quality: int = 95):
        self.base    = base_log_dir
        self.quality = image_quality
        self._dirs   = {
            "entry":      os.path.join(base_log_dir, "entries"),
            "exit":       os.path.join(base_log_dir, "exits"),
            "registered": os.path.join(base_log_dir, "registered"),
        }
        for d in self._dirs.values():
            os.makedirs(d, exist_ok=True)
        self.jsonl = os.path.join(base_log_dir, "events.jsonl")
        logger.info(f"[EventLogger] Image root → {base_log_dir}")

    # ── public API ─────────────────────────────────────────────────────────
    def log_registration(self, face_id: str, frame: np.ndarray,
                         bbox: tuple, frame_number: int = 0) -> Optional[str]:
        path = self._save(face_id, frame, bbox, "registered")
        self._jsonl(face_id, "registered", path, frame_number, 1.0)
        logger.info(
            f"[REGISTERED] id={face_id}  frame={frame_number}  img={path}"
        )
        return path

    def log_entry(self, face_id: str, frame: np.ndarray,
                  bbox: tuple, frame_number: int = 0,
                  confidence: float = 1.0) -> Optional[str]:
        path = self._save(face_id, frame, bbox, "entry")
        self._jsonl(face_id, "entry", path, frame_number, confidence)
        logger.info(
            f"[ENTRY]      id={face_id}  frame={frame_number}"
            f"  conf={confidence:.2f}  img={path}"
        )
        return path

    def log_exit(self, face_id: str, frame: np.ndarray,
                 bbox: tuple, frame_number: int = 0,
                 confidence: float = 1.0) -> Optional[str]:
        path = self._save(face_id, frame, bbox, "exit")
        self._jsonl(face_id, "exit", path, frame_number, confidence)
        logger.info(
            f"[EXIT]       id={face_id}  frame={frame_number}"
            f"  conf={confidence:.2f}  img={path}"
        )
        return path

    def log_recognition(self, face_id: str, frame_number: int, similarity: float):
        self._jsonl(face_id, "recognized", None, frame_number, similarity)
        logger.debug(
            f"[RECOGNIZED] id={face_id}  frame={frame_number}  sim={similarity:.3f}"
        )

    def log_system(self, message: str, level: str = "info"):
        getattr(logger, level, logger.info)(f"[SYSTEM] {message}")

    # ── internals ──────────────────────────────────────────────────────────
    def _save(self, face_id: str, frame: np.ndarray,
              bbox: tuple, event_type: str) -> Optional[str]:
        try:
            if frame is None or frame.size == 0:
                return None
            x1, y1, x2, y2 = bbox[:4]
            h, w = frame.shape[:2]
            # 10 % padding
            px = max(int((x2 - x1) * 0.10), 5)
            py = max(int((y2 - y1) * 0.10), 5)
            x1 = max(0, x1 - px); y1 = max(0, y1 - py)
            x2 = min(w, x2 + px); y2 = min(h, y2 + py)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None

            today   = datetime.now().strftime("%Y-%m-%d")
            day_dir = os.path.join(self._dirs[event_type], today)
            os.makedirs(day_dir, exist_ok=True)

            ts   = datetime.now().strftime("%H%M%S_%f")[:13]
            path = os.path.join(day_dir, f"{face_id}_{ts}.jpg")
            cv2.imwrite(path, crop, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            return path
        except Exception as e:
            logger.error(f"[EventLogger] Save failed for {face_id}: {e}")
            return None

    def _jsonl(self, face_id: str, event_type: str,
               image_path: Optional[str], frame_number: int, confidence: float):
        record = {
            "timestamp":  datetime.now().isoformat(),
            "face_id":    face_id,
            "event_type": event_type,
            "frame":      frame_number,
            "confidence": round(confidence, 4),
            "image_path": image_path,
        }
        try:
            with open(self.jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.error(f"[EventLogger] JSONL write failed: {e}")
