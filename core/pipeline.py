"""
core/pipeline.py
--------------------------------------------------------------
Orchestrates the full face-tracking pipeline:

  Frame -> Detector -> Recognizer -> Tracker -> Logger -> DB

Key design choices:
  - Detection runs every (skip_frames+1) frames; tracker
    interpolates with the cached bbox list in between.
  - Embedding cache: all registered embeddings are kept in
    memory so recognition is a single matrix dot-product scan.
  - Exactly ONE entry + ONE exit event per track lifetime,
    enforced by Track.entry_logged / exit_logged flags.
  - min_register_frames prevents one-frame ghost registrations.
  - Folder mode: processes all videos in a folder in sequence,
    sharing the same DB so faces are never double-counted.
"""

import json
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Callable, List, Optional

import cv2
import numpy as np

from core.face_detector   import FaceDetector
from core.face_recognizer import FaceRecognizer
from core.face_tracker    import FaceTracker, Track
from database.db_manager  import DatabaseManager
from logging_system.event_logger import EventLogger

logger = logging.getLogger("face_tracker.pipeline")


def load_config(path: str = "config/config.json") -> dict:
    with open(path) as f:
        return json.load(f)


class FaceTrackingPipeline:
    """End-to-end pipeline. Call run() to process a video source."""

    def __init__(self, config_path: str = "config/config.json"):
        self.cfg          = load_config(config_path)
        self._running     = False
        self.frame_number = 0
        self._last_dets: list = []
        self._emb_cache:  list = []

        self._build_components()
        logger.info("[Pipeline] Initialised and ready.")

    # ------------------------------------------------------------------ #
    # Component wiring
    # ------------------------------------------------------------------ #
    def _build_components(self):
        d  = self.cfg["detection"]
        r  = self.cfg["recognition"]
        t  = self.cfg["tracking"]
        lg = self.cfg["logging"]
        db = self.cfg["database"]

        self.skip_frames = d["skip_frames"]

        self.detector = FaceDetector(
            model_path=d["yolo_model"],
            confidence=d["confidence_threshold"],
            iou_threshold=d["iou_threshold"],
            min_face_size=d.get("min_face_size", 30),
        )
        self.recognizer = FaceRecognizer(
            model_name=r["model_name"],
            similarity_threshold=r["similarity_threshold"],
            device=r.get("device", "cpu"),
        )
        self.tracker = FaceTracker(
            max_disappeared=t["max_disappeared"],
            iou_threshold=t["iou_threshold"],
            min_register_frames=t.get("min_register_frames", 3),
        )
        self.db   = DatabaseManager(db["path"])
        self.elog = EventLogger(
            base_log_dir=lg["base_log_dir"],
            image_quality=lg.get("image_quality", 95),
        )

        self._emb_cache = self.db.get_all_embeddings()
        logger.info(f"[Pipeline] Loaded {len(self._emb_cache)} known faces from DB.")

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #
    def run(self, source=None, frame_callback: Optional[Callable] = None):
        """
        Process a single video file, a folder of videos, or RTSP stream.

        Folder mode: set use_folder=true and video_folder in config.json.
        All videos share the same database so the same face is only counted
        ONCE across all 25 videos.
        """
        vcfg = self.cfg["video"]

        if source is not None:
            sources = [source]
        elif vcfg.get("use_rtsp"):
            sources = [vcfg["rtsp_url"]]
        elif vcfg.get("use_folder") and vcfg.get("video_folder"):
            sources = self._collect_videos(
                vcfg["video_folder"],
                vcfg.get("video_extensions",
                          [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"])
            )
            if not sources:
                raise RuntimeError(
                    f"No video files found in folder: {vcfg['video_folder']}"
                )
            logger.info(f"[Pipeline] Folder mode — {len(sources)} videos found.")
        else:
            sources = [vcfg["source"]]

        self._running = True
        grand_t0 = time.time()

        for idx, src in enumerate(sources):
            if not self._running:
                break
            logger.info(
                f"[Pipeline] -- Video {idx+1}/{len(sources)}: "
                f"{os.path.basename(src)} --"
            )
            self._process_one(src, vcfg, frame_callback)

        total_elapsed = time.time() - grand_t0
        stats = self.get_stats()
        logger.info(
            f"[Pipeline] ALL DONE -- {len(sources)} video(s), "
            f"{self.frame_number} total frames in {total_elapsed:.1f}s | "
            f"unique visitors: {stats['unique_visitors']}"
        )
        self.elog.log_system(
            f"All sessions complete. videos={len(sources)} "
            f"frames={self.frame_number} elapsed={total_elapsed:.1f}s "
            f"unique_visitors={stats['unique_visitors']}"
        )

    def stop(self):
        self._running = False

    # ------------------------------------------------------------------ #
    # Single video processing
    # ------------------------------------------------------------------ #
    def _process_one(self, source: str, vcfg: dict,
                     frame_callback: Optional[Callable] = None):
        """Open and fully process one video file or RTSP stream."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error(f"[Pipeline] Cannot open: {source} -- skipping.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"[Pipeline] Opened: {source}  ({w}x{h} @ {fps:.1f} fps)")
        self.elog.log_system(f"Processing started: {source}")

        writer = None
        if vcfg.get("save_output"):
            base     = os.path.splitext(os.path.basename(source))[0]
            out_path = os.path.join("output", f"{base}_tracked.mp4")
            os.makedirs("output", exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        # Reset tracker per video (so exit events fire at end of each video)
        # but keep _emb_cache so faces are recognised across all videos.
        self.tracker = FaceTracker(
            max_disappeared=self.cfg["tracking"]["max_disappeared"],
            iou_threshold=self.cfg["tracking"]["iou_threshold"],
            min_register_frames=self.cfg["tracking"].get("min_register_frames", 3),
        )
        self._last_dets = []
        frame = None
        t0    = time.time()

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    break

                annotated = self.process_frame(frame)

                if writer:
                    writer.write(annotated)

                if frame_callback:
                    frame_callback(annotated, self.get_stats())

                if vcfg.get("display_output", False):
                    win = f"Face Tracker -- {os.path.basename(source)}  [q=quit]"
                    cv2.imshow(win, annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("[Pipeline] User quit.")
                        self._running = False
                        break

                self.frame_number += 1

        finally:
            self._flush_remaining(frame)
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            logger.info(
                f"[Pipeline] Finished: {os.path.basename(source)} "
                f"in {time.time()-t0:.1f}s"
            )

    # ------------------------------------------------------------------ #
    # Per-frame processing
    # ------------------------------------------------------------------ #
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Core logic for a single frame. Returns annotated frame."""

        # 1. Detect (every skip_frames+1 frames)
        if self.frame_number % (self.skip_frames + 1) == 0:
            self._last_dets = self.detector.detect(frame)

        # 2. Update tracker
        active, newly_exited = self.tracker.update(self._last_dets, self.frame_number)

        # 3. Identify unregistered tracks
        for track in active:
            if not track.is_registered and self.tracker.ready_for_registration(track):
                self._identify(track, frame)

        # 4. Log entries (once per track lifetime)
        for track in active:
            if track.is_registered and not track.entry_logged:
                img = self.elog.log_entry(
                    track.face_id, frame, track.bbox,
                    self.frame_number, track.confidence
                )
                self.db.log_event(
                    track.face_id, "entry", img,
                    self.frame_number, track.confidence
                )
                track.entry_logged = True

        # 5. Log exits (once per track lifetime)
        for track in newly_exited:
            if track.is_registered and not track.exit_logged:
                img = self.elog.log_exit(
                    track.face_id, frame, track.bbox,
                    self.frame_number, track.confidence
                )
                self.db.log_event(
                    track.face_id, "exit", img,
                    self.frame_number, track.confidence
                )
                self.db.update_last_seen(track.face_id)
                track.exit_logged = True

        return self._annotate(frame.copy(), active)

    # ------------------------------------------------------------------ #
    # Identity management
    # ------------------------------------------------------------------ #
    def _identify(self, track: Track, frame: np.ndarray):
        """Try to recognise a track; register it as new if unknown."""
        emb = self.recognizer.get_embedding(frame, track.bbox)
        if emb is None:
            return

        matched_id, sim = self.recognizer.find_match(emb, self._emb_cache)

        if matched_id:
            self.tracker.assign_identity(track.track_id, matched_id, emb, is_new=False)
            self.elog.log_recognition(matched_id, self.frame_number, sim)
        else:
            new_id = self._make_id()
            thumb  = self.elog.log_registration(new_id, frame, track.bbox, self.frame_number)
            if self.db.register_face(new_id, emb, thumb):
                self._emb_cache.append({"face_id": new_id, "embedding": emb})
                self.tracker.assign_identity(track.track_id, new_id, emb, is_new=True)

    def _flush_remaining(self, last_frame):
        """At stream end, log exits for all still-active tracks."""
        blank = np.zeros((10, 10, 3), np.uint8) if last_frame is None else last_frame
        for track in list(self.tracker.tracks.values()):
            if track.is_registered and not track.exit_logged:
                img = self.elog.log_exit(
                    track.face_id, blank, track.bbox,
                    self.frame_number, track.confidence
                )
                self.db.log_event(
                    track.face_id, "exit", img,
                    self.frame_number, track.confidence
                )
                self.db.update_last_seen(track.face_id)
                track.exit_logged = True

    # ------------------------------------------------------------------ #
    # Annotation
    # ------------------------------------------------------------------ #
    def _annotate(self, frame: np.ndarray, tracks: List[Track]) -> np.ndarray:
        """Draw bounding boxes, labels, and stats overlay on frame."""
        for t in tracks:
            x1, y1, x2, y2 = t.bbox
            color = (0, 220, 90) if t.is_registered else (0, 140, 255)
            label = t.face_id[-14:] if t.face_id else f"Track#{t.track_id}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1)

        s = self.get_stats()
        for i, line in enumerate([
            f"Frame: {self.frame_number}",
            f"Unique Visitors: {s['unique_visitors']}",
            f"Active Tracks:   {s['active_tracks']}",
            f"Total Entries:   {s['total_entries']}",
        ]):
            y = 26 + i * 22
            cv2.putText(frame, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
            cv2.putText(frame, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (20, 20, 20), 1)
        return frame

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #
    def get_stats(self) -> dict:
        db = self.db.get_stats()
        return {
            "unique_visitors": int(db.get("unique_visitors", 0)),
            "total_entries":   int(db.get("total_entries",   0)),
            "total_exits":     int(db.get("total_exits",     0)),
            "active_tracks":   len(self.tracker.get_active()),
            "frame_number":    self.frame_number,
            "session_start":   db.get("session_start", ""),
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _make_id() -> str:
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        uid = uuid.uuid4().hex[:6].upper()
        return f"FACE_{ts}_{uid}"

    @staticmethod
    def _collect_videos(folder: str, extensions: List[str]) -> List[str]:
        """
        Return all video files in folder sorted alphabetically.
        Works with spaces and special characters in the folder path.
        """
        ext_set = {e.lower() for e in extensions}
        videos  = []
        for fname in sorted(os.listdir(folder)):
            if os.path.splitext(fname)[1].lower() in ext_set:
                videos.append(os.path.join(folder, fname))
        return videos
