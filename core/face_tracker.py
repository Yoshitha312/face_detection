"""
core/face_tracker.py
--------------------------------------------------------------
Lightweight IoU-based multi-face tracker.

Lifecycle
---------
  detection → new Track (state=active)
  track unseen for max_disappeared frames → state=exited
  track.entry_logged / exit_logged flags ensure exactly ONE
  entry event and ONE exit event per track, regardless of
  how many frames the face appears in.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("face_tracker.tracker")


@dataclass
class Track:
    track_id:          int
    bbox:              Tuple[int, int, int, int]
    confidence:        float        = 1.0
    disappeared:       int          = 0
    frame_count:       int          = 1
    first_frame:       int          = 0
    last_frame:        int          = 0

    # identity
    face_id:           Optional[str]        = None
    embedding:         Optional[np.ndarray] = None
    is_registered:     bool                 = False

    # event guards (exactly one entry + one exit)
    entry_logged:      bool = False
    exit_logged:       bool = False

    state:             str  = "active"   # "active" | "exited"


class FaceTracker:
    def __init__(self,
                 max_disappeared:     int   = 30,
                 iou_threshold:       float = 0.30,
                 min_register_frames: int   = 3):
        self.max_disappeared      = max_disappeared
        self.iou_threshold        = iou_threshold
        self.min_register_frames  = min_register_frames

        self._next_id              = 1
        self.tracks: Dict[int, Track] = {}
        self.exited: List[Track]      = []

        logger.info(
            f"[Tracker] Ready  max_disappeared={max_disappeared}"
            f"  iou={iou_threshold}  min_reg_frames={min_register_frames}"
        )

    # ── public update ─────────────────────────────────────────────────────
    def update(self,
               detections: list,
               frame_number: int = 0) -> Tuple[List[Track], List[Track]]:
        """
        Args:
            detections:   list of (x1,y1,x2,y2,conf)
            frame_number: current frame index

        Returns:
            (active_tracks, newly_exited_tracks)
        """
        newly_exited: List[Track] = []

        if not detections:
            for t in list(self.tracks.values()):
                t.disappeared += 1
                if t.disappeared >= self.max_disappeared:
                    exited = self._expire(t.track_id, frame_number)
                    if exited:
                        newly_exited.append(exited)
            return list(self.tracks.values()), newly_exited

        det_bboxes = [(d[0], d[1], d[2], d[3]) for d in detections]
        det_confs  = [d[4] for d in detections]

        if not self.tracks:
            for bbox, conf in zip(det_bboxes, det_confs):
                self._create(bbox, conf, frame_number)
            return list(self.tracks.values()), []

        track_ids    = list(self.tracks.keys())
        track_bboxes = [self.tracks[tid].bbox for tid in track_ids]
        iou_mat      = self._iou_matrix(track_bboxes, det_bboxes)

        matched, unmatched_t, unmatched_d = self._greedy_match(
            iou_mat, track_ids, list(range(len(det_bboxes)))
        )

        for t_id, d_idx in matched:
            t            = self.tracks[t_id]
            t.bbox       = det_bboxes[d_idx]
            t.confidence = det_confs[d_idx]
            t.disappeared = 0
            t.last_frame  = frame_number
            t.frame_count += 1

        for t_id in unmatched_t:
            t = self.tracks[t_id]
            t.disappeared += 1
            if t.disappeared >= self.max_disappeared:
                exited = self._expire(t_id, frame_number)
                if exited:
                    newly_exited.append(exited)

        for d_idx in unmatched_d:
            self._create(det_bboxes[d_idx], det_confs[d_idx], frame_number)

        return [t for t in self.tracks.values() if t.state == "active"], newly_exited

    # ── identity ──────────────────────────────────────────────────────────
    def assign_identity(self, track_id: int, face_id: str,
                        embedding: np.ndarray, is_new: bool = False):
        if track_id not in self.tracks:
            return
        t              = self.tracks[track_id]
        t.face_id      = face_id
        t.embedding    = embedding
        t.is_registered = True
        verb = "NEW" if is_new else "recognized"
        logger.debug(f"[Tracker] #{track_id} → {verb} {face_id}")

    def ready_for_registration(self, track: Track) -> bool:
        """True once track has been stable for min_register_frames."""
        return track.frame_count >= self.min_register_frames

    def get_active(self) -> List[Track]:
        return [t for t in self.tracks.values() if t.state == "active"]

    # ── internals ─────────────────────────────────────────────────────────
    def _create(self, bbox, conf, frame_number):
        t = Track(
            track_id=self._next_id,
            bbox=bbox, confidence=conf,
            first_frame=frame_number, last_frame=frame_number
        )
        self.tracks[self._next_id] = t
        logger.debug(f"[Tracker] New #{self._next_id} @ frame {frame_number}")
        self._next_id += 1

    def _expire(self, track_id: int, frame_number: int) -> Optional[Track]:
        t = self.tracks.pop(track_id, None)
        if t:
            t.state      = "exited"
            t.last_frame = frame_number
            self.exited.append(t)
            logger.debug(f"[Tracker] #{track_id} exited @ frame {frame_number}")
        return t

    @staticmethod
    def _iou_matrix(track_bboxes, det_bboxes) -> np.ndarray:
        m = np.zeros((len(track_bboxes), len(det_bboxes)), dtype=np.float32)
        for i, tb in enumerate(track_bboxes):
            for j, db in enumerate(det_bboxes):
                xA = max(tb[0], db[0]); yA = max(tb[1], db[1])
                xB = min(tb[2], db[2]); yB = min(tb[3], db[3])
                inter = max(0, xB - xA) * max(0, yB - yA)
                if inter == 0:
                    continue
                aA = (tb[2] - tb[0]) * (tb[3] - tb[1])
                aB = (db[2] - db[0]) * (db[3] - db[1])
                m[i, j] = inter / (aA + aB - inter)
        return m

    def _greedy_match(self, iou_mat, track_ids, det_indices):
        matched, used_t, used_d = [], set(), set()
        pairs = [
            (iou_mat[ti, di], track_ids[ti], di)
            for ti in range(len(track_ids))
            for di in det_indices
            if iou_mat[ti, di] >= self.iou_threshold
        ]
        pairs.sort(key=lambda x: -x[0])
        for _, t_id, d_idx in pairs:
            if t_id not in used_t and d_idx not in used_d:
                matched.append((t_id, d_idx))
                used_t.add(t_id); used_d.add(d_idx)
        return (
            matched,
            [t for t in track_ids   if t not in used_t],
            [d for d in det_indices if d not in used_d],
        )
