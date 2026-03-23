"""
frontend/app.py
--------------------------------------------------------------
Flask + SocketIO web dashboard.

Routes:
  GET /              → live dashboard HTML
  GET /video_feed    → MJPEG stream of annotated frames
  GET /api/stats     → JSON stats snapshot
  GET /api/faces     → JSON list of all registered faces
  GET /api/events    → JSON list of recent events
  GET /api/count     → JSON unique visitor count

The processing loop runs in a background daemon thread so the
Flask server can serve HTTP concurrently.
"""

import logging
import os
import threading
import time
from typing import Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template
from flask_cors import CORS
from flask_socketio import SocketIO

logger = logging.getLogger("face_tracker.web")

# Shared MJPEG buffer
_frame_lock   = threading.Lock()
_latest_jpg   : bytes = b""


def create_app(pipeline, config: dict):
    """Build and return (Flask app, SocketIO instance)."""

    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    app = Flask(__name__, template_folder=template_dir)
    app.config["SECRET_KEY"] = "ft-secret-2024"
    CORS(app)
    sio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

    # ── background processing thread ─────────────────────────────────────
    def _process():
        global _latest_jpg
        vcfg   = config["video"]
        source = vcfg["rtsp_url"] if vcfg.get("use_rtsp") else vcfg["source"]

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error(f"[Web] Cannot open: {source}")
            return

        logger.info(f"[Web] Processing thread started: {source}")
        fn = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("[Web] Stream ended.")
                break

            annotated = pipeline.process_frame(frame)
            pipeline.frame_number += 1
            fn += 1

            # Encode for MJPEG
            ok, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 78])
            if ok:
                with _frame_lock:
                    _latest_jpg = jpg.tobytes()

            # Emit stats every 8 frames
            if fn % 8 == 0:
                sio.emit("stats_update", pipeline.get_stats())

        cap.release()
        pipeline._flush_remaining(None)

    t = threading.Thread(target=_process, daemon=True)
    t.start()

    # ── routes ────────────────────────────────────────────────────────────
    @app.route("/")
    def index():
        return render_template("dashboard.html")

    @app.route("/video_feed")
    def video_feed():
        def gen():
            while True:
                with _frame_lock:
                    jpg = _latest_jpg
                if jpg:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                time.sleep(0.033)
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/api/stats")
    def api_stats():
        return jsonify(pipeline.get_stats())

    @app.route("/api/faces")
    def api_faces():
        return jsonify(pipeline.db.get_all_faces())

    @app.route("/api/events")
    def api_events():
        return jsonify(pipeline.db.get_recent_events(50))

    @app.route("/api/count")
    def api_count():
        return jsonify({"unique_visitors": pipeline.db.get_unique_visitor_count()})

    # ── socket event forwarding ───────────────────────────────────────────
    # Monkey-patch the event logger to also push via SocketIO
    orig_entry = pipeline.elog.log_entry
    orig_exit  = pipeline.elog.log_exit
    orig_reg   = pipeline.elog.log_registration

    def _patched_entry(face_id, frame, bbox, frame_number=0, confidence=1.0):
        path = orig_entry(face_id, frame, bbox, frame_number, confidence)
        sio.emit("new_event", {"face_id": face_id, "event_type": "entry",
                               "frame": frame_number, "timestamp": _now()})
        return path

    def _patched_exit(face_id, frame, bbox, frame_number=0, confidence=1.0):
        path = orig_exit(face_id, frame, bbox, frame_number, confidence)
        sio.emit("new_event", {"face_id": face_id, "event_type": "exit",
                               "frame": frame_number, "timestamp": _now()})
        return path

    def _patched_reg(face_id, frame, bbox, frame_number=0):
        path = orig_reg(face_id, frame, bbox, frame_number)
        sio.emit("new_event", {"face_id": face_id, "event_type": "registered",
                               "frame": frame_number, "timestamp": _now()})
        return path

    pipeline.elog.log_entry        = _patched_entry
    pipeline.elog.log_exit         = _patched_exit
    pipeline.elog.log_registration = _patched_reg

    return app, sio


def _now():
    from datetime import datetime
    return datetime.now().isoformat()
