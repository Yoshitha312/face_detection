"""
main.py
--------------------------------------------------------------
Entry point for the Intelligent Face Tracker.

Usage examples:
  python main.py                                        # use config defaults
  python main.py --video path/to/video.mp4              # single video file
  python main.py --folder path/to/videos_folder         # process all videos in folder
  python main.py --rtsp rtsp://ip:port/stream           # RTSP live camera
  python main.py --web                                  # + live web dashboard
  python main.py --no-display                           # no OpenCV popup window
  python main.py --folder "C:/videos" --web --no-display  # full recommended run
"""

import argparse
import json
import logging
import os
import sys


def parse_args():
    p = argparse.ArgumentParser(
        description="Intelligent Face Tracker with Auto Registration & Visitor Counting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config",     default="config/config.json",
                   help="Path to config.json (default: config/config.json)")
    p.add_argument("--video",      default=None,
                   help="Path to a single video file")
    p.add_argument("--folder",     default=None,
                   help="Path to folder containing multiple video files")
    p.add_argument("--rtsp",       default=None,
                   help="RTSP stream URL for live camera")
    p.add_argument("--web",        action="store_true",
                   help="Start Flask web dashboard at http://localhost:5000")
    p.add_argument("--no-display", action="store_true",
                   help="Disable OpenCV popup window (use for headless/server runs)")
    p.add_argument("--log-level",  default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Logging verbosity level")
    return p.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------ #
    # Load config.json
    # ------------------------------------------------------------------ #
    if not os.path.exists(args.config):
        print(f"[ERROR] Config file not found: {args.config}")
        print("        Make sure you are running from inside the face_tracker folder.")
        sys.exit(1)

    with open(args.config) as f:
        config = json.load(f)

    # ------------------------------------------------------------------ #
    # Setup logging (file + console)
    # ------------------------------------------------------------------ #
    from logging_system.event_logger import setup_logging
    log_file = config["logging"]["log_file"]
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    setup_logging(log_file, level=getattr(logging, args.log_level))
    logger = logging.getLogger("face_tracker.main")

    logger.info("=" * 60)
    logger.info("  Intelligent Face Tracker -- Starting")
    logger.info("=" * 60)

    # ------------------------------------------------------------------ #
    # Override config from CLI arguments
    # Priority: --rtsp > --folder > --video > config defaults
    # ------------------------------------------------------------------ #
    if args.folder:
        # Process all videos in the given folder
        config["video"]["video_folder"] = args.folder
        config["video"]["use_folder"]   = True
        config["video"]["use_rtsp"]     = False
        logger.info(f"[Main] Folder mode: {args.folder}")

    elif args.video:
        # Process a single specific video file
        config["video"]["source"]     = args.video
        config["video"]["use_rtsp"]   = False
        config["video"]["use_folder"] = False
        logger.info(f"[Main] Single video: {args.video}")

    if args.rtsp:
        # Live RTSP camera stream (overrides folder/video)
        config["video"]["rtsp_url"]   = args.rtsp
        config["video"]["use_rtsp"]   = True
        config["video"]["use_folder"] = False
        logger.info(f"[Main] RTSP mode: {args.rtsp}")

    if args.no_display:
        config["video"]["display_output"] = False

    # Save overrides back to config so pipeline reads them
    with open(args.config, "w") as f:
        json.dump(config, f, indent=2)

    # ------------------------------------------------------------------ #
    # Build pipeline
    # ------------------------------------------------------------------ #
    from core.pipeline import FaceTrackingPipeline
    pipeline = FaceTrackingPipeline(config_path=args.config)

    # ------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------ #
    if args.web:
        from frontend.app import create_app
        app, sio = create_app(pipeline, config)
        host = config["frontend"]["host"]
        port = config["frontend"]["port"]
        logger.info(f"[Main] Web dashboard -> http://localhost:{port}")
        logger.info(f"[Main] Open that URL in your browser now.")
        try:
            sio.run(app, host=host, port=port, debug=False,
                    allow_unsafe_werkzeug=True)
        except KeyboardInterrupt:
            logger.info("[Main] Shutting down web server.")
            pipeline.stop()
    else:
        try:
            pipeline.run()
        except KeyboardInterrupt:
            logger.info("[Main] Interrupted by user (Ctrl+C).")
            pipeline.stop()
        except Exception as e:
            logger.exception(f"[Main] Fatal error: {e}")
            sys.exit(1)

    # ------------------------------------------------------------------ #
    # Final summary printed to terminal
    # ------------------------------------------------------------------ #
    stats = pipeline.db.get_stats()
    logger.info("=" * 60)
    logger.info(f"  UNIQUE VISITORS : {stats.get('unique_visitors', 0)}")
    logger.info(f"  TOTAL ENTRIES   : {stats.get('total_entries',   0)}")
    logger.info(f"  TOTAL EXITS     : {stats.get('total_exits',     0)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
