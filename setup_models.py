"""
setup_models.py
--------------------------------------------------------------
Run once before starting the tracker:
  python setup_models.py

Downloads / verifies:
  • yolov8n-face.pt  (YOLOv8 face-tuned model)
  • buffalo_l        (InsightFace ArcFace, auto-downloaded by insightface)
"""

import logging
import os
import sys
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

YOLO_FACE_URLS = [
    "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt",
    "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt",
]


def ensure_dirs():
    for d in ["logs/entries","logs/exits","data","config","output","utils"]:
        os.makedirs(d, exist_ok=True)
    log.info("✓ Directories ready.")


def download_yolo_face():
    dst = "yolov8n-face.pt"
    if os.path.exists(dst):
        log.info(f"✓ YOLO face model already present: {dst}")
        return True
    for url in YOLO_FACE_URLS:
        try:
            log.info(f"  Downloading from {url} …")
            urllib.request.urlretrieve(url, dst)
            log.info(f"✓ Saved → {dst}")
            return True
        except Exception as e:
            log.warning(f"  Failed: {e}")
    log.error(
        "Could not download yolov8n-face.pt automatically.\n"
        "  Manual download: https://github.com/akanametov/yolo-face/releases\n"
        "  Place file as: yolov8n-face.pt (project root)"
    )
    return False


def check_ultralytics():
    try:
        from ultralytics import YOLO
        log.info("✓ ultralytics OK.")
        return True
    except ImportError:
        log.error("ultralytics missing → pip install ultralytics")
        return False


def check_insightface():
    try:
        from insightface.app import FaceAnalysis
        log.info("Verifying InsightFace buffalo_l (may download ~300 MB on first run)…")
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        log.info("✓ InsightFace buffalo_l ready.")
        return True
    except ImportError:
        log.error("insightface missing → pip install insightface onnxruntime")
        return False
    except Exception as e:
        log.error(f"InsightFace error: {e}")
        return False


def check_opencv():
    try:
        import cv2
        log.info(f"✓ OpenCV {cv2.__version__} OK.")
        return True
    except ImportError:
        log.error("opencv-python missing → pip install opencv-python")
        return False


if __name__ == "__main__":
    log.info("=" * 50)
    log.info("  Face Tracker — Setup")
    log.info("=" * 50)

    ensure_dirs()
    ok_cv   = check_opencv()
    ok_yolo = check_ultralytics()
    ok_ins  = check_insightface()
    ok_mdl  = download_yolo_face()

    log.info("")
    log.info("=" * 50)
    log.info("  Results")
    log.info("=" * 50)
    log.info(f"  OpenCV        : {'✓' if ok_cv   else '✗'}")
    log.info(f"  ultralytics   : {'✓' if ok_yolo else '✗'}")
    log.info(f"  InsightFace   : {'✓' if ok_ins  else '✗'}")
    log.info(f"  YOLO model    : {'✓' if ok_mdl  else '✗ (download manually)'}")

    if all([ok_cv, ok_yolo, ok_ins]):
        log.info("\n✓ Setup complete.  Next: python main.py")
    else:
        log.info("\n✗ Fix errors above, then re-run this script.")
        sys.exit(1)
