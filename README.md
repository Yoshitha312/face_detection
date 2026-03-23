# 🎯 Intelligent Face Tracker — Auto Registration & Visitor Counting

Real-time AI system that detects, recognises, and tracks unique faces in video or RTSP streams.  
Every new face is registered automatically. Every entry and exit is logged with a timestamped image.

---

## Architecture Diagram

```
  Video File / RTSP Stream
          │
          ▼  every (skip_frames + 1) frames
  ┌───────────────────┐
  │  YOLOv8 Face      │  ← core/face_detector.py
  │  Detector         │    yolov8n-face.pt
  └────────┬──────────┘
           │  (x1,y1,x2,y2, conf) list
           ▼
  ┌───────────────────┐
  │  IoU Tracker      │  ← core/face_tracker.py
  │  (per-frame)      │    assigns track IDs, detects entry/exit
  └────────┬──────────┘
           │  Track objects
           ▼
  ┌───────────────────┐
  │  InsightFace      │  ← core/face_recognizer.py
  │  ArcFace Embedder │    512-dim cosine similarity
  └────────┬──────────┘
           │
     ┌─────┴─────┐
     │           │
  matched    no match
     │           │
     ▼           ▼
  Recognized  Register new
  (log only)  face_id in DB
     │           │
     └─────┬─────┘
           ▼
  ┌───────────────────────────────┐
  │  Event Logger                 │  ← logging_system/event_logger.py
  │  • logs/entries/YYYY-MM-DD/   │
  │  • logs/exits/YYYY-MM-DD/     │
  │  • logs/events.log            │
  │  • logs/events.jsonl          │
  └────────┬──────────────────────┘
           │
           ▼
  ┌───────────────────┐
  │  SQLite DB        │  ← database/db_manager.py
  │  faces / events / │    data/face_tracker.db
  │  stats            │
  └────────┬──────────┘
           │
           ▼  (optional)
  ┌───────────────────┐
  │  Flask Dashboard  │  ← frontend/app.py
  │  MJPEG + SocketIO │    http://localhost:5000
  └───────────────────┘
```

---

## Project Structure

```
face_tracker/
├── main.py                         # Entry point (CLI)
├── setup_models.py                 # Download model weights
├── requirements.txt
├── config/
│   └── config.json                 # ← All tunable parameters
├── core/
│   ├── face_detector.py            # YOLOv8 detection
│   ├── face_recognizer.py          # InsightFace ArcFace
│   ├── face_tracker.py             # IoU multi-object tracker
│   └── pipeline.py                 # Orchestration loop
├── database/
│   └── db_manager.py               # SQLite CRUD
├── logging_system/
│   └── event_logger.py             # Image saving + log files
├── frontend/
│   ├── app.py                      # Flask + SocketIO server
│   └── templates/dashboard.html   # Live web UI
├── utils/
│   └── query_results.py            # CLI DB query tool
├── logs/
│   ├── events.log                  # Human-readable system log
│   ├── events.jsonl                # Machine-readable event records
│   ├── entries/YYYY-MM-DD/         # Entry face crops
│   ├── exits/YYYY-MM-DD/           # Exit face crops
│   └── registered/YYYY-MM-DD/     # Registration thumbnails
└── data/
    └── face_tracker.db             # SQLite database
```

---

## Sample `config.json`

```json
{
  "detection": {
    "skip_frames": 3,
    "confidence_threshold": 0.50,
    "yolo_model": "yolov8n-face.pt",
    "iou_threshold": 0.45,
    "min_face_size": 30
  },
  "recognition": {
    "model_name": "buffalo_l",
    "similarity_threshold": 0.45,
    "embedding_size": 512,
    "device": "cpu"
  },
  "tracking": {
    "max_disappeared": 30,
    "iou_threshold": 0.30,
    "min_register_frames": 3
  },
  "logging": {
    "log_file": "logs/events.log",
    "image_quality": 95,
    "base_log_dir": "logs"
  },
  "database": {
    "path": "data/face_tracker.db"
  },
  "video": {
    "source": "sample_video.mp4",
    "rtsp_url": "rtsp://username:password@ip:port/stream",
    "use_rtsp": false,
    "display_output": true,
    "save_output": false,
    "output_video": "output/processed_output.mp4"
  },
  "frontend": {
    "host": "0.0.0.0",
    "port": 5000
  }
}
```

**Key parameter:** `detection.skip_frames` — number of frames to skip between YOLO detection cycles.  
- `0` = detect every frame (most accurate, highest CPU)  
- `3` = detect every 4th frame (recommended for real-time)  
- `9` = detect every 10th frame (lowest CPU, less accurate tracking)

---

## Setup Instructions

### 1. Prerequisites
- Python **3.9** or **3.10** (3.11+ works but InsightFace may need extra steps)
- pip 22+
- (Optional) NVIDIA GPU with CUDA 11.8+ for faster inference

### 2. Create virtual environment
```bash
cd face_tracker
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download model weights
```bash
python setup_models.py
```
This downloads:
- `yolov8n-face.pt` — YOLOv8 face detector (~6 MB)
- `buffalo_l` — InsightFace ArcFace model (~300 MB, saved to `~/.insightface/models/`)

### 5. Place your video file
Download the sample video from the provided Google Drive link and save it as:
```
face_tracker/sample_video.mp4
```
Or update `config/config.json` → `video.source` to your actual path.

---

## How to Run

### Option A — Video file with OpenCV window
```bash
python main.py
```

### Option B — Video file, headless (no GUI window)
```bash
python main.py --no-display
```

### Option C — Video file + live web dashboard
```bash
python main.py --web
# Open: http://localhost:5000
```

### Option D — RTSP camera stream (interview mode)
```bash
python main.py --rtsp rtsp://username:password@camera_ip:554/stream
# Add --web for dashboard
python main.py --rtsp rtsp://... --web --no-display
```

### Option E — Custom video file path
```bash
python main.py --video /path/to/your/video.mp4 --web
```

### Debug mode
```bash
python main.py --log-level DEBUG
```

---

## Query Results

```bash
# Full session summary
python utils/query_results.py

# Unique visitor count only
python utils/query_results.py --count

# List all registered faces
python utils/query_results.py --faces

# Last 50 events
python utils/query_results.py --events 50

# Export events to JSON
python utils/query_results.py --export events_export.json
```

---

## What to Change / Configure

| Parameter | Config key | When to change |
|-----------|-----------|----------------|
| Video file path | `video.source` | Point to your actual video |
| RTSP URL | `video.rtsp_url` + `use_rtsp: true` | Interview / live camera |
| Frame skip | `detection.skip_frames` | Higher = less CPU, lower = more accurate |
| Detection confidence | `detection.confidence_threshold` | Lower (0.3) = catch more faces, more false positives |
| Re-ID threshold | `recognition.similarity_threshold` | Lower (0.35) = stricter same-person match |
| Track patience | `tracking.max_disappeared` | Higher = slower exit detection |
| GPU inference | `recognition.device` | Change `"cpu"` → `"cuda"` |
| Web port | `frontend.port` | If 5000 is taken |

---

## Compute Load Estimates

### CPU-only (i5/i7 laptop)
| Component | CPU % | RAM |
|-----------|-------|-----|
| YOLOv8n detection | 15–25% (1 core per cycle) | 200 MB |
| InsightFace ArcFace | 20–35% (1 core per face) | 350 MB |
| IoU Tracker + logging | < 2% | 50 MB |
| Flask dashboard | < 3% | 80 MB |
| **Total @ skip=3, 3 faces** | **~45–65% single core** | **~700 MB** |

### GPU (NVIDIA 4GB+ VRAM)
| Component | GPU % | VRAM |
|-----------|-------|------|
| YOLOv8n | 8–15% | 400 MB |
| InsightFace ArcFace | 10–18% | 650 MB |
| **Total** | **~25–35%** | **~1.1 GB** |

**Recommended:** Intel Core i5 8th gen+ or any NVIDIA GPU ≥ 4 GB VRAM for smooth 25+ fps.

---

## AI Planning Document

### Problem Decomposition
| Step | Responsibility | Module |
|------|---------------|--------|
| 1 | Find face bounding boxes | `FaceDetector` |
| 2 | Assign consistent track IDs across frames | `FaceTracker` |
| 3 | Generate identity fingerprint (embedding) | `FaceRecognizer` |
| 4 | Match embedding to known people | `FaceRecognizer.find_match` |
| 5 | Register unknown faces | `Pipeline._identify` |
| 6 | Log entry on track birth, exit on track death | `Pipeline.process_frame` |
| 7 | Store everything in DB + filesystem | `DatabaseManager` + `EventLogger` |
| 8 | Expose stats via API + live UI | `Frontend` |

### Technology Choices & Rationale

| Need | Choice | Why not the alternative |
|------|--------|------------------------|
| Detection | YOLOv8n-face | Face-tuned, fast, easy API |
| Recognition | InsightFace buffalo_l (ArcFace) | 99.83% LFW; `face_recognition` lib is ~98% |
| Tracking | Custom IoU greedy match | Zero extra deps; sufficient for single-camera |
| Database | SQLite + WAL mode | Zero-config, ACID, resilient to interruption |
| Config | JSON | Human-editable, no extra parser needed |

### Key Design Decisions

1. **Embedding cache in RAM** — All known embeddings loaded at start; recognition is an in-memory dot-product scan. No DB queries per frame.

2. **Skip-frame detection** — YOLO runs every N frames; tracker reuses last detections in between. Reduces CPU by 60–80% with minimal accuracy loss.

3. **min_register_frames** — A track must be seen for ≥ 3 consecutive frames before embedding is generated and registration attempted. Prevents ghost registrations from single-frame noise.

4. **Exactly-one entry + exit** — `track.entry_logged` and `track.exit_logged` boolean flags ensure the DB and image logs record exactly one of each per person per visit, regardless of how many frames they appear in.

5. **WAL-mode SQLite** — Journal mode WAL means writes don't block reads, and the DB stays consistent even if the process is killed mid-write.

---

## Assumptions

1. One video source at a time (single camera).
2. Faces must be ≥ 30×30 pixels to generate a reliable embedding.
3. Similarity threshold 0.45 is appropriate for typical indoor video; adjust for extreme lighting or disguises.
4. The system counts "visits" not "people present simultaneously"; the same person re-entering after leaving counts as one unique visitor.
5. Python 3.9/3.10 is used; 3.11+ should work but was not the primary target.

---



## Demo Video-loom link:

https://www.loom.com/share/cde10973fce94520b3c266cd87544a1e



“This project is a part of a hackathon run by https://katomaran.com ”
