# 🏈 Football Player Tracking with Kalman Filter \& Re-ID

> Real-time multi-object tracking for football footage — stable player IDs across occlusion, crowd clusters, and frame edges using a custom Kalman filter tracker built on top of a YOLOv8-style PyTorch model.

![Football Player Tracking Demo](output_clip.gif)



## 📽️ Demo

|Input|Output|
|-|-|
|Raw 720p football footage|Players tracked with stable per-class IDs|

> Blue Player #1, #2 … and Red Player #1, #2 … are assigned once and held through the full clip — even through crowd clusters and partial edge crops.



## ✨ Features

* **Per-class stable IDs** — Blue and Red players get independent ID counters so IDs never cross-contaminate between teams
* **Crowd fix** — blended IoU + centroid proximity score breaks Hungarian matching ties when players cluster together
* **Edge fix** — detections near frame borders use a looser re-ID threshold so partially-cropped players reconnect to their track
* **Ghost box prevention** — degenerate Kalman states (NaN, zero-area) are caught before they reach the output; duplicate detections are absorbed rather than given new IDs
* **End-of-video stability** — fixed centroid scale (no median collapse on sparse detections) and class-locked re-ID stop ID switching in the final frames
* **Pure PyTorch NMS** — no external ONNX runtime or TorchVision dependency



## 🏗️ Architecture

```
Input Video
    │
    ▼
YOLOv8-style PyTorch Model (.pth)
    │  per-frame detections \\\[x1,y1,x2,y2, conf, cls]
    ▼
Pure PyTorch NMS (per-class)
    │
    ▼
StableTracker
    ├── Predict  — Kalman predict step (velocity decay)
    ├── Match 1  — High-conf dets × all tracks  (blended IoU+centroid, strict threshold)
    ├── Match 2  — Low-conf  dets × unmatched tracks  (loose threshold)
    ├── Re-ID    — Unmatched dets × lost tracks  (class-locked, edge-aware)
    └── Spawn    — Remaining dets → new KalmanBox (ghost-suppressed)
    │
    ▼
Annotated Output Video
```



## 🔧 Installation

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/football-player-tracking.git
cd football-player-tracking

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\\\\Scripts\\\\activate

# 3. Install dependencies
pip install -r requirements.txt
```

**requirements.txt**

```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.7.0
numpy>=1.24.0
scipy>=1.10.0
```

> GPU (recommended): install the CUDA build of PyTorch from https://pytorch.org/get-started/locally/



## 🚀 Quick Start

1. Place your `.pth` model file in the same level as .py files and your input video file in `input` folder.
2. Edit the file names at the top of `football_match.py`:

```python
PTH_PATH      = os.path.join(BASE_DIR, "your .pth file")
INPUT_VIDEO   = os.path.join(BASE_DIR, "input", "your input video file")
OUTPUT_VIDEO  = os.path.join(BASE_DIR, "output", "your output video file")
```

3. Run:

```bash
python football_match.py
```

The annotated video is written to `output_video`. Press **ESC** in the preview window to stop early.



## ⚙️ Configuration Reference

|Parameter|Default|Description|
|-|-|-|
|`CONF_THRESHOLD`|`0.35`|Minimum detection confidence|
|`HIGH_CONF`|`0.45`|Threshold separating strict vs loose match pass|
|`IOU_STRICT`|`0.40`|IoU threshold for high-conf detections|
|`IOU_LOOSE`|`0.25`|IoU threshold for low-conf detections|
|`IOU_REID`|`0.25`|Re-ID threshold (centre of frame)|
|`IOU_REID_EDGE`|`0.10`|Re-ID threshold near frame border|
|`MAX_AGE`|`8`|Frames before active track moves to lost|
|`RE_ID_AGE`|`90`|Frames a lost track is remembered for re-ID|
|`MIN_HITS`|`3`|Detections required before a box is rendered|
|`VISIBILITY_LIMIT`|`1`|Hide box if not seen in this many frames|
|`CENTROID_WEIGHT`|`0.55`|Blend weight for centroid proximity in matching|
|`CENTROID_FIXED_SCALE`|`160.0`|Normalisation scale (px) for centroid score|
|`EDGE_MARGIN`|`60`|Pixels from border that defines "edge zone"|
|`VELOCITY_DECAY`|`0.90`|Per-frame decay applied to Kalman velocity|



## 📁 Project Structure

```

football_match.py   # Main tracking script
football_training.py # Model Training Script
requirements.txt
README.md
input/
  └── (place your input videos here)
output/
  └── (tracked videos are written here)
```



## 🧠 How It Works

### Kalman Filter

Each player track maintains a 7-dimensional state vector `[cx, cy, area, aspect_ratio, vx, vy, v_area]`. The filter predicts the next position every frame and corrects it when a detection is matched. Velocity decays each frame (`VELOCITY_DECAY = 0.82`) so coasting tracks don't drift far when detections are briefly lost.

### Two-Pass Hungarian Matching

High-confidence detections are matched first using a strict IoU threshold, then low-confidence detections fill remaining unmatched tracks with a looser threshold. Both passes use a **blended score** (IoU + centroid proximity) to break ties in crowded regions.

### Class-Locked Re-ID

When a player briefly exits frame or becomes fully occluded, their track moves to a "lost" list for up to 90 frames. Any returning detection of the **same class** with sufficient overlap reactivates the original ID — preventing ID inflation. Cross-class re-ID is hard-blocked.

### Ghost Suppression

Unmatched detections that heavily overlap a confirmed track (or lie within 0.8× its diagonal) are absorbed rather than spawning new IDs.



## 🐛 Known Limitations

* The model file (`.pth`) is not included — you need to supply your own YOLOv8-compatible checkpoint trained on football players
* Tracking is single-camera only
* Performance tested on 720p @ 30fps; 1080p will be slower without a GPU




## 🙏 Acknowledgements

* SORT / DeepSORT for the foundational multi-object tracking ideas
* Roboflow for dataset
* The PyTorch team

