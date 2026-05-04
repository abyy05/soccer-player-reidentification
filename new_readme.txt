Football Player Tracking System
YOLOv8 + Kalman Filter + Stable Multi-Object Tracking

--------------------------------------------------

DEMO
Place your demo GIF at:
assets/demo.gif

Example (GitHub will render this automatically if renamed to README.md):
<img src="assets/demo.gif" width="800"/>

--------------------------------------------------

HIGHLIGHTS

- YOLOv8 custom-trained player detection
- Stable multi-object tracking with Kalman Filter
- IoU + centroid blended matching (reduces ID swaps)
- Class-locked IDs (prevents team switching)
- Edge-aware re-identification
- Ghost/duplicate suppression

--------------------------------------------------

PROJECT STRUCTURE

project/

    train_model.py
    inference_video_stable.py

    dataset/
        data.yaml
        train/
        valid/
        test/

    input/
        match.mp4

    outputs/
        sports_model.pth

    output/
        tracked_video.mp4

    assets/
        demo.gif

--------------------------------------------------

INSTALLATION

pip install ultralytics torch opencv-python numpy scipy

--------------------------------------------------

TRAINING

python train_model.py

--------------------------------------------------

INFERENCE (TRACKING)

python inference_video_stable.py

--------------------------------------------------

CONFIGURATION

PTH_PATH     = outputs/sports_model.pth
INPUT_VIDEO  = input/match.mp4
OUTPUT_VIDEO = output/tracked_video.mp4

--------------------------------------------------

OUTPUT

- Bounding boxes
- Player IDs
- Team classification (Blue / Red)
- Saved video: output/tracked_video.mp4

--------------------------------------------------

CHALLENGES SOLVED

ID switching in crowds
→ Fixed using IoU + centroid matching

Players disappearing
→ Re-identification logic

Duplicate detections
→ Absorption mechanism

Cross-team ID confusion
→ Class-locked tracking

--------------------------------------------------

TECH STACK

- Python
- PyTorch
- YOLOv8 (Ultralytics)
- OpenCV
- NumPy / SciPy

--------------------------------------------------

TIPS

Use GPU:
DEVICE = "cuda"

Disable preview window:
SHOW_WINDOW = False

--------------------------------------------------

FUTURE WORK

- Ball tracking
- Player re-identification (deep embeddings)
- Heatmaps and analytics
- Web dashboard

--------------------------------------------------

AUTHOR

Abhik
Computer Vision / Deep Learning

--------------------------------------------------
