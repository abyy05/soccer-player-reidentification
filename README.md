# soccer-player-reidentification



# Grid-Based Player Tracking with YOLO

This project implements a **player tracking system** using [YOLO](https://github.com/ultralytics/ultralytics) and a **grid-based matching algorithm** to assign persistent IDs to players across video frames.

### 🔧 Features

* Detects and classifies objects: `Player`, `Goalkeeper`, `Referee`, and `Ball`
* Assigns unique IDs to players based on spatial grids
* Tracks player movement across frames using position matching
* Saves output with bounding boxes and labels

### 📁 Project Structure

* **Input:**
  `input/15sec_input_720p.mp4` — input video
  `input/best.pt` — custom YOLO model
* **Output:**
  `output/output_tracked_video.mp4` — annotated video with tracking

### 🚀 Requirements

* Python 3.8+
* OpenCV
* NumPy
* Ultralytics YOLO

Install dependencies:

```bash
pip install opencv-python numpy ultralytics
```

### ▶️ Run

```bash
python final_tracking_up.py
```

### ⚙️ Tracking Logic

* The frame is divided into a grid (`GRID_SIZE = 1000`)
* Each detected player's center is mapped to a grid cell
* Matching is done based on grid proximity (`POSITION_THRESHOLD = 50`) and frame gap tolerance (`MAX_MISSING_FRAMES = 20`)

### 📌 Notes

* Press `Esc` to exit while running
* Model must support custom classes: Player (2), Goalkeeper (1), Referee (3), Ball (0)
  
![Demo](/output.gif)



