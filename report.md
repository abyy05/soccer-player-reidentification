
# Football Player Tracking with YOLOv8

##  Overview and Methodology

This project focuses on tracking football players using a custom-trained YOLO model that detects key entities like players, goalkeepers, referees, and the ball. To maintain consistent identities across frames, it leverages a **grid-based position matching** technique instead of relying on computationally heavy appearance-based models.

###  Tracking Pipeline:

1. **Object Detection**
   A YOLO model (`best.pt`) is used to detect objects in each frame.

2. **Class Filtering**
   Only detections with class label `2` (Player) are tracked. Other detected classes (ball, referee, goalkeeper) are visualized but not assigned IDs.

3. **Grid-Based Identity Matching**
   Player centers are projected onto a 1000x1000 grid. If a detection falls within a defined spatial range of a previously tracked position (and within a frame threshold), it is considered the same player.

4. **Track Management**
   Player IDs are maintained using a dictionary of their last known position and the last frame they were seen in. If a player disappears for more than `MAX_MISSING_FRAMES`, their ID is retired and reassigned upon re-entry.

---

##  Techniques Explored

###  Grid Normalization with Distance Threshold

* Simple and efficient.
* Works well in low-occlusion environments.
* Scales across varying resolutions.

###  Direct Pixel Distance

* Lacked consistency across videos of different dimensions.
* Failed in maintaining correct IDs, especially in motion-heavy scenes.

---

##  Common Issues Faced

* **Close Proximity/Overlapping Players**: Grid-based tracking struggles to distinguish players when they’re too close.
* **Fast Movement or Occlusion**: Quick transitions can break tracking if players move beyond the defined threshold.
* **No Visual Feature Matching**: While fast, the current approach lacks robustness in crowded or dynamic settings.

---

##  Current Limitations & Future Enhancements

### Missing Features:

* Visual Re-ID using deep embeddings (e.g., ResNet50).
* Motion continuity across multiple frames.
* Occlusion-aware ID assignment.

### Planned Improvements:

* Temporal smoothing using tracking history.
* Hybrid re-ID logic combining grid location + visual features.
* Better handling of dense scenarios using motion prediction models.

---

##  Final Thoughts

This tracking system is a lightweight and practical solution for football scenarios where speed is critical and full-scale re-ID is unnecessary. It forms a solid baseline, and with enhancements like feature-based matching and trajectory prediction, it can evolve into a much more accurate and robust tracker.

