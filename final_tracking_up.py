import cv2
import numpy as np
from ultralytics import YOLO


def get_grid_position(center, frame_shape):
    h, w = frame_shape[:2]
    gx = int(center[0] * GRID_SIZE / w)
    gy = int(center[1] * GRID_SIZE / h)
    return gx, gy


def match_player(center, frame_shape):
    gx, gy = get_grid_position(center, frame_shape)
    for pid, data in player_tracks.items():
        prev_gx, prev_gy = get_grid_position(data['center'], frame_shape)
        dist = np.linalg.norm([gx - prev_gx, gy - prev_gy])
        if dist < POSITION_THRESHOLD and frame_count - data['last_seen'] <= MAX_MISSING_FRAMES:
            return pid
    return None


CLASS_NAMES = {0: "Ball", 1: "Goalkeeper", 2: "Player", 3: "Referee"}

cap = cv2.VideoCapture("C:/Users/Sports_opencv/input/15sec_input_720p.mp4")


model = YOLO("C:/Users/Sports_opencv/input/best.pt")


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


out = cv2.VideoWriter("C:/Users/Sports_opencv/output/output_tracked_video.mp4", fourcc, fps, (width, height))

GRID_SIZE = 1000

player_id_counter = 0
player_tracks = {} 
frame_count = 0


MAX_MISSING_FRAMES = 20
POSITION_THRESHOLD = 50  



while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    results = model(frame)[0]
    detections = []

    for det in results.boxes.data:
        x1, y1, x2, y2, conf, cls = det
        cls = int(cls)

        if cls == 2:  
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            matched_id = match_player(center, frame.shape)
            if matched_id is None:
                matched_id = player_id_counter
                player_id_counter += 1

            player_tracks[matched_id] = {'center': center, 'last_seen': frame_count}
            bbox = [x1.item(), y1.item(), x2.item(), y2.item()]
            detections.append((bbox, cls, matched_id))
        elif cls in CLASS_NAMES:
            bbox = [x1.item(), y1.item(), x2.item(), y2.item()]
            detections.append((bbox, cls, -1))  

    
    for bbox, cls, obj_id in detections:
        x1, y1, x2, y2 = map(int, bbox)
        if cls == 2:
            label = f"Player {obj_id}"
            color = (0, 255, 0)
        else:
            label = f"{CLASS_NAMES[cls]}"
            color = (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Grid-based Player Tracking", frame)
    out.write(frame)  

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()