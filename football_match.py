
import os
import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment



BASE_DIR = os.path.dirname(os.path.abspath(__file__))


PTH_PATH      = os.path.join(BASE_DIR, "sports_model.pth")
INPUT_VIDEO   = os.path.join(BASE_DIR, "input", "15sec_input_720p.mp4")
OUTPUT_VIDEO  = os.path.join(BASE_DIR, "output", "output_video55.mp4")

CONF_THRESHOLD = 0.35
HIGH_CONF      = 0.45
IOU_STRICT     = 0.40
IOU_LOOSE      = 0.25
IOU_REID       = 0.25
IMG_SIZE       = 640
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"


MAX_AGE          = 8
RE_ID_AGE        = 90
MIN_HITS         = 3
VISIBILITY_LIMIT = 1


CENTROID_WEIGHT  = 0.55   
EDGE_MARGIN      = 60
IOU_REID_EDGE    = 0.10

VELOCITY_DECAY = 0.90
MIN_BOX_AREA   = 800
SHOW_WINDOW    = True

CLASS_NAMES  = {0: "Blue Player", 1: "Red Player"}
CLASS_COLORS = {0: (255, 150, 0), 1: (0, 0, 200)}


def run_nms(preds, conf_thres, iou_thres=0.45):
    if isinstance(preds, (list, tuple)): preds = preds[0]
    if preds.ndim == 2: preds = preds.unsqueeze(0)
    if preds.ndim == 3 and preds.shape[1] < preds.shape[2]:
        preds = preds.permute(0, 2, 1).contiguous()

    output = []
    for pred in preds:
        cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        cls_scores = pred[:, 4:]
        conf, cls_id = cls_scores.max(1)
        mask = conf > conf_thres
        conf, cls_id = conf[mask], cls_id[mask]
        cx, cy, w, h = cx[mask], cy[mask], w[mask], h[mask]

        if conf.numel() == 0:
            output.append(torch.zeros((0, 6), device=pred.device))
            continue

        x1, y1 = cx - w/2, cy - h/2
        x2, y2 = cx + w/2, cy + h/2
        boxes = torch.stack([x1, y1, x2, y2], 1)

        keeps = []
        for cid in cls_id.unique():
            m = cls_id == cid
            idx = m.nonzero(as_tuple=False).squeeze(1)
            cur_boxes  = boxes[m]
            cur_scores = conf[m]
            order = cur_scores.argsort(descending=True)
            keep = []
            while order.numel() > 0:
                i = order[0].item()
                keep.append(i)
                if order.numel() == 1: break
                ix1 = torch.max(cur_boxes[i, 0], cur_boxes[order[1:], 0])
                iy1 = torch.max(cur_boxes[i, 1], cur_boxes[order[1:], 1])
                ix2 = torch.min(cur_boxes[i, 2], cur_boxes[order[1:], 2])
                iy2 = torch.min(cur_boxes[i, 3], cur_boxes[order[1:], 3])
                inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
                a1 = (cur_boxes[i,2]-cur_boxes[i,0])*(cur_boxes[i,3]-cur_boxes[i,1])
                a2 = (cur_boxes[order[1:],2]-cur_boxes[order[1:],0])*(cur_boxes[order[1:],3]-cur_boxes[order[1:],1])
                iou = inter / (a1 + a2 - inter).clamp(1e-6)
                order = order[1:][iou <= iou_thres]
            keeps.append(idx[torch.tensor(keep, device=pred.device)])

        if not keeps:
            output.append(torch.zeros((0, 6), device=pred.device))
        else:
            keep   = torch.cat(keeps)
            result = torch.cat([boxes[keep], conf[keep].unsqueeze(1), cls_id[keep].float().unsqueeze(1)], 1)
            output.append(result)
    return output


class KalmanBox:
    count        = 0  
    class_counts = {} 

    def __init__(self, bbox, cls_id):
        cx   = (bbox[0]+bbox[2])/2
        cy   = (bbox[1]+bbox[3])/2
        area = max((bbox[2]-bbox[0])*(bbox[3]-bbox[1]), 1.0)
        rat  = max((bbox[2]-bbox[0])/float(bbox[3]-bbox[1]+1e-6), 0.1)

        self.x = np.array([[cx],[cy],[area],[rat],[0.],[0.],[0.]], dtype=float)
        self.F = np.eye(7); self.F[0,4]=1; self.F[1,5]=1; self.F[2,6]=1
        self.H = np.zeros((4,7)); self.H[0,0]=1; self.H[1,1]=1; self.H[2,2]=1; self.H[3,3]=1
        self.P = np.diag([5., 5., 50., 0.5, 1e3, 1e3, 500.])
        self.Q = np.diag([0.5, 0.5, 5., 0.05, 0.005, 0.005, 0.5])
        self.R = np.diag([3., 3., 20., 0.2])

        KalmanBox.count += 1
        KalmanBox.class_counts[cls_id] = KalmanBox.class_counts.get(cls_id, 0) + 1
        self.id                  = KalmanBox.class_counts[cls_id] 
        self.age                 = 0
        self.hits                = 1
        self.hit_streak          = 1
        self.cls_id              = cls_id  
        self.frames_since_update = 0

    def predict(self):
        self.x[4] *= VELOCITY_DECAY
        self.x[5] *= VELOCITY_DECAY
        self.x[6] *= VELOCITY_DECAY
        if self.x[2] <= 0: self.x[2] = 1000
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.frames_since_update += 1
        self.hit_streak = 0

    def update(self, bbox):
    
   
        cx   = (bbox[0]+bbox[2])/2
        cy   = (bbox[1]+bbox[3])/2
        area = max((bbox[2]-bbox[0])*(bbox[3]-bbox[1]), 1.0)
        rat  = max((bbox[2]-bbox[0])/float(bbox[3]-bbox[1]+1e-6), 0.1)
        z = np.array([[cx],[cy],[area],[rat]])
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(7) - K @ self.H) @ self.P
        self.hits  += 1
        self.hit_streak += 1
        self.age   = 0
        self.frames_since_update = 0

    def get_bbox(self):
        cx, cy, area, rat = self.x[0,0], self.x[1,0], self.x[2,0], self.x[3,0]
        area = max(area, 100.0); rat = max(rat, 0.1)
        w = np.sqrt(area * rat); h = area / (w + 1e-6)
        return [cx-w/2, cy-h/2, cx+w/2, cy+h/2]


def iou_matrix(boxes_a, boxes_b):
    N, M = len(boxes_a), len(boxes_b)
    if N == 0 or M == 0: return np.zeros((N, M))
    x1 = np.maximum(boxes_a[:,0,None], boxes_b[None,:,0])
    y1 = np.maximum(boxes_a[:,1,None], boxes_b[None,:,1])
    x2 = np.minimum(boxes_a[:,2,None], boxes_b[None,:,2])
    y2 = np.minimum(boxes_a[:,3,None], boxes_b[None,:,3])
    inter  = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
    area_a = (boxes_a[:,2]-boxes_a[:,0]) * (boxes_a[:,3]-boxes_a[:,1])
    area_b = (boxes_b[:,2]-boxes_b[:,0]) * (boxes_b[:,3]-boxes_b[:,1])
    return inter / (area_a[:,None] + area_b[None,:] - inter + 1e-6)

def blended_score_matrix(boxes_a, boxes_b):
  
    N, M = len(boxes_a), len(boxes_b)
    if N == 0 or M == 0: return np.zeros((N, M))

    iou = iou_matrix(boxes_a, boxes_b)

    ca = np.stack([(boxes_a[:,0]+boxes_a[:,2])/2, (boxes_a[:,1]+boxes_a[:,3])/2], axis=1)
    cb = np.stack([(boxes_b[:,0]+boxes_b[:,2])/2, (boxes_b[:,1]+boxes_b[:,3])/2], axis=1)
    dx   = ca[:,0,None] - cb[None,:,0]
    dy   = ca[:,1,None] - cb[None,:,1]
    dist = np.sqrt(dx**2 + dy**2)
    scale      = np.median(dist[dist > 0]) if np.any(dist > 0) else 1.0
    dist_score = 1.0 / (1.0 + dist / (scale + 1e-6))

    return (1.0 - CENTROID_WEIGHT) * iou + CENTROID_WEIGHT * dist_score

def hungarian_match(score, threshold):
    if score.size == 0:
        return [], list(range(score.shape[0])), list(range(score.shape[1]))
    row_ind, col_ind = linear_sum_assignment(-score)
    matched, matched_d, matched_t = [], set(), set()
    for r, c in zip(row_ind, col_ind):
        if score[r, c] >= threshold:
            matched.append((r, c))
            matched_d.add(r); matched_t.add(c)
    unmatched_d = [d for d in range(score.shape[0]) if d not in matched_d]
    unmatched_t = [t for t in range(score.shape[1]) if t not in matched_t]
    return matched, unmatched_d, unmatched_t

def _iou_pair(b1, b2):
    ix1 = max(b1[0],b2[0]); iy1 = max(b1[1],b2[1])
    ix2 = min(b1[2],b2[2]); iy2 = min(b1[3],b2[3])
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter/(a1+a2-inter+1e-6)

def is_near_edge(box, frame_w, frame_h, margin=EDGE_MARGIN):
    x1, y1, x2, y2 = box
    return x1 < margin or y1 < margin or x2 > frame_w-margin or y2 > frame_h-margin


class StableTracker:
    def __init__(self, iou_strict=IOU_STRICT, iou_loose=IOU_LOOSE,
                 iou_reid=IOU_REID, max_age=MAX_AGE, min_hits=MIN_HITS,
                 re_id_age=RE_ID_AGE, frame_w=1280, frame_h=720):
        self.tracks     = []
        self.lost       = []
        self.iou_strict = iou_strict
        self.iou_loose  = iou_loose
        self.iou_reid   = iou_reid
        self.max_age    = max_age
        self.min_hits   = min_hits
        self.re_id_age  = re_id_age
        self.frame_w    = frame_w
        self.frame_h    = frame_h
        

    def _match_class_locked(self, det_indices, trk_indices, detections, threshold):
        if not det_indices or not trk_indices:
            return [], det_indices, trk_indices

        dets_by_cls = {}
        for idx in det_indices:
            dets_by_cls.setdefault(int(detections[idx][5]), []).append(idx)

        trks_by_cls = {}
        for idx in trk_indices:
            trks_by_cls.setdefault(self.tracks[idx].cls_id, []).append(idx)

        all_pairs = []
        all_rem_d = list(det_indices)
        all_rem_t = list(trk_indices)

        for cls_id in set(list(dets_by_cls.keys()) + list(trks_by_cls.keys())):
            c_dets = dets_by_cls.get(cls_id, [])
            c_trks = trks_by_cls.get(cls_id, [])
            if not c_dets or not c_trks: continue

            d_boxes = np.array([detections[i][:4] for i in c_dets])
            t_boxes = np.array([self.tracks[j].get_bbox() for j in c_trks])

            score = blended_score_matrix(d_boxes, t_boxes)
            pairs_local, _, _ = hungarian_match(score, threshold)

            for r, c in pairs_local:
                global_d = c_dets[r]
                global_t = c_trks[c]
                all_pairs.append((global_d, global_t))
                if global_d in all_rem_d: all_rem_d.remove(global_d)
                if global_t in all_rem_t: all_rem_t.remove(global_t)

        return all_pairs, all_rem_d, all_rem_t

    def _try_reid(self, det_indices, detections):
        remaining_dets = []
        for d_idx in det_indices:
            d_cls = int(detections[d_idx][5])
            d_box = detections[d_idx][:4]
            at_edge   = is_near_edge(d_box, self.frame_w, self.frame_h)
            threshold = IOU_REID_EDGE if at_edge else self.iou_reid

            best_iou   = -1
            best_trk   = None
            best_l_idx = -1
            for l_idx, track in enumerate(self.lost):
                if track.cls_id != d_cls: continue
                iou = _iou_pair(d_box, track.get_bbox())
                if iou > best_iou:
                    best_iou = iou; best_trk = track; best_l_idx = l_idx

            if best_iou >= threshold:
                best_trk.update(d_box)
                best_trk.age = 0
                best_trk.frames_since_update = 0
                self.tracks.append(best_trk)
                self.lost.pop(best_l_idx)
            else:
                remaining_dets.append(d_idx)
        return remaining_dets

    def update(self, detections):
        detections = [d for d in detections if (d[2]-d[0])*(d[3]-d[1]) >= MIN_BOX_AREA]

        for t in self.tracks: t.predict()

        all_d  = list(range(len(detections)))
        all_t  = list(range(len(self.tracks)))
        high_d = [i for i in all_d if detections[i][4] >= HIGH_CONF]
        low_d  = [i for i in all_d if detections[i][4] <  HIGH_CONF]

        pairs1, rem_d1, rem_t1 = self._match_class_locked(high_d, all_t,  detections, self.iou_strict)
        for d_idx, t_idx in pairs1:
            self.tracks[t_idx].update(detections[d_idx][:4])  

        pairs2, rem_d2, rem_t2 = self._match_class_locked(low_d, rem_t1, detections, self.iou_loose)
        for d_idx, t_idx in pairs2:
            self.tracks[t_idx].update(detections[d_idx][:4])

        all_unmatched = rem_d1 + rem_d2
        still_new     = self._try_reid(all_unmatched, detections)

        for d_idx in still_new:
            d_box = np.array(detections[d_idx][:4])
            d_cls = int(detections[d_idx][5])
            d_cx  = (d_box[0]+d_box[2])/2
            d_cy  = (d_box[1]+d_box[3])/2

           
            absorbed          = False
            best_overlap      = 0.0
            best_existing_trk = None
            for t in self.tracks:
                if t.cls_id != d_cls: continue
                iou      = _iou_pair(d_box, np.array(t.get_bbox()))
                weighted = iou * (1.0 + 0.05 * min(t.hits, 20))  
                if weighted > best_overlap:
                    best_overlap      = weighted
                    best_existing_trk = t

            if best_overlap >= 0.35 and best_existing_trk is not None:
                best_existing_trk.update(d_box)
                absorbed = True

           
            if not absorbed:
                for t in [t for t in self.tracks if t.cls_id == d_cls and t.hits >= self.min_hits]:
                    t_box = np.array(t.get_bbox())
                    t_cx  = (t_box[0]+t_box[2])/2
                    t_cy  = (t_box[1]+t_box[3])/2
                    diag  = np.sqrt((t_box[2]-t_box[0])**2 + (t_box[3]-t_box[1])**2)
                    dist  = np.sqrt((d_cx-t_cx)**2 + (d_cy-t_cy)**2)
                    if dist < diag * 0.8: 
                        t.update(d_box)
                        absorbed = True
                        break

            if not absorbed:
                self.tracks.append(KalmanBox(detections[d_idx][:4], d_cls))

       
        alive = []
        for t in self.tracks:
            if t.frames_since_update > self.max_age:
                self.lost.append(t)
            else:
                alive.append(t)
        self.tracks = alive
        self.lost   = [t for t in self.lost if t.frames_since_update <= self.re_id_age]

        return [[*t.get_bbox(), t.id, t.cls_id] for t in self.tracks
                if t.hits >= self.min_hits and t.frames_since_update <= VISIBILITY_LIMIT]


def load_model(pth_path, device):
    print(f"\n[1/3] Loading model: {pth_path}")
    if not os.path.exists(pth_path): raise FileNotFoundError(pth_path)
    ckpt  = torch.load(pth_path, map_location=device, weights_only=False)
    model = ckpt["model"].to(device).eval()
    print(f"  Device: {device}")
    return model

def draw_track(frame, bbox, track_id, cls_id):
    x1, y1, x2, y2 = map(int, bbox)
    color = CLASS_COLORS.get(cls_id, (200, 200, 200))
    label = f"{CLASS_NAMES.get(cls_id, 'Player')} #{track_id}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x1, y1-th-10), (x1+tw+8, y1), color, -1)
    cv2.putText(frame, label, (x1+4, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

def run_inference(model, input_video, output_video):
    print(f"\n[2/3] Opening video: {input_video}")
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened(): raise FileNotFoundError(input_video)

    fps   = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    w, h  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sx, sy = w / IMG_SIZE, h / IMG_SIZE
    print(f"  {w}×{h} @ {fps}fps  |  edge margin: {EDGE_MARGIN}px")

    os.makedirs(os.path.dirname(output_video) or ".", exist_ok=True)
    writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    tracker     = StableTracker(frame_w=w, frame_h=h)
    frame_count = 0
    print(f"\n[3/3] Running inference …\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1

        blob = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        blob = blob[:,:,::-1].transpose(2,0,1)
        blob = torch.from_numpy(blob.copy()).float() / 255.0
        blob = blob.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            raw = model(blob)
        preds = raw[0] if isinstance(raw, (list, tuple)) else raw
        if preds.ndim == 3 and preds.shape[1] < preds.shape[2]:
            preds = preds.permute(0, 2, 1).contiguous()

        nms_out    = run_nms(preds, CONF_THRESHOLD)[0]
        detections = []
        for det in nms_out:
            x1 = float(det[0])*sx; y1 = float(det[1])*sy
            x2 = float(det[2])*sx; y2 = float(det[3])*sy
            detections.append([x1, y1, x2, y2, float(det[4]), int(det[5])])

        tracks = tracker.update(detections)

        for trk in tracks:
            draw_track(frame, (trk[0], trk[1], trk[2], trk[3]), int(trk[4]), int(trk[5]))

        blue_ids = KalmanBox.class_counts.get(0, 0)
        red_ids  = KalmanBox.class_counts.get(1, 0)
        cv2.putText(frame, f"Frame {frame_count}/{total} | Blue IDs: {blue_ids}  Red IDs: {red_ids}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        writer.write(frame)
        if SHOW_WINDOW:
            cv2.imshow("Stable Tracker", frame)
            if cv2.waitKey(1) & 0xFF == 27: break

    cap.release(); writer.release()
    if SHOW_WINDOW: cv2.destroyAllWindows()
    print(f"\n✅ Done! Output: {output_video}")

if __name__ == "__main__":
    model = load_model(PTH_PATH, DEVICE)
    run_inference(model, INPUT_VIDEO, OUTPUT_VIDEO)
