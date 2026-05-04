"""
train_model.py
==============
Train a YOLOv8 model on your sports dataset and save weights as .pth

Usage:
    python train_model.py

Requirements:
    pip install ultralytics torch opencv-python numpy

Dataset layout (YOLOv8 format):
    dataset/
        data.yaml
        train/images/*.jpg   train/labels/*.txt
        valid/images/*.jpg   valid/labels/*.txt
        test/images/*.jpg    test/labels/*.txt
"""

import os
import torch
from ultralytics import YOLO

# ─────────────────────────────────────────────
#  CONFIGURATION  — edit these before running
# ─────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_YAML  = os.path.join(BASE_DIR, "dataset", "data.yaml")       # path to your data.yaml
BASE_MODEL    = "yolov8n.pt"              # base pretrained model to fine-tune
                                           # options: yolov8n / s / m / l / x
                                           # (n=fastest, x=most accurate)

SAVE_DIR      = os.path.join(BASE_DIR, "roboflow_football")             # folder where .pth will be saved
PTH_FILENAME  = "sports_model1.pth"        # output .pth file name

# --- Training hyper-parameters ---
EPOCHS        = 20
IMG_SIZE      = 640
BATCH         = 10
DEVICE        = "cpu"     # GPU id (e.g. "0"); use "cpu" if no CUDA available
PATIENCE      = 15      # early-stop patience (epochs with no improvement)
WORKERS       = 4       # dataloader workers (reduce to 0 on Windows if errors)

# ─────────────────────────────────────────────


def train_and_save():
    print("\n" + "=" * 60)
    print("  YOLOv8 Sports Tracker — Training Script")
    print("=" * 60)
    print(f"  Dataset  : {DATASET_YAML}")
    print(f"  Base model  : {BASE_MODEL}")
    print(f"  Epochs      : {EPOCHS}  |  Batch : {BATCH}  |  Img : {IMG_SIZE}")
    print(f"  Device      : {DEVICE}")
    print("=" * 60 + "\n")

    # ── 1. Load pretrained YOLOv8 ───────────────────────────────
    print("[1/4] Loading base model …")
    model = YOLO(BASE_MODEL)   # downloads automatically on first run

    # ── 2. Fine-tune on your dataset ────────────────────────────
    print("[2/4] Starting training …\n")
    results = model.train(
        data      = DATASET_YAML,
        epochs    = EPOCHS,
        imgsz     = IMG_SIZE,
        batch     = BATCH,
        device    = DEVICE,
        patience  = PATIENCE,
        workers   = WORKERS,
        name      = "sports_tracking",   # saved under runs/detect/sports_tracking/
        exist_ok  = True,
        augment   = True,                # mosaic, flip, colour-jitter augmentations
        cache     = False,               # set True to cache images in RAM
        verbose   = True,
    )

    best_pt_path = str(results.save_dir / "weights" / "best.pt")
    print(f"\n[2/4] Training complete. Best weights (ultralytics format): {best_pt_path}")

    # ── 3. Evaluate on validation set ───────────────────────────
    print("\n[3/4] Evaluating on validation set …")
    best_model = YOLO(best_pt_path)
    metrics    = best_model.val(data=DATASET_YAML)
    print(f"  mAP50     : {metrics.box.map50:.4f}")
    print(f"  mAP50-95  : {metrics.box.map:.4f}")

    # ── 4. Save full model as .pth ───────────────────────────────
    print(f"\n[4/4] Saving model as .pth …")
    os.makedirs(SAVE_DIR, exist_ok=True)
    pth_path = os.path.join(SAVE_DIR, PTH_FILENAME)

    # Save the complete YOLO model object (handles fused/unfused automatically)
    torch.save(
        {
            "model"    : best_model.model,   # full model object, not just state dict
            "nc"       : best_model.model.nc,
            "names"    : best_model.model.names,
            "imgsz"    : IMG_SIZE,
            "map50"    : metrics.box.map50,
            "map50_95" : metrics.box.map,
        },
        pth_path,
    )

    # Clean up the Ultralytics runs/ folder — we only need the .pth
    import shutil
    runs_dir = str(results.save_dir.parent.parent)   # e.g. runs/detect
    if os.path.exists(runs_dir):
        shutil.rmtree(runs_dir)
        print(f"  Cleaned up temporary runs folder: {runs_dir}")

    print("\n" + "=" * 60)
    print("  ✅  Saved successfully!")
    print(f"  .pth file : {pth_path}")
    print("=" * 60 + "\n")
    print("Next step → run  inference_video.py  to process your video.\n")

    return pth_path


if __name__ == "__main__":
    train_and_save()