

import os
import torch
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_YAML  = os.path.join(BASE_DIR, "dataset", "data.yaml")      
BASE_MODEL    = "yolov8n.pt"             
                                                                                
SAVE_DIR      = os.path.join(BASE_DIR, "roboflow_football")            
PTH_FILENAME  = "sports_model1.pth"       
# --- --
EPOCHS        = 20
IMG_SIZE      = 640
BATCH         = 10
DEVICE        = "cpu"     
PATIENCE      = 15      
WORKERS       = 4       

def train_and_save():
    print("\n" + "=" * 60)
    print("  YOLOv8 Sports Tracker — Training Script")
    print("=" * 60)
    print(f"  Dataset  : {DATASET_YAML}")
    print(f"  Base model  : {BASE_MODEL}")
    print(f"  Epochs      : {EPOCHS}  |  Batch : {BATCH}  |  Img : {IMG_SIZE}")
    print(f"  Device      : {DEVICE}")
    print("=" * 60 + "\n")


    print("[1/4] Loading base model …")
    model = YOLO(BASE_MODEL)   

    print("[2/4] Starting training …\n")
    results = model.train(
        data      = DATASET_YAML,
        epochs    = EPOCHS,
        imgsz     = IMG_SIZE,
        batch     = BATCH,
        device    = DEVICE,
        patience  = PATIENCE,
        workers   = WORKERS,
        name      = "sports_tracking",   
        exist_ok  = True,
        augment   = True,                
        cache     = False,              
        verbose   = True,
    )

    best_pt_path = str(results.save_dir / "weights" / "best.pt")
    print(f"\n[2/4] Training complete. Best weights (ultralytics format): {best_pt_path}")

    print("\n[3/4] Evaluating on validation set …")
    best_model = YOLO(best_pt_path)
    metrics    = best_model.val(data=DATASET_YAML)
    print(f"  mAP50     : {metrics.box.map50:.4f}")
    print(f"  mAP50-95  : {metrics.box.map:.4f}")
   
    print(f"\n[4/4] Saving model as .pth …")
    os.makedirs(SAVE_DIR, exist_ok=True)
    pth_path = os.path.join(SAVE_DIR, PTH_FILENAME)

   
    torch.save(
        {
            "model"    : best_model.model,   #
            "nc"       : best_model.model.nc,
            "names"    : best_model.model.names,
            "imgsz"    : IMG_SIZE,
            "map50"    : metrics.box.map50,
            "map50_95" : metrics.box.map,
        },
        pth_path,
    )
 
    import shutil
    runs_dir = str(results.save_dir.parent.parent)   
        shutil.rmtree(runs_dir)
        print(f"  Cleaned up temporary runs folder: {runs_dir}")

    print("\n" + "=" * 60)
    print("    Saved successfully!")
    print(f"  .pth file : {pth_path}")
    print("=" * 60 + "\n")
    print("Next step → run  inference_video.py  to process your video.\n")

    return pth_path

if __name__ == "__main__":
    train_and_save()
