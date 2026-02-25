"""
04_model.py
------------
Trains a YOLO26 OBB (Oriented Bounding Box) model on the boat detection dataset
prepared by scripts 01 → 03.

Model variants (trade-off speed vs accuracy):
  yolo26n-obb.pt   nano   – fastest, least accurate, good for prototyping
  yolo26s-obb.pt   small  – good balance for most use-cases
  yolo26m-obb.pt   medium
  yolo26l-obb.pt   large
  yolo26x-obb.pt   xlarge – most accurate, slowest

The pretrained weights are downloaded automatically from Ultralytics on first run.
Training resumes from the last checkpoint if RESUME = True.

Outputs are saved under runs/obb/<RUN_NAME>/:
  weights/best.pt    – best checkpoint (use this for inference)
  weights/last.pt    – last epoch checkpoint
  results.csv        – per-epoch metrics
  confusion_matrix.png, PR_curve.png, …
"""

from pathlib import Path
from ultralytics import YOLO


# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Dataset -----------------------------------------------------------------
DATASET_YAML = "data/dataset.yaml"   # path to the dataset config file

# --- Model -------------------------------------------------------------------
# Pretrained checkpoint to start from.
# Use a .pt file to fine-tune from DOTA-pretrained weights (recommended).
# Use a .yaml file to train from scratch (slower convergence).
MODEL_WEIGHTS = "yolo26s-obb.pt"     # nano=n  small=s  medium=m  large=l  xlarge=x

# --- Training hyperparameters ------------------------------------------------
EPOCHS        = 100
IMGSZ         = 1024      # must match TILE_SIZE in scripts 01/02
BATCH_SIZE    = 8         # reduce if you run out of GPU memory (try 4 or 2)
WORKERS       = 4         # dataloader worker threads

LEARNING_RATE = 0.01      # initial learning rate (lr0)
LR_FINAL      = 0.01      # final lr as a fraction of lr0  (lrf = LR_FINAL * lr0)
MOMENTUM      = 0.937
WEIGHT_DECAY  = 0.0005
WARMUP_EPOCHS = 3.0

# --- Augmentation ------------------------------------------------------------
# These defaults work well for satellite imagery; tune if needed.
AUGMENT       = True
MOSAIC        = 1.0       # mosaic augmentation probability (0.0 to disable)
DEGREES       = 180.0     # rotation range – important for OBB on overhead imagery
FLIPLR        = 0.5       # horizontal flip probability
FLIPUD        = 0.5       # vertical flip probability   (useful for overhead views)
SCALE         = 0.5       # image scale (+/- fraction)
HSV_H         = 0.015     # hue augmentation
HSV_S         = 0.7       # saturation augmentation
HSV_V         = 0.4       # value (brightness) augmentation

# --- Run management ----------------------------------------------------------
RUN_NAME      = "boat_obb_v1"           # subfolder name under runs/obb/
PROJECT       = "runs/obb"
DEVICE        = 0                        # GPU id (int) or "cpu" or "0,1" for multi-GPU
RESUME        = False                    # set True to resume from last checkpoint
SAVE_PERIOD   = 10                       # save checkpoint every N epochs (0 = only best/last)

# =============================================================================


def main():
    print(f"{'='*60}")
    print(f"  YOLO26 OBB Training")
    print(f"{'='*60}")
    print(f"  Model    : {MODEL_WEIGHTS}")
    print(f"  Dataset  : {DATASET_YAML}")
    print(f"  Epochs   : {EPOCHS}")
    print(f"  Img size : {IMGSZ}")
    print(f"  Batch    : {BATCH_SIZE}")
    print(f"  Device   : {DEVICE}")
    print(f"  Run name : {RUN_NAME}")
    print()

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    if RESUME:
        # Resume from the last saved checkpoint of this run
        last_ckpt = Path(PROJECT) / RUN_NAME / "weights" / "last.pt"
        if not last_ckpt.exists():
            raise FileNotFoundError(
                f"Cannot resume: checkpoint not found at '{last_ckpt}'. "
                "Set RESUME = False to start a new run."
            )
        print(f"  Resuming from {last_ckpt}")
        model = YOLO(str(last_ckpt))
    else:
        model = YOLO(MODEL_WEIGHTS)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    results = model.train(
        data          = DATASET_YAML,
        epochs        = EPOCHS,
        imgsz         = IMGSZ,
        batch         = BATCH_SIZE,
        workers       = WORKERS,
        device        = DEVICE,
        project       = PROJECT,
        name          = RUN_NAME,
        resume        = RESUME,
        save_period   = SAVE_PERIOD,

        # Optimiser
        lr0           = LEARNING_RATE,
        lrf           = LR_FINAL,
        momentum      = MOMENTUM,
        weight_decay  = WEIGHT_DECAY,
        warmup_epochs = WARMUP_EPOCHS,

        # Augmentation
        augment       = AUGMENT,
        mosaic        = MOSAIC,
        degrees       = DEGREES,
        fliplr        = FLIPLR,
        flipud        = FLIPUD,
        scale         = SCALE,
        hsv_h         = HSV_H,
        hsv_s         = HSV_S,
        hsv_v         = HSV_V,

        # Misc
        verbose       = True,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    best_ckpt = Path(PROJECT) / RUN_NAME / "weights" / "best.pt"
    print()
    print(f"{'='*60}")
    print(f"  Training complete.")
    print(f"  Best checkpoint : {best_ckpt}")
    print(f"  Results         : {Path(PROJECT) / RUN_NAME}")
    print(f"{'='*60}")

    return results


if __name__ == "__main__":
    main()