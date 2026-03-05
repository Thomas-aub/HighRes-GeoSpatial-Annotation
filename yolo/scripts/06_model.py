"""
06_model.py
------------
Trains a YOLO26 OBB (Oriented Bounding Box) model on the boat detection dataset
prepared by scripts 01 → 05.

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

import albumentations as A


# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Dataset -----------------------------------------------------------------
DATASET_YAML = "data/dataset.yaml"

# --- Model -------------------------------------------------------------------
MODEL_WEIGHTS = "yolo26m-obb.pt"     # nano=n  small=s  medium=m  large=l  xlarge=x

# --- Training hyperparameters ------------------------------------------------
EPOCHS        = 300
IMGSZ         = 1536      # matches the output of 05_upsample.py
BATCH_SIZE    = 4         # reduce to 2 if OOM
WORKERS       = 4
PATIENCE      = 30        # early stopping patience (epochs without improvement)

LEARNING_RATE = 0.005     # initial learning rate (lr0)
LR_FINAL      = 0.1       # final LR = lr0 * lrf = 0.005 * 0.1 = 0.0005
                           # (lrf is a multiplier fraction of lr0, not an absolute value)
MOMENTUM      = 0.937
WEIGHT_DECAY  = 0.0005
WARMUP_EPOCHS = 3.0

# --- Augmentation ------------------------------------------------------------
# Geometric augmentations are handled entirely by YOLO natively.
# Albumentations is restricted to PHOTOMETRIC transforms only — geometric
# transforms (rotate, shift, scale) must NOT be applied via Albumentations for
# OBB tasks because Albumentations uses the standard 4-value YOLO box format
# [x, y, w, h] and cannot correctly transform oriented 8-coordinate OBB labels.

MOSAIC        = 0.5       # default 1.0; mosaic is very effective for small datasets
CLOSE_MOSAIC  = 30        # disable mosaic for the last N epochs to stabilise training
DEGREES       = 180.0     # full rotation range — critical for overhead satellite imagery
FLIPLR        = 0.5       # horizontal flip
FLIPUD        = 0.5       # vertical flip (valid for top-down views)
SCALE         = 0.5       # random scale ± fraction
MULTI_SCALE   = 0.0       # disabled — images are already pre-upsampled to a fixed size
                           # in step 05; rescaling again would undo that work.
                           # If enabled, must be a float (e.g. 0.2), NOT a bool.
PERSPECTIVE   = 0.0       # disabled — orthorectified satellite images have no
                           # perspective distortion; this augmentation adds noise only.
HSV_H         = 0.015
HSV_S         = 0.4
HSV_V         = 0.3

# --- Run management ----------------------------------------------------------
RUN_NAME      = "boat_obb_v1"
PROJECT       = "runs/obb"
DEVICE        = 0                   # GPU id (int), "cpu", or "0,1" for multi-GPU
RESUME        = False               # set True to resume from last checkpoint
SAVE_PERIOD   = 30                  # save checkpoint every N epochs (0 = only best/last)

# =============================================================================
# Albumentations pipeline  —  PHOTOMETRIC TRANSFORMS ONLY
# =============================================================================
# Rules for OBB + Albumentations:
#   ✓  Colour / brightness / blur / noise transforms  →  safe
#   ✗  Any geometric transform (rotate, shift, scale, flip, crop, perspective)
#      →  will silently corrupt OBB corner coordinates; use YOLO's native params instead.
#
# No bbox_params are needed here because we are not transforming bounding boxes
# through Albumentations — YOLO manages all label transforms internally.
# =============================================================================

ALBUMENTATIONS_TRANSFORMS = [
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.5,
    ),
    A.CLAHE(
        clip_limit=2.0,
        p=0.3,
    ),
    A.OneOf(
        [
            A.Blur(blur_limit=3),
            A.MedianBlur(blur_limit=3),
        ],
        p=0.3,
    ),
    # CoarseDropout API changed in albumentations >= 1.4:
    #   old: max_holes, min_holes, fill_value
    #   new: num_holes_range, hole_height_range, hole_width_range, fill
    A.CoarseDropout(
        num_holes_range=(1, 1),
        hole_height_range=(16, 16),
        hole_width_range=(16, 16),
        fill=0,
        p=0.2,
    ),
]

# Note on `erasing`:
#   The Ultralytics `erasing` parameter is only supported for the `classify`
#   task. It is NOT available for OBB and has been removed from this config.

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
        patience      = PATIENCE,

        # Optimiser
        lr0           = LEARNING_RATE,
        lrf           = LR_FINAL,
        momentum      = MOMENTUM,
        weight_decay  = WEIGHT_DECAY,
        warmup_epochs = WARMUP_EPOCHS,

        # YOLO native augmentation
        # `augment=True` keeps all built-in YOLO augmentations active.
        # `augmentations` injects the Albumentations photometric pipeline on top.
        augment       = True,
        augmentations = ALBUMENTATIONS_TRANSFORMS,
        mosaic        = MOSAIC,
        close_mosaic  = CLOSE_MOSAIC,
        degrees       = DEGREES,
        fliplr        = FLIPLR,
        flipud        = FLIPUD,
        scale         = SCALE,
        multi_scale   = MULTI_SCALE,
        perspective   = PERSPECTIVE,
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