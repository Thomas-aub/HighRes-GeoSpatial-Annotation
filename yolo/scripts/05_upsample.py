"""
04_upsample.py
--------------
Upsamples all tile images in data/processed/images/{train,val,test}/ from
TILE_SIZE to TARGET_SIZE using a high-quality interpolation filter.

WHY UPSAMPLE MANUALLY INSTEAD OF INSIDE YOLO
──────────────────────────────────────────────
YOLO's internal upsampling (imgsz > stored tile size) is done with a fast but
low-quality bilinear resize applied at every forward pass.  Pre-upsampling once
with Lanczos-4 or bicubic produces sharper edges and finer texture, giving the
model better gradient signal at training time and consistent inference quality.
It also removes the per-batch resize overhead during training.

LABEL FILES
────────────
YOLO OBB labels store coordinates as normalised [0, 1] fractions of the image
size.  Upsampling does NOT change those fractions, so label .txt files are
left untouched.

INTERPOLATION OPTIONS
──────────────────────
  INTER_LANCZOS4   (default) – highest quality, slight ringing on very hard edges
  INTER_CUBIC                – slightly softer, no ringing, still very good
  INTER_LINEAR               – fast bilinear, noticeably softer at 2× scale

PERFORMANCE
────────────
All available CPU cores are used via a ProcessPoolExecutor.

OUTPUT
───────
Images are resized IN-PLACE by default (OVERWRITE = True).
Set OVERWRITE = False to write resized copies alongside the originals with a
suffix, e.g. tile_0_0_1536.png, and keep the originals untouched.
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np


# =============================================================================
# CONFIGURATION
# =============================================================================

PROCESSED_DIR = "data/processed"       # root of the processed dataset
SPLITS        = ["train", "val", "test"]

TILE_SIZE     = 768                    # current (source) tile size in pixels
TARGET_SIZE   = 1536                   # desired output size in pixels

# Interpolation method – one of:
#   cv2.INTER_LANCZOS4   highest quality  (recommended for satellite imagery)
#   cv2.INTER_CUBIC      good quality, slightly softer
#   cv2.INTER_LINEAR     fast bilinear, noticeably softer
INTERPOLATION = cv2.INTER_LANCZOS4

# True  → overwrite each file in-place (saves disk space)
# False → write a new file with TARGET_SIZE suffix, keep the original
OVERWRITE     = True

# Number of worker processes (None = use all logical CPU cores)
MAX_WORKERS   = None

# =============================================================================


# ---------------------------------------------------------------------------
# Worker function (runs in a subprocess)
# ---------------------------------------------------------------------------

def _resize_one(args: tuple) -> tuple[str, bool, str]:
    """
    Resize a single image file.

    Returns (path_str, success, error_message).
    Defined at module level so it can be pickled by ProcessPoolExecutor.
    """
    src_path_str, target_size, interpolation, overwrite = args
    src_path = Path(src_path_str)

    try:
        img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
        if img is None:
            return src_path_str, False, "cv2.imread returned None"

        h, w = img.shape[:2]

        # Sanity check: skip if already at target size
        if h == target_size and w == target_size:
            return src_path_str, True, "already target size, skipped"

        # Resize
        resized = cv2.resize(
            img,
            (target_size, target_size),
            interpolation=interpolation,
        )

        # Destination path
        if overwrite:
            dst_path = src_path
        else:
            dst_path = src_path.with_name(f"{src_path.stem}_{target_size}{src_path.suffix}")

        # Write as lossless PNG
        cv2.imwrite(str(dst_path), resized, [cv2.IMWRITE_PNG_COMPRESSION, 3])

        return src_path_str, True, ""

    except Exception as exc:
        return src_path_str, False, str(exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    processed_dir = Path(PROCESSED_DIR)
    workers       = MAX_WORKERS or os.cpu_count() or 1

    if TARGET_SIZE <= TILE_SIZE:
        raise ValueError(
            f"TARGET_SIZE ({TARGET_SIZE}) must be larger than TILE_SIZE ({TILE_SIZE})."
        )

    interp_name = {
        cv2.INTER_LANCZOS4: "INTER_LANCZOS4",
        cv2.INTER_CUBIC:    "INTER_CUBIC",
        cv2.INTER_LINEAR:   "INTER_LINEAR",
    }.get(INTERPOLATION, str(INTERPOLATION))

    print("=" * 60)
    print("  Image Upsampling")
    print("=" * 60)
    print(f"  Source size   : {TILE_SIZE} × {TILE_SIZE} px")
    print(f"  Target size   : {TARGET_SIZE} × {TARGET_SIZE} px  "
          f"(scale factor ×{TARGET_SIZE / TILE_SIZE:.2g})")
    print(f"  Interpolation : {interp_name}")
    print(f"  Mode          : {'overwrite in-place' if OVERWRITE else 'write new file (keep originals)'}")
    print(f"  Workers       : {workers}")
    print()

    # Collect all image paths
    all_paths: list[Path] = []
    for split in SPLITS:
        split_dir = processed_dir / "images" / split
        if not split_dir.exists():
            print(f"  [WARN] Split directory not found: {split_dir} — skipping.")
            continue
        pngs = sorted(split_dir.glob("*.png"))
        print(f"  {split:<6} : {len(pngs):>6} images  ({split_dir})")
        all_paths.extend(pngs)

    if not all_paths:
        print("\n[ERROR] No images found. Check PROCESSED_DIR and SPLITS.")
        return

    total = len(all_paths)
    print(f"\n  Total images to process : {total}")
    print()

    # Build argument tuples for the worker
    task_args = [
        (str(p), TARGET_SIZE, INTERPOLATION, OVERWRITE)
        for p in all_paths
    ]

    # Process with progress tracking
    t0        = time.perf_counter()
    done      = 0
    succeeded = 0
    failed    = 0
    errors: list[tuple[str, str]] = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_resize_one, a): a[0] for a in task_args}

        for future in as_completed(futures):
            path_str, ok, msg = future.result()
            done += 1

            if ok:
                succeeded += 1
            else:
                failed += 1
                errors.append((path_str, msg))

            # Progress bar
            pct    = done / total * 100
            filled = int(pct / 2)           # 50-char bar
            bar    = "█" * filled + "░" * (50 - filled)
            elapsed = time.perf_counter() - t0
            eta     = (elapsed / done) * (total - done) if done else 0
            print(
                f"\r  [{bar}] {pct:5.1f}%  "
                f"{done}/{total}  "
                f"elapsed {elapsed:5.0f}s  ETA {eta:5.0f}s  ",
                end="",
                flush=True,
            )

    elapsed_total = time.perf_counter() - t0
    print()   # newline after progress bar
    print()
    print("=" * 60)
    print("  Done.")
    print("=" * 60)
    print(f"  Succeeded : {succeeded}")
    print(f"  Failed    : {failed}")
    print(f"  Total time: {elapsed_total:.1f}s  "
          f"({elapsed_total / total * 1000:.1f} ms/image)")

    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for path_str, msg in errors[:20]:
            print(f"    {Path(path_str).name}: {msg}")
        if len(errors) > 20:
            print(f"    … and {len(errors) - 20} more.")

    print()
    print("  Labels (.txt) were NOT modified — YOLO OBB coordinates are")
    print("  normalised [0, 1] fractions and are scale-invariant.")
    print()


if __name__ == "__main__":
    main()