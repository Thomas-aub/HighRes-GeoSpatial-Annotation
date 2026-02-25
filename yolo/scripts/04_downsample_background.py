"""
03b_downsample_background.py
-----------------------------
Reduces the number of empty (background) tiles in the train, val, and test splits
to prevent the YOLO model from overfitting to empty water/land.

Instead of deleting, excess background images and labels are safely moved to 
data/processed/archive/.

Target:
  Adjust BACKGROUND_RATIO to control what percentage of the final dataset 
  should consist of empty images. (Default: 0.10 = 10%)
"""

import random
import shutil
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

PROCESSED_DIR   = "data/processed"
BACKGROUND_RATIO = 0.10     # 10% of the final split will be background images
RANDOM_SEED     = 42        # For reproducible downsampling

# =============================================================================

def is_empty(label_path: Path) -> bool:
    """Check if a YOLO label file is completely empty."""
    if not label_path.exists():
        return True
    return len(label_path.read_text().strip()) == 0


def process_split(split_name: str, processed_dir: Path, rng: random.Random):
    images_dir = processed_dir / "images" / split_name
    labels_dir = processed_dir / "labels" / split_name
    
    archive_img_dir = processed_dir / "archive" / "images" / split_name
    archive_lbl_dir = processed_dir / "archive" / "labels" / split_name

    if not images_dir.exists() or not labels_dir.exists():
        return

    # 1. Separate annotated vs background stems
    annotated = []
    background = []
    
    for img_path in images_dir.glob("*.png"):
        stem = img_path.stem
        label_path = labels_dir / f"{stem}.txt"
        
        if is_empty(label_path):
            background.append(stem)
        else:
            annotated.append(stem)

    n_annotated = len(annotated)
    n_background = len(background)
    n_total_current = n_annotated + n_background

    if n_annotated == 0:
        print(f"  [{split_name.upper()}] No annotated images found. Skipping.")
        return

    # 2. Calculate target background count
    # Formula: BG = Annotated * (Ratio / (1 - Ratio))
    target_bg_count = int(n_annotated * (BACKGROUND_RATIO / (1.0 - BACKGROUND_RATIO)))
    
    # We can't keep more backgrounds than we actually have
    target_bg_count = min(target_bg_count, n_background)
    
    # 3. Shuffle and split backgrounds
    rng.shuffle(background)
    keep_bg = background[:target_bg_count]
    move_bg = background[target_bg_count:]

    # 4. Move excess files to archive
    if move_bg:
        archive_img_dir.mkdir(parents=True, exist_ok=True)
        archive_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        for stem in move_bg:
            # Move PNG
            src_img = images_dir / f"{stem}.png"
            if src_img.exists():
                shutil.move(str(src_img), str(archive_img_dir / f"{stem}.png"))
                
            # Move TXT
            src_lbl = labels_dir / f"{stem}.txt"
            if src_lbl.exists():
                shutil.move(str(src_lbl), str(archive_lbl_dir / f"{stem}.txt"))

    # 5. Print summary
    n_total_new = n_annotated + target_bg_count
    actual_ratio = (target_bg_count / n_total_new * 100) if n_total_new > 0 else 0

    print(f"  {split_name.upper()}:")
    print(f"    Annotated images : {n_annotated}")
    print(f"    Background kept  : {target_bg_count}  ({actual_ratio:.1f}% of new split)")
    print(f"    Background moved : {len(move_bg)}  -> archive/")
    print(f"    Old split size   : {n_total_current}")
    print(f"    New split size   : {n_total_new}\n")


def main():
    print(f"{'='*60}")
    print(f"  Downsampling Background Tiles (Target: {BACKGROUND_RATIO*100:.1f}%)")
    print(f"{'='*60}\n")
    
    rng = random.Random(RANDOM_SEED)
    base_dir = Path(PROCESSED_DIR)
    
    for split in ["train", "val", "test"]:
        process_split(split, base_dir, rng)
        
    print(f"{'='*60}")
    print(f"  Done. Excess files safely moved to '{base_dir / 'archive'}'")


if __name__ == "__main__":
    main()