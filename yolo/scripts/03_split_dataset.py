"""
03_split_dataset.py
--------------------
Randomly splits tiles from data/processed/images/train/ into train / val / test
while preserving approximate class balance across all three splits.

Strategy:
  - Annotated tiles (at least one label line) are split with multi-label
    stratification: tiles are grouped by their "class signature" (frozenset of
    class IDs present) and each group is distributed proportionally across splits.
    This ensures every class appears in every split at the right ratio.
  - Background tiles (empty .txt) are split purely at random with the same
    proportions, independently of the annotated pool.

The script MOVES both the .png (images/) and the .txt (labels/) files.
It is safe to re-run: it only touches files still sitting in the train/ folder.

Output structure (already created by scripts 01 & 02):
  data/processed/images/train/   val/   test/
  data/processed/labels/train/   val/   test/
"""

import random
import shutil
from collections import defaultdict
from pathlib import Path


# =============================================================================
# CONFIGURATION
# =============================================================================

PROCESSED_DIR = "data/processed"   # root of the processed dataset

TRAIN_RATIO   = 0.70               # fraction kept in train
VAL_RATIO     = 0.15               # fraction moved to val
TEST_RATIO    = 0.15               # fraction moved to test
# TRAIN + VAL + TEST must sum to 1.0

RANDOM_SEED   = 42                 # for reproducibility

# =============================================================================

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, \
    "TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_classes(label_path: Path) -> frozenset:
    """Return the set of class IDs present in a YOLO .txt label file."""
    classes = set()
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            line = line.strip()
            if line:
                classes.add(int(line.split()[0]))
    return frozenset(classes)


def split_indices(n: int, train_r: float, val_r: float, rng: random.Random):
    """
    Randomly partition range(n) into three index lists.
    Returns (train_idx, val_idx, test_idx).
    """
    indices = list(range(n))
    rng.shuffle(indices)
    n_val   = max(1, round(n * val_r))  if n >= 3 else 0
    n_test  = max(1, round(n * (1 - train_r - val_r))) if n >= 3 else 0
    n_train = n - n_val - n_test
    return indices[:n_train], indices[n_train:n_train + n_val], indices[n_train + n_val:]


def move_tile(stem: str, src_split: str, dst_split: str, processed_dir: Path):
    """Move image .png and label .txt from one split folder to another."""
    for subdir, ext in [("images", ".png"), ("labels", ".txt")]:
        src = processed_dir / subdir / src_split / f"{stem}{ext}"
        dst = processed_dir / subdir / dst_split / f"{stem}{ext}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.move(str(src), str(dst))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng           = random.Random(RANDOM_SEED)
    processed_dir = Path(PROCESSED_DIR)
    images_train  = processed_dir / "images" / "train"
    labels_train  = processed_dir / "labels" / "train"

    # Ensure destination folders exist
    for split in ("val", "test"):
        (processed_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (processed_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Collect all tile stems currently in train/
    # ------------------------------------------------------------------
    all_stems = [p.stem for p in sorted(images_train.glob("*.png"))]
    if not all_stems:
        print(f"[ERROR] No .png files found in '{images_train}'")
        return

    print(f"Total tiles in train/ : {len(all_stems)}")

    # ------------------------------------------------------------------
    # 2. Separate annotated vs background tiles
    # ------------------------------------------------------------------
    annotated   = []   # (stem, frozenset_of_classes)
    background  = []   # stem

    for stem in all_stems:
        label_path = labels_train / f"{stem}.txt"
        classes    = read_classes(label_path)
        if classes:
            annotated.append((stem, classes))
        else:
            background.append(stem)

    print(f"  Annotated tiles     : {len(annotated)}")
    print(f"  Background tiles    : {len(background)}")

    # ------------------------------------------------------------------
    # 3. Stratified split for annotated tiles
    #    Group by class signature, then split each group proportionally.
    # ------------------------------------------------------------------
    groups: dict[frozenset, list] = defaultdict(list)
    for stem, classes in annotated:
        groups[classes].append(stem)

    train_stems, val_stems, test_stems = [], [], []

    print(f"\n  Annotated split (stratified by class signature):")
    for sig, stems in sorted(groups.items(), key=lambda x: -len(x[1])):
        tr_idx, va_idx, te_idx = split_indices(len(stems), TRAIN_RATIO, VAL_RATIO, rng)
        tr = [stems[i] for i in tr_idx]
        va = [stems[i] for i in va_idx]
        te = [stems[i] for i in te_idx]
        train_stems += tr
        val_stems   += va
        test_stems  += te
        sig_str = "{" + ",".join(str(c) for c in sorted(sig)) + "}"
        print(f"    classes {sig_str:20s} : {len(stems):4d} total  ->  "
              f"train {len(tr)}, val {len(va)}, test {len(te)}")

    # ------------------------------------------------------------------
    # 4. Random split for background tiles
    # ------------------------------------------------------------------
    tr_idx, va_idx, te_idx = split_indices(len(background), TRAIN_RATIO, VAL_RATIO, rng)
    bg_train = [background[i] for i in tr_idx]
    bg_val   = [background[i] for i in va_idx]
    bg_test  = [background[i] for i in te_idx]

    train_stems += bg_train
    val_stems   += bg_val
    test_stems  += bg_test

    print(f"\n  Background split (random):")
    print(f"    {len(background):5d} total  ->  "
          f"train {len(bg_train)}, val {len(bg_val)}, test {len(bg_test)}")

    # ------------------------------------------------------------------
    # 5. Move files
    # ------------------------------------------------------------------
    print(f"\n  Moving files ...")
    for stem in val_stems:
        move_tile(stem, "train", "val", processed_dir)
    for stem in test_stems:
        move_tile(stem, "train", "test", processed_dir)

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    total = len(train_stems) + len(val_stems) + len(test_stems)
    print(f"\n{'='*60}")
    print(f"  Split complete  (seed={RANDOM_SEED})")
    print(f"{'='*60}")
    print(f"  {'Split':<8}  {'Total':>7}  {'Annotated':>10}  {'Background':>11}  {'Ratio':>7}")
    print(f"  {'-'*50}")

    val_set  = set(val_stems)
    test_set = set(test_stems)

    ann_val   = sum(1 for s, _ in annotated if s in val_set)
    ann_test  = sum(1 for s, _ in annotated if s in test_set)
    ann_train = len(annotated) - ann_val - ann_test

    for split_name, stems, bg_count, ann_count in [
        ("train", train_stems, len(bg_train), ann_train),
        ("val",   val_stems,   len(bg_val),   ann_val),
        ("test",  test_stems,  len(bg_test),  ann_test),
    ]:
        pct = len(stems) / total * 100 if total else 0
        print(f"  {split_name:<8}  {len(stems):>7}  {ann_count:>10}  {bg_count:>11}  {pct:>6.1f}%")

    print(f"  {'-'*50}")
    print(f"  {'TOTAL':<8}  {total:>7}  {len(annotated):>10}  {len(background):>11}")
    print()


if __name__ == "__main__":
    main()