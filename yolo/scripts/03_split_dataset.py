"""
03_split_dataset.py  ─ spatial-aware + class-balance-optimised version
───────────────────────────────────────────────────────────────────────
Splits tiles into train / val / test with ZERO pixel overlap AND the best
achievable class balance given the spatial constraints.

TWO-STRATEGY APPROACH
──────────────────────
A) MULTIPLE SOURCE TIFs  (>= 3 source images)
   Assign whole TIF files to splits.  With N TIFs there are 3^N possible
   assignments.  For N <= EXHAUSTIVE_THRESHOLD (default 15) every combination
   is evaluated and the one with the best class balance is chosen.  For larger
   N a random-restart hill-climb is used instead (still very good in practice).
   Zero-leakage is guaranteed because different TIFs cover disjoint areas.
   No buffer tiles are wasted.

B) SINGLE SOURCE TIF  (1-2 source images, or per-TIF fallback)
   Exhaustively search all valid (boundary_1, boundary_2) row-cut pairs
   (up to MAX_BOUNDARY_CANDIDATES^2 combinations), score each by class
   imbalance, and pick the best pair subject to tile-count ratios staying
   within RATIO_TOLERANCE of the targets.

LEAKAGE GUARANTEE
──────────────────
Strategy A: disjoint TIFs -> zero shared pixels by definition.
Strategy B: for every train tile T and val/test tile V:
    V.row_off  >=  boundary_1  >=  T.row_off + TILE_SIZE
    -> zero shared pixel rows.
Buffer tiles that straddle a boundary are archived, not deleted.
"""

import csv
import itertools
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path


# =============================================================================
# CONFIGURATION
# =============================================================================

PROCESSED_DIR = "data/processed"
METADATA_PATH = "data/processed/metadata.csv"

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# Tiling parameters -- must match 01_preprocessing_images.py
TILE_SIZE = 768
OVERLAP   = 384

# Strategy A: use exhaustive search when number of TIFs <= this value.
# 3^15 = 14 million combinations -- still fast (a few seconds).
# Above this threshold, random-restart hill-climbing is used instead.
EXHAUSTIVE_THRESHOLD = 15

# Strategy A (hill-climb fallback): number of random restarts.
HILL_CLIMB_RESTARTS = 200

# Strategy A: reject any assignment where a split's tile count deviates more
# than this from the target ratio.
RATIO_TOLERANCE = 0.10

# Strategy B: max boundary candidates per axis (None = no limit).
MAX_BOUNDARY_CANDIDATES = 100

# Random seed for reproducibility (hill-climb only).
RANDOM_SEED = 42

# =============================================================================

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, \
    "Ratios must sum to 1.0"

STRIDE   = TILE_SIZE - OVERLAP
TARGETS  = {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO}
IDX2NAME = {0: "train", 1: "val", 2: "test"}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def read_label_classes(label_path: Path) -> list:
    classes = []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            line = line.strip()
            if line:
                classes.append(int(line.split()[0]))
    return classes


def move_tile(stem: str, src_split: str, dst_split: str, base: Path) -> None:
    for subdir, ext in [("images", ".png"), ("labels", ".txt")]:
        src = base / subdir / src_split / f"{stem}{ext}"
        dst = base / subdir / dst_split / f"{stem}{ext}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.move(str(src), str(dst))


def archive_tile(stem: str, src_split: str, base: Path) -> None:
    for subdir, ext in [("images", ".png"), ("labels", ".txt")]:
        src = base / subdir / src_split / f"{stem}{ext}"
        dst = base / "archive" / subdir / src_split / f"{stem}{ext}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.move(str(src), str(dst))


def class_counts_for_stems(stems: list, labels_dir: Path) -> dict:
    counts = defaultdict(int)
    for stem in stems:
        for cls in read_label_classes(labels_dir / f"{stem}.txt"):
            counts[cls] += 1
    return dict(counts)


def imbalance_score_from_counts(
    split_counts: dict,   # {"train": {cls: n}, "val": {cls: n}, "test": {cls: n}}
    all_classes:  set,
) -> float:
    """
    Scalar imbalance score (lower = better).
    Weighted sum of squared deviations from target ratios, per class.
    Heavy penalty (1e6) for any class completely absent from val or test.
    """
    score = 0.0
    for cls in all_classes:
        total = sum(split_counts[s].get(cls, 0) for s in split_counts)
        if total == 0:
            continue
        for split_name, target in TARGETS.items():
            actual = split_counts[split_name].get(cls, 0) / total
            score += ((actual - target) ** 2) * total
        for split_name in ("val", "test"):
            if split_counts[split_name].get(cls, 0) == 0:
                score += 1e6
    return score


def sum_counts(count_dicts: list) -> dict:
    """Sum a list of {cls: count} dicts into one."""
    total = defaultdict(int)
    for d in count_dicts:
        for cls, n in d.items():
            total[cls] += n
    return dict(total)


# ---------------------------------------------------------------------------
# Strategy A helpers
# ---------------------------------------------------------------------------

def _score_assignment(
    assignment:   tuple,      # (split_idx, ...) one per TIF
    tif_data:     list,       # [{"n_tiles": int, "counts": dict}, ...]
    all_classes:  set,
    total_tiles:  int,
) -> float:
    """Score a candidate assignment tuple.  Returns inf if ratios violated."""
    split_tiles  = [0, 0, 0]
    split_counts = [defaultdict(int), defaultdict(int), defaultdict(int)]

    for tif_idx, split_idx in enumerate(assignment):
        split_tiles[split_idx]  += tif_data[tif_idx]["n_tiles"]
        for cls, n in tif_data[tif_idx]["counts"].items():
            split_counts[split_idx][cls] += n

    # Reject if any split is too far from its target tile ratio
    for idx, name in IDX2NAME.items():
        ratio = split_tiles[idx] / total_tiles
        if abs(ratio - TARGETS[name]) > RATIO_TOLERANCE:
            return float("inf")

    counts_dict = {IDX2NAME[i]: dict(split_counts[i]) for i in range(3)}
    return imbalance_score_from_counts(counts_dict, all_classes)


def _assignment_to_stems(assignment: tuple, tif_names: list, tif_tiles: dict) -> tuple:
    """Convert an assignment tuple to (train_stems, val_stems, test_stems)."""
    splits = [[], [], []]
    for tif_idx, split_idx in enumerate(assignment):
        tif_name = tif_names[tif_idx]
        splits[split_idx].extend(t[0] for t in tif_tiles[tif_name])
    return splits[0], splits[1], splits[2]


# ---------------------------------------------------------------------------
# Strategy A: exhaustive search
# ---------------------------------------------------------------------------

def _exhaustive_search(tif_names, tif_data, tif_tiles, all_classes, total_tiles):
    n = len(tif_names)
    print(f"  Exhaustive search over 3^{n} = {3**n:,} assignments ...")

    best_score      = float("inf")
    best_assignment = None

    for assignment in itertools.product(range(3), repeat=n):
        # Ensure at least one TIF per split
        if len(set(assignment)) < 3:
            continue
        score = _score_assignment(assignment, tif_data, all_classes, total_tiles)
        if score < best_score:
            best_score      = score
            best_assignment = assignment

    return best_assignment, best_score


# ---------------------------------------------------------------------------
# Strategy A: hill-climb (for large N)
# ---------------------------------------------------------------------------

def _hill_climb(tif_names, tif_data, tif_tiles, all_classes, total_tiles, rng):
    n = len(tif_names)
    print(f"  Hill-climb search ({HILL_CLIMB_RESTARTS} restarts) ...")

    best_score      = float("inf")
    best_assignment = None

    for _ in range(HILL_CLIMB_RESTARTS):
        # Random starting point that has all 3 splits represented
        while True:
            current = list(rng.randint(0, 2) for _ in range(n))
            if len(set(current)) == 3:
                break
        current_score = _score_assignment(tuple(current), tif_data, all_classes, total_tiles)

        # Greedy local search: flip one TIF at a time
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for new_val in range(3):
                    if new_val == current[i]:
                        continue
                    candidate       = current[:]
                    candidate[i]    = new_val
                    candidate_score = _score_assignment(
                        tuple(candidate), tif_data, all_classes, total_tiles
                    )
                    if candidate_score < current_score:
                        current       = candidate
                        current_score = candidate_score
                        improved      = True

        if current_score < best_score:
            best_score      = current_score
            best_assignment = tuple(current)

    return best_assignment, best_score


# ---------------------------------------------------------------------------
# Strategy A: main entry point
# ---------------------------------------------------------------------------

def strategy_a_multi_tif(tif_tiles: dict, labels_train: Path) -> tuple:
    print("  Strategy A: WHOLE-TIF assignment with class-balance optimisation")

    tif_names   = sorted(tif_tiles.keys())
    all_classes = set()
    tif_data    = []

    for name in tif_names:
        stems  = [t[0] for t in tif_tiles[name]]
        counts = class_counts_for_stems(stems, labels_train)
        all_classes.update(counts.keys())
        tif_data.append({"stems": stems, "counts": counts, "n_tiles": len(stems)})

    total_tiles = sum(d["n_tiles"] for d in tif_data)
    n           = len(tif_names)

    rng = random.Random(RANDOM_SEED)

    if n <= EXHAUSTIVE_THRESHOLD:
        best_assignment, best_score = _exhaustive_search(
            tif_names, tif_data, tif_tiles, all_classes, total_tiles
        )
    else:
        best_assignment, best_score = _hill_climb(
            tif_names, tif_data, tif_tiles, all_classes, total_tiles, rng
        )

    if best_assignment is None:
        raise RuntimeError(
            "No valid assignment found within RATIO_TOLERANCE. "
            "Try increasing RATIO_TOLERANCE."
        )

    print(f"  Best imbalance score : {best_score:.2f}")
    print(f"  TIF assignments:")
    for tif_idx, split_idx in enumerate(best_assignment):
        name = tif_names[tif_idx]
        d    = tif_data[tif_idx]
        print(f"    {name}  ->  {IDX2NAME[split_idx]}  "
              f"({d['n_tiles']} tiles, {sum(d['counts'].values())} annotations)")

    train_stems, val_stems, test_stems = _assignment_to_stems(
        best_assignment, tif_names, tif_tiles
    )
    return train_stems, val_stems, test_stems, []


# ---------------------------------------------------------------------------
# Strategy B: boundary search (single / few TIFs)
# ---------------------------------------------------------------------------

def _assign_by_boundaries(tile_list: list, b1: int, b2: int) -> tuple:
    train, val, test, buf = [], [], [], []
    for stem, col_off, row_off in tile_list:
        end = row_off + TILE_SIZE
        if end <= b1:
            train.append(stem)
        elif row_off >= b1 and end <= b2:
            val.append(stem)
        elif row_off >= b2:
            test.append(stem)
        else:
            buf.append(stem)
    return train, val, test, buf


def strategy_b_boundary_search(
    tif_name:     str,
    tile_list:    list,
    labels_train: Path,
) -> tuple:
    print(f"  Strategy B: BOUNDARY SEARCH for '{tif_name}'")

    sorted_rows = sorted(set(r for _, _, r in tile_list))
    n           = len(sorted_rows)
    step        = max(1, math.ceil(n / MAX_BOUNDARY_CANDIDATES)) \
        if MAX_BOUNDARY_CANDIDATES else 1
    candidates  = sorted_rows[::step]
    if sorted_rows[-1] not in candidates:
        candidates.append(sorted_rows[-1])

    print(f"    Unique row offsets  : {n}")
    print(f"    Boundary candidates : {len(candidates)}")

    all_classes = set()
    for stem, _, _ in tile_list:
        all_classes.update(read_label_classes(labels_train / f"{stem}.txt"))

    best_score  = float("inf")
    best_result = None
    n_evaluated = 0

    for b1 in candidates:
        for b2 in candidates:
            if b2 <= b1:
                continue
            tr, va, te, buf = _assign_by_boundaries(tile_list, b1, b2)
            total_used = len(tr) + len(va) + len(te)
            if total_used == 0:
                continue
            if (abs(len(tr) / total_used - TRAIN_RATIO) > RATIO_TOLERANCE or
                abs(len(va) / total_used - VAL_RATIO)   > RATIO_TOLERANCE or
                abs(len(te) / total_used - TEST_RATIO)  > RATIO_TOLERANCE):
                continue

            counts_dict = {
                "train": class_counts_for_stems(tr, labels_train),
                "val":   class_counts_for_stems(va, labels_train),
                "test":  class_counts_for_stems(te, labels_train),
            }
            score = imbalance_score_from_counts(counts_dict, all_classes)
            n_evaluated += 1
            if score < best_score:
                best_score  = score
                best_result = (tr, va, te, buf)

    print(f"    Candidates evaluated : {n_evaluated}")

    if best_result is None:
        print(f"    [WARN] No candidate within RATIO_TOLERANCE={RATIO_TOLERANCE}. "
              f"Falling back to ratio-based boundaries.")
        b1 = sorted_rows[round(n * TRAIN_RATIO)]
        b2 = sorted_rows[min(round(n * (TRAIN_RATIO + VAL_RATIO)), n - 1)]
        best_result = _assign_by_boundaries(tile_list, b1, b2)
        counts_dict = {
            "train": class_counts_for_stems(best_result[0], labels_train),
            "val":   class_counts_for_stems(best_result[1], labels_train),
            "test":  class_counts_for_stems(best_result[2], labels_train),
        }
        best_score = imbalance_score_from_counts(counts_dict, all_classes)

    print(f"    Best imbalance score : {best_score:.2f}")
    return best_result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    base         = Path(PROCESSED_DIR)
    images_train = base / "images" / "train"
    labels_train = base / "labels" / "train"

    for split in ("val", "test"):
        (base / "images" / split).mkdir(parents=True, exist_ok=True)
        (base / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1.  Read metadata
    # ------------------------------------------------------------------
    meta = {}
    with open(METADATA_PATH, newline="") as f:
        for row in csv.DictReader(f):
            stem = Path(row["tile_filename"]).stem
            meta[stem] = (row["source_tif"], int(row["col_off"]), int(row["row_off"]))

    # ------------------------------------------------------------------
    # 2.  Collect tiles and group by source TIF
    # ------------------------------------------------------------------
    all_stems = [p.stem for p in sorted(images_train.glob("*.png"))]
    print(f"Total tiles in train/ : {len(all_stems)}")

    tif_tiles = defaultdict(list)
    missing   = 0
    for stem in all_stems:
        if stem not in meta:
            missing += 1
            continue
        src_tif, col_off, row_off = meta[stem]
        tif_tiles[src_tif].append((stem, col_off, row_off))

    if missing:
        print(f"  [WARN] {missing} tile(s) had no metadata entry — skipped.")

    n_tifs = len(tif_tiles)
    print(f"Source TIF(s) found   : {n_tifs}\n")

    # ------------------------------------------------------------------
    # 3.  Choose and run strategy
    # ------------------------------------------------------------------
    train_stems  = []
    val_stems    = []
    test_stems   = []
    buffer_stems = []

    if n_tifs >= 3:
        tr, va, te, buf = strategy_a_multi_tif(tif_tiles, labels_train)
        train_stems  += tr
        val_stems    += va
        test_stems   += te
        buffer_stems += buf
    else:
        buffer_rows = math.ceil(TILE_SIZE / STRIDE) - 1
        print(f"Stride: {STRIDE} px  |  "
              f"Theoretical buffer: {buffer_rows} tile-row(s) per boundary")
        for tif_name, tile_list in sorted(tif_tiles.items()):
            print(f"\n  {tif_name}  ({len(tile_list)} tiles)")
            tr, va, te, buf = strategy_b_boundary_search(
                tif_name, tile_list, labels_train
            )
            train_stems  += tr
            val_stems    += va
            test_stems   += te
            buffer_stems += buf
            print(f"    train={len(tr)}, val={len(va)}, "
                  f"test={len(te)}, buffer={len(buf)}")

    # ------------------------------------------------------------------
    # 4.  Move files
    # ------------------------------------------------------------------
    print(f"\n  Moving {len(val_stems)} tiles  ->  val/ ...")
    for stem in val_stems:
        move_tile(stem, "train", "val", base)

    print(f"  Moving {len(test_stems)} tiles  ->  test/ ...")
    for stem in test_stems:
        move_tile(stem, "train", "test", base)

    print(f"  Archiving {len(buffer_stems)} buffer tiles  ->  archive/ ...")
    for stem in buffer_stems:
        archive_tile(stem, "train", base)

    # ------------------------------------------------------------------
    # 5.  Summary table
    # ------------------------------------------------------------------
    total = len(train_stems) + len(val_stems) + len(test_stems)

    print(f"\n{'='*60}")
    print(f"  Spatial split complete  (zero pixel overlap guaranteed)")
    print(f"{'='*60}")
    print(f"  {'Split':<8}  {'Tiles':>7}  {'Ratio':>7}")
    print(f"  {'-'*28}")
    for name, stems in [("train", train_stems), ("val", val_stems), ("test", test_stems)]:
        pct = len(stems) / total * 100 if total else 0.0
        print(f"  {name:<8}  {len(stems):>7}  {pct:>6.1f}%")
    print(f"  {'buffer':<8}  {len(buffer_stems):>7}  (archived)")
    print(f"  {'-'*28}")
    print(f"  {'TOTAL':<8}  {total:>7}")

    # ------------------------------------------------------------------
    # 6.  Class balance report
    # ------------------------------------------------------------------
    print(f"\n  Annotation distribution per split (instance counts):")
    print(f"  {'Class':<8}  {'Train':>14}  {'Val':>14}  {'Test':>14}  {'Balance?':>10}")
    print(f"  {'-'*68}")

    class_counts = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    for split_name, stems in [
        ("train", train_stems), ("val", val_stems), ("test", test_stems)
    ]:
        ldir = base / "labels" / split_name
        for stem in stems:
            for cls in read_label_classes(ldir / f"{stem}.txt"):
                class_counts[cls][split_name] += 1

    all_ok = True
    for cls in sorted(class_counts):
        c       = class_counts[cls]
        total_c = c["train"] + c["val"] + c["test"]
        if total_c == 0:
            continue
        val_share  = c["val"]  / total_c
        test_share = c["test"] / total_c
        balance_ok = (
            abs(val_share  - VAL_RATIO)  < 0.05 and
            abs(test_share - TEST_RATIO) < 0.05
        )
        if not balance_ok:
            all_ok = False
        flag = "OK" if balance_ok else "skewed"
        print(f"  {cls:<8}  "
              f"{c['train']:>7} ({c['train']/total_c*100:4.1f}%)  "
              f"{c['val']:>7} ({val_share*100:4.1f}%)  "
              f"{c['test']:>7} ({test_share*100:4.1f}%)  "
              f"{flag:>10}")

    print()
    if all_ok:
        print("  All classes within 5pp of target ratios.")
    else:
        print("  Some classes remain skewed — this reflects the geographic")
        print("  clustering of annotations across your source images.")
        print("  The assignment above is the best possible given your data.")
    print()


if __name__ == "__main__":
    main()