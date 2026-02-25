"""
count.py
--------
Counts the number of labelled objects per class in each dataset split
(train / val / test) by reading the YOLO OBB .txt label files produced
by 02_geojson_to_txt.py.

Outputs:
  - A formatted table printed to stdout
  - A bar chart saved as  data/processed/class_distribution.png

Class mapping reminder (from 02_geojson_to_txt.py):
  0 → boat_small      1 → boat_medium     2 → boat_large
  3 → boat_sailing    4 → boat_motorboat  5 → boat_other
  (class 6 was merged into 4;  class 9 was discarded)
"""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# CONFIGURATION
# =============================================================================

PROCESSED_DIR = "data/processed"

# Update these names to match your dataset.yaml
CLASS_NAMES = {
    0: "pirogue",
    1: "Double hulled",
    2: "Small Motorboat",
    3: "Medium Motorboat",
    4: "Large Motorboat",
    5: "Sailing Boat",
}

SPLITS = ["train", "val", "test"]

# =============================================================================


def count_split(labels_dir: Path) -> dict:
    """
    Count objects per class in all .txt files inside labels_dir.
    Returns {class_id: count}.
    """
    counts = defaultdict(int)
    txt_files = list(labels_dir.glob("*.txt"))
    annotated = 0

    for txt in txt_files:
        lines = [l.strip() for l in txt.read_text().splitlines() if l.strip()]
        if lines:
            annotated += 1
        for line in lines:
            class_id = int(line.split()[0])
            counts[class_id] += 1

    return dict(counts), len(txt_files), annotated


def print_table(results: dict):
    """Pretty-print a summary table to stdout."""
    # Collect all class ids seen across all splits
    all_classes = sorted({cid for split_counts in results.values()
                          for cid in split_counts["counts"].keys()})

    col_w   = 16
    cls_w   = 18
    sep     = "─" * (cls_w + col_w * len(SPLITS) + 3)

    header  = f"{'Class':<{cls_w}}" + "".join(f"{s:>{col_w}}" for s in SPLITS) + f"{'TOTAL':>{col_w}}"
    print()
    print("  Object count per class per split")
    print("  " + sep)
    print("  " + header)
    print("  " + sep)

    totals_by_split = defaultdict(int)
    grand_total = 0

    for cid in all_classes:
        name = CLASS_NAMES.get(cid, f"class_{cid}")
        row  = f"{f'[{cid}] {name}':<{cls_w}}"
        row_total = 0
        for split in SPLITS:
            n = results[split]["counts"].get(cid, 0)
            row += f"{n:>{col_w},}"
            totals_by_split[split] += n
            row_total += n
        row += f"{row_total:>{col_w},}"
        grand_total += row_total
        print("  " + row)

    print("  " + sep)

    # Totals row
    tot_row = f"{'TOTAL objects':<{cls_w}}"
    for split in SPLITS:
        tot_row += f"{totals_by_split[split]:>{col_w},}"
    tot_row += f"{grand_total:>{col_w},}"
    print("  " + tot_row)

    print("  " + sep)

    # Tile stats row
    tile_row = f"{'Total tiles':<{cls_w}}"
    for split in SPLITS:
        tile_row += f"{results[split]['n_tiles']:>{col_w},}"
    tile_row += f"{sum(r['n_tiles'] for r in results.values()):>{col_w},}"
    print("  " + tile_row)

    ann_row = f"{'Annotated tiles':<{cls_w}}"
    for split in SPLITS:
        ann_row += f"{results[split]['n_annotated']:>{col_w},}"
    ann_row += f"{sum(r['n_annotated'] for r in results.values()):>{col_w},}"
    print("  " + ann_row)

    print("  " + sep)
    print()


def plot_distribution(results: dict, output_path: Path):
    """Bar chart with one group per split, one bar per class."""
    all_classes = sorted({cid for split_counts in results.values()
                          for cid in split_counts["counts"].keys()})
    if not all_classes:
        print("  No annotations found — skipping plot.")
        return

    labels  = [f"[{c}] {CLASS_NAMES.get(c, f'class_{c}')}" for c in all_classes]
    x       = np.arange(len(all_classes))
    n_splits = len(SPLITS)
    w       = 0.72 / n_splits

    colors  = ["#2563EB", "#DC2626", "#16A34A"]   # blue / red / green

    fig, axes = plt.subplots(1, 2, figsize=(16, 5),
                             gridspec_kw={"width_ratios": [3, 1]})
    fig.patch.set_facecolor("#F8FAFC")

    # ── left: per-class count per split ──────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#FFFFFF")
    for sp in ax.spines.values():
        sp.set_edgecolor("#E2E8F0")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#F1F5F9", zorder=0)

    for i, (split, color) in enumerate(zip(SPLITS, colors)):
        counts = [results[split]["counts"].get(c, 0) for c in all_classes]
        offset = (i - n_splits / 2 + 0.5) * w
        bars   = ax.bar(x + offset, counts, w, label=split.capitalize(),
                        color=color, alpha=0.85, zorder=2)
        for bar, v in zip(bars, counts):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        str(v), ha="center", va="bottom",
                        fontsize=8, color="#334155")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("Object count", color="#334155")
    ax.set_title("Object Count per Class per Split",
                 fontsize=12, fontweight="bold", color="#1E293B", pad=10)
    ax.legend(frameon=False)

    # ── right: total per split (stacked) ─────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#FFFFFF")
    for sp in ax2.spines.values():
        sp.set_edgecolor("#E2E8F0")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.yaxis.grid(True, color="#F1F5F9", zorder=0)

    split_totals = [sum(results[s]["counts"].values()) for s in SPLITS]
    split_labels = [s.capitalize() for s in SPLITS]
    bars2 = ax2.bar(split_labels, split_totals, color=colors[:n_splits],
                    alpha=0.85, zorder=2, width=0.5)
    for bar, v in zip(bars2, split_totals):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.5,
                 f"{v:,}", ha="center", va="bottom",
                 fontsize=10, fontweight="bold", color="#334155")

    ax2.set_ylabel("Total objects", color="#334155")
    ax2.set_title("Total per Split",
                  fontsize=12, fontweight="bold", color="#1E293B", pad=10)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#F8FAFC")
    print(f"  Chart saved → {output_path}")
    plt.show()


def main():
    processed_dir = Path(PROCESSED_DIR)
    results = {}

    for split in SPLITS:
        labels_dir = processed_dir / "labels" / split
        if not labels_dir.exists():
            print(f"  [WARN] '{labels_dir}' not found — skipping.")
            results[split] = {"counts": {}, "n_tiles": 0, "n_annotated": 0}
            continue
        counts, n_tiles, n_annotated = count_split(labels_dir)
        results[split] = {
            "counts":      counts,
            "n_tiles":     n_tiles,
            "n_annotated": n_annotated,
        }

    print_table(results)
    plot_distribution(results, processed_dir / "class_distribution.png")


if __name__ == "__main__":
    main()