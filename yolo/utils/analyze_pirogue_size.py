"""
analyze_pirogue_size.py
-----------------------
Analyse la taille réelle (en pixels) des pirogues (class_id = 0)
dans les labels YOLO OBB.

Calcule :
- longueur moyenne (px)
- largeur moyenne (px)
- aire moyenne (px²)
- aspect ratio moyen
- statistiques par split
"""

from pathlib import Path
import numpy as np
from collections import defaultdict

# =============================================================================
# CONFIGURATION
# =============================================================================

PROCESSED_DIR = "data/processed"
SPLITS = ["train", "val", "test"]
PIROGUE_CLASS_ID = 0
TILE_SIZE = 1024

# =============================================================================


def polygon_area(pts):
    """Shoelace formula"""
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def side_lengths(pts):
    """Return the 4 side lengths of the quadrilateral"""
    lengths = []
    for i in range(4):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % 4]
        lengths.append(np.hypot(x2 - x1, y2 - y1))
    return lengths


def analyze_split(labels_dir: Path):
    lengths_long = []
    lengths_short = []
    areas = []

    for txt in labels_dir.glob("*.txt"):
        lines = [l.strip() for l in txt.read_text().splitlines() if l.strip()]
        for line in lines:
            parts = line.split()
            class_id = int(parts[0])
            if class_id != PIROGUE_CLASS_ID:
                continue

            coords = list(map(float, parts[1:]))
            pts_norm = [(coords[i], coords[i+1]) for i in range(0, 8, 2)]

            # Convert normalized → pixels
            pts_px = [(x * TILE_SIZE, y * TILE_SIZE) for x, y in pts_norm]

            lengths = side_lengths(pts_px)
            long_side = max(lengths)
            short_side = min(lengths)

            area = polygon_area(pts_px)

            lengths_long.append(long_side)
            lengths_short.append(short_side)
            areas.append(area)

    return {
        "count": len(lengths_long),
        "long_mean": np.mean(lengths_long) if lengths_long else 0,
        "long_std": np.std(lengths_long) if lengths_long else 0,
        "short_mean": np.mean(lengths_short) if lengths_short else 0,
        "short_std": np.std(lengths_short) if lengths_short else 0,
        "area_mean": np.mean(areas) if areas else 0,
        "aspect_ratio_mean": np.mean(np.array(lengths_long) / np.array(lengths_short)) if lengths_short else 0,
    }


def main():
    processed_dir = Path(PROCESSED_DIR)

    print("\nPIROGUE SIZE ANALYSIS (pixels)")
    print("─" * 70)

    for split in SPLITS:
        labels_dir = processed_dir / "labels" / split
        if not labels_dir.exists():
            continue

        stats = analyze_split(labels_dir)

        print(f"\nSplit: {split}")
        print(f"  Count                : {stats['count']}")
        print(f"  Long side mean       : {stats['long_mean']:.2f} px  (± {stats['long_std']:.2f})")
        print(f"  Short side mean      : {stats['short_mean']:.2f} px  (± {stats['short_std']:.2f})")
        print(f"  Area mean            : {stats['area_mean']:.2f} px²")
        print(f"  Aspect ratio mean    : {stats['aspect_ratio_mean']:.2f}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()