"""
02_geojson_to_txt.py
---------------------
Converts GeoJSON OBB boat annotations into YOLO OBB .txt label files,
one .txt per image tile produced by 01_preprocessing_images.py.

Class mapping applied:
  GeoJSON class_id  ->  YOLO class
  0                 ->  0
  1                 ->  1
  2                 ->  2
  3                 ->  3
  4                 ->  4   (kept as-is)
  5                 ->  5
  6                 ->  4   (merged with class 4)
  9                 ->  SKIP

YOLO OBB label format (one line per object):
  class_id  x1 y1  x2 y2  x3 y3  x4 y4
  where xi, yi are normalised to [0, 1] relative to the padded TILE_SIZE.

One .txt is written for every tile in metadata.csv, even if it contains no
annotations (empty file = background tile, required by YOLO).
"""

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import pyproj
from affine import Affine
from shapely.geometry import Polygon


# =============================================================================
# CONFIGURATION  –  must match the values used in 01_preprocessing_images.py
# =============================================================================

METADATA_PATH = "data/processed/metadata.csv"  # CSV produced by script 01
RAW_DIR       = "data/raw"                      # folder containing the .geojson files
OUTPUT_DIR    = "data/processed"                # root processed folder

TILE_SIZE     = 1024    # padded tile size used in script 01 (pixels)

# Minimum fraction of an OBB area that must fall inside a tile to keep it.
MIN_VISIBLE   = 0.15

# Class remapping: {geojson_class_id: yolo_class_id}
CLASS_MAP   = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 4}
SKIP_CLASSES = {9}

# =============================================================================


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def str_to_affine(s: str) -> Affine:
    """Parse a comma-separated affine string (written by script 01) back to Affine."""
    a, b, c, d, e, f = [float(v) for v in s.split(",")]
    return Affine(a, b, c, d, e, f)


def load_geojson(path: Path) -> list:
    """Return the feature list from a GeoJSON FeatureCollection."""
    with open(path) as f:
        gj = json.load(f)
    return gj.get("features", [])


def obb_corners_geo(feature: dict) -> Optional[List[Tuple[float, float]]]:
    """
    Extract exactly 4 (lon, lat) corner pairs from a GeoJSON polygon.
    If the polygon has more or fewer than 4 points,
    it uses Shapely's minimum_rotated_rectangle to force it into a valid OBB.
    """
    try:
        coords = feature["geometry"]["coordinates"][0]          # outer ring
        poly = Polygon(coords)
        
        # Attempt to fix self-intersecting or invalid polygons
        if not poly.is_valid:
            poly = poly.buffer(0)

        # Force the geometry into an Oriented Bounding Box
        obb = poly.minimum_rotated_rectangle
        obb_coords = list(obb.exterior.coords)
        
        # GeoJSON closed rings repeat the first vertex at the end, so we strip it
        pts = obb_coords[:-1] if obb_coords[0] == obb_coords[-1] else obb_coords
        
        if len(pts) != 4:
            return None
            
        return [(p[0], p[1]) for p in pts]
    except Exception:
        return None


def normalize_corners(
    corners_px: List[Tuple[float, float]],
    tile_size: int,
) -> List[Tuple[float, float]]:
    """Normalise pixel coordinates to [0, 1] relative to tile_size."""
    return [(x / tile_size, y / tile_size) for x, y in corners_px]


def corners_to_yolo_line(class_id: int, corners: List[Tuple[float, float]]) -> str:
    """Serialize to YOLO OBB format:  class_id x1 y1 x2 y2 x3 y3 x4 y4"""
    flat = " ".join(f"{x:.6f} {y:.6f}" for x, y in corners)
    return f"{class_id} {flat}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    metadata_path = Path(METADATA_PATH)
    raw_dir       = Path(RAW_DIR)
    output_dir    = Path(OUTPUT_DIR)

    if not metadata_path.exists():
        print(f"[ERROR] Metadata file not found: {metadata_path}")
        return

    # ------------------------------------------------------------------
    # 1. Load metadata and group tiles by their source .tif
    # ------------------------------------------------------------------
    tiles_by_tif = defaultdict(list)
    with open(metadata_path, newline="") as f:
        for row in csv.DictReader(f):
            tiles_by_tif[row["source_tif"]].append(row)

    total_tiles = sum(len(v) for v in tiles_by_tif.values())
    print(f"Loaded {total_tiles} tile entries from metadata ({len(tiles_by_tif)} source image(s)).")

    total_label_files  = 0
    total_annotations  = 0
    total_dropped_vis  = 0
    total_empty_files  = 0

    # ------------------------------------------------------------------
    # 2. For each source TIF, load its paired GeoJSON and process tiles
    # ------------------------------------------------------------------
    for tif_name, tif_tiles in tiles_by_tif.items():
        tif_stem     = Path(tif_name).stem
        geojson_path = raw_dir / f"{tif_stem}.geojson"
        
        # Get the CRS for this specific TIF from the metadata
        crs_str = tif_tiles[0]["crs"]
        
        # Setup coordinate transformer: GeoJSON (EPSG:4326) -> TIF Native CRS
        try:
            transformer = pyproj.Transformer.from_crs("EPSG:4326", crs_str, always_xy=True)
        except Exception as e:
            print(f"\n  [WARN] Could not parse CRS '{crs_str}' for {tif_name}: {e}. Assuming EPSG:4326.")
            transformer = None

        if not geojson_path.exists():
            print(f"\n  [WARN] No GeoJSON found for '{tif_name}' — label files will be empty.")
            features = []
        else:
            features = load_geojson(geojson_path)
            print(f"\n{'='*60}")
            print(f"  {tif_name}  ->  {geojson_path.name}  ({len(features)} raw annotations)")
            print(f"  Target CRS: {crs_str}")

        # Pre-filter: skip unwanted classes and malformed geometries
        valid_features = []
        dropped_class = 0
        dropped_geom = 0
        
        for feat in features:
            cid = feat.get("properties", {}).get("class_id")
            if cid in SKIP_CLASSES or cid not in CLASS_MAP:
                dropped_class += 1
                continue
                
            corners = obb_corners_geo(feat)
            if corners is None:
                dropped_geom += 1
                continue
                
            valid_features.append((CLASS_MAP[cid], corners))

        print(f"  Valid annotations after filtering: {len(valid_features)}")
        print(f"    - Dropped by class ID   : {dropped_class}")
        print(f"    - Dropped by geometry   : {dropped_geom}")

        # Build Shapely polygons once (reused across all tiles of this image)
        geo_polys = []
        for yolo_cls, corners in valid_features:
            try:
                # Project corners from WGS84 to the TIF's native coordinate system
                if transformer:
                    corners_proj = [transformer.transform(x, y) for x, y in corners]
                else:
                    corners_proj = corners
                    
                poly = Polygon(corners_proj)
                if not poly.is_valid:
                    poly = poly.buffer(0)   # attempt auto-repair
                
                # Store the projected corners and poly
                geo_polys.append((yolo_cls, corners_proj, poly))
            except Exception:
                pass

        # ------------------------------------------------------------------
        # 3. For each tile, find overlapping annotations and write .txt
        # ------------------------------------------------------------------
        for tile in tif_tiles:
            tile_transform = str_to_affine(tile["tile_transform"])
            inv_transform  = ~tile_transform

            tile_w = int(tile["tile_width"])
            tile_h = int(tile["tile_height"])

            # Geographic footprint of this tile (now in native CRS, not degrees)
            tl = tile_transform * (0,      0)
            tr = tile_transform * (tile_w, 0)
            br = tile_transform * (tile_w, tile_h)
            bl = tile_transform * (0,      tile_h)
            tile_poly_geo = Polygon([tl, tr, br, bl])

            lines = []

            for yolo_cls, corners_proj, ann_poly in geo_polys:

                if not tile_poly_geo.intersects(ann_poly):
                    continue

                intersection = tile_poly_geo.intersection(ann_poly)
                if intersection.is_empty:
                    continue

                vis_frac = intersection.area / ann_poly.area if ann_poly.area > 0 else 0.0
                if vis_frac < MIN_VISIBLE:
                    total_dropped_vis += 1
                    continue

                # Convert native map coords -> pixel coords relative to this tile's origin
                corners_px = [inv_transform * (x, y) for x, y in corners_proj]

                norm_corners = normalize_corners(corners_px, TILE_SIZE)

                if Polygon(norm_corners).area < 1e-9:
                    continue

                lines.append(corners_to_yolo_line(yolo_cls, norm_corners))

            tile_rel   = Path(tile["tile_filename"])
            split_name = tile_rel.parts[1]
            tile_stem  = tile_rel.stem

            label_dir = output_dir / "labels" / split_name
            label_dir.mkdir(parents=True, exist_ok=True)
            label_path = label_dir / f"{tile_stem}.txt"

            with open(label_path, "w") as lf:
                if lines:
                    lf.write("\n".join(lines))
                else:
                    total_empty_files += 1

            total_label_files += 1
            total_annotations += len(lines)

    print(f"\nDone.")
    print(f"  Label files written : {total_label_files}")
    print(f"  Total annotations   : {total_annotations}")
    print(f"  Empty (background)  : {total_empty_files}")
    print(f"  Total clipped drops : {total_dropped_vis} (visibility < {MIN_VISIBLE})")


if __name__ == "__main__":
    main()