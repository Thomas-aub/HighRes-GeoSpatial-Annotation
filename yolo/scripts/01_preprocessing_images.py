"""
01_preprocessing_images.py
---------------------------
Tiles large GeoTIFF satellite images into smaller square tiles for YOLO training.

For each .tif found in RAW_DIR:
  - Stretches the full image with percentile-based contrast (computed once per image,
    so all tiles share the same colour rendering — no per-tile colour drift)
  - Slides a window of TILE_SIZE × TILE_SIZE pixels across the stretched image
  - Skips tiles where all pixel values are identical (empty / nodata)
  - Zero-pads edge tiles to TILE_SIZE so YOLO always gets a fixed input size
  - Saves each tile as a PNG under OUTPUT_DIR/images/<SPLIT>/
  - Appends one row per tile to metadata.csv so tiles can be geo-reconstructed later

Metadata CSV columns:
  tile_filename  – relative path from OUTPUT_DIR  (e.g. images/train/stem_0_0.png)
  source_tif     – basename of the source .tif
  col_off        – pixel column offset in the source image (x)
  row_off        – pixel row offset in the source image (y)
  tile_width     – actual tile width  (may be < TILE_SIZE at right edge)
  tile_height    – actual tile height (may be < TILE_SIZE at bottom edge)
  src_width      – full source image width  (pixels)
  src_height     – full source image height (pixels)
  crs            – CRS string of the source image
  transform      – affine transform of the source image (6 comma-separated values)
  tile_transform – affine transform of this specific tile (6 comma-separated values)
"""

import csv
import math
import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image


# =============================================================================
# CONFIGURATION  –  edit these values, no command-line arguments needed
# =============================================================================

RAW_DIR    = "data/raw"        # folder that contains the .tif source images
OUTPUT_DIR = "data/processed"  # root processed folder
SPLIT      = "train"           # "train" | "val" | "test"

TILE_SIZE  = 1024              # tile width and height in pixels 
OVERLAP    = 64                 # pixel overlap between adjacent tiles (0 = no overlap)

# 1-based band indices to select for R, G, B.
# Set to None to auto-detect:  >= 3 bands -> first 3 bands;  1 band (PAN) -> replicated to RGB
BANDS = None                   # e.g. [3, 2, 1] for a 4-band image with BGR ordering

# Percentile contrast stretching applied to the FULL image before tiling
# (same values used by every tile -> consistent colours across the dataset)
MIN_PERCENTILE = 1.0
MAX_PERCENTILE = 99.0

# =============================================================================


# ---------------------------------------------------------------------------
# Band selection
# ---------------------------------------------------------------------------

def select_bands(data: np.ndarray, n_src_bands: int, bands_cfg) -> np.ndarray:
    """
    data  : (n_bands, H, W)
    Returns (3, H, W) – always 3 channels for PNG / YOLO compatibility.
    """
    if bands_cfg is not None:
        idx = [b - 1 for b in bands_cfg]     # 1-based -> 0-based
        selected = data[idx]
    elif n_src_bands >= 3:
        selected = data[:3]                   # take first 3 bands
    else:
        selected = data[:1]                   # PAN -> 1 band

    # Single-channel (PAN) -> replicate to 3 channels so YOLO gets RGB
    if selected.shape[0] == 1:
        selected = np.repeat(selected, 3, axis=0)

    return selected                           # (3, H, W)


# ---------------------------------------------------------------------------
# Padding  (mirrors pad_array from functions.py)
# ---------------------------------------------------------------------------

def pad_array(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Zero-pad (3, H, W) to (3, target_h, target_w)."""
    pad_h = target_h - arr.shape[1]
    pad_w = target_w - arr.shape[2]
    if pad_h == 0 and pad_w == 0:
        return arr
    return np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant", constant_values=0)


# ---------------------------------------------------------------------------
# Affine helpers
# ---------------------------------------------------------------------------

def affine_to_str(t) -> str:
    return ",".join(str(v) for v in [t.a, t.b, t.c, t.d, t.e, t.f])


# ---------------------------------------------------------------------------
# Core tiling function
# ---------------------------------------------------------------------------

def tile_image(tif_path: Path, writer: csv.DictWriter, output_images_dir: Path):
    stride = TILE_SIZE - OVERLAP

    with rasterio.open(tif_path) as src:
        W, H          = src.width, src.height
        n_bands       = src.count
        crs_str       = str(src.crs)
        transform_str = affine_to_str(src.transform)

        print(f"\n{'='*60}")
        print(f"  Source : {tif_path.name}")
        print(f"  Size   : {W} x {H} px   |   {n_bands} band(s)   |   CRS: {src.crs}")

        # ------------------------------------------------------------------
        # 1. Read the FULL image once and compute global stretch parameters.
        #    This mirrors render_raster() in functions.py: percentiles are
        #    computed on the whole image so every tile gets identical colour
        #    rendering (no per-tile contrast drift).
        # ------------------------------------------------------------------
        print("  Reading full image for global contrast stretch ...")
        full_data = src.read()                                     # (n_bands, H, W)
        rgb_full  = select_bands(full_data, n_bands, BANDS)        # (3, H, W)
        del full_data

        print("  Computing global percentiles ...")
        stretch_params = []
        for i in range(rgb_full.shape[0]):
            band = rgb_full[i].astype(np.float32)
            lo   = float(np.percentile(band, MIN_PERCENTILE))
            hi   = float(np.percentile(band, MAX_PERCENTILE))
            print(f"    Band {i + 1}: raw range [{lo:.1f}, {hi:.1f}]")
            stretch_params.append((lo, hi))

        # ------------------------------------------------------------------
        # 2. Slide a window over the (still raw) full image, apply the
        #    global stretch per-tile, then save.
        # ------------------------------------------------------------------
        n_cols  = math.ceil(W / stride)
        n_rows  = math.ceil(H / stride)
        n_total = n_cols * n_rows
        kept    = 0
        stem    = tif_path.stem

        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                x_off = col_idx * stride
                y_off = row_idx * stride

                win_w = min(TILE_SIZE, W - x_off)
                win_h = min(TILE_SIZE, H - y_off)

                # Slice from the already-loaded full RGB array
                tile_raw = rgb_full[:, y_off:y_off + win_h, x_off:x_off + win_w].copy()

                # Skip empty / nodata tiles  (mirrors skip_empty_tiles in slice_raster)
                if np.min(tile_raw) == np.max(tile_raw):
                    continue

                # Apply the GLOBAL stretch (same params for every tile)
                tile_uint8 = np.zeros_like(tile_raw, dtype=np.uint8)
                for i, (lo, hi) in enumerate(stretch_params):
                    band = tile_raw[i].astype(np.float32)
                    if hi - lo == 0:
                        tile_uint8[i] = np.zeros(band.shape, dtype=np.uint8)
                    else:
                        s = (band - lo) / (hi - lo) * 255.0
                        s = np.clip(s, 1, 254)          # same clip as render_raster
                        tile_uint8[i] = s.astype(np.uint8)

                # Zero-pad edge tiles to the full TILE_SIZE
                tile_uint8 = pad_array(tile_uint8, TILE_SIZE, TILE_SIZE)

                # Save as PNG  ->  transpose to (H, W, 3) for PIL
                tile_name = f"{stem}_{x_off}_{y_off}.png"
                out_path  = output_images_dir / tile_name
                Image.fromarray(tile_uint8.transpose(1, 2, 0)).save(out_path)

                # Tile-specific affine transform (needed by script 02)
                window             = Window(x_off, y_off, win_w, win_h)
                tile_transform     = src.window_transform(window)
                tile_transform_str = affine_to_str(tile_transform)

                writer.writerow({
                    "tile_filename":  str(out_path.relative_to(Path(OUTPUT_DIR))),
                    "source_tif":     tif_path.name,
                    "col_off":        x_off,
                    "row_off":        y_off,
                    "tile_width":     win_w,
                    "tile_height":    win_h,
                    "src_width":      W,
                    "src_height":     H,
                    "crs":            crs_str,
                    "transform":      transform_str,
                    "tile_transform": tile_transform_str,
                })
                kept += 1

        del rgb_full
        print(f"  Tiles  : {kept} kept  /  {n_total} candidates")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    raw_dir      = Path(RAW_DIR)
    output_dir   = Path(OUTPUT_DIR)
    images_split = output_dir / "images" / SPLIT
    images_split.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "metadata.csv"
    file_exists   = metadata_path.exists()

    fieldnames = [
        "tile_filename", "source_tif",
        "col_off", "row_off", "tile_width", "tile_height",
        "src_width", "src_height",
        "crs", "transform", "tile_transform",
    ]

    tif_files = sorted(raw_dir.glob("*.tif"))
    if not tif_files:
        print(f"[ERROR] No .tif files found in '{raw_dir}'")
        return

    print(f"Found {len(tif_files)} .tif file(s) in '{raw_dir}'")
    print(f"Tile size : {TILE_SIZE}px  |  Overlap : {OVERLAP}px  |  Split : {SPLIT}")

    with open(metadata_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for tif_path in tif_files:
            tile_image(tif_path, writer, images_split)

    print(f"\nDone. Metadata saved to '{metadata_path}'")


if __name__ == "__main__":
    main()