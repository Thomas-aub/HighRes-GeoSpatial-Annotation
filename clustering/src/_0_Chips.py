import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
import cv2

def process_single_pair(
    tif_path: str, 
    geojson_path: str, 
    output_dir: str, 
    chip_size_px: tuple[int, int],
    existing_ids: set
) -> list[dict]:
    """
    Extracts image chips from a .tif file based on geometries in a .geojson.

    For each feature, it computes a square bounding box, extracts the corresponding
    image data, resizes it, and saves it as a .npy file. It also compiles
    metadata for each chip.

    Args:
        tif_path: Path to the input GeoTIFF file.
        geojson_path: Path to the input GeoJSON file with polygons.
        output_dir: Directory to save the output .npy chips.
        chip_size_px: A tuple (width, height) for the output chip dimensions.

    Returns:
        A list of dictionaries, where each dictionary contains metadata
        for a single created chip.
    """
    metadata = []
    gdf = gpd.read_file(geojson_path)
    base_name = os.path.splitext(os.path.basename(tif_path))[0]
    
    # 1. Filter for specific class_ids (0 and 2)
    # Ensuring class_id exists and matches our requirements
    if 'class_id' in gdf.columns:
        gdf = gdf[gdf['class_id'].isin([0, 2])]
    
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]

    with rasterio.open(tif_path) as src:
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)
        
        gdf_wgs84 = gdf.to_crs("EPSG:4326")

        for idx, row in gdf.iterrows():

            chip_id = f"{base_name}_{idx}"
            if chip_id in existing_ids:
                continue

            # 2. Get the initial bounding box
            minx, miny, maxx, maxy = row.geometry.bounds
            
            width_geo = maxx - minx
            height_geo = maxy - miny
            
            # 3. Calculate square dimensions in map units
            # We take the maximum dimension to ensure we cover the whole polygon
            side_length = max(width_geo, height_geo)
            
            # Calculate centers
            center_x = (minx + maxx) / 2
            center_y = (miny + maxy) / 2
            
            # Create new square bounds
            sq_minx = center_x - (side_length / 2)
            sq_maxx = center_x + (side_length / 2)
            sq_miny = center_y - (side_length / 2)
            sq_maxy = center_y + (side_length / 2)
            
            # 4. Define the window based on the SQUARE bounds
            window = from_bounds(sq_minx, sq_miny, sq_maxx, sq_maxy, src.transform)
            
            try:
                # Read the expanded square area
                # boundless=True handles cases where the square goes outside the image edge
                img_chip = src.read(window=window, boundless=True, fill_value=0)
                img_chip = np.moveaxis(img_chip, 0, -1) # (C,H,W) -> (H,W,C)
                
                if img_chip.size == 0: continue

                # 5. Resize to target size (no padding needed as it's already square)
                img_resized = cv2.resize(img_chip, chip_size_px, interpolation=cv2.INTER_CUBIC)
                
                base_name = os.path.splitext(os.path.basename(tif_path))[0]
                save_path = os.path.join(output_dir, f"{base_name}_{idx}.npy")
                np.save(save_path, img_resized)
                
                metadata.append({
                    'chip_id': f"{base_name}_{idx}",
                    'chip_path': save_path,
                    'class_id': row['class_id'] if 'class_id' in row else None,
                    'orig_width': width_geo,
                    'orig_height': height_geo,
                    'lat': gdf_wgs84.loc[idx].geometry.centroid.y,
                    'lon': gdf_wgs84.loc[idx].geometry.centroid.x
                })
            except Exception as e: 
                print(f"Error processing index {idx}: {e}")
                continue
    return metadata

if __name__ == "__main__":
    INPUT_DIR = "../data/raw"
    OUTPUT_DIR = "../data/extracted/"
    CACHE_FILE = os.path.join(OUTPUT_DIR, "metadata_cache.csv")
    
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    # --- LOAD EXISTING CACHE ---
    existing_meta = []
    existing_ids = set()
    if os.path.exists(CACHE_FILE):
        print(f"Found existing cache: {CACHE_FILE}. Skipping existing chips.")
        existing_df = pd.read_csv(CACHE_FILE)
        existing_meta = existing_df.to_dict('records')
        existing_ids = set(existing_df['chip_id'].astype(str))

    tiffs = glob.glob(os.path.join(INPUT_DIR, "*.tif"))
    new_meta = []
    
    for t in tiffs:
        g = t.replace(".tif", ".geojson")
        if os.path.exists(g):
            print(f"Processing {t}...")
            new_meta.extend(process_single_pair(t, g, OUTPUT_DIR, (224, 224), existing_ids))

    # --- COMBINE AND SAVE ---
    if new_meta:
        combined_meta = existing_meta + new_meta
        pd.DataFrame(combined_meta).to_csv(CACHE_FILE, index=False)
        print(f"Done. Added {len(new_meta)} new chips.")
    else:
        print("No new chips to add.")