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
    existing_ids: set,
    fixed_buffer_meters: float = 100.0  
) -> list[dict]:
    """
    Extracts image chips using a fixed geographic window to preserve scale.

    Instead of resizing based on the boat's bounding box, this captures a 
    fixed area (e.g., 100m x 100m) around the object center. This ensures 
    that larger objects occupy more pixels than smaller objects.

    Args:
        tif_path: Path to the input GeoTIFF file.
        geojson_path: Path to the input GeoJSON file with polygons.
        output_dir: Directory to save the output .npy chips.
        chip_size_px: A tuple (width, height) for the output chip dimensions.
        fixed_buffer_meters: The physical size (in meters) of the square crop.

    Returns:
        A list of dictionaries containing metadata for each chip.
    """
    metadata = []
    gdf = gpd.read_file(geojson_path)
    base_name = os.path.basename(tif_path)
    base_no_ext = os.path.splitext(base_name)[0]
    
    if 'class_id' in gdf.columns:
        gdf = gdf[gdf['class_id'].isin([0, 2])]
    
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]

    with rasterio.open(tif_path) as src:
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        gdf_wgs84 = gdf.to_crs("EPSG:4326")
        
        # Get pixel resolution to calculate metadata
        pixel_resolution = src.res[0] 

        for idx, row in gdf.iterrows():
            chip_filename = f"{base_no_ext}_{idx}.npy"
            chip_id = f"{base_no_ext}_{idx}"
            
            if chip_id in existing_ids:
                continue

            # Get geometry bounds for metadata only
            minx, miny, maxx, maxy = row.geometry.bounds
            width_geo = maxx - minx
            height_geo = maxy - miny
            
            # --- FIXED SCALE CALCULATION ---
            # We use fixed_buffer_meters instead of side_length_geo
            center_x = (minx + maxx) / 2
            center_y = (miny + maxy) / 2
            
            sq_minx = center_x - (fixed_buffer_meters / 2)
            sq_maxx = center_x + (fixed_buffer_meters / 2)
            sq_miny = center_y - (fixed_buffer_meters / 2)
            sq_maxy = center_y + (fixed_buffer_meters / 2)
            
            # The scale factor is now constant for the whole dataset
            orig_size_px = fixed_buffer_meters / pixel_resolution
            scale_factor = chip_size_px[0] / orig_size_px if orig_size_px > 0 else 0
            
            window = from_bounds(sq_minx, sq_miny, sq_maxx, sq_maxy, src.transform)
            
            try:
                img_chip = src.read(window=window, boundless=True, fill_value=0)
                img_chip = np.moveaxis(img_chip, 0, -1) 
                
                if img_chip.size == 0: continue

                img_resized = cv2.resize(img_chip, chip_size_px, interpolation=cv2.INTER_CUBIC)
                
                save_path = os.path.join(output_dir, chip_filename)
                np.save(save_path, img_resized)
                
                metadata.append({
                    'name of picture': base_name,
                    'name of chip': chip_filename,
                    'chip_id': f"{base_name}_{idx}",
                    'chip_path': save_path,
                    'original width': width_geo,   
                    'original height': height_geo, 
                    'size variation': round(scale_factor, 3),
                    'class_id': row.get('class_id', None),
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
    METADATA_FILE = os.path.join(OUTPUT_DIR, "meta_data.csv")
    
    # Define how large the "world view" should be for each chip
    # For boats, 150m-200m is usually enough to fit most vessels.
    FIXED_WINDOW_METERS = 150.0 
    
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    existing_meta = []
    existing_ids = set()
    if os.path.exists(METADATA_FILE):
        existing_df = pd.read_csv(METADATA_FILE)
        existing_meta = existing_df.to_dict('records')
        existing_ids = set(existing_df['chip_id'].astype(str))

    tiffs = glob.glob(os.path.join(INPUT_DIR, "*.tif"))
    new_meta = []
    
    for t in tiffs:
        g = t.replace(".tif", ".geojson")
        if os.path.exists(g):
            print(f"Processing {t}...")
            new_meta.extend(process_single_pair(
                t, g, OUTPUT_DIR, (224, 224), existing_ids, 
                fixed_buffer_meters=FIXED_WINDOW_METERS
            ))

    if new_meta:
        combined_meta = existing_meta + new_meta
        pd.DataFrame(combined_meta).to_csv(METADATA_FILE, index=False)
        print(f"Done. Saved to {METADATA_FILE}")
    else:
        print("No new chips to add.")