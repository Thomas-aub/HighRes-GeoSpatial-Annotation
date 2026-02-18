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
    base_name = os.path.basename(tif_path) # e.g., "image1.tif"
    base_no_ext = os.path.splitext(base_name)[0]
    
    if 'class_id' in gdf.columns:
        gdf = gdf[gdf['class_id'].isin([0, 2])]

    gdf_wgs84 = gdf.to_crs("EPSG:4326")
    
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]

    with rasterio.open(tif_path) as src:
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)
        
        # Get pixel resolution to calculate scale factors
        pixel_resolution = src.res[0] # Assuming square pixels

        for idx, row in gdf.iterrows():
            chip_filename = f"{base_no_ext}_{idx}.npy"
            chip_id = f"{base_no_ext}_{idx}"
            
            if chip_id in existing_ids:
                continue

            minx, miny, maxx, maxy = row.geometry.bounds
            width_geo = maxx - minx
            height_geo = maxy - miny
            side_length_geo = max(width_geo, height_geo)
            
            # --- SCALE CALCULATION ---
            # Original size in pixels before resizing to chip_size_px
            orig_size_px = side_length_geo / pixel_resolution
            # Scale factor: how much the image was stretched or shrunk
            # e.g., if orig is 112px and target is 224px, scale is 2.0
            scale_factor = chip_size_px[0] / orig_size_px if orig_size_px > 0 else 0

            center_x = (minx + maxx) / 2
            center_y = (miny + maxy) / 2
            
            sq_minx = center_x - (side_length_geo / 2)
            sq_maxx = center_x + (side_length_geo / 2)
            sq_miny = center_y - (side_length_geo / 2)
            sq_maxy = center_y + (side_length_geo / 2)
            
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
    # Updated filename as requested
    METADATA_FILE = os.path.join(OUTPUT_DIR, "meta_data.csv")
    
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
            new_meta.extend(process_single_pair(t, g, OUTPUT_DIR, (224, 224), existing_ids))

    if new_meta:
        combined_meta = existing_meta + new_meta
        pd.DataFrame(combined_meta).to_csv(METADATA_FILE, index=False)
        print(f"Done. Saved to {METADATA_FILE}")
    else:
        print("No new chips to add.")



