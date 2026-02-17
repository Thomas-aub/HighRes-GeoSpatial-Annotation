import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window, from_bounds
import cv2

def resize_pad_black(img, target_size):
    """
    Resizes image keeping aspect ratio and pads the rest with black pixels (zeros).
    The image is centered within the black square.
    
    Args:
        img: Input image (H, W, C)
        target_size: Tuple (W, H) ex: (224, 224)
    """
    h, w = img.shape[:2]
    t_w, t_h = target_size
    
    # 1. Calculate scale to fit within target dimensions
    scale = min(t_w / w, t_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # 2. Resize keeping aspect ratio
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # 3. Calculate padding to center the image
    delta_w = t_w - new_w
    delta_h = t_h - new_h
    
    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left
    
    # 4. Apply constant black padding
    new_img = cv2.copyMakeBorder(
        resized, 
        top, bottom, 
        left, right, 
        cv2.BORDER_CONSTANT, 
        value=[0, 0, 0] # Black
    )
    
    return new_img
    
def process_single_pair(tif_path, geojson_path, output_dir, chip_size_px):
    """
    Extracts chips strictly limited to the boat's bounding box.
    Pads the rectangle to a square by replicating edge pixels to avoid 
    including external elements or distorting the boat.
    """
    metadata = []
    gdf = gpd.read_file(geojson_path)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]

    with rasterio.open(tif_path) as src:
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)
        
        gdf_wgs84 = gdf.to_crs("EPSG:4326")

        for idx, row in gdf.iterrows():
            # 1. Get the strict bounding box of the geometry
            minx, miny, maxx, maxy = row.geometry.bounds
            
            # 2. Define the window based ONLY on these bounds
            window = from_bounds(minx, miny, maxx, maxy, src.transform)
            
            try:
                # Read only the boat area
                img_chip = src.read(window=window, boundless=True, fill_value=0)
                img_chip = np.moveaxis(img_chip, 0, -1) # (C,H,W) -> (H,W,C)
                
                if img_chip.size == 0: continue

                # 3. Use the smart resize/pad function
                # This will take the 15x40 boat and make it 224x224 
                # by stretching the edges, not by taking more context.
                img_resized = resize_pad_black(img_chip, chip_size_px)
                
                base_name = os.path.splitext(os.path.basename(tif_path))[0]
                save_path = os.path.join(output_dir, f"{base_name}_{idx}.npy")
                np.save(save_path, img_resized)
                
                metadata.append({
                    'chip_id': f"{base_name}_{idx}",
                    'chip_path': save_path,
                    'width': maxx - minx,
                    'height': maxy - miny,
                    'lat': gdf_wgs84.loc[idx].geometry.centroid.y,
                    'lon': gdf_wgs84.loc[idx].geometry.centroid.x
                })
            except Exception: continue
    return metadata
if __name__ == "__main__":
    # Example usage for testing
    INPUT_DIR = "../data/raw"
    OUTPUT_DIR = "../data/extracted/"
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    
    tiffs = glob.glob(os.path.join(INPUT_DIR, "*.tif"))
    all_meta = []
    for t in tiffs:
        g = t.replace(".tif", ".geojson")
        if os.path.exists(g):
            print(f"Processing {t}...")
            all_meta.extend(process_single_pair(t, g, OUTPUT_DIR, (224, 224)))

    
    pd.DataFrame(all_meta).to_csv(os.path.join(OUTPUT_DIR, "metadata_cache.csv"), index=False)