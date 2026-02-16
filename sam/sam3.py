import os
import json
import glob
import rasterio
import numpy as np
from ultralytics.models.sam import SAM3SemanticPredictor

# Configuration
DIR_TILES = "data/tiles"
OUTPUT_GEOJSON = "results/try.geojson"

# Prompts are now crucial for labeling
PROMPTS = [
    "Small elongated objects on water", 
    "Elongated vessel shapes in water", 
    "Traditional pirogue boats at sea", 
    "Narrow vessels on open water", 
    "Pirogue", 
    "Traditional pirogue boats at sea",
    "Dark wooden pirogues at sea",
    "Dark wooden pirogues beached on sand"
]

def run_batch_inference():
    # 1. Initialize Predictor
    overrides = dict(
        conf=0.25,      
        task="segment",
        mode="predict",
        model="asset/sam3.pt",
        half=True,      
        save=False,     
        imgsz=1024
    )
    predictor = SAM3SemanticPredictor(overrides=overrides)
    
    all_features = []
    
    # Get list of images
    list_images = sorted(glob.glob(os.path.join(DIR_TILES, "*.tif")))
    
    if not list_images:
        print(f"No TIF files found in {DIR_TILES}")
        return

    print(f"Found {len(list_images)} tiles to process...")

    # 2. Process each tile
    for image_path in list_images:
        filename = os.path.basename(image_path)
        print(f"Processing: {filename}")
        
        with rasterio.open(image_path) as src:
            transform = src.transform
            
            # 3. Run SAM3 Inference
            predictor.set_image(image_path)
            results = predictor(text=PROMPTS)

            # 4. Parse Results & Georeference
            for result in results:
                if result.masks is None:
                    continue

                # Extract confidences and class indices
                if result.boxes is not None:
                    if result.boxes.conf is not None:
                        confidences = result.boxes.conf.cpu().numpy().tolist()
                    else:
                        confidences = []
                    
                    # 'cls' contains the index of the prompt that triggered the detection
                    if result.boxes.cls is not None:
                        class_indices = result.boxes.cls.cpu().numpy().tolist()
                    else:
                        class_indices = []
                else:
                    confidences = []
                    class_indices = []

                # Iterate through each detected mask
                # result.masks.xy is a list of arrays (one per mask)
                for i, mask_segments in enumerate(result.masks.xy):
                    # Default confidence if missing
                    conf_val = float(confidences[i]) if i < len(confidences) else 1.0
                    
                    # Get the class index to find the matching prompt
                    if i < len(class_indices):
                        cls_idx = int(class_indices[i])
                        # Map index back to the PROMPTS list
                        label = PROMPTS[cls_idx] if 0 <= cls_idx < len(PROMPTS) else "unknown"
                    else:
                        label = "unknown"

                    # Convert polygon points: Pixel (x, y) -> Geo (Lon, Lat / Projected)
                    geo_coords = []
                    
                    for point in mask_segments:
                        # point is [x, y]
                        px, py = point[0], point[1]
                        # Apply affine transform
                        gx, gy = transform * (px, py)
                        geo_coords.append([float(gx), float(gy)])
                    
                    # Skip empty or malformed geometries (needs at least 3 points for a polygon)
                    if len(geo_coords) < 3:
                        continue

                    # Create GeoJSON Feature
                    feature = {
                        "type": "Feature",
                        "properties": {
                            "source_tile": filename,
                            "label": label,  # NOW USING THE SPECIFIC PROMPT TEXT
                            "conf": conf_val
                        },
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [geo_coords] 
                        }
                    }
                    all_features.append(feature)

    # 5. Save Consolidate Results
    output_data = {
        "type": "FeatureCollection",
        "features": all_features
    }

    with open(OUTPUT_GEOJSON, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Processing complete. Saved {len(all_features)} detections to {OUTPUT_GEOJSON}")

if __name__ == "__main__":
    run_batch_inference()