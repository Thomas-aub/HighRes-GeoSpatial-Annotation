import os
import json
import glob
import rasterio
import numpy as np
from ultralytics.models.sam import SAM3SemanticPredictor
import torch

# Enable mixed precision
torch.set_float32_matmul_precision('medium')

# Configuration
DIR_TILES = "data/sud2/tiles1024"
OUTPUT_GEOJSON = "results/sud/try1_multi_label.geojson" 

# Prompts are now crucial for labeling
PROMPTS = [
    "Solid elongated floating on water",        # Faux positifs
    "Thin linear shapes on the sea surface",    # Faux positifs
    "Dark wooden pirogues at sea",              # Faux positifs
    "Dark wooden pirogues beached on sand",     # Faux positifs
    "Rigid boat hulls on water",                # Bien 
    "Small elongated objects on water",         # Bien 
    "Elongated vessel shapes in water",         # Bien 
    "Boat",                                     # Test
    "Waves"                                     # Test

]

def run_batch_inference():
    # 1. Initialize Predictor
    overrides = dict(
        conf=0.25,
        task="segment",
        mode="predict",
        model="asset/sam3.pt",
        half=True,  # Enable half-precision inference
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

            # Iterate through each prompt individually
            for current_prompt in PROMPTS:
                print(f"  - Running inference for prompt: '{current_prompt}'")

                predictor.set_image(image_path)
                results_for_prompt = predictor(text=[current_prompt])  # Predict for a single prompt (must be a list)

                for res in results_for_prompt:  # 'res' is a single Results object for the current image and prompt
                    if res.masks is None:
                        continue

                    confidences = []
                    if res.boxes is not None and res.boxes.conf is not None:
                        confidences = res.boxes.conf.cpu().numpy().tolist()

                    # Iterate through each detected mask for this prompt
                    for i, mask_segments in enumerate(res.masks.xy):
                        conf_val = float(confidences[i]) if i < len(confidences) else 1.0

                        # Assign the current prompt as the label
                        label = current_prompt

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

                # Free memory after processing each prompt
                del results_for_prompt
                torch.cuda.empty_cache()

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