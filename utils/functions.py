import os
import json
import math
import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry import Polygon
from sahi.predict import get_sliced_prediction
import geopandas as gpd



def render_raster(input_tif, output_tif, rgb_bands, min_percentile=1.0, max_percentile=99.0):

    # Renders an image with multiple bands to rgb 8-bit format using percentile-based contrast stretching.
    #
    # input_tif: path to input multi-band GeoTIFF
    # output_tif: path to output RGB GeoTIFF
    # rgb_bands: list of band indices to use for R, G, B (1-based indexing) [e.g., [3, 2, 1] for PlanetScope RGB]
    # min_percentile: lower percentile for contrast stretching
    # max_percentile: upper percentile for contrast stretching

    with rasterio.open(input_tif) as src:
        profile = src.profile
        bands_data = []

        data = src.read(rgb_bands)  
        print(data.shape)  # Debugging: print shape of data


        for i in range(len(rgb_bands)):
            band = data[i].astype(np.float32)
            band_min = np.percentile(band, min_percentile)
            band_max = np.percentile(band, max_percentile)

            print(f"Band {i}: min = {band_min}, max = {band_max}")

            # Avoid division by zero
            if band_max - band_min == 0:
                stretched = np.zeros(band.shape, dtype=np.uint8)
            else:
                # Stretch to 0–255
                stretched = ((band - band_min) / (band_max - band_min)) * 255.0
                stretched = np.clip(stretched, 1, 254)
                stretched = stretched.astype(np.uint8)
            
            new_band_max = stretched.max()
            new_band_min = stretched.min()
            print(f"Stretched Band {i}: new min = {new_band_min}, new max = {new_band_max}")

            bands_data.append(stretched)


        # Update profile for 8-bit output
        profile.update(
            dtype=rasterio.uint8,
            count=len(bands_data)
        )

        with rasterio.open(output_tif, 'w', **profile) as dst:
            for i, stretched_band in enumerate(bands_data, start=1):
                dst.write(stretched_band, i)

    print(f"Stretched image written to {output_tif}")






def pad_array(arr, target_height, target_width):

    # Pads the input array to the target height and width

    pad_height = target_height - arr.shape[1]
    pad_width = target_width - arr.shape[2]
    if pad_height == 0 and pad_width == 0:
        return arr
    pad = ((0, 0), (0, pad_height), (0, pad_width))
    return np.pad(arr, pad, mode='constant', constant_values=0)





def slice_raster(raster_in_path, raster_out_dir, tile_size, skip_empty_tiles=False):

    # Slices the raster input into tiles and optionnaly remove the empty ones
    #
    # raster_in_path: path to input raster
    # raster_out_dir: directory to save output tiles
    # tile_size: size of the output tiles (e.g., 512 for 512x512 tiles)
    # skip_empty_tiles: if True, skip tiles that are empty (all values are the same)

    os.makedirs(raster_out_dir, exist_ok=True)
    
    with rasterio.open(raster_in_path) as src:
        meta = src.meta.copy()
        width = src.width
        height = src.height
        n_cols = math.ceil(width / tile_size)
        n_rows = math.ceil(height / tile_size)

        for row in range(n_rows):
            for col in range(n_cols):
                x_off = col * tile_size
                y_off = row * tile_size
                win_width = min(tile_size, width - x_off)
                win_height = min(tile_size, height - y_off)

                window = Window(x_off, y_off, win_width, win_height)
                data = src.read(window=window)

                # Check if the tile is empty
                if skip_empty_tiles and np.min(data) == np.max(data):
                    continue # Skip empty tiles

                # Pad if needed
                if win_width < tile_size or win_height < tile_size:
                    data = pad_array(data, tile_size, tile_size)

                # Update transform
                transform = src.window_transform(window)

                out_meta = meta.copy()
                out_meta.update({
                    "height": tile_size,
                    "width": tile_size,
                    "transform": transform,
                    "count": 3
                })

                out_path = os.path.join(
                    raster_out_dir,
                    f"{os.path.basename(raster_in_path)[:-4]}_{x_off}_{y_off}.tif"
                )

                with rasterio.open(out_path, "w", **out_meta) as dst:
                    data = data[:3, :, :]
                    dst.write(data)






def georeference_detections(predictions, transform, crs):
    
    # Convert SAHI predictions to GeoJSON format

    features = []

    for pred in predictions:

        # Use rotated box
        class_id = pred.category.id
        confidence = pred.score.value
        x1,y1,x2,y2,x3,y3,x4,y4 = pred.mask.segmentation[0] 

        # Convert pixel coordinates to geographic coordinates
        geo_x1, geo_y1 = rasterio.transform.xy(transform, y1, x1)
        geo_x2, geo_y2 = rasterio.transform.xy(transform, y2, x2)
        geo_x3, geo_y3 = rasterio.transform.xy(transform, y3, x3)
        geo_x4, geo_y4 = rasterio.transform.xy(transform, y4, x4)

        polygon = Polygon([(geo_x1, geo_y1), (geo_x2, geo_y2), (geo_x3, geo_y3), (geo_x4, geo_y4)])

        feature = {
            "type": "Feature",
            "properties": {
                "class_id": class_id,
                "confidence": confidence
            },
            "geometry": polygon.__geo_interface__
        }
        features.append(feature)

    
    geojson = {
        "type": "FeatureCollection",
        "name": "auto_labeling",
        "crs": {
            "type": "name",
            "properties": {
                "name": f"EPSG:{crs.to_epsg()}"
            }
        },
        "features": features
    }

    return geojson


def read_raster(image_path):

    # Read a large TIFF image and return it as a numpy array

    with rasterio.open(image_path) as src:
        image = src.read()
        image = np.transpose(image, (1, 2, 0))  # (bands, h, w) → (h, w, bands)
        image = image[:, :, :3]
        transform = src.transform
        crs = src.crs
    return image, transform, crs


def yolo_obb_predict(image_file, labels_file, detection_model, tile_size, overlap_ratio, classes_to_keep):
    
    # Make predictions on a large TIFF image using sahi to slice the image

    if(os.path.exists(labels_file)):
        print(f"Labels file {labels_file} already exists. Skipping...")
        return
    print(f"Making predictions for {image_file}...")

    exclude_classes_by_id = [i for i in range(100) if i not in classes_to_keep]

    # Get sliced prediction
    image, transform, crs = read_raster(image_file)
    result = get_sliced_prediction(
        image=image,
        detection_model=detection_model,
        slice_height=tile_size,
        slice_width=tile_size,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
        perform_standard_pred=False,
        exclude_classes_by_id=exclude_classes_by_id
    )

    # Convert to geojson format
    output_geojson = georeference_detections(result.object_prediction_list, transform, crs)

    # Save to labels file
    with open(labels_file, "w") as f:
        json.dump(output_geojson, f, indent=4)

    # Cleanup to prevent memory issues
    del image
    del result

