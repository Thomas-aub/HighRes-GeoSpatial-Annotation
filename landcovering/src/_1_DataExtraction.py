import geopandas as gpd
import requests
import mercantile
import os
import time
import csv
from shapely.geometry import box
from tqdm import tqdm

# --------------------------------------------------
# 1. Load Madagascar boundary
# --------------------------------------------------

url = "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip"
world = gpd.read_file(url)
madagascar_wgs84 = world[world["ADMIN"] == "Madagascar"]

if madagascar_wgs84.empty:
    raise ValueError("Madagascar not found in dataset.")

# --------------------------------------------------
# 2. Extract geometries for processing
# --------------------------------------------------

# Get the coastline (boundary) to quickly filter tiles that cross the coast
coastline_wgs84 = madagascar_wgs84.boundary

# Reproject the main polygon to a metric CRS (UTM 38S) for accurate area calculations
madagascar_utm = madagascar_wgs84.to_crs(epsg=32738)
madagascar_utm_geom = madagascar_utm.geometry.iloc[0]

# --------------------------------------------------
# 3. Generate candidate tiles
# --------------------------------------------------

minx, miny, maxx, maxy = madagascar_wgs84.total_bounds
print("Bounds (WGS84):", minx, miny, maxx, maxy)

zoom = 15
tiles = list(mercantile.tiles(minx, miny, maxx, maxy, zoom))
print(f"Total tiles in bounding box: {len(tiles)}")

# --------------------------------------------------
# 4. Prepare folders and metadata file
# --------------------------------------------------

os.makedirs("tiles", exist_ok=True)
metadata_path = "metadata.csv"

# Configuration for the land/sea ratio (0.5 means exactly 50/50)
MIN_LAND_RATIO = 0.30
MAX_LAND_RATIO = 0.65

with open(metadata_path, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "filename", "zoom", "tile_x", "tile_y", 
        "west", "south", "east", "north", "crs", "land_ratio"
    ])

    arcgis_url = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile"

    print("Filtering and downloading tiles...")

    for tile in tqdm(tiles):

        bounds = mercantile.bounds(tile)
        tile_geom_wgs84 = box(bounds.west, bounds.south, bounds.east, bounds.north)

        # FAST PASS: Skip the tile if it doesn't cross the coastline at all
        if not coastline_wgs84.intersects(tile_geom_wgs84).any():
            continue

        # AREA CHECK: Calculate exactly how much of the tile is land
        # Convert the tile to a GeoSeries and reproject to metric (UTM 38S)
        tile_gs_wgs84 = gpd.GeoSeries([tile_geom_wgs84], crs="EPSG:4326")
        tile_gs_utm = tile_gs_wgs84.to_crs(epsg=32738)
        tile_geom_utm = tile_gs_utm.iloc[0]
        
        # Calculate how much of the tile overlaps with the Madagascar polygon
        intersection_geom = tile_geom_utm.intersection(madagascar_utm_geom)
        
        tile_area = tile_geom_utm.area
        land_area = intersection_geom.area
        ratio = land_area / tile_area
        
        # Keep only tiles with a good balance of land and sea
        if not (MIN_LAND_RATIO <= ratio <= MAX_LAND_RATIO):
            continue

        # Download Logic
        tile_url = f"{arcgis_url}/{tile.z}/{tile.y}/{tile.x}"
        filename = f"{tile.z}_{tile.x}_{tile.y}.jpg"  # Changed to .jpg
        filepath = os.path.join("tiles", filename)

        try:
            r = requests.get(tile_url, timeout=10)

            # Some empty ocean/land tiles are tiny, so > 5000 bytes ensures it's real imagery
            if r.status_code == 200 and len(r.content) > 5000:

                with open(filepath, "wb") as f:
                    f.write(r.content)

                # Save metadata, now including the calculated ratio
                writer.writerow([
                    filename, tile.z, tile.x, tile.y,
                    bounds.west, bounds.south, bounds.east, bounds.north,
                    "EPSG:4326", round(ratio, 3)
                ])

                time.sleep(0.05)

        except Exception as e:
            print("Error downloading {filename}:", e)

print("Download complete.")
print("Metadata saved to metadata.csv")