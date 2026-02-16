import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE

# --- CONFIGURATION ---
CONFIG = {
    'INPUT_DIR': './data/raw',
    'OUTPUT_DIR': './data/extracted/',
    'CHIP_SIZE': (64, 64),   # Fixed size for PCA analysis
    'N_COMPONENTS_PCA': 1,   # Extreme reduction as requested
    'N_CLUSTERS_GMM': 4,     # Number of Gaussian components
    'RANDOM_STATE': 42
}

# ---------------------------------------------------------
# 1. FILE & DIRECTORY MANAGEMENT
# ---------------------------------------------------------

def setup_directories(path):
    """Creates the output directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[Info] Created directory: {path}")

def get_file_pairs(directory):
    """Finds matching .tif and .geojson files."""
    tiffs = sorted(glob.glob(os.path.join(directory, "*.tif")))
    pairs = []
    for tif in tiffs:
        base_name = os.path.splitext(tif)[0]
        geojson = base_name + ".geojson"
        if os.path.exists(geojson):
            pairs.append((tif, geojson))
    return pairs

# ---------------------------------------------------------
# 2. CORE EXTRACTION LOGIC
# ---------------------------------------------------------

def process_single_pair(tif_path, geojson_path, output_dir, chip_size):
    """
    Process one pair of files:
    1. Reads GeoJSON.
    2. Projects to Image CRS.
    3. Extracts chips, saves them to disk.
    4. Calculates Metadata (Pos, W, H).
    """
    metadata_batch = []
    
    # Load GeoJSON
    gdf = gpd.read_file(geojson_path)
    
    # --- FIX: Drop rows with missing or empty geometries ---
    initial_len = len(gdf)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    if len(gdf) < initial_len:
        print(f"  [Warning] Dropped {initial_len - len(gdf)} empty geometries in {os.path.basename(geojson_path)}")
    
    if gdf.empty:
        return []

    with rasterio.open(tif_path) as src:
        # 1. Align Coordinate Reference Systems
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)
        
        # We also need Lat/Lon for the "Position" feature later
        # (Assuming src.crs is projected (meters), we want Lon/Lat for global context)
        try:
            gdf_wgs84 = gdf.to_crs("EPSG:4326")
        except Exception as e:
            print(f"  [Error] Could not reprojection to WGS84: {e}")
            return []

        for idx, row in gdf.iterrows():
            geom = row.geometry
            
            # Double check to be safe
            if geom is None:
                continue

            minx, miny, maxx, maxy = geom.bounds
            
            # --- Spatial Features ---
            width = maxx - minx
            height = maxy - miny
            
            # Get Centroid (Position)
            centroid = geom.centroid
            pos_x = centroid.x
            pos_y = centroid.y
            
            # Get Lat/Lon for generic "Position" feature
            # We use the same index 'idx' to grab the corresponding WGS84 geometry
            centroid_wgs = gdf_wgs84.loc[idx].geometry.centroid
            lat, lon = centroid_wgs.y, centroid_wgs.x

            # --- Image Extraction ---
            window = from_bounds(minx, miny, maxx, maxy, src.transform)
            
            try:
                img_chip = src.read(window=window)
                # Move channels: (C, H, W) -> (H, W, C)
                img_chip = np.moveaxis(img_chip, 0, -1)
                
                if img_chip.size == 0: continue

                # Resize to fixed size
                img_resized = cv2.resize(img_chip, chip_size)
                
                # Construct unique filename
                base_name = os.path.splitext(os.path.basename(tif_path))[0]
                chip_filename = f"{base_name}_{idx}.npy"
                save_path = os.path.join(output_dir, chip_filename)
                
                # Save Raw Chip
                np.save(save_path, img_resized)
                
                metadata_batch.append({
                    'chip_id': f"{base_name}_{idx}",
                    'chip_path': save_path,
                    'width': width,
                    'height': height,
                    'pos_x': pos_x, 
                    'pos_y': pos_y, 
                    'lat': lat,     
                    'lon': lon      
                })
                
            except Exception as e:
                # Edge cases (partial windows or out of bounds)
                pass

    return metadata_batch

def build_dataset(pairs, output_dir, chip_size):
    """Iterates all files and builds the master dataframe."""
    print(f"--- Starting Extraction to {output_dir} ---")
    setup_directories(output_dir)
    
    all_metadata = []
    
    for tif, geojson in pairs:
        print(f"Processing {os.path.basename(tif)}...")
        batch = process_single_pair(tif, geojson, output_dir, chip_size)
        all_metadata.extend(batch)
        
    return pd.DataFrame(all_metadata)

# ---------------------------------------------------------
# 3. DIMENSIONALITY REDUCTION
# ---------------------------------------------------------

def load_and_reduce_images(df):
    """
    Loads all saved .npy chips, flattens them, and runs PCA.
    Reduces entire image to N dimensions (User asked for 1).
    """
    print("--- Loading Chips & Reducing Dimensions ---")
    
    # Load all images into memory
    img_list = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        try:
            # Load .npy file
            img = np.load(row['chip_path'])
            img_list.append(img.flatten())
            valid_indices.append(idx)
        except:
            continue
            
    X_pixels = np.array(img_list)
    
    # Scale Pixels (0-255 -> Standard Normal)
    print("Scaling pixel data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pixels)
    
    # PCA to 1 Dimension
    print(f"Running PCA (Target Dimensions: {CONFIG['N_COMPONENTS_PCA']})...")
    pca = PCA(n_components=CONFIG['N_COMPONENTS_PCA'], random_state=CONFIG['RANDOM_STATE'])
    X_pca = pca.fit_transform(X_scaled)
    
    # Add the PCA feature(s) back to DataFrame
    # Since n=1, we just add one column
    df.loc[valid_indices, 'img_feature'] = X_pca[:, 0]
    
    return df.dropna(subset=['img_feature'])

# ---------------------------------------------------------
# 4. CLUSTERING (GMM)
# ---------------------------------------------------------

def cluster_data(df):
    """
    Clusters based on: [Image_Feature, Lat, Lon, Width, Height]
    Using Gaussian Mixture Model.
    """
    print("--- Clustering (Gaussian Mixture) ---")
    
    # Select Features
    # Note: Using Lat/Lon for position so it's global, rather than projected X/Y 
    # which might overlap incorrectly if files are from different zones.
    features = ['img_feature', 'lat', 'lon', 'width', 'height']
    X = df[features].values
    
    # CRITICAL: Scale features so "Lat" (45.0) and "Width" (10.0) and "Img" (3.0) 
    # are comparable.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # GMM
    gmm = GaussianMixture(n_components=CONFIG['N_CLUSTERS_GMM'], 
                          random_state=CONFIG['RANDOM_STATE'])
    labels = gmm.fit_predict(X_scaled)
    
    df['cluster'] = labels
    return df, X_scaled, scaler

# ---------------------------------------------------------
# 5. VISUALIZATION (t-SNE)
# ---------------------------------------------------------

def plot_tsne(X_scaled, labels, output_file='tsne_plot.png'):
    """Projects the 5D scaled data to 2D and plots."""
    print("--- Running t-SNE ---")
    
    # Handle perplexity for small datasets
    n_samples = X_scaled.shape[0]
    perp = min(30, n_samples - 1)
    
    tsne = TSNE(n_components=2, perplexity=perp, random_state=CONFIG['RANDOM_STATE'])
    X_2d = tsne.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
    plt.colorbar(scatter, label='GMM Cluster')
    plt.title(f'Boat Clusters (t-SNE)\nFeatures: Image(1D) + Pos + Size')
    plt.xlabel('t-SNE Dim 1')
    plt.ylabel('t-SNE Dim 2')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_file)
    print(f"[Info] Plot saved to {output_file}")
    plt.close()

# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------

def main():
    # 1. Setup
    pairs = get_file_pairs(CONFIG['INPUT_DIR'])
    if not pairs:
        print("No pairs found in input directory.")
        return

    # 2. Extract & Build Dataset
    df = build_dataset(pairs, CONFIG['OUTPUT_DIR'], CONFIG['CHIP_SIZE'])
    
    if df.empty:
        print("No boats extracted.")
        return
        
    print(f"Extracted {len(df)} samples.")

    # 3. Reduce Image Dimensions (Pixel -> 1D)
    df = load_and_reduce_images(df)

    # 4. Cluster (GMM)
    df, X_scaled, scaler = cluster_data(df)

    # 5. Visualize
    plot_tsne(X_scaled, df['cluster'], 'cluster_tsne.png')
    
    # 6. Save final stats
    print("\n--- Cluster Means (Scaled Domain) ---")
    print(df.groupby('cluster')[['img_feature', 'width', 'height', 'lat']].mean())
    df.to_csv("clustering_results.csv", index=False)
    print("\n[Info] Results saved to clustering_results.csv")

if __name__ == "__main__":
    main()