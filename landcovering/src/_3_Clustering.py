import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from typing import Tuple

def run_clustering(
    df: pd.DataFrame, 
    n_clusters: int = 4
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Performs clustering on combined image and metadata features.
    """
    # 1. Compute geographic centers from tile bounds
    df['center_lat'] = (df['north'] + df['south']) / 2.0
    df['center_lon'] = (df['east'] + df['west']) / 2.0

    # 2. Prepare features by combining image embeddings and spatial metadata
    img_feats = np.stack(df['img_feature'].values)
    meta_feats = df[['center_lat', 'center_lon']].values
    
    # Weight the image features heavier than the spatial features if desired,
    # but Standard Scaler will level them out.
    X_combined = np.hstack([img_feats, meta_feats])
    
    # 3. Standardize the combined feature set
    X_scaled = StandardScaler().fit_transform(X_combined)
    
    # 4. Apply PCA to reduce dimensionality for stable clustering
    pca_model = PCA(n_components=30, random_state=42)
    X_pca = pca_model.fit_transform(X_scaled)
    
    # 5. Store the first principal component (PC1)
    df['pca_1d'] = X_pca[:, 0]
    
    # 6. Perform GMM clustering on the PCA-reduced data
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    df['cluster'] = gmm.fit_predict(X_pca)
    
    return df, X_pca

if __name__ == "__main__":
    PKL_FILE = "./data/embeddings.pkl"
    RESULT_CSV = "./data/clustering_results.csv"
    PCA_FILE = "./data/X_pca.npy"
    
    if os.path.exists(PKL_FILE):
        print("Loading embeddings...")
        df = pd.read_pickle(PKL_FILE)
        
        # Run clustering (adjust n_clusters based on your Madagascar landscape needs)
        df_res, X_reduced = run_clustering(df, n_clusters=5)
        
        # Ensure results directory exists
        os.makedirs("../data", exist_ok=True)
        
        # Save updated CSV (now contains 'pca_1d', 'cluster', 'center_lat', 'center_lon')
        # Dropping the heavy embedding column before saving to CSV
        df_res_clean = df_res.drop(columns=['img_feature', 'tile_path'])
        df_res_clean.to_csv(RESULT_CSV, index=False)
        
        # Save the 50D PCA matrix for t-SNE in the next step
        np.save(PCA_FILE, X_reduced)
        
        print(f"Clustering finished. Results saved to {RESULT_CSV}")
    else:
        print(f"Error: {PKL_FILE} not found. Run _2_Embedding.py first.")