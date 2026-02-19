import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from typing import Tuple

def run_clustering(
    df: pd.DataFrame, 
    min_clusters: int = 4,
    max_clusters: int = 15  
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Performs clustering on combined image and metadata features.
    Automatically determines the optimal number of clusters using Silhouette Score
    to favor fewer, more distinct clusters.
    """
    # 1. Compute geographic centers from tile bounds
    df['center_lat'] = (df['north'] + df['south']) / 2.0
    df['center_lon'] = (df['east'] + df['west']) / 2.0

    # 2. Prepare features by combining image embeddings and spatial metadata
    img_feats = np.stack(df['img_feature'].values)
    meta_feats = df[['center_lat', 'center_lon']].values
    
    # Standardize the combined feature set
    # X_combined = np.hstack([img_feats, meta_feats])
    X_combined = np.hstack([img_feats])
    X_scaled = StandardScaler().fit_transform(X_combined)
    
    # 3. Apply PCA to reduce dimensionality for stable clustering
    pca_model = PCA(n_components=30, random_state=42)
    X_pca = pca_model.fit_transform(X_scaled)
    
    # Store the first principal component (PC1)
    df['pca_1d'] = X_pca[:, 0]
    
    # 4. Find the optimal number of clusters using Silhouette Score
    print(f"Evaluating optimal number of clusters between {min_clusters} and {max_clusters}...")
    best_gmm = None
    best_score = -1.0  # Silhouette score ranges from -1 to 1 (higher is better)
    best_k = min_clusters
    
    for k in range(min_clusters, max_clusters + 1):
        gmm = GaussianMixture(n_components=k, random_state=42)
        # Fit and get labels to compute the silhouette score
        labels = gmm.fit_predict(X_pca)
        
        # Calculate Silhouette Score
        score = silhouette_score(X_pca, labels)
        
        print(f"  -> Testing k={k}: Silhouette Score = {score:.4f}")
        
        # We want the HIGHEST silhouette score
        if score > best_score:
            best_score = score
            best_gmm = gmm
            best_k = k
            
    print(f"\n✅ Optimal number of clusters selected: {best_k} (Highest Silhouette: {best_score:.4f})")
    
    # 5. Apply the best GMM model to generate final cluster labels
    df['cluster'] = best_gmm.predict(X_pca)
    
    return df, X_pca

if __name__ == "__main__":
    PKL_FILE = "./data/embeddings.pkl"
    RESULT_CSV = "./data/clustering_results.csv"
    PCA_FILE = "./data/X_pca.npy"
    
    if os.path.exists(PKL_FILE):
        print("Loading embeddings...")
        df = pd.read_pickle(PKL_FILE)
        
        # Run clustering 
        df_res, X_reduced = run_clustering(df, min_clusters=4, max_clusters=8)
        
        # Ensure results directory exists
        os.makedirs("./data", exist_ok=True)
        
        # Save updated CSV (now contains 'pca_1d', 'cluster', 'center_lat', 'center_lon')
        if 'img_feature' in df_res.columns:
            df_res_clean = df_res.drop(columns=['img_feature'])
            if 'tile_path' in df_res_clean.columns:
                df_res_clean = df_res_clean.drop(columns=['tile_path'])
        else:
            df_res_clean = df_res
            
        df_res_clean.to_csv(RESULT_CSV, index=False)
        np.save(PCA_FILE, X_reduced)
        
        print(f"Clustering finished. Results saved to {RESULT_CSV}")
    else:
        print(f"Error: {PKL_FILE} not found. Run _2_Embedding.py first.")