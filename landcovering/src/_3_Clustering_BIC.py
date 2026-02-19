import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from typing import Tuple

def run_clustering(
    df: pd.DataFrame, 
    min_clusters: int = 2,
    max_clusters: int = 15
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Performs clustering on combined image and metadata features.
    Automatically determines the optimal number of clusters using BIC.
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
    pca_model = PCA(n_components=50, random_state=42)
    X_pca = pca_model.fit_transform(X_scaled)
    
    # 5. Store the first principal component (PC1)
    df['pca_1d'] = X_pca[:, 0]
    
    # 6. Find the optimal number of clusters using the Bayesian Information Criterion (BIC)
    print(f"Evaluating optimal number of clusters between {min_clusters} and {max_clusters}...")
    best_gmm = None
    lowest_bic = np.inf
    best_k = min_clusters
    
    for k in range(min_clusters, max_clusters + 1):
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(X_pca)
        bic = gmm.bic(X_pca)
        
        print(f"  -> Testing k={k}: BIC = {bic:.2f}")
        
        if bic < lowest_bic:
            lowest_bic = bic
            best_gmm = gmm
            best_k = k
            
    print(f"\n✅ Optimal number of clusters selected: {best_k} (Lowest BIC: {lowest_bic:.2f})")
    
    # 7. Apply the best GMM model to generate final cluster labels
    df['cluster'] = best_gmm.predict(X_pca)
    
    return df, X_pca

if __name__ == "__main__":
    PKL_FILE = "./data/embeddings.pkl"
    RESULT_CSV = "./data/clustering_results.csv"
    PCA_FILE = "./data/X_pca.npy"
    
    if os.path.exists(PKL_FILE):
        print("Loading embeddings...")
        df = pd.read_pickle(PKL_FILE)
        
        # Run clustering (it will search for the best k between 2 and 15 by default)
        df_res, X_reduced = run_clustering(df, min_clusters=2, max_clusters=15)
        
        # Ensure results directory exists
        os.makedirs("./data", exist_ok=True)
        
        # Save updated CSV (now contains 'pca_1d', 'cluster', 'center_lat', 'center_lon')
        # Dropping the heavy embedding column before saving to CSV
        if 'img_feature' in df_res.columns:
            df_res_clean = df_res.drop(columns=['img_feature'])
            # Also drop tile_path if it was generated in the previous step
            if 'tile_path' in df_res_clean.columns:
                df_res_clean = df_res_clean.drop(columns=['tile_path'])
        else:
            df_res_clean = df_res
            
        df_res_clean.to_csv(RESULT_CSV, index=False)
        
        # Save the 50D PCA matrix for t-SNE in the next step
        np.save(PCA_FILE, X_reduced)
        
        print(f"Clustering finished. Results saved to {RESULT_CSV}")
    else:
        print(f"Error: {PKL_FILE} not found. Run _2_Embedding.py first.")