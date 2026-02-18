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

    This function takes a DataFrame with image embeddings and metadata,
    combines them, applies scaling and PCA, and then performs GMM clustering.
    It adds the cluster labels and the first principal component to the
    DataFrame.

    Args:
        df: DataFrame containing 'img_feature' and metadata columns 
            ('lat', 'lon', 'orig_width', 'orig_height').
        n_clusters: The number of clusters for the Gaussian Mixture Model.

    Returns:
        A tuple containing:
        - The updated DataFrame with 'cluster' and 'pca_1d' columns.
        - The PCA-reduced data array (50 dimensions).
    """
    # 1. Prepare features by combining image embeddings and metadata
    img_feats = np.stack(df['img_feature'].values)
    meta_feats = df[[ 'original width', 'original height']].values # 'lat', 'lon',
    X_combined = np.hstack([img_feats, meta_feats])
    
    # 2. Standardize the combined feature set
    X_scaled = StandardScaler().fit_transform(X_combined)
    
    # 3. Apply PCA to reduce dimensionality for stable clustering
    pca_model = PCA(n_components=50, random_state=42)
    X_pca = pca_model.fit_transform(X_scaled)
    
    # 4. Store the first principal component (PC1) for ranking or sorting
    df['pca_1d'] = X_pca[:, 0]
    
    # 5. Perform GMM clustering on the PCA-reduced data
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    df['cluster'] = gmm.fit_predict(X_pca)
    
    return df, X_pca

if __name__ == "__main__":
    PKL_FILE = "../data/extracted/embeddings.pkl"
    RESULT_CSV = "../results/clustering_results.csv"
    PCA_FILE = "../results/X_pca.npy"
    
    if os.path.exists(PKL_FILE):
        print("Loading embeddings...")
        df = pd.read_pickle(PKL_FILE)
        
        # Run clustering (you can adjust n_clusters here)
        df_res, X_reduced = run_clustering(df, n_clusters=10)
        
        # Ensure results directory exists
        os.makedirs("../results", exist_ok=True)
        
        # Save updated CSV (now contains 'pca_1d' and 'cluster')
        df_res.to_csv(RESULT_CSV, index=False)
        
        # Save the 50D PCA matrix for t-SNE in the next step
        np.save(PCA_FILE, X_reduced)
        
        print(f"Clustering finished. 1D PCA and Clusters saved to {RESULT_CSV}")
    else:
        print(f"Error: {PKL_FILE} not found. Run _1_Embedding.py first.")