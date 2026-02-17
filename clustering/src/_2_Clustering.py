import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

def run_clustering(df, n_clusters=4):
    """
    Combines image features and metadata, scales them, and performs GMM clustering.
    Also adds a 1D PCA component for ranking and sorting.
    """
    # 1. Prepare features (Embeddings + Metadata)
    img_feats = np.stack(df['img_feature'].values)
    meta_feats = df[['lat', 'lon', 'width', 'height']].values
    X_combined = np.hstack([img_feats, meta_feats])
    
    # 2. Standardize data
    X_scaled = StandardScaler().fit_transform(X_combined)
    
    # 3. PCA for clustering stability (50 dimensions)
    pca_model = PCA(n_components=50, random_state=42)
    X_pca = pca_model.fit_transform(X_scaled)
    
    # --- NEW STEP: 1D REDUCTION ---
    # We take the very first component (PC1) which represents the 
    # maximum variance in a single dimension.
    df['pca_1d'] = X_pca[:, 0]
    
    # 4. GMM clustering (using the 50 dimensions for better accuracy)
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
        df_res, X_reduced = run_clustering(df, n_clusters=4)
        
        # Ensure results directory exists
        os.makedirs("../results", exist_ok=True)
        
        # Save updated CSV (now contains 'pca_1d' and 'cluster')
        df_res.to_csv(RESULT_CSV, index=False)
        
        # Save the 50D PCA matrix for t-SNE in the next step
        np.save(PCA_FILE, X_reduced)
        
        print(f"Clustering finished. 1D PCA and Clusters saved to {RESULT_CSV}")
    else:
        print(f"Error: {PKL_FILE} not found. Run _1_Embedding.py first.")