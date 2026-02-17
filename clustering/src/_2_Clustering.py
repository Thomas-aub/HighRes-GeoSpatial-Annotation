import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

def run_clustering(df, n_clusters=4):
    """
    Combines image features and metadata, scales them, and performs GMM clustering.
    """
    # Extraction des caractéristiques
    img_feats = np.stack(df['img_feature'].values)
    meta_feats = df[['lat', 'lon', 'width', 'height']].values
    X_combined = np.hstack([img_feats, meta_feats])
    
    # Standardisation
    X_scaled = StandardScaler().fit_transform(X_combined)
    
    # PCA pour stabiliser le clustering
    X_pca = PCA(n_components=50, random_state=42).fit_transform(X_scaled)
    
    # GMM clustering
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    df['cluster'] = gmm.fit_predict(X_pca)
    
    return df, X_pca

if __name__ == "__main__":
    PKL_FILE = "../data/extracted/embeddings.pkl"
    RESULT_CSV = "../results/clustering_results.csv"
    PCA_FILE = "../results/X_pca.npy"
    
    if os.path.exists(PKL_FILE):
        df = pd.read_pickle(PKL_FILE)
        df_res, X_reduced = run_clustering(df, n_clusters=10)
        
        # S'assurer que le dossier results existe
        os.makedirs("../results", exist_ok=True)
        
        # SAUVEGARDE CRUCIALE POUR L'INTERACTIF : 
        # On garde chip_path et on prépare les colonnes pour les coordonnées t-SNE
        df_res.to_csv(RESULT_CSV, index=False)
        np.save(PCA_FILE, X_reduced)
        print(f"Clustering finished. Results saved to {RESULT_CSV}")