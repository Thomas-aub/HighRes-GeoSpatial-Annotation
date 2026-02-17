import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

def compute_and_save_tsne(csv_path, pca_path, png_path):
    """
    Computes t-SNE from PCA data and updates the results CSV with coordinates.
    """
    print("Loading data for t-SNE...")
    df = pd.read_csv(csv_path)
    X_pca = np.load(pca_path)
    
    # Ajout de jitter pour éviter les doublons exacts
    X_pca += np.random.normal(0, 1e-5, X_pca.shape)
    
    print("Computing t-SNE (this might take a minute)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_2d = tsne.fit_transform(X_pca)
    
    # Ajout des coordonnées au DataFrame
    df['tsne_1'] = X_2d[:, 0]
    df['tsne_2'] = X_2d[:, 1]
    
    # Sauvegarde du CSV mis à jour (écrase l'ancien)
    df.to_csv(csv_path, index=False)
    print(f"t-SNE coordinates added to {csv_path}")
    
    # Affichage statique simple
    plt.figure(figsize=(10, 8))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=df['cluster'], cmap='tab10', s=10)
    plt.title("Static t-SNE Visualization")
    plt.savefig(png_path)
    plt.show()

if __name__ == "__main__":
    CSV_FILE = "../results/clustering_results.csv"
    PCA_FILE = "../results/X_pca.npy"
    PNG_PATH = "../resultstsne_static.png"

    
    if os.path.exists(CSV_FILE) and os.path.exists(PCA_FILE):
        compute_and_save_tsne(CSV_FILE, PCA_FILE, PNG_PATH)
    else:
        print("Error: Clustering results not found. Run _2_Clustering.py first.")