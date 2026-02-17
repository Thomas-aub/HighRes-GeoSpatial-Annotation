import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

def compute_and_save_tsne(csv_path: str, pca_path: str, png_path: str) -> None:
    """
    Computes 2D t-SNE, updates the CSV, and saves a static plot.

    This function loads the PCA-reduced data and clustering results,
    computes 2D t-SNE coordinates, adds them to the results CSV, and
    saves a static scatter plot of the t-SNE visualization.

    Args:
        csv_path: Path to the clustering results CSV file. This file will
                  be overwritten with the new t-SNE coordinates.
        pca_path: Path to the .npy file containing the PCA-reduced data.
        png_path: Path to save the output static t-SNE plot.
    """
    print("Loading data for t-SNE...")
    df = pd.read_csv(csv_path)
    X_pca = np.load(pca_path)

    # Ensure data consistency between the CSV and the PCA array
    if len(df) != X_pca.shape[0]:
        raise ValueError(
            f"Row count mismatch: CSV has {len(df)} rows but X_pca has "
            f"{X_pca.shape[0]} rows. Re-run Step 2 (clustering) to ensure "
            f"both files are synchronized."
        )
    
    # Add a small amount of jitter to prevent issues with identical points
    X_pca += np.random.normal(0, 1e-5, X_pca.shape)
    
    print("Computing t-SNE (this might take a minute)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    X_2d = tsne.fit_transform(X_pca)
    
    # Add the new t-SNE coordinates to the DataFrame
    df['tsne_1'] = X_2d[:, 0]
    df['tsne_2'] = X_2d[:, 1]
    
    # Overwrite the CSV with the updated DataFrame
    df.to_csv(csv_path, index=False)
    print(f"t-SNE coordinates have been added to {csv_path}")
    
    # Generate and save a static t-SNE plot
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=df['cluster'], cmap='tab20', s=15, alpha=0.8)
    plt.title("Static t-SNE Visualization of Clusters", fontsize=14)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.grid(True, alpha=0.3)
    plt.savefig(png_path)
    print(f"Static t-SNE plot saved to {png_path}")
    plt.show()

if __name__ == "__main__":
    CSV_FILE = "../results/clustering_results.csv"
    PCA_FILE = "../results/X_pca.npy"
    PNG_PATH = "../results/tsne_static.png"

    if os.path.exists(CSV_FILE) and os.path.exists(PCA_FILE):
        compute_and_save_tsne(CSV_FILE, PCA_FILE, PNG_PATH)
    else:
        print("Error: Clustering results not found. Run _2_Clustering.py first.")