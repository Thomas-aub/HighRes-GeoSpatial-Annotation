import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import os

# --- CONFIGURATION ---
CSV_FILE = "clustering_results.csv"
RANDOM_STATE = 42
PERPLEXITY = 30

def load_and_prep_data(csv_path):
    print("Loading data...")
    df = pd.read_csv(csv_path)
    
    # Re-create features for t-SNE
    features = ['img_feature', 'lat', 'lon', 'width', 'height']
    X = df[features].values
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return df, X_scaled

def compute_tsne(X_scaled):
    print("Computing t-SNE (this might take a moment)...")
    n_samples = X_scaled.shape[0]
    perp = min(PERPLEXITY, n_samples - 1)
    
    tsne = TSNE(n_components=2, perplexity=perp, random_state=RANDOM_STATE)
    X_2d = tsne.fit_transform(X_scaled)
    return X_2d

def normalize_for_display(img_data):
    """
    Converts 16-bit/float satellite data (e.g. 0-4096) 
    to viewable 8-bit RGB (0-255).
    """
    # 1. Handle Multispectral (Take only first 3 bands if > 3)
    if img_data.ndim == 3 and img_data.shape[2] > 3:
        img_data = img_data[:, :, :3]
        
    # 2. Min-Max Normalization to 0-255
    img_min = img_data.min()
    img_max = img_data.max()
    
    if img_max > img_min:
        # Scale to 0-1
        img_norm = (img_data - img_min) / (img_max - img_min)
        # Scale to 0-255
        img_8bit = (img_norm * 255).astype(np.uint8)
    else:
        # If image is flat (all one color), just return as is
        img_8bit = img_data.astype(np.uint8)
        
    return img_8bit

def on_pick(event):
    ind = event.ind[0]
    row = df.iloc[ind]
    chip_path = row['chip_path']
    cluster_id = row['cluster']
    
    print(f"Clicked Point #{ind}: Cluster {cluster_id}")
    
    try:
        if os.path.exists(chip_path):
            img_data = np.load(chip_path)
            
            # --- FIX APPLIED HERE ---
            img_to_show = normalize_for_display(img_data)

            # Popup
            fig_popup, ax_popup = plt.subplots(figsize=(4, 4))
            ax_popup.imshow(img_to_show)
            ax_popup.set_title(f"ID: {row['chip_id']}\nCluster: {cluster_id}\nSize: {row['width']:.1f}m x {row['height']:.1f}m")
            ax_popup.axis('off')
            plt.show()
        else:
            print(f"  [Error] File not found: {chip_path}")
            
    except Exception as e:
        print(f"  [Error] Could not open image: {e}")

# --- MAIN ---
if __name__ == "__main__":
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found. Run the clustering script first.")
        exit()

    df, X_scaled = load_and_prep_data(CSV_FILE)
    X_2d = compute_tsne(X_scaled)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], 
                         c=df['cluster'], 
                         cmap='viridis', 
                         s=50, 
                         alpha=0.7, 
                         picker=5)
    
    plt.colorbar(scatter, label='Cluster ID')
    plt.title('Interactive Boat Clusters\nCLICK a dot to see the image!')
    plt.xlabel('t-SNE Dim 1')
    plt.ylabel('t-SNE Dim 2')
    plt.grid(True, alpha=0.3)

    fig.canvas.mpl_connect('pick_event', on_pick)
    
    print("\nPlot open. Click on points to view images.")
    plt.show()