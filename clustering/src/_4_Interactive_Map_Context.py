import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import rasterio
from rasterio.warp import transform
from rasterio.plot import reshape_as_image
import os

# --- CONFIGURATION ---
CSV_FILE = "../results/clustering_results.csv"
RAW_DATA_DIR = "../data/raw"

def normalize_for_display(img_data):
    """
    Normalizes image data (float/uint16) to uint8 for display.
    Handles multispectral by taking the first 3 bands.
    """
    if img_data.ndim == 3 and img_data.shape[2] > 3:
        img_data = img_data[:, :, :3]
    elif img_data.ndim == 2:
        img_data = np.stack((img_data,)*3, axis=-1)

    img_data = img_data.astype(float)
    p2, p98 = np.percentile(img_data, (2, 98))
    if p98 > p2:
        img_norm = np.clip((img_data - p2) / (p98 - p2), 0, 1)
        return (img_norm * 255).astype(np.uint8)
    return img_data.astype(np.uint8)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data", "extracted")

class UltimateInteractivePlot:
    def __init__(self, csv_file):
        """
        Initializes the interactive plot with data from the clustering results.
        """
        self.df = pd.read_csv(csv_file)
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.setup_main_plot()

    def setup_main_plot(self):
        """
        Sets up the t-SNE scatter plot.
        """
        print(f"Plotting {len(self.df)} boats...")
        self.scatter = self.ax.scatter(
            self.df['tsne_1'], 
            self.df['tsne_2'], 
            c=self.df['cluster'], 
            cmap='tab10', 
            s=15, alpha=0.7, picker=5
        )
        
        self.ax.legend(*self.scatter.legend_elements(), title="Clusters")
        self.ax.set_title('Interactive Map: Click for Chip & Global Context', fontsize=14)
        self.ax.grid(True, alpha=0.3)
        
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)

    def show_global_context(self, row):
        """
        Opens a second window showing the boat's location in the original TIFF.
        """
        chip_name = os.path.basename(row['chip_path'])
        base_name = chip_name.rsplit('_', 1)[0]
        tif_path = os.path.join(RAW_DATA_DIR, base_name + ".tif")

        if not os.path.exists(tif_path):
            print(f"TIFF not found: {tif_path}")
            return

        with rasterio.open(tif_path) as src:
            # Convert WGS84 Lat/Lon to Image Pixels
            x_img, y_img = transform('EPSG:4326', src.crs, [row['lon']], [row['lat']])
            py, px = src.index(x_img[0], y_img[0])
            
            # Extract 2000x2000 context window
            window_size = 2000
            window = rasterio.windows.Window(
                px - window_size // 2, py - window_size // 2, 
                window_size, window_size
            )
            
            img_context = src.read(window=window, boundless=True, fill_value=0)
            if img_context.shape[0] > 3: img_context = img_context[:3, :, :]
            
            img_show = reshape_as_image(img_context)
            img_show = (img_show - img_show.min()) / (img_show.max() - img_show.min())

            fig_ctx, ax_ctx = plt.subplots(figsize=(7, 7))
            ax_ctx.imshow(img_show)
            
            # Red square at center
            rect = patches.Rectangle(
                (window_size//2 - 50, window_size//2 - 50), 100, 100, 
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax_ctx.add_patch(rect)
            ax_ctx.set_title(f"Global Context\nLat: {row['lat']:.5f} Lon: {row['lon']:.5f}")
            plt.show()

    

    def on_pick(self, event):
        ind = event.ind[0]
        row = self.df.iloc[ind]

        chip_filename = os.path.basename(row['chip_path'])
        chip_path = os.path.join(DATA_DIR, chip_filename)

        print(f"Looking for chip at: {chip_path}")
        print(f"Does chip path exist? {os.path.exists(chip_path)}")

        if os.path.exists(chip_path):
            img_raw = np.load(chip_path)
            img_display = normalize_for_display(img_raw)

            fig_chip, ax_chip = plt.subplots(figsize=(4, 4))
            ax_chip.imshow(img_display)
            ax_chip.set_title(f"Chip View\nCluster: {row['cluster']} | {row['width']:.1f}m")
            ax_chip.axis('off')
            # Position the chip window
            fig_chip.canvas.manager.window.move(100, 100)  # Top-left

            plt.show(block=False)
        else:
            print(f"Chip not found at: {chip_path}")

        self.show_global_context(row)
         # Position the context window
        plt.gcf().canvas.manager.window.move(100, 1400)  # Bottom-left


if __name__ == "__main__":

    if os.path.exists(CSV_FILE):
        app = UltimateInteractivePlot(CSV_FILE)
        plt.show()
    else:
        print(f"Error: Run the previous scripts first to generate {CSV_FILE}")