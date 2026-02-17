import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backend_bases import PickEvent
import rasterio
from rasterio.warp import transform
from rasterio.plot import reshape_as_image
import os
from typing import Optional

# --- CONFIGURATION ---
CSV_FILE = "../results/clustering_results.csv"
RAW_DATA_DIR = "../data/raw"
# Get the directory of the current script and construct the path to the data
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data", "extracted")

def normalize_for_display(img_data: np.ndarray) -> np.ndarray:
    """
    Normalizes raw image data (e.g., uint16, float) to a displayable uint8 format.

    It handles multi-channel images by using the first 3 channels and performs
    a percentile stretch to enhance contrast.

    Args:
        img_data: The input image as a NumPy array.

    Returns:
        The normalized image as a uint8 NumPy array, ready for display.
    """
    if img_data.ndim == 3 and img_data.shape[2] > 3:
        img_data = img_data[:, :, :3]  # Keep only the first 3 channels
    elif img_data.ndim == 2:
        img_data = np.stack([img_data] * 3, axis=-1)  # Convert grayscale to RGB

    img_data = img_data.astype(np.float32)
    p2, p98 = np.percentile(img_data, (2, 98))
    
    # Avoid division by zero if the image is flat
    if p98 > p2:
        img_norm = np.clip((img_data - p2) / (p98 - p2), 0, 1)
        return (img_norm * 255).astype(np.uint8)
    
    # Return as is if data is uniform
    return img_data.astype(np.uint8)


class UltimateInteractivePlot:
    """
    Manages an interactive t-SNE plot for exploring clustering results.

    This class creates a scatter plot where each point represents an image chip.
    Clicking a point triggers the display of the chip itself and its broader
    geospatial context from the original satellite image.
    """
    def __init__(self, csv_file: str):
        """
        Initializes the plot and loads data.

        Args:
            csv_file: Path to the clustering results CSV file, which must
                      contain t-SNE coordinates, cluster labels, and metadata.
        """
        self.df: pd.DataFrame = pd.read_csv(csv_file)
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        
        # Track figure references to manage windows
        self.fig_chip: Optional[plt.Figure] = None
        self.fig_ctx: Optional[plt.Figure] = None
        
        self.setup_main_plot()

    def setup_main_plot(self) -> None:
        """
        Configures and displays the main t-SNE scatter plot.
        """
        print(f"Plotting {len(self.df)} data points...")
        self.scatter = self.ax.scatter(
            self.df['tsne_1'], 
            self.df['tsne_2'], 
            c=self.df['cluster'], 
            cmap='tab20', 
            s=15, alpha=0.7, picker=5
        )
        
        self.ax.legend(*self.scatter.legend_elements(), title="Clusters")
        self.ax.set_title('Interactive t-SNE: Click a point to see details', fontsize=14)
        self.ax.grid(True, alpha=0.3)
        
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)

    def show_global_context(self, row: pd.Series) -> None:
        """
        Displays the chip's location within the original, large TIFF image.

        Args:
            row: The pandas Series corresponding to the selected point.
        """
        # Close the previous context window if it exists
        if self.fig_ctx and plt.fignum_exists(self.fig_ctx.number):
            plt.close(self.fig_ctx)

        chip_name = os.path.basename(row['chip_path'])
        base_name = chip_name.rsplit('_', 1)[0]
        tif_path = os.path.join(RAW_DATA_DIR, base_name + ".tif")

        if not os.path.exists(tif_path):
            print(f"Context image not found: {tif_path}")
            return

        with rasterio.open(tif_path) as src:
            # Transform point from WGS84 to the image's CRS
            x_img, y_img = transform('EPSG:4326', src.crs, [row['lon']], [row['lat']])
            py, px = src.index(x_img[0], y_img[0])
            
            # Define a window around the point
            window_size = 2000
            window = rasterio.windows.Window(
                px - window_size // 2, py - window_size // 2, 
                window_size, window_size
            )
            
            # Read the data, ensuring we only take 3 bands for RGB display
            img_context = src.read(window=window, boundless=True, fill_value=0)
            if img_context.shape[0] > 3:
                img_context = img_context[:3, :, :]
            
            img_show = reshape_as_image(img_context)
            img_show = normalize_for_display(img_show)

            # Create and display the context plot
            self.fig_ctx, ax_ctx = plt.subplots(figsize=(8, 8))
            ax_ctx.imshow(img_show)
            
            # Draw a box indicating the chip's approximate location
            rect = patches.Rectangle(
                (window_size // 2 - 50, window_size // 2 - 50), 100, 100, 
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax_ctx.add_patch(rect)
            ax_ctx.set_title(f"Global Context: {base_name}\nLat: {row['lat']:.5f}, Lon: {row['lon']:.5f}")
            
            # Position the context window
            self.fig_ctx.canvas.manager.window.move(100, 1400) # Bottom-left
            plt.show()

    def on_pick(self, event: PickEvent) -> None:
        """
        Handles click events on the scatter plot.

        Args:
            event: The matplotlib PickEvent triggered by clicking a point.
        """
        
            
        ind = event.ind[0]
        row = self.df.iloc[ind]

        # Close the previous chip window if it exists
        if self.fig_chip and plt.fignum_exists(self.fig_chip.number):
            plt.close(self.fig_chip)

        chip_filename = os.path.basename(row['chip_path'])
        chip_path = os.path.join(DATA_DIR, chip_filename)

        if os.path.exists(chip_path):
            img_raw = np.load(chip_path)
            img_display = normalize_for_display(img_raw)

            # Create and display the chip plot
            self.fig_chip, ax_chip = plt.subplots(figsize=(5, 5))
            ax_chip.imshow(img_display)
            ax_chip.set_title(
                f"Chip: {os.path.basename(row['chip_path'])}\n"
                f"Cluster: {row['cluster']} | Size: {row['orig_width']:.1f}x{row['orig_height']:.1f}m"
            )
            ax_chip.axis('off')
            # Position the chip window
            self.fig_chip.canvas.manager.window.move(100, 100)  # Top-left
            plt.show(block=False)
        else:
            print(f"Chip image not found: {chip_path}")

        # Show the global context for the selected chip
        self.show_global_context(row)

if __name__ == "__main__":
    if os.path.exists(CSV_FILE):
        app = UltimateInteractivePlot(CSV_FILE)
        plt.show()
    else:
        print(f"Error: Run the previous scripts first to generate {CSV_FILE}")