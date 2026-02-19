import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.backend_bases import PickEvent
from matplotlib.colors import ListedColormap 
from PIL import Image 
from typing import Optional

class InteractiveTileMap:
    """
    Manages an interactive map for exploring clustered satellite tiles.
    """
    def __init__(self, csv_file: str, tiles_dir: str):
        self.tiles_dir = tiles_dir
        self.df = pd.read_csv(csv_file)
        
        required_cols = ['center_lat', 'center_lon', 'cluster', 'filename']
        if not all(col in self.df.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns: {required_cols}")

        # --- FILTER FOR SPECIFIC CLASSES AND RESET INDEX ---
        self.df = self.df[self.df['cluster'].isin([0, 1, 2, 3, 4])]
        # Resetting the index is critical so the picker event maps to the correct row!
        self.df = self.df.reset_index(drop=True) 
        # ---------------------------------------------------

        self.fig, self.ax = plt.subplots(figsize=(10, 15))
        self.fig_tile: Optional[plt.Figure] = None
        
        self.setup_main_map()

    def setup_main_map(self) -> None:
        print("Loading Madagascar boundaries...")
        url = "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip"
        world = gpd.read_file(url)
        madagascar = world[world["ADMIN"] == "Madagascar"]

        madagascar.plot(ax=self.ax, color='whitesmoke', edgecolor='dimgrey', linewidth=1, zorder=1)

        print(f"Plotting {len(self.df)} interactive data points (Classes 0, 1, 2)...")
        
        # --- DEFINE CUSTOM COLORS (Trimmed to 3 colors) ---
        custom_hex_colors = ['#D6A520', '#0F5F49', '#88CCEE', "#D62920", "#0F5F49"]
        custom_cmap = ListedColormap(custom_hex_colors)
        # --------------------------------------------------

        self.scatter = self.ax.scatter(
            self.df['center_lon'], 
            self.df['center_lat'], 
            c=self.df['cluster'], 
            cmap=custom_cmap,  
            s=30, alpha=1.0, picker=5, zorder=2  
        )
        
        self.ax.legend(*self.scatter.legend_elements(), title="Clusters", loc='lower right')
        self.ax.set_title('Interactive Tile Map: Click a point to view the image', fontsize=14)
        self.ax.set_xlabel("Longitude")
        self.ax.set_ylabel("Latitude")
        self.ax.grid(True, linestyle='--', alpha=0.4)
        
        minx, miny, maxx, maxy = madagascar.total_bounds
        self.ax.set_xlim(minx - 1, maxx + 1)
        self.ax.set_ylim(miny - 1, maxy + 1)
        
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)

    def on_pick(self, event: PickEvent) -> None:
        ind = event.ind[0]
        row = self.df.iloc[ind]

        if self.fig_tile and plt.fignum_exists(self.fig_tile.number):
            plt.close(self.fig_tile)

        tile_filename = row['filename']
        tile_path = os.path.join(self.tiles_dir, tile_filename)

        if os.path.exists(tile_path):
            try:
                img_display = Image.open(tile_path).convert('RGB')
            except Exception as e:
                print(f"⚠️ Could not open {tile_filename}. It may be corrupted or an HTML error page. ({e})")
                return

            self.fig_tile, ax_tile = plt.subplots(figsize=(6, 6))
            ax_tile.imshow(img_display)
            ax_tile.set_title(
                f"Tile: {tile_filename}\n"
                f"Cluster: {row['cluster']} | Lat: {row['center_lat']:.4f}, Lon: {row['center_lon']:.4f}"
            )
            ax_tile.axis('off')
            plt.show(block=False)
        else:
            print(f"Warning: Tile image not found at {tile_path}")

if __name__ == "__main__":
    RESULTS_CSV = "./data/clustering_results.csv"
    TILES_DIR = "./tiles/"
    
    if os.path.exists(RESULTS_CSV):
        app = InteractiveTileMap(RESULTS_CSV, TILES_DIR)
        plt.show()
    else:
        print(f"Error: {RESULTS_CSV} not found. Run the clustering script first.")