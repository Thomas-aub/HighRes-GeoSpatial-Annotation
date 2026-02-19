import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap # <-- Added for custom colors

def plot_clusters_on_map(csv_path: str, output_path: str):
    """
    Plots the clustered image tiles on a map of Madagascar.
    """
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please run the clustering script first.")
        return

    print("Loading clustering results...")
    df = pd.read_csv(csv_path)
    
    # Ensure required columns exist
    required_cols = ['center_lat', 'center_lon', 'cluster']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Missing one of the required columns {required_cols} in CSV.")
        return

    # 1. Convert the DataFrame to a GeoDataFrame
    gdf_points = gpd.GeoDataFrame(
        df, 
        geometry=gpd.points_from_xy(df.center_lon, df.center_lat),
        crs="EPSG:4326" # WGS84 coordinate system
    )

    # 2. Load Madagascar boundary
    print("Loading Madagascar boundaries...")
    url = "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip"
    world = gpd.read_file(url)
    madagascar = world[world["ADMIN"] == "Madagascar"]

    if madagascar.empty:
        raise ValueError("Madagascar not found in the Natural Earth dataset.")

    # --- DEFINE CUSTOM COLORS ---
    custom_hex_colors = ['#88CCEE', '#bb73ff', '#D6A520', '#E61354', '#0F5F49', "#1D9C1B", '#0F5F49']
    custom_cmap = ListedColormap(custom_hex_colors)
    # ----------------------------

    # 3. Create the plot
    print("Generating map...")
    fig, ax = plt.subplots(figsize=(10, 15)) 
    
    madagascar.plot(ax=ax, color='whitesmoke', edgecolor='dimgrey', linewidth=1, zorder=1)
    
    # Plot the cluster points using the custom colormap
    gdf_points.plot(
        ax=ax, 
        column='cluster', 
        cmap=custom_cmap,      # <-- Applied custom colormap here
        categorical=True, 
        legend=True,
        legend_kwds={'title': 'Cluster ID', 'loc': 'lower right'},
        markersize=15,         
        alpha=1.0,             
        zorder=2
    )

    # 4. Formatting and aesthetics
    plt.title("Satellite Tile Clusters Along the Coast of Madagascar", fontsize=15, pad=15)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, linestyle='--', alpha=0.4)
    
    minx, miny, maxx, maxy = madagascar.total_bounds
    ax.set_xlim(minx - 1, maxx + 1)
    ax.set_ylim(miny - 1, maxy + 1)

    # 5. Save and display
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Map successfully saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    RESULTS_CSV = "./data/clustering_results.csv"
    OUTPUT_MAP = "./data/cluster_map.png"
    
    plot_clusters_on_map(RESULTS_CSV, OUTPUT_MAP)