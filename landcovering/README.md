# High-Resolution Geospatial Annotation for Land Covering

This project performs unsupervised clustering on high-resolution satellite imagery of the Madagascar coastline. The goal is to automatically identify and group different types of land and sea cover without pre-existing labels.

## Approach

The methodology is divided into four main stages:

1.  **Data Extraction**: Relevant satellite image tiles along the coast of Madagascar are identified and downloaded. The selection process specifically targets tiles that contain a mix of land and sea to focus on the coastal interface.

2.  **Embedding Generation**: Each downloaded image is processed by a deep learning model (SatDINO or ResNet50) to generate a high-dimensional feature vector, or "embedding". This embedding represents the visual content of the image in a numerical format that is suitable for machine learning.

3.  **Clustering**: The generated embeddings, combined with the geographical coordinates of each tile, are grouped into clusters using a Gaussian Mixture Model (GMM). Principal Component Analysis (PCA) is used for dimensionality reduction before clustering. The optimal number of clusters can be determined using metrics like the Bayesian Information Criterion (BIC) or Silhouette Score.

4.  **Visualization**: The results are visualized in two ways:
    *   A static map showing the spatial distribution of the identified clusters across Madagascar.
    *   An interactive map that allows for the inspection of individual image tiles by clicking on their corresponding points.

## Code Structure

The project is structured as a sequence of Python scripts located in the `src/` directory. They are numbered to indicate the order of execution.

*   **`_1_DataExtraction.py`**:
    *   Fetches the geographical boundary of Madagascar.
    *   Identifies map tiles at a specific zoom level that intersect with the coastline.
    *   Filters tiles to ensure a desired land/sea ratio (e.g., between 30% and 65% land).
    *   Downloads the selected tiles from an ArcGIS map server.
    *   Saves metadata for each tile (coordinates, filename, land ratio) into `metadata.csv`.

*   **`_2_Embedding.py`**:
    *   Loads the `metadata.csv`.
    *   For each image tile, it computes an embedding using either the **SatDINO** (default) or **ResNet50** model.
    *   Saves the DataFrame, now including the image embeddings, to `data/embeddings.pkl`.

*   **`_3_Clustering.py` / `_3_Clustering_BIC.py` / `_3_Clustering_Silhouette.py`**:
    *   Loads the embeddings from `data/embeddings.pkl`.
    *   Combines image features with geographical center coordinates for each tile.
    *   Applies StandardScaler and PCA for feature normalization and dimensionality reduction.
    *   Performs clustering using a Gaussian Mixture Model:
        *   `_3_Clustering.py`: Uses a fixed number of clusters.
        *   `_3_Clustering_BIC.py`: Automatically finds the best number of clusters using BIC.
        *   `_3_Clustering_Silhouette.py`: Automatically finds the best number of clusters using the Silhouette Score.
    *   Saves the clustering results (including cluster labels) to `data/clustering_results.csv` and the PCA-reduced data to `data/X_pca.npy`.

*   **`_4_PlotMap.py`**:
    *   Loads the `data/clustering_results.csv`.
    *   Generates and saves a static map (`data/cluster_map.png`) that plots the location of each tile, colored by its assigned cluster.

*   **`_4_Interactive_Map.py`**:
    *   Loads the `data/clustering_results.csv`.
    *   Creates an interactive matplotlib window displaying the clustered points on the map of Madagascar.
    *   Allows the user to click on any point to open a new window displaying the corresponding satellite image tile.
