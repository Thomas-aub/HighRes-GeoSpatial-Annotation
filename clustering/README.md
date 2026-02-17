# High-Resolution Geo-Spatial Annotation Clustering

This project is a pipeline for analyzing high-resolution geospatial imagery. It automatically extracts regions of interest (like boats or buildings) from large satellite images, generates feature embeddings, clusters them to find patterns, and provides powerful visualization tools to inspect the results.

## Features

- **Chip Extraction**: Automatically extracts small image "chips" from large TIFF images based on vector annotations (GeoJSON).
- **Deep Learning Embeddings**: Uses pre-trained models (SatDINO or ResNet50) to generate meaningful feature vectors (embeddings) for each chip.
- **Clustering**: Applies Gaussian Mixture Models (GMM) to group similar chips together based on their embeddings and metadata.
- **Dimensionality Reduction**: Uses PCA and t-SNE to reduce the high-dimensional embedding data into 2D for visualization.
- **Interactive Visualization**: Provides an interactive t-SNE plot. Clicking a point on the plot displays:
    1.  The individual image chip.
    2.  The chip's location within the original, larger satellite image for global context.

## Workflow

The project is structured as a sequential pipeline. You must run the scripts in the `src/` directory in order.

1.  **`_0_Chips.py`**:
    -   Reads raw `.tif` images and `.geojson` annotation files from `data/raw/`.
    -   Extracts image chips for each annotated feature.
    -   Saves the chips as `.npy` files in `data/extracted/`.
    -   Creates a `metadata_cache.csv` file with information about each chip (ID, path, coordinates, etc.).

2.  **`_1_Embedding.py`**:
    -   Loads the metadata from `metadata_cache.csv`.
    -   For each chip, it loads the image and passes it through a deep learning model (e.g., SatDINO) to generate a feature embedding.
    -   Saves the metadata along with the embeddings to `data/extracted/embeddings.pkl`.

3.  **`_2_Clustering.py`**:
    -   Loads the embeddings from `embeddings.pkl`.
    -   Combines image embeddings with metadata (latitude, longitude, etc.).
    -   Applies PCA for dimensionality reduction.
    -   Performs GMM clustering to group similar items.
    -   Saves the results (including cluster labels) to `results/clustering_results.csv` and the PCA-reduced data to `results/X_pca.npy`.

4.  **`_3_Visualization.py`**:
    -   Loads the clustering results and PCA data.
    -   Computes 2D t-SNE coordinates for visualization.
    -   Saves an updated `clustering_results.csv` with the t-SNE coordinates.
    -   Creates a static t-SNE scatter plot image (`results/tsne_static.png`).

5.  **`_4_Interactive_Map_Context.py`**:
    -   Launches an interactive matplotlib window with the t-SNE plot.
    -   Clicking on any point in the plot will open two new windows:
        -   A view of the selected image chip.
        -   A view of the surrounding area in the original satellite image, highlighting the chip's location.

## Directory Structure

```
.
├── data/
│   ├── extracted/      # For processed data like chips and embeddings
│   └── raw/            # Place your initial .tif and .geojson files here
├── results/            # Output files (CSV, plots, etc.)
└── src/                # Python scripts for the pipeline
    ├── _0_Chips.py
    ├── _1_Embedding.py
    ├── _2_Clustering.py
    ├── _3_Visualization.py
    └── _4_Interactive_Map_Context.py
```

## How to Run

1.  **Setup**:
    -   Install the required Python libraries (e.g., `pandas`, `numpy`, `scikit-learn`, `tensorflow`, `torch`, `transformers`, `matplotlib`, `rasterio`, `geopandas`, `opencv-python`).
    -   Place your satellite images (`.tif`) and corresponding annotations (`.geojson`) into the `data/raw/` directory.

2.  **Execute the pipeline**:
    -   Run the scripts from the `src/` directory in sequential order:
    ```bash
    python src/_0_Chips.py
    python src/_1_Embedding.py
    python src/_2_Clustering.py
    python src/_3_Visualization.py
    python src/_4_Interactive_Map_Context.py
    ```

## Dependencies
This project relies on several open-source libraries, including:
- pandas and geopandas for data manipulation.
- scikit-learn for clustering and dimensionality reduction.
- rasterio for geospatial image processing.
- matplotlib for plotting.
- cv2 (OpenCV) for image manipulation.
- Tensorflow and/or PyTorch for deep learning models.
- transformers for the SatDINO model.
