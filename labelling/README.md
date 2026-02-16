# High-Resolution Geo-Spatial Annotation with SAM

## Overview

This project focuses on automating the detection and classification of boats in high-resolution geospatial imagery using the Segment Anything Model (SAM). The primary goal is to evaluate SAM's effectiveness in identifying various types of vessels and categorizing them.

## Features

*   **Automated Boat Detection:** Utilizes SAM to identify boats in `.tif` satellite images.

*   **Multi-Label Classification:** Classifies detected boats into the following categories:
    *   Traditional Pirogue
    *   Motorboat
    *   Sailboat
    *   Catamaran
    *   Large fishing vessel
    *   Merchant ship
    *   Other vessel
  
*   **Evaluation Pipeline:** Provides scripts to evaluate the model's performance using metrics like False Positives, True Positives, and False Negatives.
  
*   **Geospatial Output:** Saves detection results in GeoJSON format for easy integration with GIS software.


## Methodology

The project follows these steps:

1.  **Manual Labeling:** A small subset of images is manually labeled to create a ground truth dataset for evaluation.
2.  **Image Preprocessing:** A ground mask is applied to the images to exclude land areas, focusing the detection on water bodies.
3.  **Prompt Engineering:** Different text prompts and preprocessing techniques are tested to optimize the detection accuracy.
    *   Prompts can be used for 2 things: To get high RealPositives or to get high FakeNegatives (and low real positive). The point is to remove the FakeNegatives from the labeling.
4.  **Automated Grid Search:** A grid search is performed to find the optimal combination of preprocessing steps and prompts to maximize True Positives and minimize False Negatives.
5.  **Outliers and processing:** Implement a computational check (shape of prediction, ...).
6.  **Pipeline Construction:** The best-performing configuration is integrated into a clean and reusable processing pipeline.





-----

Start :

download sam3.pt and put it in asset -> https://huggingface.co/facebook/sam3 