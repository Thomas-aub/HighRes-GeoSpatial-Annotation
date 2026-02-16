# Code Structure Documentation

This document outlines the structure of the Python source code for this project.

## `data` Directory

The `data` directory stores all raw and preprocessed data used by the project. It's organized to manage different stages of data processing.

### `data/ground_truth`
Contains ground truth labels or annotations used for training and evaluation.

### `data/img1`, `data/img2`, etc.
These directories hold raw and preprocessed data for specific images or datasets.

-   `raw/`: Stores the original, untouched image data.
-   `preprocess1/`, `preprocess2/`, etc.: Store data after specific preprocessing steps have been applied (e.g., till1.tiff, till2.tiff are processed versions of raw data).

## `results` Directory

The `results` directory mirrors the structure of the `data` directory and is used to store the output of the model's predictions.

-   `img1/`, `img2/`, etc.: Each subdirectory corresponds to an image or dataset from the `data` directory and holds the model's output for that data.

## `src` Directory

The `src` directory contains all the Python source code for the project, organized into several packages:

### `src/data_preparation`

This package is responsible for all data handling before the modeling phase.

-   `preprocessing.py`: Contains functions for preprocessing the raw data.
-   `tiling.py`: Contains functions for cutting the larger images into smaller tiles for processing.

### `src/evaluation`

This package contains modules for evaluating the performance of the model.

-   `evaluation_metrics.py`: Implements metrics such as precision, recall, and F1-score.

### `src/modeling`

This package contains the core model implementation.

-   `model.py`: Defines the neural network architecture and training logic.

### `src/postprocessing`

This package is for any processing that needs to happen after the model has made its predictions.

-   `refinement.py`: Contains functions for refining the raw model output.

### `src/utils`

A collection of utility functions that are used across the project.

-   `file_utils.py`: Helper functions for file I/O and path manipulation.
