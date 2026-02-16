import sys
import os
import pandas as pd
import geopandas as gpd
from typing import Dict, Tuple, Optional, List

# --- PATH CONFIGURATION ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))

try:
    from src.utils.file_utils import calculate_iou, load_and_fix_geojson, align_projections
except ImportError:
    pass

# -------------------------------------------------------------------------
# Core Matching Logic
# -------------------------------------------------------------------------

def match_predictions_to_truth(
    gdf_truth: gpd.GeoDataFrame, 
    gdf_pred: gpd.GeoDataFrame, 
    iou_threshold: float = 0.5
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Matches prediction geometries to ground truth geometries based on Intersection over Union (IoU).

    This function performs a 1-to-1 matching:
    1. Iterates through each ground truth object.
    2. Finds the prediction with the highest IoU > threshold.
    3. Marks that prediction as 'used' so it cannot be matched again.

    Args:
        gdf_truth (gpd.GeoDataFrame): GeoDataFrame containing ground truth geometries.
        gdf_pred (gpd.GeoDataFrame): GeoDataFrame containing prediction geometries.
        iou_threshold (float, optional): The minimum IoU required to consider a detection 
                                         as a True Positive. Defaults to 0.5.

    Returns:
        Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]: A tuple containing:
            - tp_gdf: True Positives (Truth objects that were detected).
            - fn_gdf: False Negatives (Truth objects that were missed).
            - fp_gdf: False Positives (Predictions that matched nothing).
    """
    matched_pred_indices = set()
    tp_indices_truth = []
    fn_indices_truth = []

    if gdf_pred.empty:
        return gdf_truth.iloc[0:0], gdf_truth, gdf_pred

    sindex_pred = gdf_pred.sindex

    for idx_true, row_true in gdf_truth.iterrows():
        true_poly = row_true.geometry
        
        if not true_poly.is_valid:
            continue

        best_iou = 0.0
        best_pred_idx = -1
        
        candidate_indexes = list(sindex_pred.query(true_poly, predicate='intersects'))
        
        for idx_pred in candidate_indexes:
            # Enforce 1-to-1 matching
            if idx_pred in matched_pred_indices:
                continue
            
            pred_poly = gdf_pred.geometry.iloc[idx_pred]
            iou = calculate_iou(true_poly, pred_poly)
            
            if iou > best_iou:
                best_iou = iou
                best_pred_idx = idx_pred
        
        if best_iou >= iou_threshold:
            tp_indices_truth.append(idx_true)
            matched_pred_indices.add(best_pred_idx)
        else:
            fn_indices_truth.append(idx_true)

    gdf_tp = gdf_truth.loc[tp_indices_truth].copy()
    gdf_fn = gdf_truth.loc[fn_indices_truth].copy()
    
    all_pred_indices = set(gdf_pred.index)
    fp_indices = list(all_pred_indices - matched_pred_indices)
    gdf_fp = gdf_pred.loc[fp_indices].copy()

    return gdf_tp, gdf_fn, gdf_fp


def compute_metrics(tp: int, fn: int, fp: int, total_truth: int, total_pred: int) -> Dict[str, float]:
    """
    Calculates standard object detection metrics (Precision, Recall, F1-Score).

    Args:
        tp (int): Count of True Positives.
        fn (int): Count of False Negatives.
        fp (int): Count of False Positives.
        total_truth (int): Total number of ground truth objects.
        total_pred (int): Total number of prediction objects.

    Returns:
        Dict[str, float]: A dictionary containing 'precision', 'recall', and 'f1'.
    """
    precision = tp / total_pred if total_pred > 0 else 0.0
    recall = tp / total_truth if total_truth > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fnr": fnr
    }

# -------------------------------------------------------------------------
# Reporting and I/O Functions
# -------------------------------------------------------------------------

def print_single_report(title: str, metrics: Dict, counts: Dict):
    """
    Prints a formatted summary of the evaluation metrics to the console.

    Args:
        title (str): The header title for the report (e.g., class name).
        metrics (Dict): Dictionary containing calculated score rates.
        counts (Dict): Dictionary containing raw counts (TP, FN, FP, etc.).
    """
    print(f"\n--- {title} ---")
    print(f"  Objects (Truth): {counts['total_truth']}")
    print(f"  Predictions:     {counts['total_pred']}")
    print(f"  TP: {counts['tp']} | FN: {counts['fn']} (Missed) | FP: {counts['fp']} (Extra)")
    print(f"  Precision: {metrics['precision']:.2%} | Recall: {metrics['recall']:.2%} | F1: {metrics['f1']:.2%}")


def save_results(output_dir: str, gdf_tp: gpd.GeoDataFrame, gdf_fn: gpd.GeoDataFrame, gdf_fp: gpd.GeoDataFrame):
    """
    Saves the classified geometries (TP, FN, FP) to GeoJSON files for visualization.

    Args:
        output_dir (str): Directory where files will be saved.
        gdf_tp (gpd.GeoDataFrame): True Positive geometries.
        gdf_fn (gpd.GeoDataFrame): False Negative geometries.
        gdf_fp (gpd.GeoDataFrame): False Positive geometries.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not gdf_tp.empty: 
        gdf_tp.to_file(os.path.join(output_dir, "true_positives.geojson"), driver="GeoJSON")
    
    if not gdf_fn.empty: 
        gdf_fn.to_file(os.path.join(output_dir, "false_negatives.geojson"), driver="GeoJSON")
    
    if not gdf_fp.empty: 
        gdf_fp.to_file(os.path.join(output_dir, "false_positives.geojson"), driver="GeoJSON")
        
    print(f"  -> Saved analysis files to {output_dir}")


def save_metrics_summary(results_list: List[Dict], output_dir: Optional[str]):
    """
    Generates a pandas DataFrame from the collected results, prints it, and saves it to CSV.

    Args:
        results_list (List[Dict]): List of dictionaries containing metrics for each run/class.
        output_dir (Optional[str]): Directory to save the CSV file.
    """
    if not results_list:
        return

    df = pd.DataFrame(results_list)
    
    # Reorder columns for better readability
    desired_order = ["Label", "Precision", "Recall", "F1", "FNR", "TP", "FN", "FP", "Total Truth", "Total Pred"]
    # Filter only columns that exist (in case keys change)
    cols = [c for c in desired_order if c in df.columns]
    df = df[cols]

    print("\n" + "="*60)
    print("FINAL SCORES SUMMARY")
    print("="*60)
    print(df.to_string(index=False, float_format="{:.2%}".format))
    print("="*60)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "evaluation_scores.csv")
        df.to_csv(csv_path, index=False)
        print(f"Summary table saved to: {csv_path}")


# -------------------------------------------------------------------------
# Main Execution Flow
# -------------------------------------------------------------------------

def run_evaluation(
    truth_path: str, 
    pred_path: str, 
    output_dir: Optional[str] = None, 
    pred_crs: str = "EPSG:32739", 
    iou_threshold: float = 0.5,
    by_class: bool = False,
    class_col: str = "label"
):
    """
    Orchestrates the evaluation pipeline.

    Loads data, aligns projections, computes metrics, and generates a summary table.

    Args:
        truth_path (str): File path to the Ground Truth GeoJSON.
        pred_path (str): File path to the Predictions GeoJSON.
        output_dir (Optional[str]): Directory to save result files. None to skip saving.
        pred_crs (str): CRS to force on predictions if missing (default: EPSG:32739).
        iou_threshold (float): IoU threshold for a positive match.
        by_class (bool): 
            - If False: Evaluates all predictions against all truth objects (binary detection).
            - If True: Splits predictions by `class_col` and evaluates each label against ALL truth objects.
        class_col (str): Column name in the prediction file to use for splitting classes.
    """
    print(f"--- Starting Evaluation (By Predicted Label vs All Truth: {by_class}) ---")
    
    gdf_truth = load_and_fix_geojson(truth_path)
    gdf_pred = load_and_fix_geojson(pred_path)
    
    if gdf_truth is None or gdf_pred is None:
        print("Error: Failed to load one or more input files.")
        return

    # Ensure consistent CRS for spatial operations
    gdf_truth, gdf_pred = align_projections(gdf_truth, gdf_pred, forced_pred_crs=pred_crs)

    # List to collect metrics for the final table
    all_results = []

    if not by_class:
        # Global Evaluation (Class Agnostic)
        gdf_tp, gdf_fn, gdf_fp = match_predictions_to_truth(gdf_truth, gdf_pred, iou_threshold)
        
        counts = {
            "tp": len(gdf_tp), "fn": len(gdf_fn), "fp": len(gdf_fp),
            "total_truth": len(gdf_truth), "total_pred": len(gdf_pred)
        }
        metrics = compute_metrics(**counts)
        
        print_single_report("Overall", metrics, counts)

        if output_dir:
            save_results(output_dir, gdf_tp, gdf_fn, gdf_fp)
        
        # Add to results list
        all_results.append({
                "Label": cls,
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1": metrics["f1"],
                "FNR": metrics["fnr"],
                "TP": counts["tp"],
                "FN": counts["fn"],
                "FP": counts["fp"],
                "Total Truth": counts["total_truth"],
                "Total Pred": counts["total_pred"]
            })

    else:
        # Per-Predicted-Label Evaluation
        print(f"\nSplitting PREDICTIONS by column: '{class_col}'")
        print("Comparing each predicted class against ALL ground truth objects.")
        
        if class_col not in gdf_pred.columns:
            print(f"Error: Column '{class_col}' not found in Prediction file.")
            print(f"Available Pred columns: {list(gdf_pred.columns)}")
            return

        gdf_pred[class_col] = gdf_pred[class_col].astype(str)
        classes = sorted(gdf_pred[class_col].unique())

        for cls in classes:
            sub_pred = gdf_pred[gdf_pred[class_col] == cls]
            sub_truth = gdf_truth.copy()  # Use full ground truth
            
            tp_df, fn_df, fp_df = match_predictions_to_truth(sub_truth, sub_pred, iou_threshold)
            
            counts = {
                "tp": len(tp_df), 
                "fn": len(fn_df), 
                "fp": len(fp_df),
                "total_truth": len(sub_truth),
                "total_pred": len(sub_pred)
            }
            metrics = compute_metrics(**counts)
            
            print_single_report(f"Predicted Label: {cls}", metrics, counts)
            
            if output_dir:
                class_dir = os.path.join(output_dir, f"pred_label_{cls}")
                save_results(class_dir, tp_df, fn_df, fp_df)

            # Add to results list
            all_results.append({
                "Label": cls,
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1": metrics["f1"],
                "FNR": metrics["fnr"],
                "TP": counts["tp"],
                "FN": counts["fn"],
                "FP": counts["fp"],
                "Total Truth": counts["total_truth"],
                "Total Pred": counts["total_pred"]
            })

    # Generate and save the visualization table
    save_metrics_summary(all_results, output_dir)


if __name__ == "__main__":
    # --- USER CONFIGURATION ---
    truth = "/home/thomas/Documents/thomas/HighRes-GeoSpatial-Annotation/labelling/data/ground_truth/south_labels_truth.geojson"
    pred = "/home/thomas/Documents/thomas/HighRes-GeoSpatial-Annotation/labelling/results/north/raw/boats_north.geojson"
    out_dir = "/home/thomas/Documents/thomas/HighRes-GeoSpatial-Annotation/labelling/results/north/evaluation"
    
    CALCULATE_PER_CLASS = True 
    CLASS_COLUMN_NAME = "label" 

    run_evaluation(
        truth, 
        pred, 
        out_dir, 
        pred_crs="EPSG:32739", 
        iou_threshold=0.5,
        by_class=CALCULATE_PER_CLASS,
        class_col=CLASS_COLUMN_NAME
    )