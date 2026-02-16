import yaml
import os
from data_preparation.tiling import tile_raster
from modeling.model import run_batch_inference
from postprocessing.refinement import refine_predictions

def main():
    """
    Main function to run the Geo-Spatial Annotation Pipeline.
    """
    # Load configuration
    with open('configs/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Create directories if they don't exist
    os.makedirs(config['data']['processed_dir'], exist_ok=True)
    os.makedirs(config['data']['results_dir'], exist_ok=True)

    # --- 1. Tiling Step ---
    # Here you would typically list your raw images and process them.
    # For this example, let's assume one raw image.
    # In a real scenario, you might loop through files in config['data']['raw_dir']
    
    raw_image_path = os.path.join(config['data']['raw_dir'], 'your_image.tif') # IMPORTANT: Change 'your_image.tif'
    tiled_output_dir = os.path.join(config['data']['processed_dir'], 'tiles')
    
    print("--- Starting Tiling Step ---")
    # tile_raster(
    #     raster_in_path=raw_image_path,
    #     raster_out_dir=tiled_output_dir,
    #     tile_size=config['tiling']['tile_size'],
    #     skip_empty_tiles=config['tiling']['skip_empty_tiles']
    # )
    print("Tiling step skipped (commented out). Please provide a raw image.")


    # --- 2. Inference Step ---
    inference_output_path = os.path.join(config['data']['results_dir'], 'detections.geojson')
    
    print("\n--- Starting Inference Step ---")
    run_batch_inference(
        dir_tiles=tiled_output_dir,
        output_geojson=inference_output_path,
        prompts=config['model']['prompts'],
        model_path=config['model']['path']
    )

    # --- 3. Post-processing Step ---
    refined_output_path = os.path.join(config['data']['results_dir'], 'detections_refined.geojson')
    print("\n--- Starting Post-processing Step ---")
    refine_predictions(
        input_geojson=inference_output_path,
        output_geojson=refined_output_path
    )

    print("\n--- Pipeline Finished ---")


if __name__ == '__main__':
    main()
