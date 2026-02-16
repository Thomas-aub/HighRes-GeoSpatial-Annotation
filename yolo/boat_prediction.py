from sahi import AutoDetectionModel
from functions import yolo_obb_predict

# Load model
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics", # Make sure ultralytics is installed: pip install ultralytics
    model_path="yolo11m-obb.pt", # This model is trained on DOTAv1 dataset (will be automatically downloaded). Use an OBB model
    confidence_threshold=0.2,
    device="cuda:0", # use GPU if available, otherwise use "cpu"
)

# Run prediction
yolo_obb_predict(
    image_file="Test_mada_tom.tif", # input georeferenced raster file (e.g. GeoTIFF)
    labels_file="test_raster_rendered.geojson", # output geojson file with georeferenced bounding boxes
    detection_model=detection_model, 
    tile_size=512, # squared tiles only, size in pixels
    overlap_ratio=0.1, # overlap between tiles, value between 0 and 1
    classes_to_keep=[1], # Keep only class ID 1 (ship class in DOTAv1 dataset)
)

