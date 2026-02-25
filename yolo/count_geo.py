import os
import json
from collections import defaultdict

def summarize_class_ids(folder_path):
    # Dictionary to store total counts
    total_counts = defaultdict(int)

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".geojson"):
            file_path = os.path.join(folder_path, filename)
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Ensure it has features
                features = data.get("features", [])
                
                for feature in features:
                    properties = feature.get("properties", {})
                    class_id = properties.get("class_id")
                    
                    if class_id is not None:
                        total_counts[class_id] += 1

            except Exception as e:
                print(f"Error reading {filename}: {e}")

    return dict(total_counts)


if __name__ == "__main__":
    folder = "data/raw"  # Change if needed
    summary = summarize_class_ids(folder)

    print("=== Class ID Summary Across All GeoJSON Files ===")
    for class_id, count in sorted(summary.items()):
        print(f"class_id {class_id}: {count}")