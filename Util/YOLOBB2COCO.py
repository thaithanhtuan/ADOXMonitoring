import os
import json

# Define paths
IMAGE_FOLDER = "AHMDPDatasetFullYOLO/JPEGImages"
LABEL_FOLDER = "AHMDPDatasetFullYOLO/labels"
CLASS_FILE = "AHMDPDatasetFullYOLO/class_names.txt"
OUTPUT_JSON = "AHMDPDatasetFullYOLO/annotations.json"

# Load class names
with open(CLASS_FILE, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# COCO structure
coco_data = {
    "images": [],
    "annotations": [],
    "categories": []
}

# Populate category information
for i, name in enumerate(class_names):
    coco_data["categories"].append({
        "id": i,
        "name": name,
        "supercategory": "object"
    })

# Helper variables
image_id = 0
annotation_id = 0

# Process each image and its label
for filename in os.listdir(IMAGE_FOLDER):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(IMAGE_FOLDER, filename)
        label_path = os.path.join(LABEL_FOLDER, os.path.splitext(filename)[0] + ".txt")

        # Skip if no corresponding label file
        if not os.path.exists(label_path):
            continue

        # Get image size (modify if using OpenCV or PIL)
        # Example: Using PIL
        from PIL import Image
        img = Image.open(img_path)
        width, height = img.size

        # Add image info
        coco_data["images"].append({
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height
        })

        # Read YOLO label file
        with open(label_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center, y_center, w, h = map(float, parts[1:])

                # Convert to COCO format (absolute pixel values)
                x_min = (x_center - w / 2) * width
                y_min = (y_center - h / 2) * height
                bbox_width = w * width
                bbox_height = h * height

                # Add annotation
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": [x_min, y_min, bbox_width, bbox_height],
                    "area": bbox_width * bbox_height,
                    "iscrowd": 0
                })

                annotation_id += 1

        image_id += 1

# Save COCO JSON file
with open(OUTPUT_JSON, "w") as f:
    json.dump(coco_data, f, indent=4)

print(f"COCO annotations saved to {OUTPUT_JSON}")
