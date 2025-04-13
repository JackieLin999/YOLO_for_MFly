from datasets import load_dataset
import os
from PIL import Image

# Load dataset
test_data = load_dataset("mfly-auton/suas-2025-synthetic-data", split="test")

# Create YOLO directory structure
os.makedirs("datasets/test/images", exist_ok=True)
os.makedirs("datasets/test/labels", exist_ok=True)

def process_example(example, idx, split):
    # Save image
    image = example["image"]
    image.save(f"datasets/{split}/images/{split}_{idx}.jpg")
    
    # Prepare YOLO label file
    label_path = f"datasets/{split}/labels/{split}_{idx}.txt"
    
    with open(label_path, "w") as f:
        # Get all bounding boxes and categories
        bboxes = example["objects"]["bbox"]
        categories = example["objects"]["category"]
        
        # Process each bounding box
        for bbox, category_id in zip(bboxes, categories):
            # Convert from [xmin, ymin, width, height] to YOLO format [x_center, y_center, width, height] (normalized)
            x_center = (bbox[0] + bbox[2]/2) / image.width
            y_center = (bbox[1] + bbox[3]/2) / image.height
            width = bbox[2] / image.width
            height = bbox[3] / image.height
            
            # Write to label file
            f.write(f"{category_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

# Process test data
for idx, example in enumerate(test_data):
    process_example(example, idx, "test")


print("Conversion complete!")
print(f"Test samples: {len(test_data)}")