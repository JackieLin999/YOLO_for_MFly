import csv
from ultralytics import YOLO

# Load the trained YOLO model (best.pth)
model = YOLO("best.pth")

# Evaluate the model on the test dataset
results = model.val(data="suas.yaml", save=True)

# Open a CSV file to write the evaluation results
csv_file = "test_results.csv"
with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    # Write header
    writer.writerow(["image_name", "bbox_x_center", "bbox_y_center", "bbox_width", "bbox_height", "category", "confidence"])

    # Iterate over the validation results (predictions and ground truths)
    for image_result in results.pred:
        image_name = image_result["filename"]
        predictions = image_result["pred"]  # Predictions (boxes and categories)
        
        for pred in predictions:
            # Extract bounding box, category, and confidence
            x_center, y_center, width, height = pred[:4]
            category = int(pred[4])  # Category ID
            confidence = pred[5]  # Confidence score

            # Write to CSV file
            writer.writerow([image_name, x_center, y_center, width, height, category, confidence])

print(f"Results saved to {csv_file}")