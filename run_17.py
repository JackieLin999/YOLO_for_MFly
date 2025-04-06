from ultralytics import YOLO
import torch

model = YOLO("yolo11n.pt")

# Augmentation parameters
augment_args = {
    # Color adjustments
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    
    # Spatial transforms
    "fliplr": 0.5,
    "mosaic": 0.75,
    "translate": 0.1,
    "scale": 0.5,
    "degrees": 5.0,
    
    # Disabled transforms
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "mixup": 0.0,
    "copy_paste": 0.0
}

model.train(
    data="suas.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    optimizer="AdamW",
    lr0=0.01,
    dfl=2.0,  # Critical DFL adjustment
    box=7.5,
    warmup_epochs=3.0,
    momentum=0.937,
    val=True,
    save=True,
    plots=True,
    project="run_17",
    **augment_args
)

torch.save(model.model.state_dict(), "run_17.pth")