from ultralytics import YOLO
import os
from datasets import load_dataset
import torch


model = YOLO("yolo11n.pt")
pretrained = True
optimizer = "AdamW"
lr0 = 0.01
save = True
device=0
warmup_epochs = 3.0
momentum = 0.937
val = True
save = True
plots = True
box = 5.0

# Default augmentation parameters (modify these values)
augment_args = {
    "hsv_h": 0.015,  # Revert to default for stability
    "hsv_s": 0.7,     # Original default
    "hsv_v": 0.4,     # Original default
    "mosaic": 0.75,   # Reduce from 1.0 to 75% probability
    "degrees": 5.0,    # Add mild rotation (5°)
    "scale": 0.5,         # Scale gain (zoom, 0-1)
}

model.train(
            data="suas.yaml",
            epochs=100,
            imgsz=640,
            batch=16,
            optimizer=optimizer,
            lr0=lr0,
            warmup_epochs=warmup_epochs,
            momentum=momentum,
            val=val,
            save=save,
            plots=plots,
            box=box,
            project="run_18",
            **augment_args
            )

torch.save(model.model.state_dict(), "run_18.pth")
