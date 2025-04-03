from ultralytics import YOLO
import os
from datasets import load_dataset
import torch

augment_args = {
    "hsv_h": 0.015,       # Hue adjustment (0-1)
    "hsv_s": 0.7,         # Saturation adjustment (0-1)
    "hsv_v": 0.4,         # Value (brightness) adjustment (0-1)
    "degrees": 30,        # Rotation (+/- 30 degrees)
    "translate": 0.1,     # Horizontal/vertical translation (10% of image size)
    "scale": 0.5,         # Scale gain (zoom in/out by 50%)
    "shear": 10,          # Shear transformation (+/- 10 degrees)
    "perspective": 0.001, # Perspective distortion (0-0.001)
    "flipud": 0.5,        # Vertical flip probability (50%)
    "fliplr": 0.5,        # Horizontal flip probability (50% - already default)
    "mosaic": 1.0,        # Mosaic augmentation probability (100%)
    "mixup": 0.2,         # MixUp augmentation probability (20%)
    "copy_paste": 0.2     # Copy-Paste augmentation probability (20%)
}

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
            **augment_args
            )

torch.save(model.model.state_dict(), "model.pth")
