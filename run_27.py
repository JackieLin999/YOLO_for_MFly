from ultralytics import YOLO
import os
from datasets import load_dataset
import torch


model = YOLO("yolo11n.pt")
pretrained = True
optimizer = "auto"
lr0 = 0.01
save = True
device=0
warmup_epochs = 3.0
momentum = 0.937
val = True
save = True
plots = True
box = 5.5
dfl = 1.65

# Default augmentation parameters (modify these values)
augment_args = {
    "hsv_h": 0.025,       # Hue adjustment (range: 0-0.1)
    "hsv_s": 0.8,         # Saturation adjustment (range: 0-1)
    "hsv_v": 0.8,         # Value (brightness) adjustment (range: 0-1)
    "fliplr": 0.5,        # Horizontal flip probability (0-1)
    "mosaic": 1.0,        # Mosaic augmentation probability (0-1)
    "translate": 0.1,     # Translation (fraction of image size, 0-0.2)
    "scale": 0.5,         # Scale gain (zoom, 0-1)
    # Disabled by default but available:
    "degrees": 0.0,       # Rotation (degrees, 0-45)
    "shear": 0.0,         # Shear (degrees, 0-10)
    "perspective": 0.0,   # Perspective (fraction, 0-0.001)
    "flipud": 0.0,        # Vertical flip probability (0-1)
    "mixup": 0.0,         # MixUp probability (0-1)
    "copy_paste": 0.0,     # Copy-Paste probability (0-1)
    "augment": False
}

model.train(
            data="suas.yaml",
            epochs=100,
            imgsz=1024,
            batch=16,
            optimizer=optimizer,
            lr0=lr0,
            warmup_epochs=warmup_epochs,
            momentum=momentum,
            val=val,
            save=save,
            plots=plots,
            box=box,
            project="run_27",
            dfl=dfl,
            **augment_args
            )

torch.save(model.model.state_dict(), "run_27.pth")
