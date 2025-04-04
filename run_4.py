from ultralytics import YOLO
import os
from datasets import load_dataset
import torch

# turning down the data aug
augment_args = {
    "hsv_h": 0.01,      # Reduced from 0.015 (subtler color changes)
    "degrees": 15,       # Reduced from 30 (smaller rotations)
    "translate": 0.05,   # Reduced from 0.1 (less shifting)
    "mosaic": 0.5,       # Lower probability (50% chance instead of 100%)
    "mixup": 0.0,        # Disable mixup (often problematic)
    "copy_paste": 0.0    # Disable copy-paste (can create unrealistic artifacts)
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
            project="run_4",
            **augment_args
            )

torch.save(model.model.state_dict(), "run_4.pth")
