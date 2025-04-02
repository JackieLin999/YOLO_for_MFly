from ultralytics import YOLO
import os
from datasets import load_dataset
import torch
'''

img = "images/test.jpg"
result = model(img)
print(result)
result[0].save(filename='result.jpg')
'''
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

model.train(
            data="/mnt/c/Users/jacki/Documents/mfly_yolo/suas.yaml",
            epochs=100,
            imgsz=640,
            batch=16,
            optimizer=optimizer,
            lr0=lr0,
            warmup_epochs=warmup_epochs,
            momentum=momentum,
            val=val,
            save=save,
            plots=plots
            )

torch.save(model.model.state_dict(), "model.pth")
