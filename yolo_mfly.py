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

model.train(data="/path/to/suas.yaml", epochs=100,
    imgsz=640,
    batch=16)

torch.save(model.model.state_dict(), "model.pth")
