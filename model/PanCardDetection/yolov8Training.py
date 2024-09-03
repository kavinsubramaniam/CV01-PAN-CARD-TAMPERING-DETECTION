from ultralytics import YOLO
import torch

model = YOLO("./yolov8n.pt")

device = torch.device('cuda')
model.to(device)

result = model.train(data='../../data/annotated/ForPanCardDetection/data.yaml', epochs=20, workers=0, amp=False)
success = model.export(format='onnx')