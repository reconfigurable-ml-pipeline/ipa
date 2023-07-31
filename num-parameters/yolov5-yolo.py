import torch

# Load the YOLOv5 model from Torch Hub
model = torch.hub.load("ultralytics/yolov5", "yolov5s")

## Count the number of parameters
num_params = sum(x.numel() for x in model.parameters())

print(f"Number of parameters in the Yolo5s model: {num_params}")
