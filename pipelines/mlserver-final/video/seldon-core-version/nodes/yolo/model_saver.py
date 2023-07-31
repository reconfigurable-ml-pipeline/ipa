import torch
import os


par_dir = "./yolov5_torchhub"
model_name = "yolov5m"
torch.hub.set_dir(par_dir)
model = torch.hub.load("ultralytics/yolov5", model_name)
loc = f"/mnt/myshareddir/torchhub/{model_name}"

dirs = os.listdir(par_dir)
for d in dirs:
    if os.path.isdir(f"{par_dir}/{d}"):
        os.system(f"sudo mkdir {loc}")
        os.system(f"sudo mv {par_dir}/{d}/* {loc}")
        os.system(f"rm -rf {par_dir}")

os.system(f"sudo mv {model_name}.pt {loc}")
