import os
import sys
import click
import yaml
from typing import List
import torch

# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
sys.path.append(os.path.normpath(os.path.join(project_dir, "..")))

from experiments.utils.constants import MODELS_METADATA_PATH, TEMP_MODELS_PATH
import shutil


def setup_model(node_name, model_name, batch_size):
    import torch
    import os

    model_local_path = "./yolov5_torchhub"
    # model_name = "yolov5n"
    torch.hub.set_dir(model_local_path)
    model = torch.hub.load("ultralytics/yolov5", model_name)
    # loc = f"/mnt/myshareddir/torchhub/{model_name}"
    loc = f"{TEMP_MODELS_PATH}/{model_name}"

    dirs = os.listdir(model_local_path)
    for d in dirs:
        if os.path.isdir(f"{model_local_path}/{d}"):
            os.system(f"mkdir {loc}")
            os.system(f"mv {model_local_path}/{d}/* {loc}")
            os.system(f"rm -rf {model_local_path}")

    os.system(f"sudo mv {model_name}.pt {loc}")

    os.system(f"mc mb minio/torchhub/yolo -p")
    os.system(f"mc cp -r {loc}" f" minio/torchhub/yolo")

    os.system(f"rm -r {loc}")


def models_processing(node_name: str, model_names: List[str], batch_size):
    for model_name in model_names:
        model_local_path = setup_model(
            node_name=node_name, model_name=model_name, batch_size=batch_size
        )


@click.command()
@click.option("--pipeline-name", required=True, type=str, default="video")
@click.option("--node-name", required=True, type=str, default="crop")
def main(pipeline_name: str, node_name: str):
    with open(MODELS_METADATA_PATH, "r") as cf:
        models_metadata = yaml.safe_load(cf)

    task_config = models_metadata[pipeline_name][node_name]
    task_type = task_config["task-type"]
    # task_name = task_config['task-name']
    model_names = task_config["model-names"]
    batch_size = 100  # TODO check if this could be a problem

    models_processing(
        node_name=node_name, model_names=model_names, batch_size=batch_size
    )


if __name__ == "__main__":
    main()
