import os
import sys
import click
import yaml
from typing import List
import torch
import torchvision.models as models


# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
sys.path.append(os.path.normpath(os.path.join(project_dir, "..")))

from experiments.utils.constants import MODELS_METADATA_PATH, TEMP_MODELS_PATH
import shutil


def setup_model(node_name, model_name, batch_size):
    import torch
    import os

    model_variants = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "resnet152": models.resnet152,
    }
    # TODO cpu and gpu from env variable
    model = model_variants[model_name](pretrained=True)

    model_dir = f"{TEMP_MODELS_PATH}/model.pt"
    torch.save(model.state_dict(), model_dir)

    os.system(f"mc mb minio/torchhub/resnet/{model_name} -p")
    os.system(f"mc cp -r {model_dir}" f" minio/torchhub/resnet/{model_name}")

    os.system(f"rm -r {model_dir}")


def models_processing(node_name: str, model_names: List[str], batch_size):
    for model_name in model_names:
        model_local_path = setup_model(
            node_name=node_name, model_name=model_name, batch_size=batch_size
        )


@click.command()
@click.option("--pipeline-name", required=True, type=str, default="video")
@click.option("--node-name", required=True, type=str, default="classification")
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
