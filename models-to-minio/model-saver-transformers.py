import os
import sys
import click
import yaml
from typing import List
from transformers import pipeline

# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
sys.path.append(os.path.normpath(os.path.join(project_dir, "..")))

from experiments.utils.constants import MODELS_METADATA_PATH, TEMP_MODELS_PATH
import shutil


def setup_model(task_name, node_name, model_name, batch_size):
    model = pipeline(task=task_name, model=model_name, batch_size=batch_size)

    model_name = model_name.replace("/", "-")

    model_local_path = os.path.join(TEMP_MODELS_PATH, task_name, model_name)
    # model.save_pretrained(f"./{dirname}")

    model.save_pretrained(model_local_path)

    return model_local_path


def upload_minio(bucket_name: str, model_local_path: str):
    """uploads model files to minio
        and removes them from the disk

    Args:
        bucket_name (str): name of the minio bucket
    """
    # copy generated models to minio
    os.system(f"mc mb minio/{bucket_name} -p")
    os.system(f"mc cp -r {model_local_path}" f" minio/{bucket_name}")
    shutil.rmtree(model_local_path)


def models_processing(
    task_name: str, node_name: str, model_names: List[str], batch_size
):
    for model_name in model_names:
        model_local_path = setup_model(
            task_name=task_name,
            node_name=node_name,
            model_name=model_name,
            batch_size=batch_size,
        )

        task_remote_path = f"huggingface/{task_name}"
        upload_minio(bucket_name=task_remote_path, model_local_path=model_local_path)


@click.command()
@click.option("--pipeline-name", required=True, type=str, default="sum-qa")
@click.option("--node-name", required=True, type=str, default="nlp-sum")
def main(pipeline_name: str, node_name: str):
    with open(MODELS_METADATA_PATH, "r") as cf:
        models_metadata = yaml.safe_load(cf)

    task_config = models_metadata[pipeline_name][node_name]
    task_type = task_config["task-type"]
    task_name = task_config["task-name"]
    model_names = task_config["model-names"]
    batch_size = 100  # TODO check if this could be a problem

    models_processing(
        task_name=task_name,
        node_name=node_name,
        model_names=model_names,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    main()
