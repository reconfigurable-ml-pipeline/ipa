import os
import sys
import yaml
from transformers import pipeline

# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
sys.path.append(os.path.normpath(os.path.join(project_dir, "..")))

from experiments.utils.constants import MODELS_METADATA_PATH, TEMP_MODELS_PATH

with open(MODELS_METADATA_PATH, "r") as cf:
    models_metadata = yaml.safe_load(cf)

pipeline_name = "audio-qa"
node_name = "nlp-qa"

task_config = models_metadata[pipeline_name][node_name]
task_type = task_config["task-type"]
task_name = task_config["task-name"]
model_names = task_config["model-names"]

num_params = {}
if task_type == "huggingface":
    for model_name in model_names:
        # Instantiate the pipeline
        hf_pipeline = pipeline(task=task_name, model=model_name)

        # Get the model
        model = hf_pipeline.model

        # Count the number of parameters
        num_params[model_name] = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

num_params = dict(sorted(num_params.items(), key=lambda x: x[1]))

print(50 * "-" + " results " + 50 * "-")
for model_name, num_params in num_params.items():
    print(f"Number of parameters in model, {model_name} is: {num_params/(10**6)} M")
