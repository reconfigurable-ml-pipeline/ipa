#!/bin/bash

experiment_number=$1

conda activate central

# Running Python script with the extracted experiment number
python runner_script.py --config-name "video-mul-$experiment_number"
sleep 60

# Drawing the results of the experiment
jupyter nbconvert --execute --to notebook --inplace ~/ipa/experiments/runner/notebooks/Jsys-reviewers-revision.ipynb
