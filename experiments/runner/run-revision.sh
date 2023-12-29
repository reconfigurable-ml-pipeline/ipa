#!/bin/bash

conda activate central

python runner_script.py --config-name video-mul-1
sleep 60
python runner_script.py --config-name video-mul-2
sleep 60
python runner_script.py --config-name video-mul-3
sleep 60
python runner_script.py --config-name video-mul-5
sleep 60
python runner_script.py --config-name video-mul-6
sleep 60
python runner_script.py --config-name video-mul-7
sleep 60
python runner_script.py --config-name video-mul-8
sleep 60
python runner_script.py --config-name video-mul-10
sleep 60
python runner_script.py --config-name video-mul-11
sleep 60
python runner_script.py --config-name video-mul-12
sleep 60
python runner_script.py --config-name video-mul-13
sleep 60
python runner_script.py --config-name video-mul-15
sleep 60
python runner_script.py --config-name video-mul-16
sleep 60
python runner_script.py --config-name video-mul-17
sleep 60
python runner_script.py --config-name video-mul-18
sleep 60
python runner_script.py --config-name video-mul-20
sleep 60

# Draw the results of the experiment
jupyter nbconvert --execute --to notebook --inplace ~/ipa/experiments/runner/notebooks/Jsys-reviewers-revision.ipynb
