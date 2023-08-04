#!/bin/bash

execute_notebooks() {
  # List of notebook paths
  notebooks=(
    ~/ipa/experiments/runner/notebooks/paper-fig2-movivation-latency-accuracy-throughput.ipynb
    ~/ipa/experiments/runner/notebooks/paper-fig7-patterns.ipynb
    ~/ipa/experiments/runner/notebooks/paper-fig8-e2e-video.ipynb
    ~/ipa/experiments/runner/notebooks/paper-fig9-e2e-audio-qa.ipynb
    ~/ipa/experiments/runner/notebooks/paper-fig10-e2e-audio-sent.ipynb
    ~/ipa/experiments/runner/notebooks/paper-fig11-e2e-sum-qa.ipynb
    ~/ipa/experiments/runner/notebooks/paper-fig12-e2e-nlp.ipynb
    ~/ipa/experiments/runner/notebooks/paper-fig13-gurobi-decision-latency.ipynb
    ~/ipa/experiments/runner/notebooks/paper-fig14-objecitve-preference.ipynb
    ~/ipa/experiments/runner/notebooks/paper-fig15-comparison-cdf.ipynb
    ~/ipa/experiments/runner/notebooks/paper-fig16-predictor-abelation.ipynb
  )

  # Loop through the notebooks and execute them
  for notebook in "${notebooks[@]}"; do
    jupyter nbconvert --execute --to notebook --inplace "$notebook"
  done
}

# Call the function to execute the notebooks
execute_notebooks
