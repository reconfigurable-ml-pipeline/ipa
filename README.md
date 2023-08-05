# Project Setup Steps
1. Go to the [infrastructure](/infrastructure/README.md) for the guide to set up the K8S cluster and related depandancies, the complete installtion takes ~30 minutes.

2. After downloading ipa data explained in 1 the log of the experiments presented in the paper will be avialable in the directory [data/results/final](data/results/final) to draw the figures in the paper go to [experiments/runner/notebooks](experiments/runner/notebooks) to draw each figure presented in the paper. Each figure is organized in a different Jupyter notebook e.g. to draw the figure 8 of the paper pipeline figure [experiments/runner/notebooks/paper-fig8-e2e-video.ipynb](experiments/runner/notebooks/paper-fig8-e2e-video.ipynb)

3. If you want to check main paper e2e experiments (figure 8-12) do the following steps:
    1. IPA use config yaml files for running experiments, the config files used in the paper are stored in the `data/configs/final` folder.
    2. To Run a specific experiment move to the [experiments/runner](experiments/runner) directory and run `python runner_script.py --config-name <name of one of the config files in data/configs/final>` or use step three to run and automated script that generate the entire figure 8 of the paper for the video pipeline
    3. Go to the `experiments/runner` and run `source run.sh`, this will take ~7 hours since each of the 20 experiments is conducted on a 20 minute load (20 * 20 = 400 minutes ~ 7 hours). The results and logs will be saved under `ipa/data/results/final/18`.
    4. Go to the `experiments/runner/notebooks/Jsys-reviewers.ipynb` notebook to generate all the figures same as the `paper-fig8-e2e-video.ipynb` that was generated from the downloaded log.


## Experiment console
A typical log of an IPA run session:

![experiment](https://github.com/reconfigurable-ml-pipeline/ipa/assets/6298780/b7511930-dbf0-4dca-b1c2-c1a064232416)



## Kubernetes pod autoscaling
Pods being added/deleted by IPA autoconfiguration module:

![log](https://github.com/reconfigurable-ml-pipeline/ipa/assets/6298780/b43ea8d5-68d9-44b6-b452-c9486878c57e)
