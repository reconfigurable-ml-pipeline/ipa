# Infrastructure Setup Manual

Do the steps in the following orders to setup the environment:

* **Server Utility Tools** [zsh](https://www.zsh.org/) and [ohmyzsh](https://ohmyz.sh/)
   1. Installation of [zsh](https://github.com/ohmyzsh/ohmyzsh/wiki/Installing-ZSH)
   2. Installation of [ohmyzsh](https://github.com/ohmyzsh/ohmyzsh)
   3. Installation of [zsh autosuggestion](https://github.com/zsh-users/zsh-autosuggestions/blob/master/INSTALL.md)

* **Python Environment**
  1. Download and install [miniconda](https://docs.conda.io/en/latest/miniconda.html)
  2. Create [conda](https://docs.conda.io/en/latest/miniconda.html) virtual-environment
     ```
      conda create --name central python=3.9.15
     ```
  3. Activate conda environment
     ```
      conda activate central
     ```

  5. Install the followings
     ```
      sudo apt install cmake libz-dev
     ```
  6. For Open-CV to work install:
    ```
      sudo apt-get update
      sudo apt-get install ffmpeg libsm6 libxext6  -y
    ```

* **Infrastracture** [Kubernetes](https://kubernetes.io/)
   1. Install [Helm](https://helm.sh/docs/intro/install/)
   3. Setup a K8S cluster [k8s-setup](manual-installation/setup-chameleon-k8s.md)
   4. Set up multi node cluster [Multi-node cluster](https://microk8s.io/docs/clustering)

* **Network Service Mesh Tool** [Istio](https://istio.io/)
   1. Setup Istio on Chameleon [istio-setup](manual-installation/setup-istio.md)

* **ML Inference DAG Technology** [Seldon Core](https://docs.seldon.io/projects/seldon-core/en/latest/)
   1. Setup the Seldon core operator on your cluster [seldon-core-installation](manual-installation/setup-seldon-core-installation.md)
   2. See [Overview of Component](https://docs.seldon.io/projects/seldon-core/en/latest/workflow/overview.html#metrics-with-prometheus) for an overview of the Seldon core framework
   3. Also see the link to the [shortlisted](manual-installation/guide-seldon.md) parts of the documentation

* **Testing Installation**
   1. Up to this point you should have a complete working installation
   2. To test the installation use [test-installation](manual-installation/test_installation.md)

* **Resources Observibility Tool** [Prometheus](https://prometheus.io/) and [Grafana](https://grafana.com/)
   1. Setup the observibitliy tools for services resource usage monitoring [setup-observibility](manual-installation/setup-prometeus-monitoring.md)
   2. check out the [guide-obervibility](manual-installation/guide-prometheus.md) for usefule information about using the observability tools

* **Docker**
   1. For compiling model images of the pipeline you'll need [Dcoker](https://www.docker.com/)
   2. Install them using the offical documentation for [docker-doc](https://docs.docker.com/engine/install/ubuntu/). Also do the [post-installation-steps](https://docs.docker.com/engine/install/linux-postinstall/)

* **Minio and nfs**
   1. [Minio](https://min.io/) and [nfs](https://en.wikipedia.org/wiki/Network_File_System) are needed for the storage
   2. Setup them using [setup-storage](manual-installation/setup-storage.md)

* **Guide to Deploy a Model and Pipeline**
   1. [Guide-model-deployment](manual-installation/guide-model-deployment.md)

* **MLServer**
   1. We use a modified version of [MLServer](https://mlserver.readthedocs.io/en/stable/) library. Go to the [MLServer](../MLServer/) folder, and install it using `make install-dev`

* **Load Tester**
   1. For load testing we use the [load_tester library saeed branch](https://github.com/reconfigurable-ml-pipeline/load_tester). Go to the [load_tester](../load_tester/) folder and install it using `pip install -e .`

* **Project Data**
Project data are stored in the google bucket [`ipa-results`](https://storage.googleapis.com/ipa-results-1) and [`ipa-models`](https://storage.googleapis.com/ipa-models)
1. Install [Gcloud CLI](https://cloud.google.com/sdk/docs/install) for downloading data.
2. Download the data using TODO

* **Installing Gurobi Sovler (not mandatory for JSys review)**
   1. [Gurobi Solver Installation](manual-installation/gurobi-installation.md)

* **Multi Node server (not mandatory for JSys review)**
    1. How to set multiNode cluster [MultiNode](manual-installation/multi-node.md)


