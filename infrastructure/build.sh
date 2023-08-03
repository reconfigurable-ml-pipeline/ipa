#!/bin/bash

# TODO some y prompt

install_packages() {
    if [ -z "$1" ];
    then
        echo "You must provide public IP: ./build.sh PUBLIC_IP"
        exit 0; 
    fi

    hack_dir="$HOME/ipa/infrastructure"
    zsh_script="${hack_dir}/hack/zsh.sh"
    repos_script="${hack_dir}/hack/repos.sh"
    kubernetes_script="${hack_dir}/hack/kubernetes.sh"
    utilities_script="${hack_dir}/hack/utilities.sh"
    storage_script="${hack_dir}/hack/storage.sh"
    gurobi_script="${hack_dir}/hack/gurobi.sh"
    download_data="${hack_dir}/hack/download_data.sh"
    run_jupyters="${hack_dir}/run_jupyters.sh"

    source "$repos_script"
    echo "repos.sh completed"
    bash "$kubernetes_script"
    echo "kubernetes.sh completed"
    bash "$utilities_script"
    echo "utilities.sh completed"
    bash "$storage_script" "$1"
    echo "storage.sh completed"
    bash install_kube_dev
    echo "install_kube_dev.sh completed"
    bash "$gurobi_script"
    echo "gurobi.sh completed"
    bash "$download_data"
    echo "download_data.sh completed"
    bash "$run_jupyters"
    echo "run_jupyters.sh completed"


    echo "Installation of all packages and dependencies completed"
}

# Call the function with the public IP as argument
install_packages "$1"
