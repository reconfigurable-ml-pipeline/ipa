#!/bin/bash

# Install Google Cloud SDK
function install_gcloud() {
    echo "Installing Google Cloud SDK"
    # wget https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-430.0.0-linux-x86_64.tar.gz -O gcloud.tar.gz
    # tar -xf gcloud.tar.gz
    # bash ./google-cloud-sdk/install.sh -q
    # other distros need different installation
    sudo apt-get install -y apt-transport-https ca-certificates gnupg curl sudo
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
    sudo apt-get update && sudo apt-get install google-cloud-cli
    echo "Google Cloud SDK installation complete"
    echo
}

# Install Helm
function install_helm() {
    echo "Installing Helm"
    wget https://get.helm.sh/helm-v3.11.3-linux-amd64.tar.gz -O helm.tar.gz
    tar -xf helm.tar.gz
    sudo mv linux-amd64/helm /usr/local/bin/helm
    rm helm.tar.gz
    echo "Helm installation complete"
    echo
}

# Install MicroK8s
function install_microk8s() {
    echo "Installing MicroK8s"
    sudo snap install microk8s --classic --channel=1.23/edge
    sudo usermod -a -G microk8s cc
    mkdir -p $HOME/.kube
    sudo chown -f -R cc ~/.kube
    microk8s config > $HOME/.kube/config
    sudo ufw allow in on cni0
    sudo ufw allow out on cni0
    sudo ufw default allow routed
    sudo microk8s enable dns
    echo "alias k='kubectl'" >> ~/.zshrc
    echo "MicroK8s installation complete"
    echo
}

# Main script
echo "Running script"

install_gcloud
install_helm
install_microk8s

echo "Script execution complete"
