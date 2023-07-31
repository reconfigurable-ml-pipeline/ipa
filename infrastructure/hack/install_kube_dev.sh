#!/bin/bash

# Function to install the specified Go version
install_go() {
    # Define the Go version you want to install
    local GO_VERSION="1.20"

    # Download the Go binary
    wget https://golang.org/dl/go$GO_VERSION.linux-amd64.tar.gz

    # Extract the archive
    sudo tar -C /usr/local -xzf go$GO_VERSION.linux-amd64.tar.gz

    # Set up Go environment variables
    echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.zshrc
    echo 'export GOPATH=$HOME/go' >> ~/.zshrc
    echo 'export PATH=$PATH:$GOPATH/bin' >> ~/.zshrc

    # Load the environment variables for the current session
    source ~/.zshrc

    # Verify the installation
    go version
}

# Call the install_go function
install_go
