#!/bin/bash

install_zsh() {
    # Install Zsh
    sudo apt update
    sudo apt install -y zsh

    # Install Oh My Zsh
    echo "Y" | sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

    # Install Autosuggestions plugin for Oh My Zsh
    git clone https://github.com/zsh-users/zsh-autosuggestions ~/.oh-my-zsh/custom/plugins/zsh-autosuggestions

    # Enable Autosuggestions plugin
    sed -i 's/plugins=(git)/plugins=(git zsh-autosuggestions)/' ~/.zshrc

    # Set Zsh as the default shell
    echo "Y" | chsh -s "$(which zsh)"

    # Restart the shell
    exec zsh
}

# Call the function
install_zsh
