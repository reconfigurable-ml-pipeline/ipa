# Install Gurobi
function install_gurobi() {
    # TODO make sure environment is activated
    echo "Installing Gurobi"
    pip install gurobipy==10.0.1

    # Step 1: Download Gurobi Optimizer
    # download_url="https://packages.gurobi.com/10.0/gurobi10.0.2_linux64.tar.gz"
    # echo "Downloading Gurobi Optimizer..."
    # wget "$download_url"
    
    # Step 2: Extract the downloaded package
    # tar -xvf gurobi10.0.2_linux64.tar.gz
    
    # Step 3: Install Gurobi Optimizer
    # cd gurobi10.0.2/linux64
    # echo "Installing Gurobi Optimizer..."
    # sudo ./install.sh
    
    # Step 4: Set up the environment variables
    # echo "Setting up environment variables..."
    # echo 'export GUROBI_HOME="/opt/gurobi10.0.2/linux64"' >> ~/.bashrc
    # echo 'export PATH="${GUROBI_HOME}/bin:${PATH}"' >> ~/.bashrc
    # echo 'export LD_LIBRARY_PATH="${GUROBI_HOME}/lib:${LD_LIBRARY_PATH}"' >> ~/.bashrc
    # source ~/.bashrc
    
    # Step 5: Activate the academic license
    # read -p "Please enter the path to your Gurobi academic license file: " license_file
    # echo "Activating the academic license..."
    # grbgetkey "$license_file"

    # echo "Gurobi installation complete"
    # echo
}

install_gurobi
