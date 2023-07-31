# Gurobi installation
[Gurobi solver](https://www.gurobi.com/) is an optimization solver that we used in our work for solving optimizatin problems.

## Steps to installation
1. Download the linux version of Gurobi from [Gurobi Solver](https://www.gurobi.com/downloads/gurobi-software/) on the server
2. unzip the downloaded file `tar -xvf gurobi10.0.1_linux64.tar.gz`
3. `cd gurobi1001/linux64` into the extracted folder
4. Install Gurobi using `sudo python setup.py install`
5. Check installation:
```bash
(base) ➜  linux64 gurobi_cl --version                                              
Gurobi Optimizer version 10.0.0 build v10.0.0rc2 (linux64)
Copyright (c) 2022, Gurobi Optimization, LLC
```
6. To activate and use academic license go the [academic-named-user-license](https://portal.gurobi.com/iam/licenses/request/?type=academic) and follow the instructions for a free account
7. Login into [Gurobi User Portal](https://portal.gurobi.com/) and generate a Named-User Academic License.
8. Use the license to activate the Gurobi solver on the command line
```
grbgetkey xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```
9. Sometimes the steps 3-5 does not work, alternively use the steps [mentioned here](https://www.gurobi.com/documentation/9.5/quickstart_linux/software_installation_guid.html#section:Installation)
```
export GUROBI_HOME="/opt/gurobi1002/linux64"
export PATH="${PATH}:${GUROBI_HOME}/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
```
11. Done!
