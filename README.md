

# Modeling and Control of Asymptotically Stable Systems Using a Foward Tracking Neural ODE Approach (FTNODE)

This repository contains the code to reproduce the examples from the paper: Modeling and feedback control of asymptotically stable nonlinear
dynamical systems 

-add reference--

## Overview

This work presents a methodology for learning dynamical systems with enforced asymptotic stability using Neural ODEs, enabling reliable short-horizon learning and accurate, noise-robust feedback control across systems exhibiting nontrivial bifurcations.

## Methodology

### Model Architecture

The approach uses a tunable model of the form:

F_θ(x, u) = f_θ(x)(x - g_θ(x, u))

Where:
- `f_θ` and `g_θ` are multi-layer perceptrons (MLPs) with Sigmoid-weighted Linear Unit (SiLU) activation functions and sigmoidal output layers
- Asymptotic stability is enforced by requiring that f_θ(x)<0 everywhere on the domain. and g_θ remains bounded under bounded input

### Training Objectives

Two training objectives are implemented:

1. **Trajectory Matching**: Standard NODE training using backpropagation to compare simulated trajectories against observed trajectories
2. **Gradient Matching**: Compares the learned model against gradient information computed using centered finite differences


## Examples

The repository includes implementations for the following systems:

- **Mixing Tanks**
- **Symmetric Hysteresis**
- **Budworm Population**
- **Genetic Toggle Switch**

## Requirements

- PyTorch
- torchode (Parallel ODE Solver for PyTorch)

## Installation & Reproducibility

This repository supports both general use as a library and exact reproducibility for the results presented in the associated paper.

### General Use 
If you want to use the Forward Tracking Neural ODE Approach (`ftnode`) in your own projects alongside your existing packages, you can install it directly. 

```bash
# Clone the repository
git clone https://github.com/pnnl/FTNODE.git
cd FTNODE

# Install the package
pip install .
```

### Reproducing Results
If you are looking to reproduce the results from `examples/`, you may use the frozen `requirements.txt` file. 

```bash
# 1. Clone the repository and navigate into it
git clone https://github.com/pnnl/FTNODE.git
cd FTNODE

# 2. Create and activate a fresh virtual environment
python -m venv ftnode_env

# On Windows:
# ftnode_env\Scripts\activate
# On macOS/Linux:
source ftnode_env/bin/activate

# 3. Install the exact dependency tree
pip install -r requirements.txt

# 4. Install the package locally in editable mode
pip install -e .
```


## Usage

Each example contains specific hyperparameters including:
- Number of training epochs
- Batch size
- Initial learning rate
- Model architecture specifications (layer widths, bounds)

Refer to individual example scripts for detailed configuration.

## Citation

If you use this code, please cite the associated paper.

## License

Copyright Battelle Memorial Institute 2026
 
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
 
1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
 
2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
 
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Disclaimer

DISCLAIMER
This material was prepared as an account of work sponsored by an agency of the
United States Government.  Neither the United States Government nor the United
States Department of Energy, nor Battelle, nor any of their employees, nor any
jurisdiction or organization that has cooperated in the development of these
materials, makes any warranty, express or implied, or assumes any legal
liability or responsibility for the accuracy, completeness, or usefulness or
any information, apparatus, product, software, or process disclosed, or
represents that its use would not infringe privately owned rights.
 
Reference herein to any specific commercial product, process, or service by
trade name, trademark, manufacturer, or otherwise does not necessarily
constitute or imply its endorsement, recommendation, or favoring by the United
States Government or any agency thereof, or Battelle Memorial Institute. The
views and opinions of authors expressed herein do not necessarily state or
reflect those of the United States Government or any agency thereof.
 
                PACIFIC NORTHWEST NATIONAL LABORATORY
                             operated by
                               BATTELLE
                               for the
                  UNITED STATES DEPARTMENT OF ENERGY
                   under Contract DE-AC05-76RL01830

