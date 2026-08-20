

# Modeling and Control of Asymptotically Stable Systems Using a Foward Tracking Neural ODE Approach (FTNODE)

This repository contains the code to reproduce the examples from the paper: 
> "Data-driven discovery and control of multistable nonlinear systems and hysteresis via structured Neural ODEs" *arXiv preprint arXiv:2603.27024* (2026).

```bibtex
@misc{salas2026datadriven,
  title         = {Data-driven discovery and control of multistable nonlinear systems and hysteresis via structured Neural ODEs},
  author        = {Salas, Ike Griss and King, Ethan},
  year          = {2026},
  eprint        = {2603.27024},
  archivePrefix = {arXiv},
  primaryClass  = {eess.SY},
  howpublished  = {\url{[https://arxiv.org/abs/2603.27024](https://arxiv.org/abs/2603.27024)}}
}
```

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
uv pip install .

# ...or, with plain pip
pip install .
```

### Reproducing Results
If you are looking to reproduce the results from `examples/`, you may recreate the exact environment. `uv` is the recommended path; `pip` and `conda` are also supported.

#### Option A: uv (recommended)
Use the committed `uv.lock`. This is the fastest path and the only one with a true lockfile: `uv` reads `.python-version` (3.10.5), downloads that interpreter if you don't have it, resolves nothing at install time, and installs `ftnode` in editable mode automatically.

```bash
# 0. Install uv (once)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 1. Clone the repository and navigate into it
git clone https://github.com/pnnl/FTNODE.git
cd FTNODE

# 2. Create the locked environment in ./.venv
#    Drop --group notebooks if you don't need the Jupyter stack for examples/
uv sync --group notebooks

# 3. Run anything inside it without activating
uv run python -c "import ftnode.node; print('ok')"

# ...or activate the venv the usual way
source .venv/bin/activate
```

This produces the same package versions as Option C. Unlike Option C it needs no `CONDA_SUBDIR` workaround: `uv` resolves per-platform, and the lock is restricted to the platforms where `torch==2.10.0` publishes wheels (macOS arm64 and Linux x86_64). macOS Intel is not supported at this pin, and `uv sync` will say so instead of failing deep in the install.

To change a dependency, edit `pyproject.toml` and run `uv lock`, then commit the updated `uv.lock`. The exact versions of the reference environment are held in `[tool.uv] constraint-dependencies`; relax those pins if you want a newer stack.

#### Option B: pip + venv
Use the frozen `requirements.txt` file.

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

#### Option C: conda
Use the pinned `environment.yaml`. This creates a Python 3.10.5 environment named `ftnode` and installs the package in editable mode (`-e .`) automatically.

```bash
# 1. Clone the repository and navigate into it
git clone https://github.com/pnnl/FTNODE.git
cd FTNODE

# 2. Create the environment (run from the repo root so `-e .` resolves to this package)
conda env create -f environment.yaml

# 3. Activate it
conda activate ftnode
```

> **Apple Silicon note:** `torch==2.10.0` publishes only an arm64 macOS wheel. If your conda/anaconda base is an Intel (osx-64) build, force an arm64 env so the torch pin resolves, then persist the subdir for later installs:
> ```bash
> CONDA_SUBDIR=osx-arm64 conda env create -f environment.yaml
> conda env config vars set CONDA_SUBDIR=osx-arm64 -n ftnode
> ```

## Usage

### Train a small model end to end

A κ-bounded latent FT-NODE on the partially observed Duffing oscillator. The model
is `F(z, u) = A(z) (z - g(z, u))`: you pick an operator `A`, pick an equilibrium
map `g`, and wrap them in an encoder/decoder pair, because only `q` is measured and
the latent state has to be inferred from a window of past measurements.

Deliberately tiny so it runs in **about 15 seconds on a laptop CPU**:

```python
import torch

from ftnode.systems import DuffingDataConfig, make_dataset
from ftnode.latent import (BoundedTanhG, ClampOperator, Encoder, KappaBudget,
                           LatentFTNODE, LatentSysID, LinearDecoder)
from ftnode.train import TrainConfig, train_one, restore_best, rollout_y

# 1. Data. Only q is measured; q_dot is never observed -- that is what makes
#    this a latent identification problem.
train = make_dataset(DuffingDataConfig(n_traj=64, L=50,  tau=8, seed=0))
val   = make_dataset(DuffingDataConfig(n_traj=16, L=100, tau=8, seed=1))

# 2. Model. The budget caps cond(A(z)) <= kappa_max by construction, not by
#    penalty, so the bound holds at every point of training.
budget = KappaBudget(sigma_min=0.1, kappa_max=25.0, skew_frac=0.6, m=4)

torch.manual_seed(0)
model = LatentSysID(
    Encoder(tau=8, m=4),                                  # measurements -> latent
    LatentFTNODE(operator=ClampOperator(m=4, budget=budget),       # A(z)
                 equilibrium=BoundedTanhG(m=4)),                      # g(z, u)
    LinearDecoder(m=4),                                   # latent -> measurement
)

# 3. Train. Validation rolls out a longer horizon than training (L_eval > L), so
#    it measures extrapolation rather than fit.
model, hist = train_one(
    model, train, val,
    TrainConfig(n_epochs=10, lr=3e-3, batch=32, lam_res=1e-2, L=50, L_eval=100),
    ckpt_path="best-model.pth",
)
restore_best(model, hist)

# 4. Evaluate: roll out from a held-out measurement window.
with torch.no_grad():
    y_hat, z = rollout_y(model, val.W, val.U, L=100, h=0.05)
print(f"best val MSE {hist['best_val']:.3e} @ epoch {hist['best_epoch']}")
print(f"rollout {tuple(y_hat.shape)}   latent {tuple(z.shape)}")
```

> **This is an API demonstration, not a result.** At 64 trajectories and 10 epochs
> the model has barely started to fit; the κ bound holds regardless, because it is
> structural. The paper settings are `n_traj=512, L=200` / `n_traj=64, L=600` with
> `n_epochs=200`, which takes hours on CPU. Do not quote numbers from the snippet
> above.

### Swap either half of the model

`A` and `g` are independent axes — any operator works with any equilibrium map, so
a different model is just a different pair. No config plumbing involved:

```python
from ftnode.latent import UnboundedOperator, YoulaOperator

# SVD-free kappa bound, same equilibrium map.
youla = LatentSysID(
  Encoder(tau=8, m=4),
  LatentFTNODE(YoulaOperator(m=4, budget=budget), BoundedTanhG(m=4)),
  LinearDecoder(m=4)
)

# No kappa cap at all -- the baseline the bounded operators are measured against.
free = LatentSysID(
  Encoder(tau=8, m=4),
  LatentFTNODE(UnboundedOperator(m=4), BoundedTanhG(m=4)),
  LinearDecoder(m=4)
)
```

Adding a variant on either axis is one module plus one entry in the corresponding
registry — see the constructor contracts in
[`ftnode/latent/operator.py`](ftnode/latent/operator.py) and
[`ftnode/latent/equilibrium.py`](ftnode/latent/equilibrium.py).

### Reproducing frozen results, and config-driven runs

The snippets above build a model directly, which is the right thing when you are
*making* one. To *reproduce* a committed checkpoint or a paper figure, go through a
config and its builder instead:

```python
from ftnode.latent import EncoderConfig, LatentModelConfig, OperatorConfig, build_clamp
from ftnode.utils import save_config, load_config

cfg = LatentModelConfig(
    m=budget.m,
    encoder=EncoderConfig(tau=8),                        # must match the dataset's tau
    operator=OperatorConfig(sigma_min=budget.sigma_min),
)
torch.manual_seed(0)
model = build_clamp(cfg, budget)                         # == build_latent_ftnode(cfg, budget)

save_config(cfg, "model.yaml")                           # nested sections round-trip
cfg = load_config(LatentModelConfig, "model.yaml")
```

Two reasons this path exists rather than being redundant:

- **Construction order is part of the result.** Every submodule draws from the
  global torch RNG as it initializes, so `torch.manual_seed(s)` only reproduces a
  frozen run if the pieces are built in the same order — encoder, then equilibrium
  map, then operator. The builders guarantee that; building by hand gives a
  perfectly good model with different weights for the same seed.
- **Checkpoints are bare state dicts** carrying no architecture metadata, so the
  config is the only record of what produced them. Save it beside the `.pth`.

`build_youla`, `build_unbounded` and `build_latent_node` are the sibling
shortcuts; `build_latent_node` is the unstructured baseline and takes no `budget`.

For a runnable version with figures and every diagnostic, see
[`examples/duffing/pkg_kappa_variants.ipynb`](examples/duffing/pkg_kappa_variants.ipynb).

## Training from the command line

A multi-seed comparison at paper settings takes hours and does not belong in a
notebook kernel. `ftnode-train` runs one from a YAML file, so it can be detached —
and a notebook then loads the finished run and plots it in seconds.

```bash
mkdir -p runs                              # the log redirect below needs it to exist
uv run ftnode-train experiments/duffing/kappa_quick.yaml          # ~5 min

nohup uv run ftnode-train experiments/duffing/kappa_full.yaml \
      > runs/kappa_full.log 2>&1 &         # hours; detach and walk away
tail -f runs/kappa_full.log
```

An experiment file names the shared configs, the variants and the seeds. A variant
is a **pair of registry keys** — one operator, one equilibrium map — so a newly
written `g` is sweepable against every operator without any code change:

```yaml
variants:
  - l-ft-k-svd-clamp                          # a named entry
  - {operator: youla, equilibrium: sym_jac}   # or an explicit pair
seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

Useful flags:

| flag | what it does |
|---|---|
| `--skip-existing` | resume after a kill; re-runs nothing already finished |
| `--only SLUG --seeds 0-4` | train one shard; repeat in another shell to parallelize |
| `--dry-run` | print the resolved job list and exit |
| `--no-rollouts` | skip caching validation rollouts |
| `ftnode-train show runs/NAME` | summarize a finished run |

Sharding is safe to fan out across tmux panes pointed at the same output
directory: each `(variant, seed)` is reseeded immediately before its model is
built, so a shard produces **bitwise the same checkpoints** as the corresponding
slice of one long run.

Then, in a notebook:

```python
from ftnode.experiments import load_run

run = load_run("runs/kappa_full")
models    = run.models(device)      # rebuilt from the config stored in the run
histories = run.histories           # per-epoch train / val / zmax / res
rollouts  = run.rollouts()          # cached validation rollouts, or None
```

A run directory is self-describing — it stores a resolved snapshot of every
config, not a pointer at the experiment file, because the checkpoints are bare
state dicts that cannot be rebuilt without it.

[`examples/duffing/pkg_kappa_analysis.ipynb`](examples/duffing/pkg_kappa_analysis.ipynb)
is a worked example: same figures as the tutorial notebook, no training.

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

