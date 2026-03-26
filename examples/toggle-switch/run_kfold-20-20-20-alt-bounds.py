import time
from typing import List
import itertools
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold

# Attempt to import tqdm, but define a dummy if missing or to allow disabling
try:
    from tqdm.auto import tqdm
except ImportError:
    # Fallback if tqdm not installed, though we will disable it anyway for logging
    def tqdm(iterable, **kwargs): return iterable

# Custom imports from your environment
# Ensure the 'ftnode' package is in your PYTHONPATH or current directory
try:
    import torchode
    from ftnode.node import (FeluSigmoidMLP, GeluSigmoidMLP, FTNODE)
    from ftnode.utils import set_global_seed
except ImportError as e:
    print(f"Error importing custom modules: {e}")
    print("Please ensure 'ftnode' and 'torchode' are installed and accessible.")
    exit(1)

# --- Configuration ---
plt.rcParams['font.family'] = 'serif'
device = 'cpu' # Change to 'cuda' if available
seed = 1234
random_state = 67
random_state = 3
set_global_seed(seed=seed)

print("[Seed] Deterministic mode enabled.", flush=True)

# --- Physics / ODE Definition ---
def genetic_toggle_switch(state, t, alpha1, alpha2, beta, gamma):
    """
    Computes derivatives for Equation 27 (Li & Lin, 2013).
    """
    x1, x2 = state
    dx1dt = -x1 + alpha1 / (1 + x2**beta)
    dx2dt = -x2 + alpha2 / (1 + x1**gamma)
    return [dx1dt, dx2dt]

# --- Data Generation ---
def generate_data():
    print("Generating simulation data...", flush=True)
    t_max = 100
    n_colloc = 501
    t = np.linspace(0, t_max, n_colloc)

    x1s = np.linspace(0, 6, 9)
    x2s = np.linspace(0, 6, 9)

    alphas1 = np.linspace(0, 5, 5)
    alphas2 = np.linspace(0, 5, 5)
    alphas1[0] = 0.1
    alphas2[0] = 0.1

    betas = np.linspace(0, 5, 5)
    betas[0] = 0.1

    gammas = np.linspace(0, 5, 5)
    gammas[0] = 0.1

    total_iter = len(alphas1) * len(alphas2) * len(betas) * len(gammas)
    
    Us = []
    Xs = []
    
    # Using simple iterator instead of tqdm for log cleanliness
    iterator = itertools.product(alphas1, alphas2, betas, gammas)
    
    for i, args in enumerate(iterator):
        if i % 100 == 0:
            print(f"Simulation progress: {i}/{total_iter}", end='\r')
            
        alpha1, alpha2, beta, gamma = args
        for x0 in itertools.product(x1s, x2s):
            sol = odeint(genetic_toggle_switch, x0, t, args=(alpha1, alpha2, beta, gamma))
            Xs.append(sol)
            Us.append(args)

    print(f"\nSimulation complete. Total samples: {len(Xs)}", flush=True)
    return np.array(Xs), np.array(Us), t

Xs, Us, t = generate_data()

# --- Preprocessing & Derivatives ---
print("Calculating derivatives...", flush=True)
dXs = np.zeros_like(Xs)
T = t[np.newaxis, :, np.newaxis]
X_diff = Xs[:, 2:, :] - Xs[:, :-2, :]
T_diff = T[:, 2:, :] - T[:, :-2, :]

dXs[:, 1:-1, :] = X_diff / T_diff
dXs[:, 0, :] = (Xs[:, 1, :] - Xs[:, 0, :]) / (T[:, 1, :] - T[:, 0, :])
dXs[:, -1, :] = (Xs[:, -1, :] - Xs[:, -2, :]) / (T[:, -1, :] - T[:, -2, :])

# --- Transient Filtering ---
eps_tol = 1e-3
transient_idx = np.argmax(np.all(np.cumsum(np.abs(dXs)[:, ::-1, :] >= eps_tol, axis=1)[:, ::-1, :] == 0, axis=2), axis=1)


# --- Tensor Conversion ---
print("Converting to tensors...", flush=True)
dX_tensor = [torch.tensor(dxi, dtype=torch.float32, device=device) for dxi in dXs]
X_tensor = [torch.tensor(xi, dtype=torch.float32, device=device) for xi in Xs]
U_tensor = [torch.tensor(ui, dtype=torch.float32, device=device) for ui in Us]
T_tensor = [torch.tensor(t, dtype=torch.float32, device=device) for _ in range(len(Xs))]

# --- Dataset Class ---
class GradDataset(torch.utils.data.Dataset):
    def __init__(self, dX: List, X: List, T: List, U: List, Transient_idx: List):
        self.dX = dX
        self.X = X
        self.T = T
        self.U = U
        self.trans_idx = Transient_idx

    def __len__(self):
        return len(self.dX)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"Index {idx} is out of bounds.")
        return self.dX[idx], self.X[idx], self.T[idx], self.U[idx], self.trans_idx[idx]

dataset = GradDataset(dX=dX_tensor, X=X_tensor, T=T_tensor, U=U_tensor, Transient_idx=transient_idx)

# --- Model Definition Helper ---
def get_fresh_model_components():
    f = FeluSigmoidMLP(
        dims=[2, 20, 20, 20, 2],
        lower_bound=-10,
        upper_bound=-0.1
    )
    g = GeluSigmoidMLP(
        dims=[6, 20, 20, 20, 2],
        lower_bound=-0.1,
        upper_bound=10
    )
    model = FTNODE(f, g)
    return f, g, model

# --- K-Fold Cross Validation Settings ---
k_folds = 10
n_epochs = 500
batch_size = 200
learning_rate = 1e-2
print_every = 10
_precision = 6

kfold = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)

val_results = []
train_results = []
avg_best_val_losses = []
avg_best_train_losses = []

print(f"Starting {k_folds}-Fold Cross Validation...", flush=True)

# --- Main Loop ---
for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
    print(f'\n--- FOLD {fold+1}/{k_folds} ---', flush=True)

    fold_seed = random_state + fold
    set_global_seed(seed=fold_seed)

    # 1. Re-initialize Model & Optimizer
    f_fold, g_fold, model_fold = get_fresh_model_components()
    model_fold.to(device)
    model_fold.train()

    loss_criteria = nn.MSELoss(reduction = 'none')

    opt = torch.optim.Adam(
        list(f_fold.parameters()) + list(g_fold.parameters()),
        lr=learning_rate
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=10
    )

    # 2. DataLoaders
    train_subsampler = SubsetRandomSampler(train_ids)
    val_subsampler = SubsetRandomSampler(val_ids)

    trainloader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
    valloader = DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler)

    # 3. Training Loop
    best_val_loss = float('inf')
    best_val_train_loss = float('inf')

    # Disable tqdm for log files, rely on manual prints
    for epoch in range(n_epochs):
        t1 = time.time()

        # --- TRAINING ---
        model_fold.train()
        train_loss = 0.0

        for batch_idx, (dXi, Xi, ti, ui, trans_indices) in enumerate(trainloader):
            # Move batch data to device if needed (dataset is already on device in this script, 
            # but good practice if logic changes)
            
            ui_expanded = ui.unsqueeze(dim=1).expand((-1, len(ti.T), -1))
            # Lambda function for u(t)
            u_func = lambda t: ui_expanded

            opt.zero_grad()

            dXi_pred = model_fold(ti, Xi, u_func)

            batch, n_time, dim = dXi.shape
            # Create mask for transients
            mask = torch.arange(n_time, device=device)[None, :] < trans_indices[:, None]
            mask = mask.unsqueeze(-1)

            loss_per_elem = loss_criteria(dXi, dXi_pred)
            masked_loss = loss_per_elem * mask
            loss = masked_loss.sum() / mask.sum().clamp(min=1)

            loss.backward()
            opt.step()

            train_loss += loss.item()

        train_loss /= len(trainloader)

        # --- VALIDATION ---
        model_fold.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (dXi, Xi, ti, ui, trans_indices) in enumerate(valloader):
                ui_expanded = ui.unsqueeze(dim=1).expand((-1, len(ti.T), -1))
                u_func = lambda t: ui_expanded

                dXi_pred = model_fold(ti, Xi, u_func)

                batch, n_time, dim = dXi.shape
                mask = torch.arange(n_time, device=device)[None, :] < trans_indices[:, None]
                mask = mask.unsqueeze(-1)

                loss_per_elem = loss_criteria(dXi, dXi_pred)
                masked_loss = loss_per_elem * mask
                loss = masked_loss.sum() / mask.sum().clamp(min=1)

                val_loss += loss.item()

        val_loss /= len(valloader)

        # --- SCHEDULER & LOGGING ---
        epoch_time = time.time() - t1
        scheduler.step(val_loss)
        cur_lr = opt.param_groups[0]['lr']

        if epoch <= 5 or epoch % print_every == 0 or epoch == n_epochs - 1:
            print(
                f"Epoch {epoch}: "
                f"Train Loss = {train_loss:.{_precision}e}, "
                f"Val Loss = {val_loss:.{_precision}e}, "
                f"Time = {epoch_time:.{_precision}e}, "
                f"lr = {cur_lr:.{_precision}e}",
                flush=True
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_train_loss = train_loss

    print(f"Fold {fold+1} Best Val Loss: {best_val_loss:.{_precision}e}", flush=True)
    val_results.append(best_val_loss)
    train_results.append(best_val_train_loss)

# --- SUMMARY ---
print("\nK-Fold Cross Validation Results:", flush=True)
avg_loss = np.mean(val_results)
avg_train_loss = np.mean(train_results)
print(f"Average Best Validation Loss: {avg_loss:.{_precision}e}", flush=True)
print(f"Average Best Train Loss: {avg_train_loss:.{_precision}e}", flush=True)