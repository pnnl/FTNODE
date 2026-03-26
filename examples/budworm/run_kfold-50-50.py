import time
from typing import List
import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp as sp_solve_ivp
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler
from sympy import symbols, Eq, solve, simplify

# Attempt to import tqdm, but define a dummy if missing or to allow disabling
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): return iterable

try:
    from ftnode.utils import set_global_seed
    from ftnode.node import (
        FTNODE, FeluSigmoidMLP,
        GeluSigmoidMLPfeaturized
    )
except ImportError as e:
    print(f"Error importing custom modules: {e}")
    print("Please ensure 'ftnode' and 'torchode' are installed and accessible.")
    exit(1)


device = 'cpu'
seed = 1234
random_state = 67
set_global_seed(seed=seed, deterministic=True)

print("[Seed] Deterministic mode enabled.", flush=True)


def budworm_ode(t, x, r, k):
    return r * x * (1 - x / k) - x**2 / (1 + x**2)

# --- Parameter Calculation (SymPy) ---
print("Calculating simulation parameters...", flush=True)
r_sym, k_sym = symbols('r k', positive=True)
a = r_sym / k_sym
b = -r_sym
c = (k_sym + r_sym) / k_sym
d = -r_sym
p = (3 * a * c - b**2) / (3 * a**2)
q = (2 * b**3 - 9 * a * b * c + 27 * a**2 * d) / (27 * a**3)

D = -(4 * p**3 + 27 * q**2)
D = simplify(D)

D_fixed = D.subs(r_sym, 0.56)
r1_sol, r2_sol = solve(Eq(D_fixed, 0), k_sym)
print(f"Bifurcation points: {r1_sol}, {r2_sol}", flush=True)

# --- Data Generation ---
print("Generating simulation data...", flush=True)
n_control = 51
n_traj = 51
r_val = 0.56

# Ensure us values are floats
us = np.linspace(float(r1_sol) - 2, float(r2_sol) + 2, n_control)
x0s = np.linspace(0.1, 10, n_traj)

t_max = 10
n_colloc = 801
t = np.linspace(0, t_max, n_colloc)

Xs = []
Us = []

# Generate data
total_sims = len(us) * len(x0s)
print(f"Total simulations to run: {total_sims}", flush=True)

for i, ui in enumerate(us):
    if i % 5 == 0: # Print progress occasionally
        print(f"Simulating control param {i+1}/{len(us)}...", end='\r')
        
    for x0 in x0s:
        sol = sp_solve_ivp(
            budworm_ode,
            t_span=[0, t_max],
            y0=np.array(x0).reshape(-1),
            t_eval=t,
            args=(r_val, ui,)
        )
        Xs.append(sol.y.T)
        Us.append([ui])

print(f"\nSimulation complete. Total samples: {len(Xs)}", flush=True)

Xs = np.array(Xs)
Us = np.array(Us)

# --- Preprocessing ---
print("Scaling data...", flush=True)
scaler = MinMaxScaler(feature_range=(0, 1))
Xs_scaled = scaler.fit_transform(Xs.reshape(-1, 1)).reshape(-1, n_colloc, 1)

print("Calculating derivatives...", flush=True)
dXs = np.zeros_like(Xs)
T = t[np.newaxis, :, np.newaxis]
X_diff = Xs_scaled[:, 2:, :] - Xs_scaled[:, :-2, :]
T_diff = T[:, 2:, :] - T[:, :-2, :]

dXs[:, 1:-1, :] = X_diff / T_diff
dXs[:, 0, :] = (Xs_scaled[:, 1, :] - Xs_scaled[:, 0, :]) / (T[:, 1, :] - T[:, 0, :])
dXs[:, -1, :] = (Xs_scaled[:, -1, :] - Xs_scaled[:, -2, :]) / (T[:, -1, :] - T[:, -2, :])

# --- Tensor Conversion ---
print("Converting to tensors...", flush=True)
dX_tensor = [torch.tensor(dxi, dtype=torch.float32, device=device) for dxi in dXs]
X_tensor = [torch.tensor(xi, dtype=torch.float32, device=device) for xi in Xs_scaled]
U_tensor = [torch.tensor(ui, dtype=torch.float32, device=device) for ui in Us]
T_tensor = [torch.tensor(t, dtype=torch.float32, device=device) for _ in range(len(Xs))]

# --- Dataset Class ---
class GradDataset(torch.utils.data.Dataset):
    def __init__(self, dX: List, X: List, T: List, U: List):
        self.dX = dX
        self.X = X
        self.T = T
        self.U = U

    def __len__(self):
        return len(self.dX)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"Index {idx} is out of bounds.")
        
        dXi = self.dX[idx]
        Xi = self.X[idx]
        ti = self.T[idx]
        ui = self.U[idx]
        return dXi, Xi, ti, ui

dataset = GradDataset(dX=dX_tensor, X=X_tensor, T=T_tensor, U=U_tensor)

# --- Model Definition Helper ---
def get_fresh_model_components():
    f = FeluSigmoidMLP(
        dims=[1, 50,50, 1],
        activation=nn.SiLU(),
        lower_bound=-1,
        upper_bound=-0.1,
    )

    g = GeluSigmoidMLPfeaturized(
        dims=[6, 50,50, 1],
        activation=nn.SiLU(),
        lower_bound=-5,
        upper_bound=2,
        freq_sample_step=1,
        feat_lower_bound=-1,
        feat_upper_bound=1.5,
    )

    model = FTNODE(f,g)
    return f, g, model 

# --- K-Fold Cross Validation Settings ---
k_folds = 10
n_epochs = 200
batch_size = 50
learning_rate = 1e-1
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

    loss_criteria = nn.MSELoss()

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

    # Using range instead of tqdm for clean logs
    for epoch in range(n_epochs):
        t1 = time.time()

        # --- TRAINING ---
        model_fold.train()
        train_loss = 0.0

        for batch_idx, (dXi, Xi, ti, ui) in enumerate(trainloader):
            ui_expanded = ui.unsqueeze(dim=1).expand(Xi.shape)
            u_func = lambda t: ui_expanded

            opt.zero_grad()
            
            dXi_pred = model_fold(ti, Xi, u_func)
            loss = loss_criteria(dXi, dXi_pred)
            
            loss.backward()
            opt.step()
            
            train_loss += loss.item()

        train_loss /= len(trainloader)

        # --- VALIDATION ---
        model_fold.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (dXi, Xi, ti, ui) in enumerate(valloader):
                ui_expanded = ui.unsqueeze(dim=1).expand(Xi.shape)
                u_func = lambda t: ui_expanded

                dXi_pred = model_fold(ti, Xi, u_func)
                loss = loss_criteria(dXi, dXi_pred)
                
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