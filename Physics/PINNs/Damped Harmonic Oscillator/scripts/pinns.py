import numpy as np
import matplotlib.pyplot as plt
import torch
import sys

import h5py

sys.path.append("../solver/")
from rk4 import load

sys.path.append("../models/")
from fc_model import FCNet, PINNs_FCNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

plot_dir = "../report/Figures/damped_harmonic/pinns/"

### Load data
data_dir = "../data/simulations/damped_harmonic/"

times, positions, velocities, config = load(stored_dir=data_dir + "data_gamma_0.2.h5")

times = torch.tensor(times, dtype=torch.float32).to(device)
positions = torch.tensor(positions, dtype=torch.float32)
velocities = torch.tensor(velocities, dtype=torch.float32)

### Select Training Points
# Boundary Conditions:
time_train_idx = config["Nsteps"]//2  ## defines the max time for training data
X_bc = torch.cat(
    (
        times[0:1],
        times[time_train_idx:time_train_idx+1],
    ),
    dim=0,
).unsqueeze(-1).to(device)
Y_bc = torch.cat(
    (
        positions[0:1],
        positions[time_train_idx:time_train_idx+1],
    ),
    dim=0,
).unsqueeze(-1).to(device)

# From Simulations:
ntrain = 64  ## defines number of points inside training interval used for training
half_points = np.arange(1, time_train_idx -1)
np.random.shuffle(half_points)
train_indices = np.sort(half_points[:ntrain])

X_train = times[train_indices].unsqueeze(-1).to(device)
Y_train = positions[train_indices].unsqueeze(-1).to(device)

times = times.unsqueeze(-1)

# Collocation Points:
def gen_collocation(tmax, Ncol, device):
    return torch.tensor(
        np.random.uniform(
            low = 0,
            high = tmax,
            size = Ncol,
        ),
        dtype = torch.float32,
        requires_grad = True,
        device = device
    ).unsqueeze(-1)
    
### Model
model_config = {
    "Nin": 1,
    "Nout": 1,
    "nlayers": 12,
    "nnodes": 32,
    "activation": torch.nn.Tanh,
    "device": device,
}

model_classical = FCNet(**model_config)
model_pinn = PINNs_FCNet(**model_config)

### Train
train_config = {
    "epochs": 200_000,
    "lr": 1e-3,
}

def mse(output, expected):
    return (output - expected).pow(2).mean()

optimizer_classical = torch.optim.Adam(model_classical.parameters(), lr=train_config["lr"])
optimizer_pinn = torch.optim.Adam(model_pinn.parameters(), lr=train_config["lr"])

sched_classical = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_classical, T_max=train_config["epochs"], eta_min=1e-6)
sched_pinn = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_pinn, T_max=train_config["epochs"], eta_min=1e-6)

# classical NN
loss_classical = {
    "mse": np.zeros(train_config["epochs"], dtype=np.float32)
}
predictions_classical = np.zeros((train_config["epochs"], times.shape[0]), dtype=np.float32)
for epoch in range(train_config["epochs"]):
    optimizer_classical.zero_grad()
    
    psi = model_classical(X_train)
    loss = mse(psi, Y_train)

    loss.backward()
    optimizer_classical.step()

    loss_classical["mse"][epoch] = loss.item()

    sched_classical.step()
        
    predictions_classical[epoch, :] = model_classical(times).detach().cpu().numpy().squeeze()
    
    if epoch % 500 == 0:
        print(f"epoch {epoch:5d} | loss = {loss.item():.3e}")

# pinn
def residual(psi, t, model):
    dpsi_dt, = torch.autograd.grad(
        psi.sum(),
        t,
        retain_graph = True,
        create_graph = True,
    )
    dpsi_dtt, = torch.autograd.grad(
        dpsi_dt.sum(),
        t,
        retain_graph = True,
        create_graph = True,
    )
    return (dpsi_dtt + model.gamma*dpsi_dt + model.kappa*psi).abs().pow(2).mean()

def ntk_weights_update(
    losses,
    model,
    prev_weights = None,
    alpha = 0.5,
    eps = 1e-16,
):
    if prev_weights is None:
        prev_weights = {k: torch.tensor(1.0, device=next(model.parameters()).device)
                        for k in losses.keys()}

    params = [p for p in model.parameters() if p.requires_grad]
    norms = {}

    for key, Li in losses.items():
        grads = torch.autograd.grad(
            Li, params,
            retain_graph=True,
            allow_unused=True,
        )
        flat = []
        for g in grads:
            if g is not None:
                flat.append(g.reshape(-1))
        if len(flat) == 0:
            norms[key] = torch.tensor(0.0, device=Li.device)
        else:
            norms[key] = torch.cat(flat).norm()

    total_norm = sum(norms.values()) + eps

    weights = {}
    for key in losses.keys():
        w_new = norms[key] / total_norm
        weights[key] = alpha * prev_weights.get(key, torch.tensor(1.0, device=w_new.device)) + (1 - alpha) * w_new

    return weights

def compute_losses(
    X_train,
    Y_train,
    X_bc,
    Y_bc,
    X_col,
    model,
    prev_weights = None
):
    psi = model(X_train)
    psi_col = model(X_col)
    psi_bc = model(X_bc)

    losses = {
        "data": mse(psi, Y_train),
        "bc": mse(psi_bc, Y_bc),
        "res": residual(psi_col, X_col, model_pinn),
    }
    
    lambdas = ntk_weights_update(losses,
                                 model_pinn,
                                 prev_weights = prev_weights,
                                 alpha = 0.5,)
    
    return losses, lambdas

step_gen_col = 200
Ncol = 128
loss_pinn = {
    "data": np.zeros(train_config["epochs"], dtype=np.float32),
    "bc": np.zeros(train_config["epochs"], dtype=np.float32),
    "res": np.zeros(train_config["epochs"], dtype=np.float32),
}
params_pinn = {
    "gamma": np.zeros(train_config["epochs"]+1, dtype=np.float32),
    "kappa": np.zeros(train_config["epochs"]+1, dtype=np.float32),
}
lambdas_pinn = {
    "data": np.zeros(train_config["epochs"], dtype=np.float32),
    "bc": np.zeros(train_config["epochs"], dtype=np.float32),
    "res": np.zeros(train_config["epochs"], dtype=np.float32),
}
params_pinn["gamma"][0] = model_pinn.gamma.item()
params_pinn["kappa"][0] = model_pinn.kappa.item()
prev_weights = None
predictions_pinn = np.zeros((train_config["epochs"], times.shape[0]), dtype=np.float32)
for epoch in range(train_config["epochs"]):
    optimizer_pinn.zero_grad()
    
    if epoch%step_gen_col == 0:
        X_col = gen_collocation(config["tmax"], Ncol=Ncol, device=device)
    
    psi = model_pinn(X_train)
    psi_col = model_pinn(X_col)
    
    losses, lambdas = compute_losses(X_train, Y_train, X_bc, Y_bc, X_col, model_pinn, prev_weights)
    
    loss = (lambdas["data"]*losses["data"] +
            lambdas["res"]*losses["res"] +
            lambdas["bc"]*losses["bc"])

    loss.backward()
    optimizer_pinn.step()

    loss_pinn["data"][epoch] = losses["data"].item()
    loss_pinn["bc"][epoch] = losses["bc"].item()
    loss_pinn["res"][epoch] = losses["res"].item()
    
    params_pinn["gamma"][epoch+1] = model_pinn.gamma.item()
    params_pinn["kappa"][epoch+1] = model_pinn.kappa.item()

    lambdas_pinn["data"][epoch] = lambdas["data"].item()
    lambdas_pinn["bc"][epoch] = lambdas["bc"].item()
    lambdas_pinn["res"][epoch] = lambdas["res"].item()

    predictions_pinn[epoch, :] = model_pinn(times).detach().cpu().numpy().squeeze()

    sched_pinn.step()

    if epoch % 500 == 0:
        print(f"epoch {epoch:5d} | loss = {loss.item():.3e} | L_data = {losses['data'].item():.3e} ({lambdas['data']*losses['data'].item():.3e}) | L_bc = {losses['bc'].item():.3e} ({lambdas['bc']*losses['bc'].item():.3e}) | L_res = {losses['res'].item():.3e} ({lambdas['res']*losses['res'].item():.3e}) | gamma = {model_pinn.gamma.item():.3e} | kappa = {model_pinn.kappa.item():.3e}")

output_classical = model_classical(times).detach().cpu().numpy()
output_pinn = model_pinn(times).detach().cpu().numpy()

times = times[:,0].detach().cpu().numpy()

fig, axs = plt.subplots()
axs.plot(times, output_classical[:, 0], label=r"$x_{classical}$")
axs.plot(times, output_pinn[:, 0], label=r"$x_{pinn}$")
axs.plot(times, positions, label=r"$x_{exp}$")
axs.scatter(X_bc.detach().cpu().numpy(), Y_bc.detach().cpu().numpy(), color="k", marker="x", s=50, label="BCs")
axs.scatter(X_train.detach().cpu().numpy(), Y_train.detach().cpu().numpy(), color="r", marker="o", s=20, label="Training Data")
axs.legend()
fig.savefig(plot_dir + "predicted_state.png", dpi=300)

epochs = np.arange(train_config["epochs"])
fig, axs = plt.subplots(2, 1)
axs[0].plot(epochs, loss_classical["mse"], label=r"$L_{total}=L_{mse}$")
axs[1].plot(epochs, loss_pinn["data"], label=r"$L_{mse}$")
axs[1].plot(epochs, loss_pinn["bc"], label=r"$L_{bc}$")
axs[1].plot(epochs, loss_pinn["res"], label=r"$L_{res}$")
axs[1].plot(epochs, loss_pinn["data"]+loss_pinn["res"], label=r"$L_{total} = L_{mse} +L_{res}$")
axs[0].set_yscale("log")
axs[1].set_yscale("log")
axs[0].legend()
axs[1].legend()
fig.savefig(plot_dir + "losses.png", dpi=300)

epochs = np.arange(train_config["epochs"]+1)
fig, axs = plt.subplots(2, 1)
axs[0].plot(epochs, params_pinn["gamma"], label=r"$\gamma$")
axs[0].plot([0,train_config["epochs"]], [config["gamma"], config["gamma"]], "--", color="k", label=r"$\gamma_{exp}$")
axs[1].plot(epochs, params_pinn["kappa"], label=r"$\kappa$")
axs[1].plot([0,train_config["epochs"]], [config["kappa"], config["kappa"]], "--", color="k", label=r"$\kappa_{exp}$")
axs[0].set_ylabel(r"$\gamma$")
axs[1].set_ylabel(r"$\kappa$")
axs[0].set_xlabel("epochs")
axs[1].set_xlabel("epochs")
fig.savefig(plot_dir + "params.png", dpi=300)

fig, axs = plt.subplots(1)
axs.plot(epochs[:-1], lambdas_pinn["data"], label=r"$\lambda_{mse}$")
axs.plot(epochs[:-1], lambdas_pinn["bc"], label=r"$\lambda_{bc}$")
axs.plot(epochs[:-1], lambdas_pinn["res"], label=r"$\lambda_{res}$")
axs.set_ylabel(r"$\lambda$")
axs.set_xlabel("epochs")
axs.legend()
fig.savefig(plot_dir + "lambdas.png", dpi=300)

with h5py.File(data_dir + "pinns_results_gamma_0.2.h5", "w") as f:
    f.create_dataset("times", data=times)
    f.create_dataset("positions", data=positions.cpu().numpy())
    f.create_dataset("velocities", data=velocities.cpu().numpy())
    
    grp_classical = f.create_group("classical_nn")
    grp_classical.create_dataset("predictions", data=predictions_classical)
    grp_classical.create_dataset("loss_mse", data=loss_classical["mse"])
    
    grp_pinn = f.create_group("pinn")
    grp_pinn.create_dataset("predictions", data=predictions_pinn)
    grp_pinn.create_dataset("loss_data", data=loss_pinn["data"])
    grp_pinn.create_dataset("loss_bc", data=loss_pinn["bc"])
    grp_pinn.create_dataset("loss_res", data=loss_pinn["res"])
    grp_pinn.create_dataset("param_gamma", data=params_pinn["gamma"])
    grp_pinn.create_dataset("param_kappa", data=params_pinn["kappa"])
    grp_pinn.create_dataset("lambda_data", data=lambdas_pinn["data"])
    grp_pinn.create_dataset("lambda_bc", data=lambdas_pinn["bc"])
    grp_pinn.create_dataset("lambda_res", data=lambdas_pinn["res"])