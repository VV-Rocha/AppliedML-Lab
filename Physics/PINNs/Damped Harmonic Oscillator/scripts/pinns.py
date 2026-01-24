import numpy as np
import matplotlib.pyplot as plt
import torch
import sys

sys.path.append("../solver/")
from rk4 import load

sys.path.append("../models/")
from fc_model import FCNet, PINNs_FCNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

plot_dir = "../report/Figures/damped_harmonic/pinns/"

### Load data
data_dir = "../data/simulations/damped_harmonic/data_gamma_0.2.h5"

times, positions, velocities, config = load(stored_dir=data_dir)

times = torch.tensor(times, dtype=torch.float32)
positions = torch.tensor(positions, dtype=torch.float32)
velocities = torch.tensor(velocities, dtype=torch.float32)

### Select Training Points
# From Simulations:
ntrain = config["Nsteps"]//2
half_points = np.arange(config["Nsteps"]//2)[::8]
np.random.shuffle(half_points)
train_indices = np.sort(half_points[:ntrain])

X_train = times[train_indices].unsqueeze(-1)
Y_train = torch.cat(
    (
        positions[train_indices].unsqueeze(-1),
        velocities[train_indices].unsqueeze(-1),
    ),
    dim = 1,
)

# Collocation Points:
def gen_collocation(tmax, Ncol):
    return torch.tensor(
        np.random.uniform(
            low = 0,
            high = tmax,
            size = Ncol,
        ),
        dtype = torch.float32,
        requires_grad = True,
    ).unsqueeze(-1)
    
### Model
model_config = {
    "Nin": 1,
    "Nout": 2,
    "nlayers": 3,
    "nnodes": 32,
    "activation": torch.nn.Sigmoid,
}

model_classical = FCNet(**model_config)
model_pinn = PINNs_FCNet(**model_config)

### Train
train_config = {
    "epochs": 10_000,
    "lr": 1e-3,
}

def mse(output, expected):
    return (output - expected).pow(2).mean()

optimizer_classical = torch.optim.Adam(model_classical.parameters(), lr=train_config["lr"])
optimizer_pinn = torch.optim.Adam(model_pinn.parameters(), lr=train_config["lr"])

# classical NN
loss_classical = {
    "mse": np.zeros(train_config["epochs"], dtype=np.float32)
}
for epoch in range(train_config["epochs"]):
    optimizer_classical.zero_grad()
    
    psi = model_classical(X_train)
    loss = mse(psi, Y_train)

    loss.backward()
    optimizer_classical.step()

    loss_classical["mse"][epoch] = loss.item()

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
    losses: dict,
    model: torch.nn.Module,
    prev_weights: dict | None = None,
    alpha: float = 0.5,
    eps: float = 1e-16,
):
    """
    losses: dict like {"data": L_data, "phys": L_phys, "ic": L_ic}
            each value is a scalar torch tensor (requires grad).
    Returns: weights dict with same keys.
    """
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
    X_col,
    model,
    prev_weights = None
):
    psi = model(X_train)
    psi_col = model(X_col)

    losses = {
        "data": mse(psi, Y_train),
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
    "res": np.zeros(train_config["epochs"], dtype=np.float32),
}
params_pinn = {
    "gamma": np.zeros(train_config["epochs"]+1, dtype=np.float32),
    "kappa": np.zeros(train_config["epochs"]+1, dtype=np.float32),
}
params_pinn["gamma"][0] = model_pinn.gamma.item()
params_pinn["kappa"][0] = model_pinn.kappa.item()
prev_weights = None
for epoch in range(train_config["epochs"]):
    optimizer_pinn.zero_grad()
    
    if epoch%step_gen_col == 0:
        X_col = gen_collocation(config["tmax"]/2, Ncol=Ncol)
    
    psi = model_pinn(X_train)
    psi_col = model_pinn(X_col)
    
    losses, lambdas = compute_losses(X_train, Y_train, X_col, model_pinn, prev_weights)
    
    loss = lambdas["data"]*losses["data"] + lambdas["res"]*losses["res"]
    # loss = losses["data"] + losses["res"]

    loss.backward()
    optimizer_pinn.step()

    loss_pinn["data"][epoch] = losses["data"].item()
    loss_pinn["res"][epoch] = losses["res"].item()
    
    params_pinn["gamma"][epoch+1] = model_pinn.gamma.item()
    params_pinn["kappa"][epoch+1] = model_pinn.kappa.item()

    if epoch % 500 == 0:
        print(f"epoch {epoch:5d} | loss = {loss.item():.3e} | L_data = {losses['data'].item():.3e} ({lambdas['data']*losses['data'].item():.3e}) | L_res = {losses['res'].item():.3e} ({lambdas['res']*losses['res'].item():.3e}) | gamma = {model_pinn.gamma.item():.3e} | kappa = {model_pinn.kappa.item():.3e}")

output_classical = model_classical(times.unsqueeze(-1)).detach().cpu().numpy()
output_pinn = model_pinn(times.unsqueeze(-1)).detach().cpu().numpy()

fig, axs = plt.subplots(1, 2)
axs[0].plot(output_classical[:, 0], label=r"$x_{classical}$")
axs[1].plot(output_classical[:, 1], label=r"$v_{classical}$")
axs[0].plot(output_pinn[:, 0], label=r"$x_{pinn}$")
axs[1].plot(output_pinn[:, 1], label=r"$v_{pinn}$")
axs[0].plot(positions, label=r"$x_{exp}$")
axs[1].plot(velocities, label=r"$v_{exp}$")
axs[0].legend()
axs[1].legend()
fig.savefig(plot_dir + "predicted_state.png", dpi=300)

epochs = np.arange(train_config["epochs"])
fig, axs = plt.subplots(1, 2)
axs[0].plot(epochs, loss_classical["mse"], label=r"$L_{total}=L_{mse}$")
axs[1].plot(epochs, loss_pinn["data"], label=r"$L_{mse}$")
axs[1].plot(epochs, loss_pinn["res"], label=r"$L_{res}$")
axs[1].plot(epochs, loss_pinn["data"]+loss_pinn["res"], label=r"$L_{total} = L_{mse} +L_{res}$")
axs[0].set_yscale("log")
axs[1].set_yscale("log")
axs[0].legend()
axs[1].legend()
fig.savefig(plot_dir + "losses.png", dpi=300)

epochs = np.arange(train_config["epochs"]+1)
fig, axs = plt.subplots(1, 2)
axs[0].plot(epochs, params_pinn["gamma"], label=r"$\gamma$")
axs[0].plot([0,train_config["epochs"]], [config["gamma"], config["gamma"]], "--", color="k", label=r"$\gamma_{exp}$")
axs[1].plot(epochs, params_pinn["kappa"], label=r"$\kappa$")
axs[1].plot([0,train_config["epochs"]], [config["kappa"], config["kappa"]], "--", color="k", label=r"$\kappa_{exp}$")
fig.savefig(plot_dir + "params.png", dpi=300)