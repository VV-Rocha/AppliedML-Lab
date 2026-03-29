import h5py
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append("../model/")

from physics_based_model import PhysicsModel

import torch
from torch import nn

torch.manual_seed(0)
np.random.seed(0)

### Load Data
data_dir = "../data/"
store_dir = "../report/Figures/multistep/"

def _load_config(file_dir):
    config = {}
    with h5py.File(file_dir, "r") as f:
        for key in f.attrs.keys():
            config[key] = f.attrs[key]
        return config
    
def load(data_dir):
    with h5py.File(data_dir + "results.h5", "r") as f:
        times = f["times"][:]
        states = f["states"][:]
        
        system_config = _load_config(data_dir + "system_config.h5")
        initial_config = _load_config(data_dir + "initial_config.h5")
        solver_config = _load_config(data_dir + "solver_config.h5")
    return times, states, system_config, initial_config, solver_config

times, states, system_config, initial_config, solver_config = load("../data/")

train_config = {
    "Ntrain": 32,  # number of points used for training
    "num_epochs": 2_500,
    "lr": 1e-3,
}

### Pre-processing
# select training states
indices = np.arange(1, states.shape[0])
np.random.shuffle(indices)
indices = indices[:train_config["Ntrain"]]
indices = np.sort(indices)

states = (1 + np.random.normal(0,1,states.shape)*.1) * states

Ystates = states[indices]
Ystates = torch.tensor(Ystates, device=solver_config["device"], dtype=(torch.float32 if solver_config["dtype"]=="float32" else torch.float64))

### Define Model
model = PhysicsModel(
    **solver_config,
)

### Training Loop
params_history = {
    "k": np.zeros(train_config["num_epochs"]+1),
    "k2": np.zeros(train_config["num_epochs"]+1),
    "gamma": np.zeros(train_config["num_epochs"]+1),
    "psi0": np.zeros((train_config["num_epochs"]+1, 2))
}
loss_history = {
    "data": np.zeros(train_config["num_epochs"]),
}
optimizer = torch.optim.Adam(model.parameters(), lr=train_config["lr"])

params_history["gamma"][0] = model.gamma.detach().cpu().numpy()
params_history["k"][0] = model.k.detach().cpu().numpy()
params_history["k2"][0] = model.k2.detach().cpu().numpy()
params_history["psi0"][0] = model.psi0.detach().cpu().numpy()
for epoch in range(train_config["num_epochs"]):
    optimizer.zero_grad()
    
    model.update_matrix()
    
    states = model.forward()
    pred_states = states[indices-1]
    
    L = (pred_states - Ystates).pow(2).mean()
    
    L.backward()
    optimizer.step()
    
    # store params
    params_history["gamma"][epoch+1] = model.gamma.detach().cpu().numpy()
    params_history["k"][epoch+1] = model.k.detach().cpu().numpy()
    params_history["k2"][epoch+1] = model.k2.detach().cpu().numpy()
    params_history["psi0"][epoch+1] = model.psi0.detach().cpu().numpy()
    
    loss_history["data"][epoch] = L.detach().cpu().numpy()
    
    print(f"Epochs: {epoch+1}/{train_config['num_epochs']} | k: {params_history['k'][epoch+1]:.3e} ({system_config['k']}) | k2: {params_history['k2'][epoch+1]:.3e} ({system_config['k2']}) | gamma: {params_history['gamma'][epoch+1]:.3e} ({system_config['gamma']:.3e}) | x0: {params_history['psi0'][epoch+1][0]:.3e} ({1}) | v0: {params_history['psi0'][epoch+1][1]:.3e} ({0})")
    

### Plots
savefig_config = {
    "dpi": 300,
    "transparent": False,
}

fig, axs = plt.subplots(1, figsize=(10, 6.4))
axs.plot(loss_history["data"])
axs.set_yscale("log")
axs.set_ylabel("Loss", fontsize=18)
axs.set_xlabel("epochs", fontsize=18)
fig.tight_layout()
fig.savefig(store_dir + "loss.png", **savefig_config)

fig, axs = plt.subplots(5, 1, figsize=(10, 25))
axs[0].plot(params_history["k"])
axs[0].plot([0, train_config["num_epochs"]+1], [system_config["k"], system_config["k"]], "--", color="k")
axs[0].set_ylabel(r"$k$", fontsize=18)
axs[0].set_xlabel("epochs", fontsize=18)

axs[1].plot(params_history["k2"])
axs[1].plot([0, train_config["num_epochs"]+1], [system_config["k2"], system_config["k2"]], "--", color="k")
axs[1].set_ylabel(r"$k2$", fontsize=18)
axs[1].set_xlabel("epochs", fontsize=18)

axs[2].plot(params_history["gamma"])
axs[2].plot([0, train_config["num_epochs"]+1], [system_config["gamma"], system_config["gamma"]], "--", color="k")
axs[2].set_ylabel(r"$\gamma$", fontsize=18)
axs[2].set_xlabel("epochs", fontsize=18)

axs[3].plot(params_history["psi0"][:, 0])
axs[3].plot([0, train_config["num_epochs"]+1], [1, 1], "--", color="k")
axs[3].set_ylabel(r"$x_0$", fontsize=18)
axs[3].set_xlabel("epochs", fontsize=18)

axs[4].plot(params_history["psi0"][:, 1])
axs[4].plot([0, train_config["num_epochs"]+1], [0, 0], "--", color="k")
axs[4].set_ylabel(r"$v_0$", fontsize=18)
axs[4].set_xlabel("epochs", fontsize=18)

for i in range(4):
    axs[i].tick_params(axis="both", labelsize=14)

fig.savefig(store_dir + "params.png", **savefig_config)