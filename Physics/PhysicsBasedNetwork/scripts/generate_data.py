import sys
sys.path.append("../solver/")
from euler_scheme import EulerSolver

import matplotlib.pyplot as plt
import numpy as np
import h5py

system_config = {
    "gamma": .2,
    "k": 1.,
    "k2": .1,
}

initial_config = {
    "x0": 1.,
    "v0": 0.,
}

solver_config = {
    "tmax": 50,
    "Nt": 1024,
    "dtype": "float64",
    "device": "cuda",
    "verbose": 1,
}

solver = EulerSolver(
    **system_config,
    **initial_config,
    **solver_config,
)

# run solver
solver.solve()

# store data
store_dir = "../data/"

def _store_config(file_dir, config):
    with h5py.File(file_dir, "w") as f:
        for key, value in config.items():
            f.attrs[key] = value
            
def store(
    times,
    states,
    system_config,
    initial_config,
    solver_config,
    store_dir,
):
    with h5py.File(store_dir + "results.h5", "w") as f:
        f.create_dataset("times", data=times.detach().cpu().numpy())
        f.create_dataset("states", data=states.detach().cpu().numpy())
        
    _store_config(store_dir + "system_config.h5", system_config)
    _store_config(store_dir + "initial_config.h5", initial_config)
    _store_config(store_dir + "solver_config.h5", solver_config)
    
store(
    solver.t,
    solver.states,
    system_config,
    initial_config,
    solver_config,
    store_dir
)