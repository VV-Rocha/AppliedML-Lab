import numpy as np
import h5py

def compute_k(
        t,
        y,
        gamma=1.,
        kappa=1.,
    ):
    x, v = y
    dxdt = v
    dvdt = -gamma*v - kappa*x
    return np.array([dxdt, dvdt], dtype=float)

def rk4_step(f, t, y, dt, **params):
    k1 = f(t, y, **params)
    k2 = f(t + 0.5*dt, y + 0.5*dt*k1, **params)
    k3 = f(t + 0.5*dt, y + 0.5*dt*k2, **params)
    k4 = f(t + dt, y + dt*k3, **params)
    return y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def store(T, y, config, store_dir):
    with h5py.File(store_dir + f"data_gamma_{config["gamma"]}", "w") as f:
        f.create_dataset("times", data = T)
        f.create_dataset("positions", data = y[:, 0])
        f.create_dataset("velocities", data = y[:, 1])
        
        for key, value in config.items():
            f.attrs[key] = value

def load(stored_dir):
    with h5py.File(stored_dir, "r") as f:
        times = f["times"][:]
        positions = f["positions"][:]
        velocities = f["velocities"][:]

        config = dict(f.attrs)
    return times, positions, velocities, config

def simulate_duffing(
    x0=1.0,
    v0=0.0,
    tmax=100.0,
    Nsteps=1024,
    kappa=1.0,
    gamma=0.0,
    store_dir=None
):
    T = np.linspace(0, tmax, Nsteps)
    dt = T[1] - T[0]
    
    # initial state
    y = np.zeros((Nsteps, 2), dtype=float)
    y[0] = [x0, v0]
    
    params = dict(kappa=kappa, gamma=gamma)

    for i in range(Nsteps-1):
        y[i+1] = rk4_step(compute_k, T[i], y[i], dt, **params)
        print(f"{i+1}/{Nsteps-1}", end="\r")
        
    if store_dir is not None:
        config = {
            "x0": x0,
            "v0": v0,
            "tmax": tmax,
            "Nsteps": Nsteps,
            "kappa": kappa,
            "gamma": gamma,
        }
        store(T, y, config, store_dir)
    return T, y[:,0], y[:,1]