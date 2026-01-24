import matplotlib.pyplot as plt
import sys

sys.path.append("../solver/")
from rk4 import simulate_duffing

config = {}

### Simulation Config
config["Nsteps"] = 512
config["tmax"] = 50

### Initial Conditions
config["x0"] = 1.
config["v0"] = 0.

### Physical Coefficients
config["kappa"] = 1.
config["gamma"] = 0.2

### Store folders
config["store_dir"] = f"../data/simulations/damped_harmonic/"

plot_dir = f"../report/Figures/damped_harmonic/simulations/"

times, positions, velocities = simulate_duffing(**config,)

fig, axs = plt.subplots(1)
axs.plot(positions)
axs.set_xlabel("t")
axs.set_ylabel("x(t)")
fig.savefig(plot_dir + f"oscillation_gamma_{config['gamma']}.png", dpi=300)