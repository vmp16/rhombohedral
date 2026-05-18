import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from model.model import McCannSystem
from model.analysis import calculate_ahe
from model.config import GAMMA0, GAMMA1, N, N_LINEAR, K_MAX_INT as K_MAX

# -------------------- CONFIGURATION ------------------------

DELTA1 = 0.05
DELTA2 = - DELTA1 # 0.3
DELTAS = [DELTA1, DELTA2]

VALLEY_IDX = [1,-1]

# General constants
kB = 8.617e-5       # Boltzmann constant in eV/K

T_real = 20         # temperature [K]
T_eff = (kB * T_real) / GAMMA0

mu_lim = 10 * np.max(np.abs(DELTAS)) / GAMMA0

# -----------------------------------------------------------

def main():
    mu_vals = np.linspace(-mu_lim, mu_lim, 50)

    sigma_vals = []

    for mu in mu_vals:
        sigma_xy_list = []

        for xi, delta in zip(VALLEY_IDX, DELTAS):
            system = McCannSystem(GAMMA0, GAMMA1, xi, delta, N)

            sigma_xy, sigma_xy_bands = calculate_ahe(system, K_MAX, N_LINEAR, T_eff, mu)
            sigma_xy_list.append(sigma_xy)

        sigma_vals.append(np.sum(sigma_xy_list))

    plt.figure(figsize=(7,6))
    plt.rcParams.update({'font.size': 16})

    plt.plot(mu_vals, sigma_vals, color='blue')
    plt.axvspan(-abs(DELTA1), abs(DELTA1), color='red', alpha=0.1, label=r'$|\mu| < \Delta_1$')
    plt.axvspan(-abs(DELTA2), abs(DELTA2), color='green', alpha=0.1, label=r'$|\mu| < \Delta_2$')
    plt.legend()

    plt.xlabel(r"$\mu / \gamma_0$")
    plt.ylabel(r"$\sigma_{xy} / (e^2/h)$")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()