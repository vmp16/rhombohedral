import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.model import McCannSystem
from model.analysis import calculate_ahe
from model.config import GAMMA0, GAMMA1, N, N_LINEAR, K_MAX_INT as K_MAX, DELTAS, VALLEY_IDX

# -------------------- CONFIGURATION ------------------------

# General constants
kB = 8.617e-5       # Boltzmann constant in eV/K

T_real = 20         # temperature [K]
T_eff = (kB * T_real) / GAMMA0

mu_lim = 3 * np.max(np.abs(DELTAS)) / GAMMA0

# -----------------------------------------------------------

def main():
    mu_vals = np.linspace(-mu_lim, mu_lim, 100)

    sigma_vals = []

    for mu in mu_vals:
        print(f"Progress: {100 * (np.where(mu_vals == mu)[0][0] + 1) / len(mu_vals):.1f}%", end='\r')

        sigma_xy_list = []

        for xi, delta_K in zip(VALLEY_IDX, DELTAS):
            # Build one system per spin
            system_up = McCannSystem(GAMMA0, GAMMA1, xi, delta_K[0], N)
            system_dn = McCannSystem(GAMMA0, GAMMA1, xi, delta_K[1], N)

            # Calculate the conductivity
            sigma_xy_up, _ = calculate_ahe(system_up, K_MAX, N_LINEAR, T_eff, mu)
            sigma_xy_dn, _ = calculate_ahe(system_dn, K_MAX, N_LINEAR, T_eff, mu)

            sigma_xy_list.append(sigma_xy_up + sigma_xy_dn)

        sigma_vals.append(np.sum(sigma_xy_list))

    plt.figure(figsize=(7,6))
    plt.rcParams.update({'font.size': 16})

    # plt.axhline(0, linestyle=':', color='grey')
    # plt.axhline(-N/2, linestyle=':', color='grey')
    # plt.axhline(-N, linestyle=':', color='grey')

    plt.scatter(mu_vals, sigma_vals, marker='o', s=5, color='blue')

    # plt.axvspan(-abs(DELTA1), abs(DELTA1), color='red', alpha=0.1, label=r'$|\mu| < \Delta_1$')
    # plt.axvspan(-abs(DELTA2), abs(DELTA2), color='green', alpha=0.1, label=r'$|\mu| < \Delta_2$')
    # plt.legend()

    plt.xlabel(r"$\mu / \gamma_0$")
    plt.ylabel(r"$\sigma_{xy} / (e^2/h)$")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()