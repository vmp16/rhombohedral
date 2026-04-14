import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.model import McCannSystem
from model.analysis import calculate_ahe
from model.config import GAMMA0, GAMMA1, N, N_LINEAR, K_MAX_INT as K_MAX

# -------------------- CONFIGURATION ------------------------

DELTA1 = 0.05
DELTA2 = -DELTA1
DELTAS = [DELTA1, DELTA2]

VALLEY_IDX = [1,-1]

# General constants
kB = 8.617e-5       # Boltzmann constant in eV/K

T_real = 20         # temperature [K]
T_eff = (kB * T_real) / GAMMA0

mu_eff = 0.0 / GAMMA0       # Fermi level [GAMMA0 units]

# -----------------------------------------------------------

def main():
    sigma_xy_list = []

    for xi, delta in zip(VALLEY_IDX, DELTAS):
        print(f"Calculating for valley = {xi} (Delta = {delta})...")

        system = McCannSystem(GAMMA0, GAMMA1, xi, delta, N)

        sigma_xy, sigma_xy_bands = calculate_ahe(system, K_MAX, N_LINEAR, T_eff, mu_eff)
        sigma_xy_list.append(sigma_xy)

        print(f"\tsigma_xy(valley={xi}) = {sigma_xy:.5f}\n")

    sigma_xy_tot = np.sum(sigma_xy_list)

    print(f"sigma_xy total = {sigma_xy_tot:.5f}")


if __name__ == "__main__":
    main()

