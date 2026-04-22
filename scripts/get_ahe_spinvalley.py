import sys
import numpy as np
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

mu_eff = 0.0 / GAMMA0       # Fermi level [GAMMA0 units]

# -----------------------------------------------------------

def main():
    print(10*'=' + f" CALCULATING THE MAGNETIC AHE FOR MU = {mu_eff} " + 10*'=')
    
    sigma_xy_list = []

    for xi, delta_K in zip(VALLEY_IDX, DELTAS):
        print(f"Calculating for valley = {xi}...")

        # Build one system per spin
        system_up = McCannSystem(GAMMA0, GAMMA1, xi, delta_K[0], N)
        system_dn = McCannSystem(GAMMA0, GAMMA1, xi, delta_K[1], N)

        # Calculate the conductivity
        sigma_xy_up, _ = calculate_ahe(system_up, K_MAX, N_LINEAR, T_eff, mu_eff)
        sigma_xy_dn, _ = calculate_ahe(system_dn, K_MAX, N_LINEAR, T_eff, mu_eff)

        print(f"\tsigma_xy(up) = {sigma_xy_up:.4f}")
        print(f"\tsigma_xy(dn) = {sigma_xy_dn:.4f}")

        sigma_xy_list.append(sigma_xy_up + sigma_xy_dn)

    sigma_xy_tot = np.sum(sigma_xy_list)

    print(f"sigma_xy total = {sigma_xy_tot:.5f}")

if __name__ == "__main__":
    main()