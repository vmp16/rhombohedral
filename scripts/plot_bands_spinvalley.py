import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.model import McCannSystem
from model.config import GAMMA0, GAMMA1, N, N_PTS, K_MIN, K_MAX

# -------------------- CONFIGURATION ------------------------

DELTA1UP = 0.1
DELTA1DN = 0.0
DELTA2UP = 0.0
DELTA2DN = -0.1

DELTAS = np.array([[DELTA1UP, DELTA1DN],
                   [DELTA2UP, DELTA2DN]])

VALLEY_IDX = [1,-1]

# -----------------------------------------------------------

def main():
    fig, ax = plt.subplots(1, 2, figsize=(13, 7), sharex=True, sharey=True)

    for i, (xi, delta_K) in enumerate(zip(VALLEY_IDX, DELTAS)):
        # Build one system per spin
        system_up = McCannSystem(GAMMA0, GAMMA1, xi, delta_K[0], N)
        system_dn = McCannSystem(GAMMA0, GAMMA1, xi, delta_K[1], N)

        # Map k-space to consider in polar coordinates (p, phi)
        # For the x direction, phi=0
        phi = 0
        k = np.linspace(K_MIN, K_MAX, N_PTS)

        # Get the energies
        energy_up = system_up.get_energy_bands(k, phi)
        energy_dn = system_dn.get_energy_bands(k, phi)

        ax[i].plot(k, energy_up[0], color='blue', label='spin up')
        ax[i].plot(k, energy_up[1], color='blue')
        ax[i].plot(k, energy_dn[0], color='red', label='spin down')
        ax[i].plot(k, energy_dn[1], color='red')
        
        ax[i].set_xlabel("k", fontsize=16)
        ax[i].set_title(f"Valley index {xi}", fontsize=16)

        ax[i].tick_params(axis='both', which='major', labelsize=16)

    ax[0].set_ylabel("Energy", fontsize=16)
    ax[1].legend(fontsize=16)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
