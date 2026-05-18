import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from model.model import McCannSystem
from model.config import GAMMA0, GAMMA1, N, N_PTS, K_MIN, K_MAX, VALLEY_IDX

# -------------------- CONFIGURATION ------------------------

DELTA1 = 0.05
DELTA2 = -DELTA1 # 0.1
DELTAS = [DELTA1, DELTA2]

# -----------------------------------------------------------

def main():
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

    for i, (xi, delta) in enumerate(zip(VALLEY_IDX, DELTAS)):
        # Build the system
        system = McCannSystem(GAMMA0, GAMMA1, xi, delta, N)

        # Map k-space to consider in polar coordinates (p, phi)
        # For the x direction, phi=0
        phi = 0
        k = np.linspace(K_MIN, K_MAX, N_PTS)

        # Get the energy
        energy = system.get_energy_bands(k, phi)

        ax[i].plot(k, energy[0], color='blue')
        ax[i].plot(k, energy[1], color='blue')

        ax[i].set_xlabel("k")
        ax[i].set_title(f"Valley index {xi}")

    ax[0].set_ylabel("Energy")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
