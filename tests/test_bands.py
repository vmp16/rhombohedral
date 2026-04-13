import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.model import McCannSystem
from model.config import GAMMA0, GAMMA1, VALLEY_IDX, DELTA, N, N_PTS, K_MIN, K_MAX

def main():
    # Build the system
    system = McCannSystem(GAMMA0, GAMMA1, -1, DELTA, N)

    # Map k-space to consider in polar coordinates (p, phi)
    # For the x direction, phi=0
    phi = 0
    k = np.linspace(K_MIN, K_MAX, N_PTS)

    # Get the energies
    energy = system.get_energy_bands(k, phi)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(k, energy, label='Positive band', color='blue')
    plt.plot(k, -energy, label='Negative band', color='red')

    plt.xlabel(r'$k$')
    plt.ylabel('Energy')
    plt.title(f'Bands for N={N} (McCann Model)')
    plt.legend()
    plt.show()
    # plt.savefig('bands_plot.png')
    # print("Bands plot saved to bands_plot.png")

if __name__ == "__main__":
    main()
