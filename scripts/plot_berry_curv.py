import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.model import McCannSystem
from model.analysis import calculate_berry_curv
from model.config import GAMMA0, GAMMA1, VALLEY_IDX, DELTA, N, N_PTS, K_MIN, K_MAX

def main():
    # Build the system
    system = McCannSystem(GAMMA0, GAMMA1, VALLEY_IDX, DELTA, N)

    # Map k-space to consider in polar coordinates (p, phi)
    # For the x direction, phi=0
    phi = 0
    k = np.linspace(K_MIN, K_MAX, N_PTS)

    # Get the energies
    energy = system.get_energy_bands(k, phi)

    # Get the Berry curvature
    Omega = calculate_berry_curv(system)
    print(f"Omega shape = {Omega.shape}")
    
    idx_max = np.argmax(np.abs(Omega[:, 0]))
    print(f"Peak Omega (Pos Band) = {Omega[idx_max, 0]} at k = {k[idx_max]}")

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(k, Omega[:, 0], label='Positive band', color='blue')
    plt.plot(k, Omega[:, 1], label='Negative band', color='red')

    plt.xlabel(r'$k$')
    plt.ylabel(r'Berry Curvature ($\Omega$)')
    plt.title(f'Berry Curvature for N={N} (McCann Model)')
    plt.legend()
    plot_path = project_root / "figures" / "berry_curv_plot.png"
    # plt.savefig(plot_path)
    plt.show()
    # print("Berry curvature plot saved to berry_curv_plot.png")

if __name__ == "__main__":
    main()