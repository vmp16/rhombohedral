import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.model import McCannSystem
from model.analysis import calculate_berry_curv, calculate_berry_integral
from model.config import GAMMA0, GAMMA1, DELTA, N, N_LINEAR, K_MAX_INT

def compare_valleys():
    # Setup k-range for plotting
    k_plot = np.linspace(-0.5, 0.5, 500)
    phi = 0  # Plot along kx
    
    valleys = [1, -1]
    results = {}

    plt.figure(figsize=(12, 5))

    for i, xi in enumerate(valleys):
        system = McCannSystem(GAMMA0, GAMMA1, xi, DELTA, N)
        
        # 1. Calculate Bands
        energies = system.get_energy_bands(k_plot, phi)
        pos_band = energies # energies is already the positive band from get_energy_bands
        
        # 2. Calculate Berry Curvature
        omega = calculate_berry_curv(system)[:, 0] # Positive band
        
        # 3. Calculate Chern Number Contribution
        # Use a circular area of radius K_MAX_INT
        integrals = calculate_berry_integral(system, K_MAX_INT, N_LINEAR)
        chern = integrals[0] / (2 * np.pi)
        
        results[xi] = {
            'chern': chern,
            'k': k_plot,
            'energy': pos_band,
            'omega': omega
        }

        # Plotting Energy Bands
        plt.subplot(1, 2, 1)
        plt.plot(k_plot, pos_band, label=f'Valley $\\xi={xi}$')
        plt.title('Positive Energy Band')
        plt.xlabel('k')
        plt.ylabel('Energy')
        plt.legend()

        # Plotting Berry Curvature
        plt.subplot(1, 2, 2)
        plt.plot(k_plot, omega, label=f'Valley $\\xi={xi}$')
        plt.title('Berry Curvature (Positive Band)')
        plt.xlabel('k')
        plt.ylabel('$\\Omega$')
        plt.legend()

    plt.tight_layout()
    plot_path = project_root / "figures" / "valley_comparison_detailed.png"
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

    # Print results
    print("\n" + "="*40)
    print(f"Results for N={N} layers:")
    for xi in valleys:
        print(f"Valley xi = {xi:2d}: Local Chern Contribution = {results[xi]['chern']:.6f}")
    print("="*40)

if __name__ == "__main__":
    compare_valleys()
