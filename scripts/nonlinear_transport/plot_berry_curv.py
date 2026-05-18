import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from model.model import McCannSystem
from model.analysis import calculate_berry_curv, get_kmesh
from model.config import GAMMA0, GAMMA1, GAMMA2, GAMMA3, VALLEY_IDX, DELTAS, N, N_PTS, K_MAX

def main():
    # Choose the valley and gap
    xi = VALLEY_IDX[0]
    delta = DELTAS[0, 0]

    # Build the system
    system = McCannSystem(GAMMA0, GAMMA1, xi, delta, N, gamma2=GAMMA2, gamma3=GAMMA3)

    # Map k-space to consider in a square mesh, as in get_sigma_PS.py
    KX, KY, k_flat, phi_flat = get_kmesh(K_MAX, N_PTS)

    # Evaluate the system
    system.get_energy_bands(k_flat, phi_flat)

    # Get the Berry curvature
    Omega = calculate_berry_curv(system)
    
    # Reshape for plotting
    Omega_pos = Omega[:, 0].reshape((N_PTS, N_PTS))
    Omega_neg = Omega[:, 1].reshape((N_PTS, N_PTS))

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Use a divergent colormap centered at zero
    vmax_pos = np.max(np.abs(Omega_pos))
    im0 = axes[0].pcolormesh(KX, KY, Omega_pos, cmap='RdBu_r', shading='auto', 
                             vmin=-vmax_pos, vmax=vmax_pos)
    fig.colorbar(im0, ax=axes[0], label=r'$\Omega$')
    axes[0].set_title(fr'Berry Curvature (Band 0), Valley {xi}, $\Delta={delta}$')
    axes[0].set_xlabel(r'$k_x$')
    axes[0].set_ylabel(r'$k_y$')
    axes[0].set_aspect('equal')

    vmax_neg = np.max(np.abs(Omega_neg))
    im1 = axes[1].pcolormesh(KX, KY, Omega_neg, cmap='RdBu_r', shading='auto', 
                             vmin=-vmax_neg, vmax=vmax_neg)
    fig.colorbar(im1, ax=axes[1], label=r'$\Omega$')
    axes[1].set_title(fr'Berry Curvature (Band 1) Valley {xi}, $\Delta={delta}$')

    axes[1].set_xlabel(r'$k_x$')
    axes[1].set_ylabel(r'$k_y$')
    axes[1].set_aspect('equal')

    plt.tight_layout()
    plot_path = project_root / "figures" / "berry_curv_2D.png"
    # plt.savefig(plot_path)
    plt.show()

if __name__ == "__main__":
    main()