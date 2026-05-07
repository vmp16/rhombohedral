import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.model import McCannSystem
from model.analysis import velocity_operator, get_kmesh
from model.config import GAMMA0, GAMMA1, VALLEY_IDX, DELTA, N, N_PTS, K_MAX

def main():
    # Build the system
    system = McCannSystem(GAMMA0, GAMMA1, VALLEY_IDX[0], DELTA, N)

    # Create a grid in k-space
    KX, KY, k_flat, phi_flat = get_kmesh(K_MAX, N_PTS)

    # Evaluate the system and calculate velocities for the valence band (idx=1)
    system.get_energy_bands(k_flat, phi_flat)
    Vx, Vy = velocity_operator(system, idx1=1, idx2=1)

    # Reshape components back to grid
    VX = Vx.reshape(N_PTS, N_PTS)
    VY = Vy.reshape(N_PTS, N_PTS)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot Vx
    im0 = axes[0].pcolormesh(KX, KY, VX, shading='auto', cmap='RdBu_r')
    axes[0].set_title(r'Group Velocity $v_x$')
    axes[0].set_xlabel(r'$k_x$')
    axes[0].set_ylabel(r'$k_y$')
    fig.colorbar(im0, ax=axes[0])

    # Plot Vy
    im1 = axes[1].pcolormesh(KX, KY, VY, shading='auto', cmap='RdBu_r')
    axes[1].set_title(r'Group Velocity $v_y$')
    axes[1].set_xlabel(r'$k_x$')
    axes[1].set_ylabel(r'$k_y$')
    fig.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
