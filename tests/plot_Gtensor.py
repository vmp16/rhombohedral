import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.model import McCannSystem
from model.analysis import get_G, get_kmesh
from model.config import GAMMA0, GAMMA1, VALLEY_IDX, DELTA, N, N_PTS, K_MAX

def main():
    # Build the system
    system = McCannSystem(GAMMA0, GAMMA1, VALLEY_IDX[1], DELTA, N)

    # Create a grid in k-space
    KX, KY, k_flat, phi_flat = get_kmesh(K_MAX, N_PTS)

    # Evaluate the system and calculate G tensor for the valence band (idx=1)
    system.get_energy_bands(k_flat, phi_flat)
    G_tensor_flat = get_G(system, band_idx=1)

    # Reshape components back to grid
    G_components = [
        G_tensor_flat[0, 0].reshape(N_PTS, N_PTS), # Gxx
        G_tensor_flat[0, 1].reshape(N_PTS, N_PTS), # Gxy
        G_tensor_flat[1, 0].reshape(N_PTS, N_PTS), # Gyx
        G_tensor_flat[1, 1].reshape(N_PTS, N_PTS)  # Gyy
    ]
    titles = ["Gxx", "Gxy", "Gyx", "Gyy"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        im = ax.pcolormesh(KX, KY, G_components[i], shading='auto', cmap='RdBu_r')
        ax.set_title(titles[i])
        ax.set_xlabel(r'$k_x$')
        ax.set_ylabel(r'$k_y$')
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
