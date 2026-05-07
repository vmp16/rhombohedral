import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.model import McCannSystem
from model.analysis import velocity_operator, get_G, get_kmesh
from model.config import GAMMA0, GAMMA1

class BrokenMcCannSystem(McCannSystem):
    def set_perturbation(self, delta_X):
        self.delta_X = delta_X
    def X_at_k(self, k, phi):
        base_X = super().X_at_k(k, phi)
        self.X = base_X + self.delta_X
        return self.X

def plot_symmetry_for_N(N_layers, C_re, ax_row):
    # Setup system
    system = BrokenMcCannSystem(GAMMA0, GAMMA1, 1, 0.05, N_layers)
    system.set_perturbation(C_re)
    
    # Grid
    K_MAX = 0.2
    N_PTS = 100
    KX, KY, k_flat, phi_flat = get_kmesh(K_MAX, N_PTS)
    
    # Calculations
    energies = system.get_energy_bands(k_flat, phi_flat)
    E = energies[0].reshape(N_PTS, N_PTS)
    
    vx, _ = velocity_operator(system, 0, 0)
    VX = vx.reshape(N_PTS, N_PTS)
    
    G = get_G(system, 0)
    GXX = G[0, 0].reshape(N_PTS, N_PTS)
    
    # Plotting
    im0 = ax_row[0].pcolormesh(KX, KY, E, shading='auto', cmap='viridis')
    ax_row[0].set_title(f"N={N_layers}: Energy")
    plt.colorbar(im0, ax=ax_row[0])
    
    im1 = ax_row[1].pcolormesh(KX, KY, VX, shading='auto', cmap='RdBu_r')
    ax_row[1].set_title(f"N={N_layers}: Velocity $v_x$")
    plt.colorbar(im1, ax=ax_row[1])
    
    im2 = ax_row[2].pcolormesh(KX, KY, GXX, shading='auto', cmap='magma')
    ax_row[2].set_title(fr"N={N_layers}: Metric $G_{{xx}}$")
    plt.colorbar(im2, ax=ax_row[2])

def main():
    C_re = 0.3
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    for i, N in enumerate([3, 4, 5]):
        plot_symmetry_for_N(N, C_re, axes[i])
        
    plt.tight_layout()
    plt.savefig("tests/CN_symmetry_check.png")
    print("Saved symmetry check plots to tests/CN_symmetry_check.png")

if __name__ == "__main__":
    main()
