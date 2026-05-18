import sys
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.model import McCannSystem
from model.analysis import deriv_fermi_distrib, get_kmesh
from model.config import GAMMA0, GAMMA1, GAMMA2, GAMMA3, VALLEY_IDX, DELTA, N, N_PTS, K_MAX

# -------------------- CONFIGURATION ------------------------
kB = 8.617e-5       # Boltzmann constant in eV/K
T_real = 40         # temperature [K]
T_eff = (kB * T_real) / GAMMA0
mu_eff = 0.5 / GAMMA0  # Fermi level moved to intersect the valence band (which peaks at -0.05)
# -----------------------------------------------------------

def main():
    # Build the system
    system = McCannSystem(GAMMA0, GAMMA1, VALLEY_IDX[0], DELTA, N, gamma2=GAMMA2, gamma3=GAMMA3)

    # Create a grid in k-space
    KX, KY, k_flat, phi_flat = get_kmesh(K_MAX, N_PTS)

    # Evaluate the system and calculate energies for the valence band (idx=1)
    energies = system.get_energy_bands(k_flat, phi_flat)
    band_E = energies[0]

    # Calculate the derivative of the Fermi distribution
    df_dE = deriv_fermi_distrib(band_E, mu_eff, T_eff)
    DF_DE = df_dE.reshape(N_PTS, N_PTS)

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(KX, KY, DF_DE, shading='auto', cmap='viridis')
    ax.set_title(r'Derivative of Fermi Distribution $\partial f / \partial E$')
    ax.set_xlabel(r'$k_x$')
    ax.set_ylabel(r'$k_y$')
    fig.colorbar(im, ax=ax, label=r'$df/dE$')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
