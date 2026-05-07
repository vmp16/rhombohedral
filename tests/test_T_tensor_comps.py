import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.model import McCannSystem
from model.analysis import velocity_operator, get_G, get_kmesh, deriv_fermi_distrib
from model.config import GAMMA0, GAMMA1, GAMMA2, GAMMA3, VALLEY_IDX, DELTA, N, N_PTS, K_MAX

# Physical constants
kB = 8.617e-5       # Boltzmann constant in eV/K
T_real = 20         # temperature [K]
T_eff = (kB * T_real) / GAMMA0
mu_eff = 0.06 / GAMMA0       # Fermi level [GAMMA0 units]

def main():
    # Build the system
    system = McCannSystem(GAMMA0, GAMMA1, VALLEY_IDX[0], DELTA, N, gamma2=GAMMA2, gamma3=GAMMA3)

    # Create a grid in k-space
    KX, KY, k_flat, phi_flat = get_kmesh(K_MAX, N_PTS)

    energies = system.get_energy_bands(k_flat, phi_flat)
    band_E = energies[0]
    E_surf = band_E.reshape(N_PTS, N_PTS)

    G_tensor = get_G(system, 0)
    # Gxx = G_tensor[0, 0].reshape(N_PTS, N_PTS)

    Vx, Vy = velocity_operator(system, 0, 0)
    # VX = Vx.reshape(N_PTS, N_PTS)

    # Get the derivative of the Fermi distribution
    df_dE = deriv_fermi_distrib(band_E, mu_eff, T_eff)

    term1 = Vx * G_tensor[1, 1] * df_dE
    term2 = - Vy * G_tensor[0, 1] * df_dE

    T_shift = 2*term1 + term2
    T_NLAHE = term1 + term2
    
    # Reshape for plotting
    TERM1 = term1.reshape(N_PTS, N_PTS)
    TERM2 = term2.reshape(N_PTS, N_PTS)
    T_SHIFT = T_shift.reshape(N_PTS, N_PTS)
    T_NLAHE = T_NLAHE.reshape(N_PTS, N_PTS)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Adjust norm based on expected values after multiplying by df_dE
    # df_dE is ~ -1/(4*T_eff) at maximum. T_eff ~ 1.7e-3. 1/4T_eff ~ 150.
    # Previous values were up to 50000. So new values might be ~ 7e6?
    # Actually df_dE is negative, so let's check signs.
    
    norm = mcolors.TwoSlopeNorm(vcenter=0)

    im0 = axes[0, 0].pcolormesh(KX, KY, TERM1, shading='auto', cmap='RdBu_r', norm=norm)
    axes[0, 0].set_title(r'Term 1: $v_x G_{yy} (\partial f / \partial E)$')
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].pcolormesh(KX, KY, TERM2, shading='auto', cmap='RdBu_r', norm=norm)
    axes[0, 1].set_title(r'Term 2: $-v_y G_{xy} (\partial f / \partial E)$')
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[1, 0].pcolormesh(KX, KY, T_SHIFT, shading='auto', cmap='RdBu_r', norm=norm)
    axes[1, 0].set_title(r'$T_{shift} \cdot (\partial f / \partial E)$')
    plt.colorbar(im2, ax=axes[1, 0])

    im3 = axes[1, 1].pcolormesh(KX, KY, T_NLAHE, shading='auto', cmap='RdBu_r', norm=norm)
    axes[1, 1].set_title(r'$T_{NLAHE} \cdot (\partial f / \partial E)$')
    plt.colorbar(im3, ax=axes[1, 1])

    plt.suptitle(fr'Tensor Contributions to the Fermi Surface ($\mu = {mu_eff}$, $T = {T_real} K$)')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()