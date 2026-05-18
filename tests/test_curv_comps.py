import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.model import McCannSystem
from model.analysis import get_kmesh
from model.config import GAMMA0, GAMMA1, GAMMA2, GAMMA3, VALLEY_IDX, DELTAS, N, N_PTS, K_MAX

def get_velocities(system):
    psi_p, psi_m = system.get_eigenstates()

    # Stack eigenstates as columns and move k to the first dimension: (k, component, eigenstate)
    U_raw = np.array((psi_p, psi_m))
    U = np.transpose(U_raw, (2, 1, 0))
    U_dag = np.conj(np.transpose(U, (0, 2, 1)))
    # print(f"U shape = {U.shape}")
    # print(f"U_dag shape = {U_dag.shape}")

    dH_dk, dH_dphi = system.derivate_system()
    # Move k to the first dimension: (k, component, component)
    dH_dk = np.transpose(dH_dk, (2, 0, 1))
    dH_dphi = np.transpose(dH_dphi, (2, 0, 1))

    Vk = U_dag @ dH_dk @ U
    Vphi = U_dag @ dH_dphi @ U

    return Vk, Vphi

def main():
    delta = DELTAS[0, 0]

    KX, KY, k_flat, phi_flat = get_kmesh(K_MAX, N_PTS)

    fig, axs = plt.subplots(2, 3, figsize=(16, 8))

    for i, xi in enumerate(VALLEY_IDX):
        system = McCannSystem(GAMMA0, GAMMA1, xi, delta, N, gamma2=GAMMA2, gamma3=GAMMA3)

        system.get_energy_bands(k_flat, phi_flat)

        Vk, Vphi = get_velocities(system)
        cross_term = Vk * Vphi.conj() - Vphi * Vk.conj()

        # Reshape for plotting
        Vk = np.imag(Vk[:, 0, 1]).reshape((N_PTS, N_PTS))
        Vphi = np.imag(Vphi[:, 0, 1]).reshape((N_PTS, N_PTS))
        cross_term = np.imag(cross_term[:, 0, 1]).reshape((N_PTS, N_PTS))

        im0 = axs[i, 0].pcolormesh(KX, KY, Vk, cmap='RdBu_r')
        # fig.colorbar(im0, ax=axs[i, 0], label=r'$V_k$')
        axs[i, 0].set_title(r'$V_k$')

        im1 = axs[i, 1].pcolormesh(KX, KY, Vphi, cmap='RdBu_r')
        # fig.colorbar(im1, ax=axs[i, 1], label=r'$V_{\phi}$')
        axs[i, 1].set_title(r'$V_{\phi}$')

        im2 = axs[i, 2].pcolormesh(KX, KY, cross_term, cmap='RdBu_r')
        # fig.colorbar(im2, ax=axs[i, 2], label='cross prod')
        axs[i, 2].set_title('cross prod')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()