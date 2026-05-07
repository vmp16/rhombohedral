import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.model import McCannSystem
from model.analysis import get_kmesh
from model.config import GAMMA0, GAMMA1, GAMMA2, GAMMA3, VALLEY_IDX, DELTA, N, K_MAX

def main():
    print(10*'=' + ' CHECKING HAMILTONIAN SYMMETRIES ' + 10*'=')
    
    # System setup
    system = McCannSystem(GAMMA0, GAMMA1, VALLEY_IDX[0], DELTA, N, gamma2=GAMMA2, gamma3=GAMMA3)
    ksi = VALLEY_IDX[0]
    
    print(f"System parameters: N={N}, ksi={ksi}, gamma2={GAMMA2}, gamma3={GAMMA3}, Delta={DELTA}")

    # Test point in k-space (away from high-symmetry lines to be general)
    k0 = 0.1
    phi0 = np.pi/7
    
    H0 = system.get_hamiltonian(k0, phi0)
    
    # 1. Test C3 rotation: H(phi + 2pi/3) = U H(phi) U^\dagger
    # For this model, U = diag(exp(i*alpha/2), exp(-i*alpha/2)) where alpha = ksi * N * 2pi/3
    phi_rot = phi0 + 2*np.pi/3
    H_rot = system.get_hamiltonian(k0, phi_rot)
    
    alpha = ksi * N * (2 * np.pi / 3)
    U_c3 = np.array([[np.exp(1j * alpha / 2), 0],
                     [0, np.exp(-1j * alpha / 2)]])
    
    # Check if U * H0 * U_dag = H_rot
    H_rot_check = U_c3 @ H0 @ np.conj(U_c3).T
    c3_ham_invariant = np.allclose(H_rot, H_rot_check, atol=1e-8)
    
    print(f"\nC3 Rotation (2pi/3):")
    print(f"Is Hamiltonian unitarily equivalent? {c3_ham_invariant}")
    if not c3_ham_invariant:
        print("Diff:\n", H_rot - H_rot_check)

    # 2. Test Mx Mirror (y -> -y): H(phi) -> H(-phi)
    # Mirroring in-plane doesn't swap A1 and BN sites in this model.
    # The expected relation is H(-phi) = H*(phi)
    phi_mir = -phi0
    H_mir = system.get_hamiltonian(k0, phi_mir)
    
    mx_ham_invariant = np.allclose(H_mir, np.conj(H0), atol=1e-8)
    
    print(f"\nMx Mirror (y -> -y):")
    print(f"Does H(-phi) == H*(phi)? {mx_ham_invariant}")
    
    # 3. Test Local Inversion : H(kx, ky) -> H(-kx, -ky)
    # This should be BROKEN by gamma3
    phi_inv = phi0 + np.pi
    H_inv = system.get_hamiltonian(k0, phi_inv)
    
    # Local inversion in 2D behaves as C2 rotation.
    alpha_c2 = ksi * N * np.pi
    U_c2 = np.array([[np.exp(1j * alpha_c2 / 2), 0],
                     [0, np.exp(-1j * alpha_c2 / 2)]])
    
    H_inv_check = U_c2 @ H0 @ np.conj(U_c2).T
    p_ham_invariant = np.allclose(H_inv, H_inv_check, atol=1e-8)
    
    # For energy to be invariant, eigenvalues must match.
    evals0 = np.sort(system.get_energy_bands(k0, phi0))
    evals_inv = np.sort(system.get_energy_bands(k0, phi_inv))
    p_energy_invariant = np.allclose(evals0, evals_inv, atol=1e-8)
    
    print(f"\nLocal Inversion (P):")
    print(f"Is Hamiltonian unitarily equivalent? {p_ham_invariant}")
    print(f"Is Energy invariant? {p_energy_invariant}")

    # Summary
    if c3_ham_invariant and mx_ham_invariant:
        print("\nCONCLUSION: The Hamiltonian correctly exhibits C3 and Mx symmetries.")
    else:
        print("\nCONCLUSION: Symmetry check FAILED for the Hamiltonian matrix.")

    # # Re-running the visual check for completeness
    # print("\nGenerating visual check plot...")
    # N_PTS = 200
    # KX, KY, k_flat, phi_flat = get_kmesh(K_MAX, N_PTS)
    
    # energies_mesh = system.get_energy_bands(k_flat, phi_flat)
    # E_mesh = energies_mesh[0].reshape(N_PTS, N_PTS)
    
    # plt.figure(figsize=(8, 7))
    # cp = plt.pcolormesh(KX, KY, E_mesh, shading='auto', cmap='viridis')
    # plt.colorbar(cp, label='Energy E(k)')
    # plt.title(f'Energy bands (Hamiltonian Symmetry Check)\nC3: {c3_ham_invariant}, Mx: {mx_ham_invariant}, P_Energy: {p_energy_invariant}')
    # plt.xlabel('kx')
    # plt.ylabel('ky')
    # plt.contour(KX, KY, E_mesh, colors='white', alpha=0.3, levels=10)
    # plt.axis('equal')
    # plt.tight_layout()
    # plt.show()
    # plt.savefig(project_root / 'tests' / 'CN_symmetry_check.png')
    # print(f"Saved visualization to tests/CN_symmetry_check.png")

if __name__ == "__main__":
    main()
