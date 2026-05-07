import sys
import numpy as np
from pathlib import Path

# Try to import matplotlib, but handle cases where it's missing
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.model import McCannSystem
from model.analysis import velocity_operator, get_G, get_kmesh
from model.config import GAMMA0, GAMMA1, VALLEY_IDX, DELTA, N, N_PTS, K_MAX

class BrokenMcCannSystem(McCannSystem):
    """
    Subclass to allow manual injection of terms into X and its derivatives
    for symmetry testing.
    """
    def set_perturbation(self, delta_X, d_delta_X_dphi):
        self.delta_X = delta_X
        self.d_delta_X_dphi = d_delta_X_dphi

    def X_at_k(self, k, phi):
        base_X = super().X_at_k(k, phi)
        self.X = base_X + self.delta_X
        return self.X

    def derivate_system(self):
        dH_dk, dH_dphi = super().derivate_system()
        # Add the perturbation derivative to the off-diagonal elements
        dH_dphi[0, 1] += self.d_delta_X_dphi
        dH_dphi[1, 0] += np.conj(self.d_delta_X_dphi)
        return dH_dk, dH_dphi

def run_test_case(name, n_parity, ax_row=None):
    print(f"\n--- Running Test: {name} ---")
    
    # Initialize system
    system = BrokenMcCannSystem(GAMMA0, GAMMA1, VALLEY_IDX[0], DELTA, N)
    
    # Create grid
    KX, KY, k_flat, phi_flat = get_kmesh(K_MAX, N_PTS)

    # Define perturbation: alpha * cos(n * phi)
    alpha = 0.05
    delta_X = alpha * np.cos(n_parity * phi_flat)
    d_delta_X_dphi = -alpha * n_parity * np.sin(n_parity * phi_flat)
    
    system.set_perturbation(delta_X, d_delta_X_dphi)
    
    # Evaluate bands and geometry
    energies = system.get_energy_bands(k_flat, phi_flat)
    E_surf = energies[0].reshape(N_PTS, N_PTS) # Conduction band
    
    Vx, Vy = velocity_operator(system, 0, 0)
    VX = Vx.reshape(N_PTS, N_PTS)
    VY = Vy.reshape(N_PTS, N_PTS)
    
    G_tensor = get_G(system, 0)
    Gxx = G_tensor[0, 0].reshape(N_PTS, N_PTS)
    Gxy = G_tensor[0, 1].reshape(N_PTS, N_PTS)

    # Symmetry checks
    E_sym = np.allclose(E_surf, np.flip(E_surf), atol=1e-8)
    Vx_odd = np.allclose(VX, -np.flip(VX), atol=1e-8)
    Gxx_even = np.allclose(Gxx, np.flip(Gxx), atol=1e-8)
    
    print(f"Is Energy E(k) symmetric? {E_sym}")
    print(f"Is Velocity v_x(k) odd? {Vx_odd}")
    print(f"Is Metric G_xx(k) even? {Gxx_even}")
    
    if E_sym and Vx_odd and Gxx_even:
        print("RESULT: Local inversion is PRESERVED. Integral of (v * G) will be ZERO.")
    else:
        print("RESULT: Local inversion is BROKEN! Integral of (v * G) can be NON-ZERO.")

    if HAS_MATPLOTLIB and ax_row is not None:
        # Plot E
        im0 = ax_row[0].pcolormesh(KX, KY, E_surf, shading='auto', cmap='viridis')
        ax_row[0].set_title(f'{name}: Energy')
        plt.colorbar(im0, ax=ax_row[0])
        
        # Plot Vx
        im1 = ax_row[1].pcolormesh(KX, KY, VX, shading='auto', cmap='RdBu_r')
        ax_row[1].set_title('Velocity $v_x$')
        plt.colorbar(im1, ax=ax_row[1])
        
        # Plot Gxx
        im2 = ax_row[2].pcolormesh(KX, KY, Gxx, shading='auto', cmap='magma')
        ax_row[2].set_title('Metric $G_{xx}$')
        plt.colorbar(im2, ax=ax_row[2])

def main():
    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        case1_axes = axes[0]
        case2_axes = axes[1]
    else:
        print("Matplotlib not found. Running numerical checks only.")
        case1_axes = case2_axes = None

    # Case 1: Add cos(phi) (Odd term)
    # Since N=5 (Odd), mixing Odd with Odd preserves inversion.
    run_test_case(r"Adding $\cos(\phi)$ (Odd)", 1, case1_axes)

    # Case 2: Add cos(2*phi) (Even term)
    # Mixing Odd with Even breaks inversion.
    run_test_case(r"Adding $\cos(2\phi)$ (Even)", 2, case2_axes)

    if HAS_MATPLOTLIB:
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
