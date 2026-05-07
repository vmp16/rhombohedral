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
from model.analysis import velocity_operator, get_G, get_electric_shift, get_electric_NLAHE, get_kmesh
from model.config import GAMMA0, GAMMA1, VALLEY_IDX, DELTA, N, N_PTS, K_MAX

# Parameters for integration
T_real = 20
kB = 8.617e-5
T_eff = (kB * T_real) / GAMMA0
mu_eff = 0.06 / GAMMA0

class BrokenMcCannSystem(McCannSystem):
    """
    Subclass to allow manual injection of a constant term into X
    for symmetry testing.
    """
    def set_perturbation(self, delta_X):
        self.delta_X = delta_X

    def X_at_k(self, k, phi):
        # We add the constant to the total X, but the derivatives 
        # (calculated in the base class using X0, Xw, etc.) remain unchanged.
        base_X = super().X_at_k(k, phi)
        self.X = base_X + self.delta_X
        return self.X

def main():
    print(10*"=" + " BREAKING C2 SYMMETRY WITH COMPLEX CONSTANT " + 10*"=")
    
    # Initialize system (h0 will be 0 as gamma2/gamma4 default to 0 in this test)
    system = BrokenMcCannSystem(GAMMA0, GAMMA1, VALLEY_IDX[0], DELTA, N)
    
    # Create grid
    KX, KY, k_flat, phi_flat = get_kmesh(K_MAX, N_PTS)

    # Define perturbation: a complex constant C_even
    # This breaks the effective local inversion and all mirror symmetries.
    C_even = 0.1  + 0.0j
    system.set_perturbation(C_even)
    
    # 1. Evaluate Symmetry Properties
    energies = system.get_energy_bands(k_flat, phi_flat)
    E_surf = energies[0].reshape(N_PTS, N_PTS)
    
    Vx, Vy = velocity_operator(system, 0, 0)
    VX = Vx.reshape(N_PTS, N_PTS)
    
    G_tensor = get_G(system, 0)
    Gxx = G_tensor[0, 0].reshape(N_PTS, N_PTS)

    E_sym = np.allclose(E_surf, np.flip(E_surf), atol=1e-8)
    Vx_odd = np.allclose(VX, -np.flip(VX), atol=1e-8)
    Gxx_even = np.allclose(Gxx, np.flip(Gxx), atol=1e-8)
    
    print(f"\nSymmetry Check (C_even = {C_even}):")
    print(f"Is Energy E(k) symmetric? {E_sym}")
    print(f"Is Velocity v_x(k) odd? {Vx_odd}")
    print(f"Is Metric G_xx(k) even? {Gxx_even}")
    
    if not E_sym:
        print("RESULT: Local inversion symmetry is BROKEN by the constant shift.")

    # 2. Calculate Positional Shift Integral
    print("\nCalculating Positional Shift Integral (mu_eff = 0.06)...")
    chi_tensor = get_electric_shift(system, 0, K_MAX, N_PTS, T_eff, mu_eff)
    
    print(f"Total Shift Tensor (selected components):")
    print(chi_tensor)
    # print(f"chi_xxx = {chi_tensor[0, 0, 0]:.6e}")
    # print(f"chi_xyy = {chi_tensor[0, 1, 1]:.6e}")

    if np.any(np.abs(chi_tensor) > 1e-5):
        print("RESULT: Positional shift is NON-ZERO, as expected.")
    else:
        print("RESULT: Positional shift is zero (check mu_eff or perturbation magnitude).")

    # 3. Plotting
    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        im0 = axes[0].pcolormesh(KX, KY, E_surf, shading='auto', cmap='viridis')
        axes[0].set_title(r'Energy $E(\mathbf{k})$')
        plt.colorbar(im0, ax=axes[0])
        
        im1 = axes[1].pcolormesh(KX, KY, VX, shading='auto', cmap='RdBu_r')
        axes[1].set_title(r'Velocity $v_x$')
        plt.colorbar(im1, ax=axes[1])
        
        im2 = axes[2].pcolormesh(KX, KY, Gxx, shading='auto', cmap='magma')
        axes[2].set_title(r'Metric $G_{xx}$')
        plt.colorbar(im2, ax=axes[2])

        plt.suptitle(f'Breaking Local Inversion with Complex Constant ($C_{{even}}={C_even}$)')
        plt.tight_layout()
        plt.show()
    else:
        print("\nMatplotlib not found. Skipping plots.")

if __name__ == "__main__":
    main()
