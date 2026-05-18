import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.model import McCannSystem
from model.analysis import velocity_operator, get_G, get_electric_shift, get_kmesh
from model.config import GAMMA0, GAMMA1, GAMMA2, GAMMA3, VALLEY_IDX, DELTA, N, N_PTS, K_MAX, T_eff as T, mu_eff as mu

band_idx = 1

SHOW_PLOTS = False

def main():
    print(10*'=' + ' BREAKING P SYMMETRY WITH TRIGONAL WARPING ' + 10*'=')
    # Build the system
    system = McCannSystem(GAMMA0, GAMMA1, VALLEY_IDX[0], DELTA, N, gamma2=GAMMA2, gamma3=GAMMA3)

    # Create a grid in k-space
    KX, KY, k_flat, phi_flat = get_kmesh(K_MAX, N_PTS)

    # 1. Evaluate the system and its Symmetry
    energies = system.get_energy_bands(k_flat, phi_flat)
    E_surf = energies[0].reshape(N_PTS, N_PTS)

    Vx, Vy = velocity_operator(system, 0, 0)
    VX = Vx.reshape(N_PTS, N_PTS)
    
    G_tensor = get_G(system, 0)
    Gxx = G_tensor[0, 0].reshape(N_PTS, N_PTS)

    E_sym = np.allclose(E_surf, np.flip(E_surf), atol=1e-8)
    Vx_odd = np.allclose(VX, -np.flip(VX), atol=1e-8)
    Gxx_even = np.allclose(Gxx, np.flip(Gxx), atol=1e-8)
    
    print(f"\nC_inf Symmetry Check:")
    print(f"Is Energy E(k) symmetric? {E_sym}")
    print(f"Is Velocity v_x(k) odd? {Vx_odd}")
    print(f"Is Metric G_xx(k) even? {Gxx_even}")
   
    # 2. Calculate Electric Integral
    print("\nCalculating Electric Posiitonal Shift (mu_eff = 0.06)...")
    chi_tensor = get_electric_shift(system, band_idx, K_MAX, N_PTS, T, mu)
    
    print(f"Total Tensor (chi_ijl):")
    print(chi_tensor)
    
    # 3. Symmetry Analysis
    print("\n--- Symmetry Analysis ---")
    
    # Mirror Symmetry Mx (y -> -y)
    # Components with an odd number of y-indices should be zero
    c_yxx = chi_tensor[1, 0, 0]
    c_yyy = chi_tensor[1, 1, 1]
    c_xyx = chi_tensor[0, 1, 0]
    c_xxy = chi_tensor[0, 0, 1]
    
    print(f"Mirror symmetry Mx check (odd y-indices should be ~0):")
    print(f"  chi_yxx (100): {c_yxx:.2e}")
    print(f"  chi_yyy (111): {c_yyy:.2e}")
    print(f"  chi_xyx (010): {c_xyx:.2e}")
    print(f"  chi_xxy (001): {c_xxy:.2e}")
    
    # C3 Symmetry relations: xxx = -xyy = -yxy = -yyx
    c_xxx = chi_tensor[0, 0, 0]
    c_xyy = chi_tensor[0, 1, 1]
    c_yxy = chi_tensor[1, 0, 1]
    c_yyx = chi_tensor[1, 1, 0]
    
    print(f"\nC3 symmetry check (xxx = -xyy = -yxy = -yyx):")
    print(f"  chi_xxx (000): {c_xxx:.2e}")
    print(f"  chi_xyy (011): {c_xyy:.2e}")
    print(f"  chi_yxy (101): {c_yxy:.2e}")
    print(f"  chi_yyx (110): {c_yyx:.2e}")
    # print(f"  Max deviation from C3: {max(abs(c_xxx+c_xyy), abs(c_xxx+c_yxy), abs(c_xxx+c_yyx)):.2e}")

    # Internal formula ratio check (Structured Noise)
    # As discussed, the specific formula structure leads to these ratios in nearly-isotropic grids
    if abs(c_yyy) > 1e-20:
        print(f"\nInternal formula ratio check (Structured Noise):")
        print(f"  chi_yxx / chi_yyy: {c_yxx/c_yyy:.4f}")
        print(f"  chi_yyy / chi_xyx: {c_yyy/c_xyx:.4f}")

    if np.any(np.abs(chi_tensor) > 1e-5):
        print("\nRESULT: Positional shift is NON-ZERO, as expected.")
    else:
        print("\nRESULT: Positional shift is zero (check mu_eff or perturbation magnitude).")

    if SHOW_PLOTS :
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

        plt.suptitle('Breaking Local Inversion with Trigonal Warping')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()