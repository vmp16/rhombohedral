import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.model import McCannSystem
from model.analysis import velocity_operator, get_G, get_electric_NLAHE, get_kmesh
from model.config import GAMMA0, GAMMA1, GAMMA2, GAMMA3, VALLEY_IDX, DELTA, N, N_PTS, K_MAX, T_eff, mu_eff

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
    
    print(f"\nSymmetry Check:")
    print(f"Is Energy E(k) symmetric? {E_sym}")
    print(f"Is Velocity v_x(k) odd? {Vx_odd}")
    print(f"Is Metric G_xx(k) even? {Gxx_even}")
   
    # 2. Calculate Electric NLAHE Integral
    print("\nCalculating Electric NLAHE Integral (mu_eff = 0.06)...")
    chi_tensor = get_electric_NLAHE(system, 0, K_MAX, N_PTS, T_eff, mu_eff)
    
    print(f"Total NLAHE Tensor (chi_ijl):")
    print(chi_tensor)
    
    # 3. Symmetry Analysis of NLAHE Tensor
    print("\n--- Symmetry Analysis ---")
    
    # Antisymmetry in i, j: chi_ijl = -chi_jil
    # This implies chi_iil = 0
    c_xxx = chi_tensor[0, 0, 0]
    c_xxy = chi_tensor[0, 0, 1]
    c_yyx = chi_tensor[1, 1, 0]
    c_yyy = chi_tensor[1, 1, 1]
    
    print(f"Antisymmetry check (chi_iil should be 0):")
    print(f"  chi_xxx: {c_xxx:.2e}")
    print(f"  chi_xxy: {c_xxy:.2e}")
    print(f"  chi_yyx: {c_yyx:.2e}")
    print(f"  chi_yyy: {c_yyy:.2e}")
    
    # Mirror Symmetry Mx (y -> -y)
    # chi_xyx: current x (even), field y (odd), field x (even) -> odd -> should be 0
    # chi_yxx: current y (odd), field x (even), field x (even) -> odd -> should be 0
    c_xyx = chi_tensor[0, 1, 0]
    c_yxx = chi_tensor[1, 0, 0]
    
    print(f"\nMirror symmetry Mx check (odd y-indices should be 0):")
    print(f"  chi_xyx: {c_xyx:.2e}")
    print(f"  chi_yxx: {c_yxx:.2e}")
    
    # Non-zero components: chi_xyy and chi_yxy
    # chi_xyy: current x (even), field y (odd), field y (odd) -> even -> non-zero
    # chi_yxy: current y (odd), field x (even), field y (odd) -> even -> non-zero
    # And chi_xyy = -chi_yxy by antisymmetry
    c_xyy = chi_tensor[0, 1, 1]
    c_yxy = chi_tensor[1, 0, 1]
    
    print(f"\nNon-zero components check:")
    print(f"  chi_xyy: {c_xyy:.2e}")
    print(f"  chi_yxy: {c_yxy:.2e}")
    print(f"  Antisymmetry chi_xyy = -chi_yxy check: {c_xyy + c_yxy:.2e}")

    if np.any(np.abs(chi_tensor) > 1e-5):
        print("\nRESULT: Electric NLAHE is NON-ZERO, as expected.")
    else:
        print("\nRESULT: Electric NLAHE is zero (check mu_eff or perturbation magnitude).")

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