import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.model import McCannSystem
from model.analysis import get_electric_shift, get_kmesh
from model.config import GAMMA0, GAMMA1, GAMMA2, VALLEY_IDX, DELTA, N, K_MAX

# Parameters for integration
T_real = 20
kB = 8.617e-5
T_eff = (kB * T_real) / GAMMA0
mu_eff = 0.06 / GAMMA0

def main():
    print("Checking chi_tensor ratios in ISOTROPIC limit (gamma3=0)...")
    # Set gamma3=0 explicitly
    system = McCannSystem(GAMMA0, GAMMA1, VALLEY_IDX[0], DELTA, N, gamma2=GAMMA2, gamma3=0.0)
    
    n_pts = 200
    chi = get_electric_shift(system, 0, K_MAX, n_pts, T_eff, mu_eff)
    
    c_100 = chi[1, 0, 0]
    c_111 = chi[1, 1, 1]
    c_010 = chi[0, 1, 0]
    c_001 = chi[0, 0, 1]
    
    print(f"\nZero components (Integrals of odd functions):")
    print(f"chi[1,0,0] (yxx) = {c_100:.2e}")
    print(f"chi[1,1,1] (yyy) = {c_111:.2e}")
    print(f"chi[0,1,0] (xyx) = {c_010:.2e}")
    print(f"chi[0,0,1] (xxy) = {c_001:.2e}")
    
    if abs(c_111) > 1e-25:
        print(f"\nRatios:")
        print(f"chi[1,0,0] / chi[1,1,1] = {c_100/c_111:.4f} (Expect 2.0 in perfect isotropic limit)")
        print(f"chi[0,1,0] / chi[1,1,1] = {c_010/c_111:.4f} (Expect -0.5 in perfect isotropic limit)")

    # In isotropic limit, ALL components should be zero
    print(f"\nDiagonal component (should be zero in isotropic limit):")
    print(f"chi[0,0,0] (xxx) = {chi[0,0,0]:.2e}")

if __name__ == "__main__":
    main()
