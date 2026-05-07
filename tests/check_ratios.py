import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.model import McCannSystem
from model.analysis import get_electric_shift
from model.config import GAMMA0, GAMMA1, GAMMA2, GAMMA3, VALLEY_IDX, DELTA, N, N_PTS, K_MAX

# Parameters for integration
T_real = 20
kB = 8.617e-5
T_eff = (kB * T_real) / GAMMA0
mu_eff = 0.06 / GAMMA0

def main():
    print("Checking chi_tensor ratios...")
    system = McCannSystem(GAMMA0, GAMMA1, VALLEY_IDX[0], DELTA, N, gamma2=GAMMA2, gamma3=GAMMA3)
    
    # We use a higher N_PTS to see if ratios persist
    n_pts = 200
    chi = get_electric_shift(system, 0, K_MAX, n_pts, T_eff, mu_eff)
    
    # Indices for components that should be zero (Mx symmetry: odd number of y indices)
    # yyy: [1, 1, 1]
    # yxx: [1, 0, 0]
    # xyx: [0, 1, 0]
    # xxy: [0, 0, 1]
    
    c_100 = chi[1, 0, 0]
    c_111 = chi[1, 1, 1]
    c_010 = chi[0, 1, 0]
    c_001 = chi[0, 0, 1]
    
    print(f"\nZero components (expected ~0):")
    print(f"chi[1,0,0] (yxx) = {c_100:.2e}")
    print(f"chi[1,1,1] (yyy) = {c_111:.2e}")
    print(f"chi[0,1,0] (xyx) = {c_010:.2e}")
    print(f"chi[0,0,1] (xxy) = {c_001:.2e}")
    
    if abs(c_111) > 1e-25:
        print(f"\nRatios:")
        print(f"chi[1,0,0] / chi[1,1,1] = {c_100/c_111:.4f} (Expected 2?)")
        print(f"chi[1,0,0] / chi[0,1,0] = {c_100/c_010:.4f} (Expected -4 or 4?)")
        print(f"chi[0,1,0] / chi[0,0,1] = {c_010/c_001:.4f} (Expected 1)")
    
    # Indices for components that should be NON-ZERO (even number of y indices)
    # xxx: [0, 0, 0]
    # xyy: [0, 1, 1]
    # yxy: [1, 0, 1]
    # yyx: [1, 1, 0]
    
    c_000 = chi[0, 0, 0]
    c_011 = chi[0, 1, 1]
    c_101 = chi[1, 0, 1]
    c_110 = chi[1, 1, 0]
    
    print(f"\nNon-zero components (should satisfy C3 relations):")
    print(f"chi[0,0,0] (xxx) = {c_000:.2e}")
    print(f"chi[0,1,1] (xyy) = {c_011:.2e}")
    print(f"chi[1,0,1] (yxy) = {c_101:.2e}")
    print(f"chi[1,1,0] (yyx) = {c_110:.2e}")
    
    print(f"\nC3 Check for non-zero components (xxx = -xyy = -yxy = -yyx):")
    print(f"xxx + xyy = {c_000 + c_011:.2e}")
    print(f"xxx + yxy = {c_000 + c_101:.2e}")
    print(f"xxx + yyx = {c_000 + c_110:.2e}")

if __name__ == "__main__":
    main()
