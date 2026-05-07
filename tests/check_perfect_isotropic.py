import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.model import McCannSystem
from model.analysis import get_electric_shift
from model.config import GAMMA0, GAMMA1, VALLEY_IDX, DELTA, N, K_MAX

# Parameters for integration
T_real = 20
kB = 8.617e-5
T_eff = (kB * T_real) / GAMMA0
mu_eff = 0.06 / GAMMA0

def main():
    print("Checking chi_tensor ratios in PERFECTLY ISOTROPIC limit (gamma2=0, gamma3=0)...")
    # Set gamma2=0 and gamma3=0
    system = McCannSystem(GAMMA0, GAMMA1, VALLEY_IDX[0], DELTA, N, gamma2=0.0, gamma3=0.0)
    
    n_pts = 200
    chi = get_electric_shift(system, 0, K_MAX, n_pts, T_eff, mu_eff)
    
    c_100 = chi[1, 0, 0]
    c_111 = chi[1, 1, 1]
    c_010 = chi[0, 1, 0]
    
    print(f"\nZero components:")
    print(f"chi[1,0,0] (yxx) = {c_100:.2e}")
    print(f"chi[1,1,1] (yyy) = {c_111:.2e}")
    print(f"chi[0,1,0] (xyx) = {c_010:.2e}")
    
    if abs(c_111) > 1e-25:
        print(f"\nRatios:")
        print(f"chi[1,0,0] / chi[1,1,1] = {c_100/c_111:.4f} (Expect 2.0)")
        print(f"chi[0,1,0] / chi[1,1,1] = {c_010/c_111:.4f} (Expect -0.5 or similar?)")

    print(f"\nDiagonal component (xxx) = {chi[0,0,0]:.2e}")

if __name__ == "__main__":
    main()
