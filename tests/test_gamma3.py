import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.model import McCannSystem
from model.analysis import get_electric_NLAHE, get_electric_shift
from model.config import GAMMA0, GAMMA1, GAMMA2, GAMMA3, GAMMA4, DELTA, N

def test_gamma3():
    # system with gamma3
    sys_w = McCannSystem(GAMMA0, GAMMA1, 1, DELTA, N, GAMMA2, 0.1, GAMMA4)
    # system without gamma3
    sys_wo = McCannSystem(GAMMA0, GAMMA1, 1, DELTA, N, GAMMA2, 0.0, GAMMA4)
    
    k_lim = 0.15
    T_eff = 0.001
    mu_eff = 0.06
    band_idx = 0
    n_pts = 61
    
    chi_w = get_electric_NLAHE(sys_w, band_idx, k_lim, n_pts, T_eff, mu_eff)
    chi_wo = get_electric_NLAHE(sys_wo, band_idx, k_lim, n_pts, T_eff, mu_eff)
    
    print("NLAHE with gamma3=0.1:")
    print(chi_w)
    print("\nNLAHE with gamma3=0.0:")
    print(chi_wo)

if __name__ == "__main__":
    test_gamma3()
