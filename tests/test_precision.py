
import numpy as np
from model.model import McCannSystem
from model.analysis import get_electric_NLAHE, get_electric_shift
from model.config import GAMMA0, GAMMA1, GAMMA2, GAMMA3, GAMMA4, DELTA, N

def test_precision():
    system = McCannSystem(GAMMA0, GAMMA1, 1, DELTA, N, GAMMA2, GAMMA3, GAMMA4)
    k_lim = 0.15
    T_eff = 0.001
    mu_eff = 0.06
    band_idx = 0
    
    n_pts = 210
    chi_nlahe = get_electric_NLAHE(system, band_idx, k_lim, n_pts, T_eff, mu_eff)
    chi_shift = get_electric_shift(system, band_idx, k_lim, n_pts, T_eff, mu_eff)
    
    print(f"N_PTS = {n_pts}")
    print("NLAHE Tensor:")
    print(chi_nlahe)
    print("\nShift Tensor:")
    print(chi_shift)

if __name__ == "__main__":
    test_precision()
