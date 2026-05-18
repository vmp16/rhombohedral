
import numpy as np
from model.model import McCannSystem
from model.analysis import get_electric_NLAHE, get_electric_shift
from model.config import GAMMA0, GAMMA1, GAMMA2, GAMMA3, GAMMA4, DELTA, N

def test_npts_detail():
    system = McCannSystem(GAMMA0, GAMMA1, 1, DELTA, N, GAMMA2, GAMMA3, GAMMA4)
    k_lim = 0.15
    T_eff = 0.001
    mu_eff = 0.06
    band_idx = 0
    
    print(f"{'N_PTS':>6} | {'yxx (NLAHE)':>15} | {'yyy (Shift)':>15}")
    print("-" * 45)
    
    for n_pts in range(20, 41):
        chi_nlahe = get_electric_NLAHE(system, band_idx, k_lim, n_pts, T_eff, mu_eff)
        chi_shift = get_electric_shift(system, band_idx, k_lim, n_pts, T_eff, mu_eff)
        
        yx_nlahe = chi_nlahe[1, 0, 0]
        y_shift = chi_shift[1, 1, 1]
        
        print(f"{n_pts:6d} | {yx_nlahe:15.6e} | {y_shift:15.6e}")

if __name__ == "__main__":
    test_npts_detail()
