
import numpy as np
from model.model import McCannSystem
from model.analysis import get_electric_NLAHE, get_electric_shift
from model.config import GAMMA0, GAMMA1, GAMMA2, GAMMA3, GAMMA4, DELTA, N

def test_npts_200_vs_30():
    system = McCannSystem(GAMMA0, GAMMA1, 1, DELTA, N, GAMMA2, GAMMA3, GAMMA4)
    k_lim = 0.15
    T_eff = 0.001
    mu_eff = 0.06
    band_idx = 0
    
    npts_list = [30, 60, 90, 120, 150, 180, 200, 210, 240]
    
    print(f"{'N_PTS':>6} | {'chi_yyy (NLAHE)':>15} | {'chi_yxx (NLAHE)':>15}")
    print("-" * 50)
    
    for n_pts in npts_list:
        chi_nlahe = get_electric_NLAHE(system, band_idx, k_lim, n_pts, T_eff, mu_eff)
        chi_yyy = chi_nlahe[1, 1, 1]
        chi_yxx = chi_nlahe[1, 0, 0]
        print(f"{n_pts:6d} | {chi_yyy:15.6e} | {chi_yxx:15.6e}")

if __name__ == "__main__":
    test_npts_200_vs_30()
