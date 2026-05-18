
import numpy as np
from model.model import McCannSystem
from model.analysis import get_electric_NLAHE
from model.config import GAMMA0, GAMMA1, GAMMA2, GAMMA3, GAMMA4, DELTA, N

def test_npts_scan():
    system = McCannSystem(GAMMA0, GAMMA1, 1, DELTA, N, GAMMA2, GAMMA3, GAMMA4)
    k_lim = 0.15
    T_eff = 0.001
    mu_eff = 0.06
    band_idx = 0
    
    npts_list = list(range(190, 221))
    
    print(f"{'N_PTS':>6} | {'chi_yxx (NLAHE)':>15}")
    print("-" * 25)
    
    for n_pts in npts_list:
        chi_nlahe = get_electric_NLAHE(system, band_idx, k_lim, n_pts, T_eff, mu_eff)
        yxx = chi_nlahe[1, 0, 0]
        print(f"{n_pts:6d} | {yxx:15.6e}")

if __name__ == "__main__":
    test_npts_scan()
