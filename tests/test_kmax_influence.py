
import numpy as np
from model.model import McCannSystem
from model.analysis import get_electric_NLAHE
from model.config import GAMMA0, GAMMA1, GAMMA2, GAMMA3, GAMMA4, DELTA, N

def test_kmax_influence():
    system = McCannSystem(GAMMA0, GAMMA1, 1, DELTA, N, GAMMA2, GAMMA3, GAMMA4)
    T_eff = 0.001
    mu_eff = 0.06
    band_idx = 0
    n_pts = 30
    
    k_lims = [0.15, 0.20, 0.25]
    
    print(f"{'K_MAX':>8} | {'N_PTS':>6} | {'chi_yxx (NLAHE)':>15}")
    print("-" * 35)
    
    for k_lim in k_lims:
        chi_nlahe = get_electric_NLAHE(system, band_idx, k_lim, n_pts, T_eff, mu_eff)
        chi_yxx = chi_nlahe[1, 0, 0]
        print(f"{k_lim:8.2f} | {n_pts:6d} | {chi_yxx:15.6e}")

if __name__ == "__main__":
    test_kmax_influence()
