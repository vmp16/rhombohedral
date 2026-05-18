
import numpy as np
from model.model import McCannSystem
from model.analysis import get_electric_NLAHE, get_kmesh, deriv_fermi_distrib
from model.config import GAMMA0, GAMMA1, GAMMA2, GAMMA3, GAMMA4, DELTA, N

def check_sampling_and_components():
    system = McCannSystem(GAMMA0, GAMMA1, 1, DELTA, N, GAMMA2, GAMMA3, GAMMA4)
    T_eff = 0.001
    mu_eff = 0.06
    band_idx = 0
    k_lim = 0.15
    
    # Range around 200 and some multiples of 30
    npts_list = [30, 60, 90, 120, 150, 180, 200, 210]
    
    print(f"{'N_PTS':>6} | {'chi_xyy':>12} | {'chi_yxx':>12}")
    print("-" * 40)
    
    for n_pts in npts_list:
        chi = get_electric_NLAHE(system, band_idx, k_lim, n_pts, T_eff, mu_eff)
        xyy = chi[0, 1, 1]
        yxx = chi[1, 0, 0]
        
        print(f"{n_pts:6d} | {xyy:12.6f} | {yxx:12.2e}")

if __name__ == "__main__":
    check_sampling_and_components()
