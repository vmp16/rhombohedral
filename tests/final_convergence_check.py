
import numpy as np
from model.model import McCannSystem
from model.analysis import get_electric_NLAHE, get_electric_shift
from model.config import GAMMA0, GAMMA1, GAMMA2, GAMMA3, GAMMA4, DELTA, N

def final_convergence_check():
    system = McCannSystem(GAMMA0, GAMMA1, 1, DELTA, N, GAMMA2, GAMMA3, GAMMA4)
    
    T_eff = 0.001
    mu_eff = 0.06
    band_idx = 0
    k_lim = 0.15
    
    # Check non-zero components convergence with N_PTS
    n_pts_list = [30, 60, 90, 120, 150, 210, 300, 450, 600]
    
    print(f"{'N_PTS':>6} | {'chi_xxx (Shift)':>15} | {'chi_xyy (Shift)':>15} | {'dx':>10}")
    print("-" * 60)
    
    for n_pts in n_pts_list:
        chi_shift = get_electric_shift(system, band_idx, k_lim, n_pts, T_eff, mu_eff)
        xxx = chi_shift[0, 0, 0]
        xyy = chi_shift[0, 1, 1]
        dx = 2 * k_lim / (n_pts - 1)
        print(f"{n_pts:6d} | {xxx:15.6f} | {xyy:15.6f} | {dx:10.6f}")

if __name__ == "__main__":
    final_convergence_check()
