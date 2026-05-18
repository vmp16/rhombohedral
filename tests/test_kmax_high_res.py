
import numpy as np
from model.model import McCannSystem
from model.analysis import get_electric_NLAHE, get_electric_shift
from model.config import GAMMA0, GAMMA1, GAMMA2, GAMMA3, GAMMA4, DELTA, N

def test_kmax_high_res():
    system = McCannSystem(GAMMA0, GAMMA1, 1, DELTA, N, GAMMA2, GAMMA3, GAMMA4)
    # No perturbation needed for this stability check, but let's use one to see non-zero values
    # system = BrokenMcCannSystem(...) # I'll just use the regular one and look at xyy
    
    T_eff = 0.001
    mu_eff = 0.06
    band_idx = 0
    
    # Try a range of K_MAX with high resolution to keep dx small
    k_max_range = [0.15, 0.20, 0.25, 0.30]
    # We want dx = 0.3 / 1000 approx 0.0003
    # n_pts = 2 * k_lim / dx + 1
    
    dx_target = 0.0003
    
    print(f"{'K_MAX':>8} | {'N_PTS':>6} | {'dx':>10} | {'xyy (NLAHE)':>15}")
    print("-" * 50)
    
    for k_lim in k_max_range:
        n_pts = int(2 * k_lim / dx_target) + 1
        if n_pts % 2 == 0: n_pts += 1
        
        chi_nlahe = get_electric_NLAHE(system, band_idx, k_lim, n_pts, T_eff, mu_eff)
        chi_shift = get_electric_shift(system, band_idx, k_lim, n_pts, T_eff, mu_eff)
        print(f"\nFull NLAHE tensor for K_MAX={k_lim}, N_PTS={n_pts}:")
        print(chi_nlahe)
        print(f"\nFull Shift tensor for K_MAX={k_lim}, N_PTS={n_pts}:")
        print(chi_shift)
        break

if __name__ == "__main__":
    test_kmax_high_res()
