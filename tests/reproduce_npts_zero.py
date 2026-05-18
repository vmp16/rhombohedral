
import numpy as np
from model.model import McCannSystem
from model.analysis import get_electric_NLAHE, get_electric_shift
from model.config import GAMMA0, GAMMA1, GAMMA2, GAMMA3, GAMMA4, DELTA, N

def test_npts():
    system = McCannSystem(GAMMA0, GAMMA1, 1, DELTA, N, GAMMA2, GAMMA3, GAMMA4)
    
    k_lim = 0.15
    T_eff = 0.001
    mu_eff = 0.06
    band_idx = 0
    
    npts_list = [29, 30, 31, 59, 60, 61, 89, 90, 91, 119, 120, 121]
    
    print(f"{'N_PTS':>6} | {'chi_yyy (NLAHE)':>15} | {'chi_yxx (NLAHE)':>15} | {'chi_yyy (Shift)':>15} | {'chi_yxx (Shift)':>15}")
    print("-" * 80)
    
    n_pts = 31
    chi_nlahe = get_electric_NLAHE(system, band_idx, k_lim, n_pts, T_eff, mu_eff)
    print("\nFull chi_nlahe tensor for N_PTS=31:")
    print(chi_nlahe)
    
    chi_shift = get_electric_shift(system, band_idx, k_lim, n_pts, T_eff, mu_eff)
    print("\nFull chi_shift tensor for N_PTS=31:")
    print(chi_shift)

if __name__ == "__main__":
    test_npts()
