import sys
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.model import McCannSystem
from model.analysis import get_electric_shift
from model.config import GAMMA0, GAMMA1, GAMMA2, GAMMA3, GAMMA4, DELTA, N

def study_kmax_mu():
    system = McCannSystem(GAMMA0, GAMMA1, 1, DELTA, N, GAMMA2, GAMMA3, GAMMA4)
    T_eff = 0.001
    band_idx = 0
    n_pts = 300 # High enough to be stable
    
    mu_list = [0.04, 0.06, 0.08, 0.10, 0.12]
    k_max_range = np.linspace(0.05, 0.3, 26)
    
    print(f"{'mu':>6} | {'Best K_MAX':>10} | {'xxx Value':>12}")
    print("-" * 35)
    
    for mu in mu_list:
        values = []
        for k_lim in k_max_range:
            chi = get_electric_shift(system, band_idx, k_lim, n_pts, T_eff, mu)
            values.append(chi[0, 0, 0])
        
        # Determine "Best K_MAX" as the first value that is within 1% of the final value
        final_val = values[-1]
        best_k = k_max_range[-1]
        for i in range(len(values)-2, -1, -1):
            if abs(values[i] - final_val) > 0.01 * abs(final_val):
                best_k = k_max_range[i+1]
                break
        
        print(f"{mu:6.2f} | {best_k:10.3f} | {final_val:12.6f}")

if __name__ == "__main__":
    study_kmax_mu()
