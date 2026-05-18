
import numpy as np
from model.model import McCannSystem
from model.analysis import get_electric_NLAHE, get_electric_shift
from model.config import GAMMA0, GAMMA1, GAMMA2, GAMMA3, GAMMA4, DELTA, N

class BrokenMcCannSystem(McCannSystem):
    def set_perturbation(self, delta_X):
        self.delta_X = delta_X
    def X_at_k(self, k, phi):
        base_X = super().X_at_k(k, phi)
        self.X = base_X + self.delta_X
        return self.X

def study_convergence():
    system = BrokenMcCannSystem(GAMMA0, GAMMA1, 1, DELTA, N, GAMMA2, GAMMA3, GAMMA4)
    system.set_perturbation(0.01 + 0.02j)
    
    T_eff = 0.001
    mu_eff = 0.06
    band_idx = 0
    
    # N_PTS study
    n_pts_range = np.arange(20, 151, 1)
    y_nlahe_list = []
    yx_nlahe_list = []
    y_shift_list = []
    yx_shift_list = []
    xyy_nlahe_list = []
    
    k_lim = 0.15
    
    print("Studying N_PTS convergence...")
    for n_pts in n_pts_range:
        chi_nlahe = get_electric_NLAHE(system, band_idx, k_lim, n_pts, T_eff, mu_eff)
        chi_shift = get_electric_shift(system, band_idx, k_lim, n_pts, T_eff, mu_eff)
        
        y_nlahe_list.append(chi_nlahe[1, 1, 1])
        yx_nlahe_list.append(chi_nlahe[1, 0, 0])
        xyy_nlahe_list.append(chi_nlahe[0, 1, 1])
        
        y_shift_list.append(chi_shift[1, 1, 1])
        yx_shift_list.append(chi_shift[1, 0, 0])

    # K_MAX study
    k_max_range = np.linspace(0.05, 0.5, 20)
    n_pts_fixed = 121
    k_conv_y_nlahe = []
    k_conv_xyy_nlahe = []
    
    print("Studying K_MAX convergence...")
    for k_lim in k_max_range:
        chi_nlahe = get_electric_NLAHE(system, band_idx, k_lim, n_pts_fixed, T_eff, mu_eff)
        k_conv_y_nlahe.append(chi_nlahe[1, 1, 1])
        k_conv_xyy_nlahe.append(chi_nlahe[0, 1, 1])

    # Since I cannot plot, I will print some key values and trends
    print("\nN_PTS Trend (NLAHE xyy):")
    for i in range(0, len(n_pts_range), 10):
        print(f"N_PTS={n_pts_range[i]:3d}: xyy={xyy_nlahe_list[i]:12.6f}, yxx={yx_nlahe_list[i]:12.6e}")

    print("\nN_PTS multiples of 30 (NLAHE yxx):")
    for n in [30, 60, 90, 120, 150]:
        idx = np.where(n_pts_range == n)[0][0]
        print(f"N_PTS={n:3d}: yxx={yx_nlahe_list[idx]:12.6e}")

    print("\nK_MAX Trend (NLAHE xyy):")
    for i in range(len(k_max_range)):
        print(f"K_MAX={k_max_range[i]:.3f}: xyy={k_conv_xyy_nlahe[i]:12.6f}")

if __name__ == "__main__":
    study_convergence()
