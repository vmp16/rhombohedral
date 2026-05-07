import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.model import McCannSystem
from model.analysis import get_electric_shift, get_electric_NLAHE
from model.config import GAMMA0, GAMMA1, VALLEY_IDX, DELTAS, N, K_MAX_INT as K_MAX, N_LINEAR

# Parameters for integration
T_real = 20
kB = 8.617e-5
T_eff = (kB * T_real) / GAMMA0
mu_eff = 0.06 / GAMMA0

class BrokenMcCannSystem(McCannSystem):
    def set_perturbation(self, delta_X):
        self.delta_X = delta_X

    def X_at_k(self, k, phi):
        base_X = super().X_at_k(k, phi)
        self.X = base_X + self.delta_X
        return self.X

def run_scan(re_vals, im_vals):
    results_nlahe = []
    results_shift = []
    
    system = BrokenMcCannSystem(GAMMA0, GAMMA1, VALLEY_IDX[0], DELTAS[0, 1], N)
    
    for re in re_vals:
        for im in im_vals:
            C = re + 1j * im
            system.set_perturbation(C)
            
            # Use smaller N_pts for speed in scan if necessary, but N_LINEAR is from config
            chi_nlahe = get_electric_NLAHE(system, 0, K_MAX, N_LINEAR, T_eff, mu_eff)
            chi_shift = get_electric_shift(system, 0, K_MAX, N_LINEAR, T_eff, mu_eff)
            
            results_nlahe.append((re, im, chi_nlahe))
            results_shift.append((re, im, chi_shift))
            
    return results_nlahe, results_shift

def main():
    # Define ranges for C = Re + i*Im
    re_vals = np.linspace(0.0, 0.5, 5)
    im_vals = np.linspace(0.0, 0.5, 5)
    
    print(f"Running scan with {len(re_vals)}x{len(im_vals)} points...")
    res_nlahe, res_shift = run_scan(re_vals, im_vals)
    
    # Analyze NLAHE [0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,1,1] etc.
    # The user said [0,0,:] and [1,1,:] should be 0.
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # NLAHE chi_{000} vs Im(C) for different Re(C)
    ax = axes[0, 0]
    for re in re_vals:
        im_plot = []
        val_plot = []
        for r, i, chi in res_nlahe:
            if r == re:
                im_plot.append(i)
                val_plot.append(chi[0, 0, 0])
        ax.plot(im_plot, val_plot, 'o-', label=f"Re(C)={re:.1f}")
    ax.set_title(r"NLAHE $\chi_{xxx}$ vs Im(C)")
    ax.set_xlabel("Im(C)")
    ax.legend()

    # NLAHE chi_{011} vs Im(C)
    ax = axes[0, 1]
    for re in re_vals:
        im_plot = []
        val_plot = []
        for r, i, chi in res_nlahe:
            if r == re:
                im_plot.append(i)
                val_plot.append(chi[0, 1, 1])
        ax.plot(im_plot, val_plot, 'o-', label=f"Re(C)={re:.1f}")
    ax.set_title(r"NLAHE $\chi_{xyy}$ vs Im(C)")
    ax.set_xlabel("Im(C)")
    ax.legend()

    # Shift chi_{000} vs Im(C)
    ax = axes[1, 0]
    for re in re_vals:
        im_plot = []
        val_plot = []
        for r, i, chi in res_shift:
            if r == re:
                im_plot.append(i)
                val_plot.append(chi[0, 0, 0])
        ax.plot(im_plot, val_plot, 'o-', label=f"Re(C)={re:.1f}")
    ax.set_title(r"Shift $\chi_{xxx}$ vs Im(C)")
    ax.set_xlabel("Im(C)")
    ax.legend()

    # Shift chi_{111} vs Im(C)
    ax = axes[1, 1]
    for re in re_vals:
        im_plot = []
        val_plot = []
        for r, i, chi in res_shift:
            if r == re:
                im_plot.append(i)
                val_plot.append(chi[1, 1, 1])
        ax.plot(im_plot, val_plot, 'o-', label=f"Re(C)={re:.1f}")
    ax.set_title(r"Shift $\chi_{yyy}$ vs Im(C)")
    ax.set_xlabel("Im(C)")
    ax.legend()

    plt.tight_layout()
    plt.savefig("tests/influence_of_C_scan.png")
    print("Saved plot to tests/influence_of_C_scan.png")

    # Print a table for some values
    print("\nTable of selected NLAHE components:")
    print(f"{'Re(C)':<8} {'Im(C)':<8} {'chi_000':<12} {'chi_011':<12} {'chi_010':<12}")
    for r, i, chi in res_nlahe:
        if i in [0.0, 0.25, 0.5] and r in [0.0, 0.25, 0.5]:
             print(f"{r:<8.2f} {i:<8.2f} {chi[0,0,0]:<12.2e} {chi[0,1,1]:<12.2e} {chi[0,1,0]:<12.2e}")

    print("\nTable of selected Shift components:")
    print(f"{'Re(C)':<8} {'Im(C)':<8} {'chi_000':<12} {'chi_111':<12} {'chi_011':<12}")
    for r, i, chi in res_shift:
        if i in [0.0, 0.25, 0.5] and r in [0.0, 0.25, 0.5]:
             print(f"{r:<8.2f} {i:<8.2f} {chi[0,0,0]:<12.2e} {chi[1,1,1]:<12.2e} {chi[0,1,1]:<12.2e}")

if __name__ == "__main__":
    main()
