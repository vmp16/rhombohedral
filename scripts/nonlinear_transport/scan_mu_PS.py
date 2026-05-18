import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from model.model import McCannSystem
from model.analysis import get_electric_shift
from model.config import GAMMA0, GAMMA1, GAMMA2, GAMMA3, VALLEY_IDX, DELTAS, N, N_PTS, K_MAX, T_eff as T

mu_lim = 0.35 # 3 * np.max(np.abs(DELTAS)) / GAMMA0

def main(custom_deltas=None, ylim=None, out_name='scan_mu_PS.png'):
    global DELTAS
    if custom_deltas is not None:
        DELTAS = custom_deltas
    
    mu_vals = np.linspace(-mu_lim, mu_lim, 150)
    sigma_xxx_vals = []

    for mu in mu_vals:
        print(f"Progress: {100 * (np.where(mu_vals == mu)[0][0] + 1) / len(mu_vals):.1f}%", end='\r')

        sigma_tensor_list = []
        for xi, delta_K in zip(VALLEY_IDX, DELTAS):
            # Selecting the pertinent band comparing the Fermi level and the gap
            # This forces sigma=0 whenever mu lies in the gap
            sigma_K_up = np.zeros((2, 2, 2))
            if mu > abs(delta_K[0]):
                system_up = McCannSystem(GAMMA0, GAMMA1, xi, delta_K[0], N, gamma2=GAMMA2, gamma3=GAMMA3)
                sigma_K_up = get_electric_shift(system_up, 0, K_MAX, N_PTS, T, mu)
            elif mu < -abs(delta_K[0]):
                system_up = McCannSystem(GAMMA0, GAMMA1, xi, delta_K[0], N, gamma2=GAMMA2, gamma3=GAMMA3)
                sigma_K_up = get_electric_shift(system_up, 1, K_MAX, N_PTS, T, mu)

            sigma_K_dn = np.zeros((2, 2, 2))
            if mu > abs(delta_K[1]):
                system_dn = McCannSystem(GAMMA0, GAMMA1, xi, delta_K[1], N, gamma2=GAMMA2, gamma3=GAMMA3)
                sigma_K_dn = get_electric_shift(system_dn, 0, K_MAX, N_PTS, T, mu)
            elif mu < -abs(delta_K[1]):
                system_dn = McCannSystem(GAMMA0, GAMMA1, xi, delta_K[1], N, gamma2=GAMMA2, gamma3=GAMMA3)
                sigma_K_dn = get_electric_shift(system_dn, 1, K_MAX, N_PTS, T, mu)

            sigma_tensor_list.append(sigma_K_up + sigma_K_dn)

        sigma_tensor_tot = np.sum(sigma_tensor_list, axis=0)
        sigma_xxx_vals.append(sigma_tensor_tot[0, 0, 0])

    plt.figure(figsize=(8,7))
    plt.rcParams.update({'font.size': 14})

    # Shaded regions for gaps
    colors = ['red', 'blue', 'green', 'orange']
    c_idx = 0
    for i, xi in enumerate(VALLEY_IDX):
        delta_K = DELTAS[i]
        plt.axvspan(-abs(delta_K[0]), abs(delta_K[0]), color=colors[c_idx], alpha=0.1, label=f'Gap K={xi} $\\uparrow$')
        c_idx += 1
        plt.axvspan(-abs(delta_K[1]), abs(delta_K[1]), color=colors[c_idx], alpha=0.1, label=f'Gap K={xi} $\\downarrow$')
        c_idx += 1

    plt.scatter(mu_vals, sigma_xxx_vals, marker='o', s=10, color='black', zorder=5)
    
    plt.xlabel(r"$\mu / \gamma_0$")
    plt.ylabel(r"$\sigma_{xxx}$")
    
    title = f"$\Delta_{{K\\uparrow}}={DELTAS[0,0]}, \Delta_{{K\\downarrow}}={DELTAS[0,1]}, \Delta_{{K'\\uparrow}}={DELTAS[1,0]}, \Delta_{{K'\\downarrow}}={DELTAS[1,1]}$"
    plt.title(title)
    
    if ylim:
        plt.ylim(ylim)
    
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()

    outdir = project_root / 'figures' / out_name
    plt.savefig(outdir)
    print(f"\nSaved plot to {outdir}")
    # plt.show() # Commented for non-interactive run

if __name__ == "__main__":
    main()