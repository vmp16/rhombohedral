import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from model.model import McCannSystem
from model.analysis import get_electric_shift
from model.config import GAMMA0, GAMMA1, GAMMA2, GAMMA3, VALLEY_IDX, DELTAS, N, N_PTS, K_MAX, T_eff as T, mu_eff as mu

SHOW_PLOTS = False

def main():
    print(12*'=' + ' CALCULATING THE CONDUCTIVITY WITH THE POSITIONAL SHIFT ' + 12*'=')

    sigma_tensor_list = []
    for xi, delta_K in zip(VALLEY_IDX, DELTAS):        
        print(f"\nCalculating for valley = {xi}...")

        # Selecting the pertinent band comparing the Fermi level and the gap
        # This forces sigma=0 whenever mu lies in the gap
        sigma_K_up = np.zeros((2, 2, 2))
        if mu > abs(delta_K[0]):
            system_up = McCannSystem(GAMMA0, GAMMA1, xi, delta_K[0], N, gamma2=GAMMA2, gamma3=GAMMA3)
            sigma_K_up = get_electric_shift(system_up, 0, K_MAX, N_PTS, T, mu)
        elif mu < -abs(delta_K[0]):
            system_up = McCannSystem(GAMMA0, GAMMA1, xi, delta_K[0], N, gamma2=GAMMA2, gamma3=GAMMA3)
            sigma_K_up = get_electric_shift(system_up, 1, K_MAX, N_PTS, T, mu)
        else:
            print(f"The Fermi level lies in the gap for the valley {xi}, spin up.")

        sigma_K_dn = np.zeros((2, 2, 2))
        if mu > abs(delta_K[1]):
            system_dn = McCannSystem(GAMMA0, GAMMA1, xi, delta_K[1], N, gamma2=GAMMA2, gamma3=GAMMA3)
            sigma_K_dn = get_electric_shift(system_dn, 0, K_MAX, N_PTS, T, mu)
        elif mu < -abs(delta_K[1]):
            system_dn = McCannSystem(GAMMA0, GAMMA1, xi, delta_K[1], N, gamma2=GAMMA2, gamma3=GAMMA3)
            sigma_K_dn = get_electric_shift(system_dn, 1, K_MAX, N_PTS, T, mu)
        else:
            print(f"The Fermi level lies in the gap for the valley {xi}, spin down.")

        # IF WE WANTED TO LET FERMI'S DERIVATIVE HANDLE THE SELECTION:

        # sigma_K_up = sum(get_electric_shift(system_up, b_idx, K_MAX, N_PTS, T, mu) for b_idx in (0, 1))
        # sigma_K_dn = sum(get_electric_shift(system_dn, b_idx, K_MAX, N_PTS, T, mu) for b_idx in (0, 1))

        print("sigma_up:")
        print(sigma_K_up)
        print("sigma_dn:")
        print(sigma_K_dn)

        sigma_tensor_list.append(sigma_K_up + sigma_K_dn)

    sigma_tensor_tot = np.sum(sigma_tensor_list, axis=0)

    print(5*'=' + " RESULTS " + 5*'=')

    print("TOTAL CONDUCTIVITY:")
    print(sigma_tensor_tot)

    print("\nExpected non-zero components:")
    print(f"sigma_xxx = {sigma_tensor_tot[0, 0, 0]}")
    print(f"sigma_xyy = {sigma_tensor_tot[0, 1, 1]}")
    print(f"sigma_yxy = sigma_yyx = {sigma_tensor_tot[1, 1, 0]}")

if __name__ == "__main__":
    main()