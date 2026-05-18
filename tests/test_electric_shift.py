import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.model import McCannSystem
from model.analysis import get_electric_shift
from model.config import GAMMA0, GAMMA1, GAMMA2, GAMMA3, N, N_LINEAR, K_MAX_INT as K_MAX, DELTAS, VALLEY_IDX

# -------------------- CONFIGURATION ------------------------

# General constants
kB = 8.617e-5       # Boltzmann constant in eV/K

T_real = 20         # temperature [K]
T_eff = (kB * T_real) / GAMMA0

mu_eff = -0.01 / GAMMA0       # Fermi level [GAMMA0 units]

# -----------------------------------------------------------

def main():
    print(10*"="+" CALCULATING ELECTRIC POSITIONAL SHIFT "+10*"=")
    system = McCannSystem(GAMMA0, GAMMA1, VALLEY_IDX[0], DELTAS[0, 1], N, gamma2=GAMMA2, gamma3=GAMMA3)

    chi_tensor = get_electric_shift(system, 1, K_MAX, N_LINEAR, T_eff, mu_eff)

    print(f"tensor shape = {chi_tensor.shape}")
    print(chi_tensor)

    # Check if any element in chi_tensor is non-zero
    threshold = 1e-10
    non_zero_indices = np.argwhere(np.abs(chi_tensor) > threshold)

    if non_zero_indices.size > 0:
        print(f"Found {non_zero_indices.shape[0]} non-zero elements in chi_tensor:")
        for idx in non_zero_indices:
            val = chi_tensor[tuple(idx)]
            print(f"Element {tuple(idx)}: {val}")
    else:
        print("No non-zero elements found in chi_tensor.")


if __name__ == "__main__":
    main()