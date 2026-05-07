import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.model import McCannSystem
from model.analysis import get_electric_shift
from model.config import GAMMA0, GAMMA1, DELTAS, N, K_MAX_INT, N_LINEAR

# Parameters for integration
T_real = 20
kB = 8.617e-5
T_eff = (kB * T_real) / GAMMA0
mu_eff = 0.0 / GAMMA0 # At the Dirac point for simplicity

class BrokenMcCannSystem(McCannSystem):
    def set_perturbation(self, delta_X):
        self.delta_X = delta_X
    def X_at_k(self, k, phi):
        base_X = super().X_at_k(k, phi)
        self.X = base_X + self.delta_X
        return self.X

def main():
    system = BrokenMcCannSystem(GAMMA0, GAMMA1, 1, 0.05, N)
    
    # Estimate |X0| at k=0.1
    v = (np.sqrt(3) / 2) * GAMMA0
    k_test = 0.1
    X0_mag = (v * k_test)**N / (GAMMA1**(N-1))
    print(f"Typical |X0| at k={k_test}: {X0_mag:.4f}")

    # Scan C_re from 0 to 0.5
    C_vals = np.linspace(0.0, 0.5, 50)
    chi_xxx = []
    
    for C in C_vals:
        system.set_perturbation(C)
        # Using a slightly smaller grid for speed in the scan
        chi = get_electric_shift(system, 0, 0.2, 50, T_eff, mu_eff)
        chi_xxx.append(chi[0, 0, 0])
    
    plt.figure(figsize=(8, 5))
    plt.plot(C_vals, chi_xxx, 'b-', lw=2)
    plt.axvline(X0_mag, color='r', linestyle='--', label=f'|X0| at k=0.1')
    plt.title("Response $\chi_{xxx}$ vs. Constant $C_{re}$")
    plt.xlabel("Constant $C_{re}$")
    plt.ylabel("$\chi_{xxx}$")
    plt.legend()
    plt.grid(True)
    plt.savefig("tests/C_sweet_spot.png")
    
    max_idx = np.argmax(np.abs(chi_xxx))
    print(f"Maximum response at C = {C_vals[max_idx]:.4f}")

if __name__ == "__main__":
    main()
