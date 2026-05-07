import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.model import McCannSystem
from model.analysis import get_electric_NLAHE, get_electric_shift
from model.config import GAMMA0, GAMMA1, VALLEY_IDX, DELTAS, N, K_MAX_INT as K_MAX, N_LINEAR

class BrokenMcCannSystem(McCannSystem):
    def set_perturbation(self, delta_X):
        self.delta_X = delta_X
    def X_at_k(self, k, phi):
        base_X = super().X_at_k(k, phi)
        self.X = base_X + self.delta_X
        return self.X

def main():
    system = BrokenMcCannSystem(GAMMA0, GAMMA1, VALLEY_IDX[0], DELTAS[0, 1], N)
    T_eff = 0.01
    mu_eff = 0.0
    
    # Test with the problematic constant mentioned by the user
    C = 0.5 + 0.05j
    system.set_perturbation(C)
    
    chi_nlahe = get_electric_NLAHE(system, 0, K_MAX, N_LINEAR, T_eff, mu_eff)
    chi_shift = get_electric_shift(system, 0, K_MAX, N_LINEAR, T_eff, mu_eff)
    
    print(f"Testing with C = {C}")
    print("\n--- get_electric_NLAHE ---")
    print(f"chi_xxx (0,0,0) = {chi_nlahe[0,0,0]:.2e}")
    print(f"chi_xxy (0,0,1) = {chi_nlahe[0,0,1]:.2e}")
    print(f"chi_yyx (1,1,0) = {chi_nlahe[1,1,0]:.2e}")
    print(f"chi_yyy (1,1,1) = {chi_nlahe[1,1,1]:.2e}")
    
    print("\n--- get_electric_shift ---")
    print(f"chi_xxx (0,0,0) = {chi_shift[0,0,0]:.2e}")
    print(f"chi_yyy (1,1,1) = {chi_shift[1,1,1]:.2e}")

if __name__ == "__main__":
    main()
