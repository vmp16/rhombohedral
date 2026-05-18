
import numpy as np
from model.model import McCannSystem
from model.config import GAMMA0, GAMMA1, GAMMA2, GAMMA3, GAMMA4, DELTA, N

def check_k_fs():
    system = McCannSystem(GAMMA0, GAMMA1, 1, DELTA, N, GAMMA2, GAMMA3, GAMMA4)
    k_vals = np.linspace(0, 0.5, 100)
    phi = 0
    energies = system.get_energy_bands(k_vals, phi)
    
    print(f"{'k':>10} | {'E_upper':>10}")
    print("-" * 25)
    for i in range(len(k_vals)):
        print(f"{k_vals[i]:10.3f} | {energies[0, i]:10.6f}")

if __name__ == "__main__":
    check_k_fs()
