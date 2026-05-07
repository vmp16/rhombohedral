import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.model import McCannSystem
from model.analysis import velocity_operator
from model.config import GAMMA0, GAMMA1, VALLEY_IDX, DELTAS, N

class BrokenMcCannSystem(McCannSystem):
    def set_perturbation(self, delta_X):
        self.delta_X = delta_X

    def X_at_k(self, k, phi):
        base_X = super().X_at_k(k, phi)
        self.X = base_X + self.delta_X
        return self.X

def main():
    system = BrokenMcCannSystem(GAMMA0, GAMMA1, VALLEY_IDX[0], DELTAS[0, 1], N)
    
    k = 0.1
    phi = 0.0 # along kx
    
    C_list = [0.0, 0.5, 0.5j, 0.5+0.5j]
    
    print(f"{'C':<15} {'v_x(k, phi=0)':<15} {'v_x(k, phi=pi)':<15} {'Is Odd?':<10}")
    
    for C in C_list:
        system.set_perturbation(C)
        
        # Eval at (k, 0)
        system.get_energy_bands(np.array([k]), np.array([0.0]))
        vx_0, _ = velocity_operator(system, 0, 0)
        
        # Eval at (k, pi) -> -k
        system.get_energy_bands(np.array([k]), np.array([np.pi]))
        vx_pi, _ = velocity_operator(system, 0, 0)
        
        is_odd = np.allclose(vx_0, -vx_pi)
        print(f"{str(C):<15} {vx_0[0]:<15.4e} {vx_pi[0]:<15.4e} {str(is_odd):<10}")

if __name__ == "__main__":
    main()
