import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.model import McCannSystem
from model.analysis import get_G
from model.config import GAMMA0, GAMMA1, GAMMA2, GAMMA3, N

def get_G_at_point(k, phi, xi, delta):
    system = McCannSystem(GAMMA0, GAMMA1, xi, delta, N, gamma2=GAMMA2, gamma3=GAMMA3)
    # get_G uses system.get_energy_bands internally via h0 and energy arrays, 
    # so we must evaluate the system first.
    system.get_energy_bands(np.array([k]), np.array([phi]))
    
    # get_G expects evaluated system
    G_tensor = get_G(system, band_idx=1) # Valence band
    return G_tensor[:, :, 0] # Return the 2x2 matrix at the single k-point

def main():
    k = 0.1
    phi = np.pi / 4
    delta = 0.1
    
    G_pos = get_G_at_point(k, phi, 1, delta)
    G_neg = get_G_at_point(k, phi, -1, delta)
    
    print(f"Point: k={k}, phi={phi}")
    print("\nValley xi=1 G tensor:")
    print(G_pos)
    print("\nValley xi=-1 G tensor:")
    print(G_neg)
    
    print("\nDifference:")
    print(np.abs(G_pos - G_neg))

if __name__ == "__main__":
    main()
