import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.model import McCannSystem
from model.analysis import calculate_berry_integral
from model.config import GAMMA0, GAMMA1, VALLEY_IDX, DELTA, N, N_LINEAR, K_MAX_INT as K_MAX

def main():
    # Build the system
    print(f"Calculating for valley index = {VALLEY_IDX}")
    system = McCannSystem(GAMMA0, GAMMA1, VALLEY_IDX, DELTA, N)

    integrals = calculate_berry_integral(system, K_MAX, N_LINEAR)
    berry_integral_pos = integrals[0]

    print(f"Integration Area: Circle of radius k = {K_MAX}")
    print(f"Positive Band Integral (Berry Phase): {berry_integral_pos:.6f}")
    print(f"Local Chern contribution (Pos Band):  {berry_integral_pos / (2 * np.pi):.6f}")


if __name__ == "__main__":
    main()