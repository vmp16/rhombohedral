import numpy as np

# Physical parameters for the McCann Model
GAMMA0 = 1.0     # In-layer hopping
GAMMA1 = 0.1     # Nearest-layer vertical hopping
GAMMA4 = 0.05    # Second-order hopping
VALLEY_IDX = 1   # Valley index (1 or -1)
N = 5            # Number of layers
DELTA = 0.05     # Gap parameter (Gap = 2*Delta)

# Numerical/Plotting Defaults
N_PTS = 200
K_MIN = -0.2
K_MAX = 0.2

# Integration Defaults
N_LINEAR = 400
K_MAX_INT = 1.0

# Spin-valley polarization parameters
DELTA1UP = 0.1
DELTA1DN = 0.0
DELTA2UP = 0.0
DELTA2DN = -0.1

DELTAS = np.array([[DELTA1UP, DELTA1DN],
                   [DELTA2UP, DELTA2DN]])

VALLEY_IDX = [1,-1]

# Amplitude of the Spin-Orbit coupling (SOC)
LAMBDA = 1e-3