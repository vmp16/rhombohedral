import numpy as np
import matplotlib.pyplot as plt

# Parameters
gamma0 = 1.0
gamma1 = 0.1
valley_idx = 1
N = 5

# Gap = 2*Delta
Delta = 0.1

# Build parameters in the expression of the hamiltonian
v = (np.sqrt(3) / 2) * gamma0

# Map k-space to consider in polar coordinates (p, phi)
# For the x direction, phi=0
phi = 0
k = np.linspace(-0.2, 0.2, 100)

X = ((v * k * np.exp(valley_idx * 1j * phi)) ** N) / ((-gamma1) ** (N-1))

energy = np.sqrt(np.abs(X)**2 + Delta)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(k, energy, label='Positive band', color='blue')
plt.plot(k, -energy, label='Negative band', color='red')

plt.xlabel(r'$k$')
plt.ylabel('Energy')
plt.title(f'Bands for N={N} (McCann Model)')
plt.legend()
plt.grid(True)
plt.show()