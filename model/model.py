import numpy as np

class McCannSystem:
    def __init__(self, gamma0, gamma1, valley_idx, Delta, N, gamma2=0.0, gamma3=0.0, gamma4=0.0):
        """
        Initialize a system following McCann's model for ABC-stacked graphene.

        Parameters:
        -----------
        gamma0 : float
            In-layer hopping
        gamma1 : float
            Nearest-layer vertical hopping
        ksi : int, 1 or -1
            Valley index
        Delta : float
            Gap = 2*Delta
        N : int
            Number of layers in the system
        gamma2 : float, optional
            Next-nearest layer hopping
        gamma3 : float, optional
            Hopping breaking rotational symmetry (trigonal warping)
        gamma4 : float, optional
            Hopping breaking electron-hole symmetry
        """

        self.gamma0 = gamma0
        self.gamma1 = gamma1
        self.valley_idx = valley_idx
        self.Delta = Delta
        self.N = N
        self.gamma2 = gamma2
        self.gamma3 = gamma3
        self.gamma4 = gamma4

    def X_at_k(self, k, phi):
        """
        Evaluate the off-diagonal term of the hamiltonian at the given k-point or k-path, expressed in polar coordinates k*exp(i*phi).

        Parameters:
        -----------
        k : float or array
            module from the polar coordinates of the momentum.
        phi : float
            Angle from the polar coordinates of the momentum.

        Returns:
        --------
        X : float or array
            Value of the off-diagonal term of the hamiltonian.
        """
        # Store the k-path
        self.k = k
        self.phi = phi

        # Build parameters in the expression of the hamiltonian
        v = (np.sqrt(3) / 2) * self.gamma0
        v3 = (np.sqrt(3) / 2) * self.gamma3
        
        pi = v * self.k * np.exp(self.valley_idx * 1j * self.phi)
        pi_dag = v * self.k * np.exp(-self.valley_idx * 1j * self.phi)

        # Leading term (isotropic)
        X0 = (pi ** self.N) / ((-self.gamma1) ** (self.N - 1))
        
        # Trigonal warping term from gamma3 (v3)
        X10 = 0
        if self.gamma3 != 0:
            X10 = (self.N - 1) * v3 * (pi ** (self.N - 2)) * pi_dag / (v * (-self.gamma1) ** (self.N - 2))
            
        # Next-nearest layer coupling from gamma2
        X01 = 0
        if self.gamma2 != 0:
            X01 = (self.N - 2) * self.gamma2 * (-pi / self.gamma1) ** (self.N - 3)

        self.X0 = X0
        self.X10 = X10
        self.X01 = X01
        self.X = X0 + X10 + X01

        return self.X
    
    def add_H3c(self, k):
        """
        Calculate the second-order term H_3c to the system, responsible for breaking the electron-hole symmetry.

        Parameters:
        -----------
        k : float or array
            Module from the polar coordinates of the momentum.

        Returns:
        --------
        h0 : float or array (k-shaped)
            Value of the additional diagonal term.
        """
        # Build the velocities in the expression of the hamiltonian
        v = (np.sqrt(3) / 2) * self.gamma0
        v4 = (np.sqrt(3) / 2) * self.gamma4

        # diagonal terms from gamma4
        h0 = (2 * v * v4 * k**2) / self.gamma1

        return h0

    def get_energy_bands(self, k, phi):
        """
        Calculate the energy bands for a given k-path given in polar coordinates by the module k and angle phi.

        Parameters:
        -----------
        k : float or array
            module from the polar coordinates of the momentum.
        phi : float
            Angle from the polar coordinates of the momentum

        Returns
        -------
        energies : array, float (2, len(k))
            The two energy bands evaluated at the given k-path.
        """

        # Calculate the different terms taking part in the energies
        X = self.X_at_k(k, phi)
        h0 = self.add_H3c(k) # is 0 when gamma4 == 0 and gamma2 == 0
        self.h0 = h0

        # self.hamiltonian = np.array([[h0 + self.Delta, X], [np.conj(X), h0 - self.Delta]])

        energy = np.sqrt(np.abs(X)**2 + self.Delta**2)
        self.energy = energy

        return np.array([h0 + energy, h0 - energy])
    
    def get_eigenstates(self):
        """
        Calculate the normalised eigenstates of the system using the analytical expression. It requires the previous calculation of the energies.

        Returns:
        --------
        psi_p, psi_m : 2D-arrays
            Eigenstates of the 2x2 hamiltonian evaluated at the same k-path than the energies previously calculated.
        """
        if not hasattr(self, 'X'):
            raise AttributeError("The system must be evaluated at a k-path using 'get_energy_bands' before derivation.")
        
        # Define the eigenstates according to their analytical expressions
        psi_p = np.array((self.energy + self.Delta, np.conj(self.X)))
        psi_m = np.array((-self.X, self.energy + self.Delta))
        
        # Normalize the eigenstates
        norm_p = np.sqrt(np.sum(np.abs(psi_p)**2, axis=0))
        norm_m = np.sqrt(np.sum(np.abs(psi_m)**2, axis=0))
        
        psi_p = psi_p / norm_p
        psi_m = psi_m / norm_m

        return psi_p, psi_m

    def derivate_system(self):
        """
        Build the derivative of the hamiltonian in polar coordinates.

        Returns:
        --------
        dH_dk : array
            2x2 matrix with the derivative of each element with respect to the first polar coordinate.
        dH_dphi : array
            2x2 matrix with the derivative of each element with respect to the second polar coordinate.
        """
        if not hasattr(self, 'X'):
            raise AttributeError("The system must be evaluated at a k-path using 'get_energy_bands' before derivation.")

        # Derivatives of X
        dX0_dk = self.N * self.X0 / self.k
        dX0_dphi = 1j * self.valley_idx * self.N * self.X0 / self.k

        dX10_dk = 0
        dX10_dphi = 0
        if self.gamma3 != 0:
            dX10_dk = (self.N - 1) * self.X10 / self.k
            dX10_dphi = 1j * self.valley_idx * (self.N - 3) * self.X10 / self.k

        dX01_dk = 0
        dX01_dphi = 0
        if self.gamma2 != 0:
            dX01_dk = (self.N - 3) * self.X01 / self.k
            dX01_dphi = 1j * self.valley_idx * (self.N - 3) * self.X01 / self.k

        dX_dk = dX0_dk + dX10_dk + dX01_dk
        dX_dphi = dX0_dphi + dX10_dphi + dX01_dphi

        dh0_dk = np.zeros_like(dX_dk)
        if self.gamma4 != 0:
            v = (np.sqrt(3) / 2) * self.gamma0
            v4 = (np.sqrt(3) / 2) * self.gamma4
            dh0_dk = (4 * v * v4 * self.k) / self.gamma1
        
        dH_dk = np.array([[dh0_dk, dX_dk], [np.conj(dX_dk), dh0_dk]])
        dH_dphi = np.array([[np.zeros_like(dX_dk), dX_dphi], [np.conj(dX_dphi), np.zeros_like(dX_dk)]])

        return dH_dk, dH_dphi
    
    def get_hamiltonian(self, k, phi):
        """
        Construct the 2x2 Hamiltonian matrix for a given k and phi.

        Parameters:
        -----------
        k : float or array
            module from the polar coordinates of the momentum.
        phi : float
            Angle from the polar coordinates of the momentum.

        Returns:
        --------
        H : array
            Hamiltonian matrix. If k is an array, returns (2, 2, len(k)).
        """
        X = self.X_at_k(k, phi)
        h0 = self.add_H3c(k)

        # Using np.atleast_1d to handle both scalar and array inputs
        k_arr = np.atleast_1d(k)
        X_arr = np.atleast_1d(X)
        h0_arr = np.atleast_1d(h0)

        H = np.zeros((2, 2, len(k_arr)), dtype=complex)
        H[0, 0, :] = h0_arr + self.Delta
        H[1, 1, :] = h0_arr - self.Delta
        H[0, 1, :] = X_arr
        H[1, 0, :] = np.conj(X_arr)

        if np.isscalar(k):
            return H[:, :, 0]
        return H