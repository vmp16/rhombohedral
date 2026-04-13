import numpy as np

class McCannSystem:
    def __init__(self, gamma0, gamma1, valley_idx, Delta, N):
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
        """

        self.gamma0 = gamma0
        self.gamma1 = gamma1
        self.valley_idx = valley_idx
        self.Delta = Delta
        self.N = N

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
        
        X = ((v * k * np.exp(self.valley_idx * 1j * phi)) ** self.N) / ((-self.gamma1) ** (self.N-1))
        self.X = X

        return X

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
        energy : array, float (as input k)
            absolute value of the symmetrical energy bands evaluated at the given k-path.
        """

        X = self.X_at_k(k, phi)
        # self.hamiltonian = np.array([[self.Delta, X], [np.conj(X), -self.Delta]])

        energy = np.sqrt(np.abs(X)**2 + self.Delta**2)
        self.energy = energy

        return energy
    
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

        dX_dk = self.N * self.X / self.k
        dX_dphi = 1j * self.valley_idx * self.N * self.X / self.k

        dH_dk = np.array([[np.zeros_like(dX_dk), dX_dk], [np.conj(dX_dk), np.zeros_like(dX_dk)]])
        dH_dphi = np.array([[np.zeros_like(dX_dk), dX_dphi], [np.conj(dX_dphi), np.zeros_like(dX_dk)]])

        return dH_dk, dH_dphi