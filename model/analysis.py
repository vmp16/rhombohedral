import numpy as np

def calculate_berry_curv(system):
    """
    Calculate the Berry curvature in the out-of-plane direction using the Kubo-like formula.

    Parameters:
    -----------
    system : McCannSystem instance

    Returns:
    --------
    Omega_per_band : array
        Value of the Berry curvature for each band.
    """
    psi_p, psi_m = system.get_eigenstates()

    # Define psi dag
    # psi_p_dag = np.conj(psi_p).T
    # psi_m_dag = np.conj(psi_m).T

    # Stack eigenstates as columns and move k to the first dimension: (k, component, eigenstate)
    U_raw = np.array((psi_p, psi_m))
    U = np.transpose(U_raw, (2, 1, 0))
    U_dag = np.conj(np.transpose(U, (0, 2, 1)))
    # print(f"U shape = {U.shape}")
    # print(f"U_dag shape = {U_dag.shape}")

    dH_dk, dH_dphi = system.derivate_system()
    # Move k to the first dimension: (k, component, component)
    dH_dk = np.transpose(dH_dk, (2, 0, 1))
    dH_dphi = np.transpose(dH_dphi, (2, 0, 1))

    Vk = U_dag @ dH_dk @ U
    Vphi = U_dag @ dH_dphi @ U

    # Cross term
    cross_term = Vk * Vphi.conj() - Vphi * Vk.conj()

    # Sum over the intermediate band index m (axis=2) to get the (100, 2) array
    summed_cross_term = np.sum(cross_term, axis=2)

    # Here, (E_n - E_l)^2 = (2E)^2 = 4E^2
    Omega_per_band = np.real(1j * summed_cross_term / (4 * system.energy[:, np.newaxis] ** 2))

    return Omega_per_band

def calculate_berry_integral(system, k_limit, n_points, e_max=None):
    """
    Calculates the integral of the Berry curvature over a region in k-space.

    Parameters:
    -----------
    system : McCannSystem instance.
    k_limit : float
        Half-width of the square grid in k-space.
    n_points : int
        Number of points along each dimension of the grid.
    e_max : float
        Optional energy cutoff. If provided, the integral is restricted to E <= e_max. If None, it's restricted to a circle of radius k_limit.

    Returns:
    --------
    integral : array
        A numpy array containing the integral for each band.
    """
    kx_vals = np.linspace(-k_limit, k_limit, n_points)
    ky_vals = np.linspace(-k_limit, k_limit, n_points)
    KX, KY = np.meshgrid(kx_vals, ky_vals)

    kx_flat = KX.flatten()
    ky_flat = KY.flatten()

    k_flat = np.sqrt(kx_flat**2 + ky_flat**2)
    phi_flat = np.arctan2(ky_flat, kx_flat)

    if e_max is not None:
        energies = system.get_energy_bands(k_flat, phi_flat)
        mask = (energies <= e_max)
    else:
        mask = (k_flat <= k_limit)

    k_in = k_flat[mask]
    phi_in = phi_flat[mask]

    # Avoid singularity at k=0
    k_in[k_in == 0] = 1e-6

    # Update system states for the selected points
    system.get_energy_bands(k_in, phi_in)
    Omega_in = calculate_berry_curv(system)

    dk = kx_vals[1] - kx_vals[0]
    area_per_point = dk**2

    integral = np.sum(Omega_in, axis=0) * area_per_point
    return integral