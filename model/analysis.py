import numpy as np

def fermi_distrib(E, mu_eff, T_eff):
    """
    Returns the Fermi distribution value.

    Parameters
    ----------
    E : float
        Energy in t1 units
    mu_eff : float
        Fermi level in t1 units
    T_eff : float
        Scalated temperature (kB * T_real / t1)

    Returns
    ----------
    f(E, T, mu) : float
        Corresponding value of the Fermi distribution
    """

    x = (E - mu_eff) / T_eff

    # Avoid overflow in exp by clipping x
    x_clipped = np.clip(x, -700, 700)
    return 1 / (1 + np.exp(x_clipped))

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

    # Convert to polar coordinates
    k_flat = np.sqrt(kx_flat**2 + ky_flat**2)
    phi_flat = np.arctan2(ky_flat, kx_flat)

    if e_max is not None:
        energies = system.get_energy_bands(k_flat, phi_flat)
        mask = (energies[0] <= e_max)
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

def calculate_ahe(system, k_lim, n_pts, T_eff, mu_eff):
    """
    Calculate the Anomalous Hall Effect using the Berry curvature, integrating it over a square.
    
    system : McCannSystem
        The physical system.
    k_limit : float
        Half-width of the square grid in k-space.
    n_points : int
        Number of points along each dimension of the grid.
    T_eff : float
        Scaled temperature (kB * T_real / t1)
    mu_eff : float
        Scaled chemical potential (mu / t1)

    Returns
    -------
    sigma_xy : float
        The total Hall conductivity.
    sigma_xy_list : array
        the Hall conductivity per band.
    """
    kx_vals = np.linspace(-k_lim, k_lim, n_pts)
    ky_vals = np.linspace(-k_lim, k_lim, n_pts)
    KX, KY = np.meshgrid(kx_vals, ky_vals)

    kx_flat = KX.flatten()
    ky_flat = KY.flatten()

    # Convert to polar coordinates
    k_flat = np.sqrt(kx_flat**2 + ky_flat**2)
    phi_flat = np.arctan2(ky_flat, kx_flat)

    # Avoid singularity at k=0
    k_flat[k_flat == 0] = 1e-6

    energies = system.get_energy_bands(k_flat, phi_flat)
    Omega = calculate_berry_curv(system)

    dk = kx_vals[1] - kx_vals[0]
    prefactor = (dk**2) / (2 * np.pi)

    sigma_xy_list = []

    for n in range(Omega.shape[-1]):
        band_E = energies[n]
        band_curv = Omega[..., n]

        # Integrate the Berry curvature ponderated by the Fermi distribution
        f = fermi_distrib(band_E, mu_eff, T_eff)
        integral = np.sum(f * band_curv)

        sigma_xy_list.append(prefactor * integral)

    sigma_xy = np.sum(sigma_xy_list)

    return sigma_xy, np.array(sigma_xy_list)

def velocity_operator(system, idx1, idx2):
    """
    Calculate the 2D velocity vector between the two bands indexed by idx1 and idx2.

    Parameters
    ----------
    system : McCannSystem
        The physical system.
    idx1, idx2 : int
        Indices of the energy bands

    Return
    ------
    Vx, Vy : array
        The velocity components in 2D cartesian coordinates at every point in the k-area.
    """
    psi_p, psi_m = system.get_eigenstates()

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

    # Stack derivatives to use matrix product: (k, 2_polar, 2_comp, 2_comp)
    dH_polar = np.stack((dH_dk, dH_dphi), axis=1)

    # Build the Jacobian matrix for the coordinate transformation
    phi = np.asarray(system.phi)
    J = np.array([[np.cos(phi), -np.sin(phi)],
                  [np.sin(phi),  np.cos(phi)]])
    
    # Apply the Jacobian to get cartesian derivatives
    if phi.ndim > 0:
        J = np.transpose(J, (2, 0, 1)) # (k, 2_cart, 2_polar)
        dH_cart = np.einsum('kij,kjlm->kilm', J, dH_polar)
    else:
        dH_cart = np.einsum('ij,kjlm->kilm', J, dH_polar)

    dH_dx = dH_cart[:, 0, :, :]
    dH_dy = dH_cart[:, 1, :, :]

    # Calculate the velocity operator elements using cartesian derivatives
    Vx = np.einsum('kc,kcd,kd->k', U_dag[:, idx1, :], dH_dx, U[:, :, idx2])
    Vy = np.einsum('kc,kcd,kd->k', U_dag[:, idx1, :], dH_dy, U[:, :, idx2])

    return Vx, Vy

def get_omega_z(system, band_idx, k_lim, n_pts):
    """
    Calculate the omega tensor taking part in the Magneto Non Linear AHE.

    Parameters
    ----------
    system : McCannSystem
        The physical system.
    band_idx : int
        Index of the band of interest.
    k_limit : float
        Half-width of the square grid in k-space.
    n_points : int
        Number of points along each dimension of the grid.

    Return
    ------
    omega_z : float
        Only non-zero component of the omega tensor for a 2D system, evaluated at every point of the k-area.
    """
    if band_idx not in (0, 1):
        print(f"Error: band_idx must be 0 or 1, not {band_idx}.")
        return
    
    n_idx = 1 - band_idx
    
    kx_vals = np.linspace(-k_lim, k_lim, n_pts)
    ky_vals = np.linspace(-k_lim, k_lim, n_pts)
    KX, KY = np.meshgrid(kx_vals, ky_vals)

    kx_flat = KX.flatten()
    ky_flat = KY.flatten()

    # Convert to polar coordinates
    k_flat = np.sqrt(kx_flat**2 + ky_flat**2)
    phi_flat = np.arctan2(ky_flat, kx_flat)

    # Avoid singularity at k=0
    k_flat[k_flat == 0] = 1e-6

    # Evaluate the system
    energies = system.get_energy_bands(k_flat, phi_flat)

    e0 = energies[band_idx]
    e1 = energies[n_idx]

    # Get the velocities
    Vx_00, Vy_00 = velocity_operator(system, band_idx, band_idx)
    Vx_11, Vy_11 = velocity_operator(system, n_idx, n_idx)
    Vx_10, Vy_10 = velocity_operator(system, n_idx, band_idx)

    # Put all together
    omega_z = -1j * ((Vx_11 + Vx_00) * Vy_10 - (Vy_11 + Vy_00) * Vx_10) / (e1 - e0)

    return omega_z

def get_G(system, band_idx, k_lim, n_pts):
    """
    Calculate the G tensor taking part in the Electric Non Linear AHE for a 2D, 2-band model.

    Parameters
    ----------
    system : McCannSystem
        The physical system.
    band_idx : int
        Index of the band of interest.
    k_limit : float
        Half-width of the square grid in k-space.
    n_points : int
        Number of points along each dimension of the grid.

    Return
    ------
    G_xy, G_yx : array
        2D cartesian components of the G tensor.
    """
    # Define the missing index for a 2-band model
    if band_idx not in (0, 1):
        print(f"Error: band_idx must be 0 or 1, not {band_idx}.")
        return
    n_idx = 1 - band_idx

    # Evaluate the system at a squared area
    kx_vals = np.linspace(-k_lim, k_lim, n_pts)
    ky_vals = np.linspace(-k_lim, k_lim, n_pts)
    KX, KY = np.meshgrid(kx_vals, ky_vals)

    kx_flat = KX.flatten()
    ky_flat = KY.flatten()

    # Convert to polar coordinates
    k_flat = np.sqrt(kx_flat**2 + ky_flat**2)
    phi_flat = np.arctan2(ky_flat, kx_flat)

    # Avoid singularity at k=0
    k_flat[k_flat == 0] = 1e-6

    # Evaluate the system
    energies = system.get_energy_bands(k_flat, phi_flat)

    # Separate the bands
    e0 = energies[band_idx]
    e1 = energies[n_idx]

    # Calculate the interband velocities for a 2D system (x, y)
    Vx_01, Vy_01 = velocity_operator(system, band_idx, n_idx)
    Vx_10, Vy_10 = velocity_operator(system, n_idx, band_idx)

    # Calculate the tensor G
    G_xy = np.real((Vx_01 * Vy_10) / ((e0 - e1)**3))
    G_yx = np.real((Vy_01 * Vx_10) / ((e0 - e1)**3))

    return G_xy, G_yx

def get_F(system, band_idx, k_lim, n_pts):
    """
    Calculate the F tensor taking part in the Magnetic Non Linear AHE for a 2D, 2-band model.

    Parameters
    ----------
    system : McCannSystem
        The physical system.
    band_idx : int
        Index of the band of interest.
    k_limit : float
        Half-width of the square grid in k-space.
    n_points : int
        Number of points along each dimension of the grid.

    Return
    ------
    F_xy, F_yx : array
        2D cartesian components of the F tensor.
    """
    # Define the missing index for a 2-band model
    if band_idx not in (0, 1):
        print(f"Error: band_idx must be 0 or 1, not {band_idx}.")
        return
    n_idx = 1 - band_idx

    # Get the different velocities taking part in the calculation
    omega_z = get_omega_z(system, band_idx, k_lim, n_pts)
    Vx_01, Vy_01 = velocity_operator(system, band_idx, n_idx)

    # get_omega_z evaluates the system, so we can use the stored system.energy
    energies = np.array([system.h0 + system.energy, system.h0 - system.energy])
    e0 = energies[band_idx]
    e1 = energies[n_idx]

    # Calculate the tensor F
    F_xz = np.imag((Vx_01 * omega_z) / (e0 - e1)**2)
    F_yz = np.imag((Vy_01 * omega_z) / (e0 - e1)**2)

    return F_xz, F_yz

# def get_orbital_momentum()
