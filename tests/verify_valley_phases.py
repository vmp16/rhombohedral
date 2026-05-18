import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.model import McCannSystem
from model.config import GAMMA0, GAMMA1, GAMMA2, GAMMA3, N

def get_velocities_at_point(k, phi, xi, delta):
    system = McCannSystem(GAMMA0, GAMMA1, xi, delta, N, gamma2=GAMMA2, gamma3=GAMMA3)
    system.get_energy_bands(np.array([k]), np.array([phi]))
    
    psi_p, psi_m = system.get_eigenstates()
    U = np.array([[psi_p[0, 0], psi_m[0, 0]],
                  [psi_p[1, 0], psi_m[1, 0]]])
    U_dag = np.conj(U.T)
    
    dH_dk, dH_dphi = system.derivate_system()
    dH_dk = dH_dk[:, :, 0]
    dH_dphi = dH_dphi[:, :, 0]
    
    Vk = U_dag @ dH_dk @ U
    Vphi = U_dag @ dH_dphi @ U
    
    return Vk[0, 1], Vphi[0, 1]

def main():
    k = 0.1
    phi = np.pi / 4
    delta = 0.1
    
    v_k_pos, v_phi_pos = get_velocities_at_point(k, phi, 1, delta)
    v_k_neg, v_phi_neg = get_velocities_at_point(k, phi, -1, delta)
    
    print(f"Point: k={k}, phi={phi}")
    print(f"Valley xi=1:  Vk={v_k_pos:.4f}, Vphi={v_phi_pos:.4f}")
    print(f"Valley xi=-1: Vk={v_k_neg:.4f}, Vphi={v_phi_neg:.4f}")
    
    print("\nComparing magnitudes:")
    print(f"|Vk(1)|  = {np.abs(v_k_pos):.4f}, |Vk(-1)|  = {np.abs(v_k_neg):.4f}")
    print(f"|Vphi(1)| = {np.abs(v_phi_pos):.4f}, |Vphi(-1)| = {np.abs(v_phi_neg):.4f}")
    
    print("\nComparing real parts:")
    print(f"Re(Vk(1))  = {np.real(v_k_pos):.4f}, Re(Vk(-1))  = {np.real(v_k_neg):.4f}")
    print(f"Re(Vphi(1)) = {np.real(v_phi_pos):.4f}, Re(Vphi(-1)) = {np.real(v_phi_neg):.4f}")
    
    print("\nCross term (imaginary part of Vk * Vphi.conj() - Vphi * Vk.conj()):")
    cross_pos = v_k_pos * np.conj(v_phi_pos) - v_phi_pos * np.conj(v_k_pos)
    cross_neg = v_k_neg * np.conj(v_phi_neg) - v_phi_neg * np.conj(v_k_neg)
    print(f"xi=1:  {cross_pos:.4f}")
    print(f"xi=-1: {cross_neg:.4f}")

if __name__ == "__main__":
    main()
