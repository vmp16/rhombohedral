import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from model.model import McCannSystem
from model.analysis import calculate_berry_curv, calculate_berry_integral
from model.config import GAMMA0, GAMMA1, DELTA

def generate_plots():
    fig_dir = project_root / "figures"
    fig_dir.mkdir(exist_ok=True)
    
    # 1. Comparison of Bands for N=1 and N=5
    k = np.linspace(-0.5, 0.5, 400)
    phi = 0 # x direction
    
    plt.figure(figsize=(10, 5))
    E_LIMIT_PLOT = 0.5 # Energy cutoff for the band plot
    for i, N in enumerate([1, 5]):
        system = McCannSystem(GAMMA0, GAMMA1, 1, DELTA, N)
        energy = system.get_energy_bands(k, phi)
        plt.subplot(1, 2, i+1)
        plt.plot(k, energy[0], 'b-', label='Pos Band')
        plt.plot(k, energy[1], 'r-', label='Neg Band')
        plt.ylim(-E_LIMIT_PLOT, E_LIMIT_PLOT) # Apply energy cutoff to the plot
        plt.title(f'Bands for N={N}')
        plt.xlabel('k')
        plt.ylabel('Energy')
        plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / 'bands_comparison.png')
    plt.close()

    # 2. Comparison of Berry Curvature for N=1 and N=5
    plt.figure(figsize=(10, 5))
    for i, N in enumerate([1, 5]):
        system = McCannSystem(GAMMA0, GAMMA1, 1, DELTA, N)
        system.get_energy_bands(k, phi)
        Omega = calculate_berry_curv(system)
        plt.subplot(1, 2, i+1)
        plt.plot(k, Omega[:, 0], 'b-', label='Pos Band')
        plt.title(f'Berry Curvature for N={N}')
        plt.xlabel('k')
        plt.ylabel(r'$\Omega$')
    plt.tight_layout()
    plt.savefig(fig_dir / 'berry_curv_comparison.png')
    plt.close()

    # 3. Valley Index Effect (N=5)
    k_v = np.linspace(-0.25, 0.25, 300)
    plt.figure(figsize=(10, 5))
    for i, v_idx in enumerate([1, -1]):
        system = McCannSystem(GAMMA0, GAMMA1, v_idx, DELTA, 5) # Changed to N=5
        system.get_energy_bands(k_v, phi)
        Omega = calculate_berry_curv(system)
        plt.subplot(1, 2, i+1)
        plt.plot(k_v, Omega[:, 0], 'g-', label=f'Valley {v_idx}')
        plt.title(f'Berry Curv. (Valley {v_idx}, N=5)')
        plt.xlabel('k')
        plt.ylabel(r'$\Omega$')
        plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / 'valley_effect.png')
    plt.close()

    # 4. Convergence of Chern Number with Energy Cutoff
    E_max_vals = np.linspace(DELTA + 0.05, 1.5, 15)
    plt.figure(figsize=(8, 6))
    for N in [1, 5]:
        chern_vals = []
        # Estimate a large enough K to cover E_max = 1.5
        # E^2 = (v*k)^{2N} / g1^{2N-2} + D^2
        v = (np.sqrt(3) / 2) * GAMMA0
        K_LIMIT = 2.0 if N==1 else 1.2
        
        for Em in E_max_vals:
            N_L = 250
            sys_int = McCannSystem(GAMMA0, GAMMA1, 1, DELTA, N)
            
            integrals = calculate_berry_integral(sys_int, K_LIMIT, N_L, e_max=Em)
            chern = integrals[0] / (2 * np.pi)
            chern_vals.append(chern)
        
        plt.plot(E_max_vals, chern_vals, 'o-', label=f'N={N}')
    
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='N/2 (N=1)')
    plt.axhline(y=2.5, color='gray', linestyle='--', alpha=0.5, label='N/2 (N=5)')
    plt.xlabel(r'Energy Cutoff $E_{max}$')
    plt.ylabel('Local Chern Contribution')
    plt.title('Convergence to N/2 (Energy Cutoff)')
    plt.legend()
    plt.savefig(fig_dir / 'chern_convergence.png')
    plt.close()

if __name__ == "__main__":
    generate_plots()
