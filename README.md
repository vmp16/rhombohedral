# Rhombohedral Graphene - McCann Model Analysis

This project implements and analyzes the **McCann model** for ABC-stacked (rhombohedral) multilayer graphene. It provides tools to calculate energy bands, Berry curvature, and local Chern number contributions for an arbitrary number of layers $N$.

## Project Overview

The McCann model describes the low-energy physics of multilayer graphene near the K and K' valleys using an effective 2x2 Hamiltonian:

$$H = \begin{pmatrix} \Delta & X(\mathbf{k}) \\ X^\dagger(\mathbf{k}) & -\Delta \end{pmatrix}$$

where $2\Delta$ is the energy gap and $X(\mathbf{k})$ represents the interlayer coupling. For $N$ layers, the off-diagonal term scales as $k^N$, leading to increasingly flat bands near the Dirac point as $N$ increases.

### Key Features
- **Energy Band Calculation**: Compute and plot the dispersion relation for different layer numbers.
- **Berry Curvature**: Analytical and numerical calculation of the Berry curvature $\Omega(\mathbf{k})$ using a Kubo-like formula.
- **Topological Analysis**: Numerical integration of the Berry curvature to verify the local Chern number contribution ($C \approx \pm N/2$ per valley).
- **Valley Symmetry**: Comparison between the $K$ ($\xi=1$) and $K'$ ($\xi=-1$) valleys, demonstrating time-reversal symmetry effects.

## Project Structure

```text
.
├── model/              # Core physics implementation
│   ├── model.py        # McCannSystem class and analytical formulas
│   └── config.py       # Physical constants and numerical parameters
├── scripts/            # Analysis and plotting scripts
│   ├── compare_valleys.py       # Compare K and K' valley properties
│   ├── generate_report_plots.py # Generate figures for the report
│   ├── get_chern_contrib.py     # Calculate Chern number via integration
│   └── plot_berry_curv.py       # Visualize Berry curvature distribution
├── figures/            # Generated plots and visualizations
├── report/             # LaTeX source for the project report
└── tests/              # Unit tests and verification scripts
```

## Getting Started

### Prerequisites
- Python 3.x
- NumPy
- Matplotlib

### Usage

1. **Configure Parameters**: Edit `model/config.py` to change physical constants like $\gamma_0$, $\gamma_1$, or the number of layers $N$.
2. **Run Analysis**:
   - To visualize the Berry curvature:
     ```bash
     python scripts/plot_berry_curv.py
     ```
   - To calculate the Chern number:
     ```bash
     python scripts/get_chern_contrib.py
     ```
   - To compare valleys:
     ```bash
     python scripts/compare_valleys.py
     ```

## Results Summary

- **Monolayer (N=1)**: Linear dispersion (Dirac cones) and Berry curvature peaked at $k=0$.
- **Multilayer (N=5)**: Flatter bands at low energy ($E \propto k^5$) and Berry curvature shifted to finite $k$.
- **Topology**: The total Berry phase for $N$ layers is $N\pi$, corresponding to a local Chern number of $N/2$ in the gapped state.

---
*Project developed as part of the DIPC (Donostia International Physics Center) research internship.*
