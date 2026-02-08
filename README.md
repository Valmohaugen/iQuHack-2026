# iQuHACK-2025: Alice & Bob Quantum Challenge

This repository is the home for the iQuHACK 2025 Alice & Bob challenge, focused on simulating and controlling cat qubits in open quantum systems. 

## Features

- **Cat Qubit Stabilization & Analysis:** Simulate a cat qubit stabilized by two-photon dissipation, build composite Hilbert spaces, and analyze quantum state evolution using the `dynamiqs` library and JAX.
- **Lab Frame Simulation & Hamiltonian Engineering:** Model quantum systems in the lab frame, engineer time-dependent Hamiltonians (including SQUID and drive terms), and visualize results.
- **Open Quantum System Simulation:** Solve the Lindblad master equation, compare simulation approaches, and implement advanced gates and optimal control.
- **Visualization:** Generate plots and animations (Wigner functions, expectation values) to interpret quantum state evolution.

## How to Use This Repository

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Valmohaugen/iQuHack-2025.git
   cd iQuHack-2025
   ```
2. **Open the challenge notebooks:**
   - Use Jupyter Notebook or VS Code to open `Challenge 1.ipynb` and `Challenge 2.ipynb`.
3. **Consult Resources:**
   - The `Resources` file contains background reading and helpful links (e.g., [iQuHACK 2025 Alice & Bob](https://github.com/iQuHACK/2025-Alice-and-Bob)).
4. Make sure you have Python, Jupyter, and the required packages (`dynamiqs`, `jax`, `matplotlib`, `numpy`, `scipy`).
   - If you use additional packages, list them in your `requirements.txt` and kernel.

## Results

Running the project end-to-end, you will:

- Simulate and visualize the stabilization of a cat qubit, including photon number, parity, and Wigner function evolution.
- Engineer and analyze lab-frame Hamiltonians, producing animations and insights into quantum state dynamics.
- Compare simulation methods, implement gates, and apply optimal control.
- **Dependencies:**
   - Ensure you have Python, Jupyter, and required packages (e.g., `dynamiqs`, `jax`, `matplotlib`, `numpy`, `scipy`) installed.