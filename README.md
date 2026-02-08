# iQuHACK-2026: Clifford+T Circuit Optimization Challenge

This repository is the home for the iQuHACK 2026 Clifford+T circuit optimization challenge, focused on pushing the limits of quantum circuit efficiency using advanced synthesis and decoding techniques. 

## Features

- **Clifford+T Circuit Optimization:**
   - Minimize T-gate count in quantum circuits using pattern recognition, brute-force search, and phase polynomial optimization.
   - Recognize and optimize common 2-qubit unitaries (QFT, Heisenberg, etc.).
   - Integrate with the `rmsynth` toolkit for deep T-count and T-depth reduction.
   - Use both command-line tools and a Python API for flexible workflows.

- **rmsynth Toolkit:**
   - Resource-optimization for Clifford+T circuits via phase polynomial methods and punctured Reed–Muller decoding.
   - Multiple decoding backends (Dumer, RPA, OSD, etc.), depth-aware optimization, and autotuning.
   - Python API and CLI, with a C++ backend for high performance.

- **Notebooks & Datasets:**
   - Jupyter notebook for step-by-step optimization and visualization of 11 challenge unitaries.
   - Example datasets and QASM outputs for benchmarking and analysis.

- **Papers & References:**
   - A curated collection of foundational papers on T-count optimization, Reed–Muller codes, and quantum circuit synthesis.

## How to Use This Repository

1. **Clone the repository:**
    ```bash
    git clone https://github.com/Valmohaugen/iQuHack-2026.git
    cd iQuHack-2026
    ```
2. **Set up your environment:**
    - Install Python 3.11+ (conda recommended)
    - Run the setup script or install requirements:
       ```bash
       conda create -n mit26 python=3.11
       conda activate mit26
       pip install -r rmsynth/requirements.txt
       pip install -r requirements.txt
       python -m pip install -e rmsynth/
       ```
    - Required packages: `qiskit`, `pygridsynth`, `numpy`, `matplotlib`, `mpmath`, `pandas`, and more (see requirements files)
3. **Optimize a unitary (command line):**
    ```bash
    python optimize_unitaries.py unitary1.npy unitary1_optimized.qasm --effort 3
    ```
4. **Explore interactively:**
    - Open `optimize_unitaries.ipynb` in Jupyter or VS Code
    - Run through the cells to see the optimization pipeline, visualize results, and compare methods
5. **Use the Python API:**
    ```python
    from optimize_unitaries import load_unitary, decompose_and_optimize
    U = load_unitary("unitary1.npy")
    qc = decompose_and_optimize(U, effort=3)
    ```
6. **Consult the `rmsynth` subfolder for advanced optimization, API docs, and implementation details.**

## Results

Running the project end-to-end, you will:

- Optimize all 11 challenge unitaries to Clifford+T circuits with minimized T-count and high fidelity.
- Generate QASM outputs and compare resource usage (T-count, CX-count, depth).
- Visualize circuit structure and performance in the notebook.
- Leverage state-of-the-art Reed–Muller decoding to push the limits of quantum circuit efficiency.
**Dependencies:**
   - Python 3.11+, Jupyter, Qiskit, pygridsynth, numpy, matplotlib, mpmath, pandas, and a C++17 compiler for rmsynth