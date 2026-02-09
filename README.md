# iQuHACK 2026: Superquantum Challenge

This repository is the home for the iQuHACK 2026 Superquantum challenge, focused on minimizing Clifford+T resources—especially T-count and T-depth—through structure-aware synthesis and Reed–Muller decoding–based optimization.

## Features

* **Clifford+T Circuit Optimization:**

  * Minimize T-count (and related resource metrics) using pattern recognition, brute-force search, and phase-polynomial optimization.
  * Recognize and optimize common 2-qubit unitaries (e.g., QFT-like blocks, Heisenberg/XX+YY structure, and other frequent motifs).
  * Integrate with the `rmsynth` toolkit for aggressive T-count and T-depth reduction.
  * Support both command-line workflows and a Python API for flexible experimentation.

* **`rmsynth` Toolkit:**

  * Optimize Clifford+T resources via phase-polynomial synthesis and punctured Reed–Muller decoding.
  * Multiple decoding backends (e.g., Dumer, RPA, OSD), depth-aware optimization modes, and autotuning controls.
  * Python API and CLI, backed by a high-performance C++ implementation.

* **Notebooks & Datasets:**

  * Jupyter notebook for step-by-step optimization and visualization of the 11 challenge unitaries.
  * Example unitary inputs and QASM outputs for benchmarking, comparison, and regression testing.

* **Papers & References:**

  * A curated collection of foundational papers on T-count optimization, Reed–Muller codes, and modern quantum circuit synthesis methods.

## How to Use This Repository

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Valmohaugen/iQuHack-2026.git
   cd iQuHack-2026
   ```
2. **Set up your environment:**

   * Install Python 3.11+ (conda recommended)
   * Create the environment and install requirements:

     ```bash
     conda create -n mit26 python=3.11
     conda activate mit26
     pip install -r rmsynth/requirements.txt
     pip install -r requirements.txt
     python -m pip install -e rmsynth/
     ```
   * Core packages include `qiskit`, `pygridsynth`, `numpy`, `matplotlib`, `mpmath`, `pandas`, and more (see the requirements files).
3. **Optimize a unitary (command line):**

   ```bash
   python optimize_unitaries.py unitary1.npy unitary1_optimized.qasm --effort 3
   ```
4. **Explore interactively:**

   * Open `optimize_unitaries.ipynb` in Jupyter or VS Code.
   * Run through the cells to inspect the optimization pipeline, visualize resource metrics, and compare methods.
5. **Use the Python API:**

   ```python
   from optimize_unitaries import load_unitary, decompose_and_optimize

   U = load_unitary("unitary1.npy")
   qc = decompose_and_optimize(U, effort=3)
   ```
6. **Dig into `rmsynth/` for advanced optimization:**

   * Deeper configuration, backend details, and implementation notes live in the `rmsynth` subfolder.

## Results

Running the project end-to-end, you will:

* Optimize all 11 challenge unitaries to Clifford+T circuits with minimized T-count and high fidelity.
* Generate QASM outputs and compare resource usage (T-count, CX-count, depth) across approaches and effort settings.
* Visualize circuit structure and performance in the notebook.
* Leverage Reed–Muller decoding–based synthesis to push circuit efficiency beyond straightforward decompositions.

**Dependencies:**

* Python 3.11+, Jupyter, Qiskit, pygridsynth, numpy, matplotlib, mpmath, pandas, and a C++17 compiler for rmsynth.
