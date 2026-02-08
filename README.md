# iQuHACK 2026: Clifford+T Circuit Optimization Challenge

This repository is for the iQuHACK 2026 challenge, focused on compiling quantum unitaries into efficient Clifford+T circuits. The tools here help minimize T-gate count while keeping high fidelity to the original unitary, and include both Python scripts and a notebook for interactive exploration.

## Features

- **Pattern Recognition & Synthesis:** Detects and optimally compiles common 2-qubit unitaries (QFT, Heisenberg, XX+YY, etc.), and recognizes special structures for efficient circuit generation.
- **Brute Force & Reed-Muller Optimization:** Searches for low-T circuits using brute force for small cases, and applies Reed-Muller synthesis (via the `rmsynth` toolkit) for phase polynomial optimization.
- **Gridsynth Rz Approximation:** Approximates arbitrary Rz rotations with Clifford+T sequences, adaptively tuning precision to avoid excessive T-counts.
- **Manual Clifford+T Conversion:** Converts non-native gates to the allowed set using known identities and gridsynth.
- **Challenge 12 Support:** Handles a special 9-qubit commuting Pauli phase program, synthesizing circuits with only {H, T, Tdg, CX} gates.
- **Visualization & Analysis:** The notebook provides circuit diagrams, fidelity metrics, and summary tables for all results.

## How to Use This Repository

1. **Clone the repository:**
	 ```bash
	 git clone https://github.com/Valmohaugen/iQuHack-2026.git
	 cd iQuHack-2026
	 ```
2. **Install dependencies:**
	 - Make sure you have Python 3.9+ and pip installed.
	 - Install required packages:
		 ```bash
		 pip install -r requirements.txt
		 ```
	 - For advanced phase polynomial optimization, install the `rmsynth` toolkit (see `rmsynth/README.md`).
3. **Optimize circuits:**
	 - **Batch mode (all 11 unitaries):**
		 ```bash
		 python optimize_unitaries.py --batch
		 ```
	 - **Single unitary:**
		 ```bash
		 python optimize_unitaries.py unitary1.npy output.qasm --effort 3
		 ```
	 - **Hard mode (brute force for specific unitaries):**
		 ```bash
		 python optimize_unitaries.py --hard 6 7 10 --max-t 5 --max-cx 2
		 ```
	 - **Challenge 12 (commuting Pauli phase program):**
		 ```bash
		 python optimize_unitaries.py --challenge12 --json unitary12.json --output unitary12_optimized.qasm
		 ```
4. **Explore interactively:**
	 - Open `optimize_unitaries.ipynb` in Jupyter or VS Code to see step-by-step optimization, circuit diagrams, and results for each unitary and Challenge 12.
5. **Python API:**
	 - Use the functions directly in your scripts:
		 ```python
		 from optimize_unitaries import load_unitary, decompose_and_optimize, analyze_unitary
		 U = load_unitary("unitary1.npy")
		 qc = decompose_and_optimize(U, effort=3)
		 analyze_unitary(U, name="My Unitary")
		 best_qc, fid = brute_force_search(U, max_t=3, max_cx=3)
		 ```

## Results

Running the scripts or notebook, you will:

- Compile all 11 challenge unitaries to high-fidelity Clifford+T circuits with minimized T-counts and CX counts.
- Solve Challenge 12, synthesizing a 9-qubit circuit using only the allowed gates, with validation and summary statistics.
- See summary tables and visualizations in the notebook, including T-counts, CX counts, fidelities, and circuit diagrams for all results.

**Dependencies:**
- Python 3.9+
- Qiskit, pylatexenc, pygridsynth (see `requirements.txt`)
- For advanced optimization: `rmsynth` (see `rmsynth/README.md`)