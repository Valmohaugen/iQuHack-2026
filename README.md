

# iQuHACK 2026 - Clifford+T Circuit Optimizer

This repository provides tools and scripts for optimizing quantum circuits to use minimal T-gates while maintaining high fidelity with target unitaries, as required for the iQuHACK 2026 challenge.

## Features
- Pattern recognition for common 2-qubit unitaries (QFT, Heisenberg, XX+YY, etc.)
- Brute force search for optimal low-T circuits
- Phase polynomial optimization via rmsynth
- Gridsynth-based Rz approximation

## Optimization Pipeline
1. **Pattern Recognition**: Detects known structures and applies optimal synthesis.
2. **Manual Clifford+T Conversion**: Converts non-native gates to the allowed set using known identities.
3. **Reed-Muller Synthesis (rmsynth)**: Optimizes phase polynomials for diagonal and near-diagonal unitaries.
4. **Brute Force Search**: Searches for low-T circuits for small unitaries when fidelity is insufficient.
5. **Gridsynth**: Approximates arbitrary Rz rotations with Clifford+T sequences, adaptively tuning precision to avoid T-count blowup.

## Usage

### Command Line
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
- **Challenge 12 (commuting Pauli phase program, only H, T, Tdg, CX allowed):**
	```bash
	python optimize_unitaries.py --challenge12 --json unitary12.json --output unitary12_optimized.qasm
	```

### Python API
```python
from optimize_unitaries import load_unitary, decompose_and_optimize, analyze_unitary
U = load_unitary("unitary1.npy")
qc = decompose_and_optimize(U, effort=3)
analyze_unitary(U, name="My Unitary")
best_qc, fid = brute_force_search(U, max_t=3, max_cx=3)
```

### Jupyter Notebook
The notebook [optimize_unitaries.ipynb](optimize_unitaries.ipynb) demonstrates the optimization process for each unitary, with code, results, and visualizations. Each unitary is processed in its own section, and Challenge 12 is handled in a dedicated block.

## Challenge 12: Commuting Pauli Phase Program
This special challenge requires synthesizing a 9-qubit unitary that is a product of exponentials of commuting Pauli terms, each with a phase multiple of π/8. The circuit must use only {H, T, Tdg, CX} gates (no S/Sdg allowed). The solution uses direct synthesis for each term, followed by circuit simplification and validation. The notebook provides summary statistics and validation for the resulting circuit.

## Gate Set
Allowed gates: H, T, Tdg, S, Sdg, CX (CNOT)
For Challenge 12: Only H, T, Tdg, CX (no S/Sdg)

## Results
- All 11 unitaries are compiled to high-fidelity Clifford+T circuits with minimized T-counts.
- Challenge 12 is synthesized with only the allowed gates and high fidelity.
- The notebook provides a summary table of T-counts, CX counts, fidelities, and other metrics for all circuits.