import argparse
import json
import subprocess
import tempfile
import os
import sys
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator, process_fidelity
from qiskit.synthesis import qs_decomposition
from scipy.linalg import expm
import qiskit.qasm2

# --- Configuration ---
RMSYNTH_CMD = "rmsynth-optimize"

def load_unitary(filepath):
    """Loads a unitary matrix from a .npy or text file."""
    try:
        if filepath.endswith('.npy'):
            U = np.load(filepath)
        else:
            # Load complex text file (handles 1+1j format)
            U = np.loadtxt(filepath, dtype=complex)
        
        # Verify shape
        if U.shape[0] != U.shape[1]:
            raise ValueError("Matrix is not square.")
        return U
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

def get_z8_phases(diagonal_unitary):
    """
    Extracts phases from a diagonal matrix and maps them to Z8 (integers 0-7).
    """
    # Extract diagonal elements
    diags = np.diag(diagonal_unitary)
    
    # Check if any magnitude is close to 0 (indicates non-unitary or non-diagonal error)
    if np.any(np.abs(diags) < 1e-5):
        return None, 999.0

    phases = np.angle(diags) # (-pi, pi]
    
    # Map [-pi, pi] to [0, 8)
    # theta = k * (pi/4) -> k = theta * 4 / pi
    k_float = phases * (4 / np.pi)
    k_int = np.round(k_float).astype(int) % 8
    
    # Calculate error
    approx_phases = k_int * (np.pi / 4)
    diff = np.abs(np.exp(1j * phases) - np.exp(1j * approx_phases))
    max_error = np.max(diff)
    
    return k_int.tolist(), max_error

def is_diagonal(op_data):
    """Checks if a matrix is diagonal."""
    return np.allclose(op_data, np.diag(np.diag(op_data)), atol=1e-6)

def run_rmsynth(phase_vector_z8, num_qubits, effort=1):
    """Calls rmsynth-optimize for a specific phase vector."""
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as tmp_in:
        json.dump(phase_vector_z8, tmp_in)
        tmp_in_path = tmp_in.name
    
    cmd = [
        RMSYNTH_CMD,
        "--n", str(num_qubits),
        "--vec-json", tmp_in_path,
        "--effort", str(effort),
        "--decoder", "ml-exact"
    ]

    try:
        # Run process
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        qasm_content = result.stdout
        
        if os.path.exists(tmp_in_path): os.remove(tmp_in_path)
        
        if not qasm_content.strip():
            return None
        return qasm_content
        
    except subprocess.CalledProcessError:
        # rmsynth failed (likely strict parameters or bad vector)
        if os.path.exists(tmp_in_path): os.remove(tmp_in_path)
        return None
    except Exception as e:
        print(f"Unexpected error calling rmsynth: {e}")
        if os.path.exists(tmp_in_path): os.remove(tmp_in_path)
        return None

def decompose_and_optimize(U, effort):
    num_qubits = int(np.log2(U.shape[0]))
    print(f"--- Processing {num_qubits}-qubit Unitary ---")
    
    # 1. Quantum Shannon Decomposition
    print("1. Performing Quantum Shannon Decomposition (QSD)...")
    qc_qsd = qs_decomposition(U)
    
    # 2. Transpile to basis
    print("2. Unrolling to Diagonal + Clifford basis...")
    qc_transpiled = transpile(qc_qsd, basis_gates=['rz', 'cx', 'h'], optimization_level=3)
    
    # 3. Identify and Optimize Diagonal Segments
    final_qc = QuantumCircuit(num_qubits)
    qubit_map = {bit: i for i, bit in enumerate(qc_transpiled.qubits)}
    current_diagonal_chunk = []
    
    print("3. Extracting diagonal blocks for rmsynth...")
    
    def process_chunk(chunk, target_qc):
        if not chunk:
            return

        # Build chunk circuit
        chunk_qc = QuantumCircuit(num_qubits)
        for op, indices in chunk:
            chunk_qc.append(op, indices)
        
        # Check if strictly diagonal
        chunk_op = Operator(chunk_qc)
        if not is_diagonal(chunk_op.data):
            # If not diagonal (e.g. contains raw CNOTs), we cannot optimize with rmsynth
            target_qc.compose(chunk_qc, inplace=True)
            return

        # It is diagonal -> Optimize
        z8_vec, error = get_z8_phases(chunk_op.data)
        
        if z8_vec and any(v != 0 for v in z8_vec):
            optimized_qasm = run_rmsynth(z8_vec, num_qubits, effort)
            if optimized_qasm:
                try:
                    # Fix potential QASM header issues
                    if "OPENQASM" not in optimized_qasm:
                        optimized_qasm = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[" + str(num_qubits) + "];\n" + optimized_qasm
                    
                    opt_circ = QuantumCircuit.from_qasm_str(optimized_qasm)
                    target_qc.compose(opt_circ, qubits=range(num_qubits), inplace=True)
                except Exception:
                    # Fallback to original
                    target_qc.compose(chunk_qc, inplace=True)
            else:
                target_qc.compose(chunk_qc, inplace=True)
        else:
            # Identity or error
            target_qc.compose(chunk_qc, inplace=True)

    # Iterate through instructions
    for instruction in qc_transpiled.data:
        gate_name = instruction.operation.name
        qubit_indices = [qubit_map[q] for q in instruction.qubits]
        
        # 'cx' is conditional: it can be part of a diagonal (sandwich) or NOT.
        # We optimistically group it, but 'is_diagonal' check will filter bad groups later.
        if gate_name in ['rz', 'cx', 'id', 'barrier']:
            current_diagonal_chunk.append((instruction.operation, qubit_indices))
        else:
            # Hit a non-diagonal gate (H, etc.) -> Process buffer
            process_chunk(current_diagonal_chunk, final_qc)
            current_diagonal_chunk = []
            
            # Append current non-diagonal gate
            final_qc.append(instruction.operation, qubit_indices)
            
    # Process leftovers
    process_chunk(current_diagonal_chunk, final_qc)

    return final_qc

def main():
    parser = argparse.ArgumentParser(description="Decompose Unitary and optimize using rmsynth.")
    parser.add_argument("input_file", help="Path to file containing Unitary matrix (.npy or .txt)")
    parser.add_argument("output_file", help="Path to save output QASM")
    parser.add_argument("--effort", type=int, default=1, help="Effort level for rmsynth (default: 1)")

    args = parser.parse_args()

    # 1. Load
    U = load_unitary(args.input_file)

    # 2. Process
    optimized_circuit = decompose_and_optimize(U, args.effort)

    # 3. Calculate Metrics
    print("\n--- Results ---")

    ops = optimized_circuit.count_ops()
    t_count = ops.get('t', 0) + ops.get('tdg', 0)
    print(f"Total T-gates: {t_count}")

    if optimized_circuit.num_qubits <= 10:
        try:
            op_target = Operator(U)
            op_actual = Operator(optimized_circuit)

            # Distance
            dist = np.linalg.norm(op_target.data - op_actual.data, ord=2)
            print(f"Distance from original: {dist:.6e}")

            # Fidelity
            fid = process_fidelity(op_target, op_actual)
            print(f"Process Fidelity: {fid:.6f}")

        except Exception as e:
            print(f"Could not calculate fidelity: {e}")
    else:
        print("Distance: (Skipped, too large to compute explicitly)")

    # 4. Save using Qiskit 1.0+ syntax
    try:
        # qiskit.qasm2.dumps is the modern replacement for circuit.qasm()
        qasm_str = qiskit.qasm2.dumps(optimized_circuit)
        with open(args.output_file, 'w') as f:
            f.write(qasm_str)
        print(f"Optimized circuit saved to {args.output_file}")
    except Exception as e:
        print(f"Error saving QASM: {e}")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
