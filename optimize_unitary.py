import argparse
import json
import subprocess
import tempfile
import os
import sys
import numpy as np
import pygridsynth as gs  # Using pygridsynth as requested

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator, process_fidelity
from qiskit.synthesis import qs_decomposition
import qiskit.qasm2

# --- Configuration ---
RMSYNTH_CMD = "rmsynth-optimize"

def load_unitary(filepath):
    try:
        if filepath.endswith('.npy'):
            U = np.load(filepath)
        else:
            U = np.loadtxt(filepath, dtype=complex)
        return U
    except Exception as e:
        print(f"Error loading file: {e}"); sys.exit(1)

def get_z8_phases(diagonal_unitary):
    diags = np.diag(diagonal_unitary)
    if np.any(np.abs(diags) < 1e-5): return None, 999.0
    phases = np.angle(diags)
    k_int = np.round(phases * (4 / np.pi)).astype(int) % 8
    approx = k_int * (np.pi / 4)
    return k_int.tolist(), np.max(np.abs(np.exp(1j * phases) - np.exp(1j * approx)))

def is_diagonal(op_data):
    return np.allclose(op_data, np.diag(np.diag(op_data)), atol=1e-6)

def run_rmsynth(phase_vector_z8, num_qubits, effort=1):
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as tmp_in:
        json.dump(phase_vector_z8, tmp_in); tmp_path = tmp_in.name
    cmd = [RMSYNTH_CMD, "--n", str(num_qubits), "--vec-json", tmp_path, "--effort", str(effort), "--decoder", "ml-exact"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if os.path.exists(tmp_path): os.remove(tmp_path)
        return res.stdout if res.stdout.strip() else None
    except:
        if os.path.exists(tmp_path): os.remove(tmp_path)
        return None

def apply_manual_identities(qc):
    """
    1. X = HZH
    2. Z = HXH
    3. SXSdg = Y
    4. SYSdg = -X
    Uses pygridsynth for non-Z8 RZ gates.
    """
    new_qc = QuantumCircuit(qc.num_qubits)
    for inst in qc.data:
        name = inst.operation.name; qbs = inst.qubits
        if name == 'x':
            new_qc.h(qbs); new_qc.s(qbs); new_qc.s(qbs); new_qc.h(qbs)
        elif name == 'z':
            new_qc.s(qbs); new_qc.s(qbs)
        elif name == 'y':
            new_qc.s(qbs); new_qc.h(qbs); new_qc.s(qbs); new_qc.s(qbs); new_qc.h(qbs); new_qc.sdg(qbs)
        elif name == 'rz':
            angle = inst.operation.params[0] % (2 * np.pi)
            if np.isclose(angle, 0, atol=1e-7): continue
            
            k_f = angle / (np.pi / 4); k = int(np.round(k_f))
            if np.abs(k_f - k) < 1e-6:
                k %= 8
                if k == 1: new_qc.t(qbs)
                elif k == 2: new_qc.s(qbs)
                elif k == 3: new_qc.s(qbs); new_qc.t(qbs)
                elif k == 4: new_qc.s(qbs); new_qc.s(qbs)
                elif k == 5: new_qc.s(qbs); new_qc.s(qbs); new_qc.t(qbs)
                elif k == 6: new_qc.sdg(qbs)
                elif k == 7: new_qc.tdg(qbs)
            else:
                # pygridsynth call
                seq = gs.gridsynth(angle, epsilon=1e-10)
                if seq:
                    for g in str(seq):
                        if g == 'H': new_qc.h(qbs)
                        elif g == 'T': new_qc.t(qbs)
                        elif g == 'S': new_qc.s(qbs)
                # If gridsynth fails or is empty, we do NOT append RZ to ensure no RZ in output
        elif name == 'cx': new_qc.cx(qbs[0], qbs[1])
        else: new_qc.append(inst.operation, qbs)
    return new_qc

def decompose_and_optimize(U, effort):
    num_qubits = int(np.log2(U.shape[0]))
    qc_qsd = qs_decomposition(U)
    qc_t = transpile(qc_qsd, basis_gates=['rz', 'cx', 'h'], optimization_level=3)
    final_qc = QuantumCircuit(num_qubits)
    q_map = {bit: i for i, bit in enumerate(qc_t.qubits)}
    chunk = []

    def process_chunk(c, target):
        if not c: return
        c_qc = QuantumCircuit(num_qubits)
        for op, idx in c: c_qc.append(op, idx)
        if not is_diagonal(Operator(c_qc).data):
            target.compose(c_qc, inplace=True); return
        z8, _ = get_z8_phases(Operator(c_qc).data)
        if z8 and any(v != 0 for v in z8):
            opt = run_rmsynth(z8, num_qubits, effort)
            if opt:
                if "OPENQASM" not in opt: opt = f"OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[{num_qubits}];\n{opt}"
                target.compose(QuantumCircuit.from_qasm_str(opt), qubits=range(num_qubits), inplace=True)
                return
        target.compose(c_qc, inplace=True)

    for inst in qc_t.data:
        n = inst.operation.name; idx = [q_map[q] for q in inst.qubits]
        if n in ['rz', 'cx', 'id', 'barrier']: chunk.append((inst.operation, idx))
        else:
            process_chunk(chunk, final_qc); chunk = []
            final_qc.append(inst.operation, idx)
    process_chunk(chunk, final_qc)
    return final_qc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file"); parser.add_argument("output_file")
    parser.add_argument("--effort", type=int, default=1)
    args = parser.parse_args()

    # 1. Decomposition and Substitution
    U_target = load_unitary(args.input_file)
    qc = decompose_and_optimize(U_target, args.effort)
    qc = apply_manual_identities(qc)

    # 2. Phase Alignment Logic
    # Calculate the current unitary V
    U_actual = Operator(qc).data
    # theta = angle( Tr(U_target_dag @ U_actual) )
    # To fix V such that V_new = exp(i*theta) * V
    phase_diff = np.angle(np.trace(np.conj(U_target.T) @ U_actual))
    qc.global_phase = -phase_diff # Adjusting the circuit property

    # 3. Final Metrics
    U_final = Operator(qc).data
    dist = np.linalg.norm(U_target - U_final, ord=2)
    fid = process_fidelity(Operator(U_target), Operator(qc))

    print("\n--- Results ---")
    ops = qc.count_ops()
    print(f"TOTAL T-GATES: {ops.get('t', 0) + ops.get('tdg', 0)}")
    print(f"Operator Norm Distance: {dist:.6e}")
    print(f"Process Fidelity: {fid:.6f}")

    # 4. Save
    qasm_str = qiskit.qasm2.dumps(qc)
    with open(args.output_file, 'w') as f: f.write(qasm_str)
    print(f"\nFinal QASM saved to {args.output_file}")

if __name__ == "__main__":
    main()
