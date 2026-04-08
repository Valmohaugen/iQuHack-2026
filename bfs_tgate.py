import numpy as np
import os
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor
from qiskit import QuantumCircuit, transpile, qasm2
from qiskit.quantum_info import Operator, process_fidelity, random_unitary
from tqdm import tqdm

# --- CONFIGURATION ---
BATCH_SIZE = 500
SAFETY_LIMIT = 5_000_000 

# --- HELPER: Phase-Invariant Hashing ---
def operator_to_key(op_data):
    """
    Hashes a 4x4 Unitary Matrix. 
    Crucial: Normalizes global phase so equivalent unitaries hash to same key.
    """
    flat = op_data.flatten()
    non_zeros = flat[np.abs(flat) > 1e-5]
    if len(non_zeros) > 0:
        phase_factor = non_zeros[0] / np.abs(non_zeros[0])
        normalized = flat / phase_factor
    else:
        normalized = flat
    return np.round(normalized, 5).tobytes()

# --- WORKER FUNCTION (Static) ---
def expand_unitary_batch(args):
    """
    Expands a batch of (Circuit, Operator, T_Count, Last_Ops).
    """
    batch_data, target_op, max_t = args
    local_res = []
    
    # Gate Set (Name, Cost, Matrix)
    # Pre-computed matrices for the MATH step (not for circuit building)
    gate_defs = [
        ('h', 0, Operator.from_label('H')),
        ('s', 0, Operator.from_label('S')),
        ('t', 1, Operator.from_label('T')),
        ('cx', 0, Operator([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]])) # CNOT 0->1
    ]
    # Add reversed CNOT
    cx10 = Operator([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
    gate_defs.append(('cx_rev', 0, cx10))

    for (qc, curr_op, t_cnt, last_ops) in batch_data:
        for name, cost, gate_mat in gate_defs:
            
            if t_cnt + cost > max_t: continue

            targets = [0, 1] 
            if 'cx' in name: targets = [None] 

            for qubit in targets:
                # --- PRUNING ---
                prev = last_ops[qubit] if qubit is not None else last_ops[0]
                
                if name == 'h' and prev == 'h': continue
                if 'cx' in name and 'cx' in str(prev): continue
                if name == 't' and prev == 't': continue
                
                # --- APPLY GATE (FIXED) ---
                branch_qc = qc.copy()
                new_ops = list(last_ops)
                
                # Use native Qiskit methods instead of opaque unitary matrices
                if name == 'h': 
                    branch_qc.h(qubit)
                    new_ops[qubit] = 'h'
                    # Matrix Math
                    if qubit == 0: full_gate = np.kron(np.eye(2), gate_mat.data) 
                    else: full_gate = np.kron(gate_mat.data, np.eye(2))
                        
                elif name == 's': 
                    branch_qc.s(qubit)
                    new_ops[qubit] = 's'
                    if qubit == 0: full_gate = np.kron(np.eye(2), gate_mat.data) 
                    else: full_gate = np.kron(gate_mat.data, np.eye(2))

                elif name == 't': 
                    branch_qc.t(qubit)
                    new_ops[qubit] = 't'
                    if qubit == 0: full_gate = np.kron(np.eye(2), gate_mat.data) 
                    else: full_gate = np.kron(gate_mat.data, np.eye(2))
                    
                elif name == 'cx':
                    branch_qc.cx(0, 1)
                    new_ops = ['cx', 'cx']
                    full_gate = gate_mat.data
                    
                elif name == 'cx_rev':
                    branch_qc.cx(1, 0)
                    new_ops = ['cx', 'cx']
                    full_gate = gate_mat.data

                # --- EVOLVE MATRIX ---
                # New = Gate * Old (Standard QM composition)
                new_matrix = np.dot(full_gate, curr_op) 
                
                # --- METRIC ---
                # Calculate Process Fidelity
                # Using Qiskit's optimized C++ implementation for speed
                fid = process_fidelity(Operator(new_matrix), target_op)

                key = operator_to_key(new_matrix)
                local_res.append((fid, key, new_matrix, branch_qc, t_cnt + cost, tuple(new_ops)))

    return local_res

# --- MAIN SOLVER ---
class UnitarySolver:
    def __init__(self, target_unitary, max_t=1):
        self.target_op = Operator(target_unitary)
        self.max_t = max_t
        self.visited = set()

    def solve(self):
        initial_op = np.eye(4, dtype=complex)
        initial_qc = QuantumCircuit(2)
        
        frontier = [(initial_qc, initial_op, 0, (None, None))]
        self.visited.add(operator_to_key(initial_op))
        
        best_fid = -1.0
        best_qc = None
        total_checked = 0
        depth = 0
        workers = os.cpu_count()

        print(f"Searching Unitary Space (Max T={self.max_t})...")
        
        with ProcessPoolExecutor(max_workers=workers) as executor:
            while frontier:
                chunks = [frontier[i:i + BATCH_SIZE] for i in range(0, len(frontier), BATCH_SIZE)]
                frontier = []

                results = executor.map(expand_unitary_batch, 
                                     [(c, self.target_op, self.max_t) for c in chunks])
                
                for batch_res in tqdm(results, total=len(chunks), desc=f"Depth {depth}"):
                    for fid, key, op, qc, t, ops in batch_res:
                        total_checked += 1
                        
                        if fid > best_fid:
                            best_fid = fid
                            best_qc = qc
                            if best_fid > 0.9999:
                                return best_qc, best_fid

                        if key not in self.visited:
                            self.visited.add(key)
                            frontier.append((qc, op, t, ops))
                
                depth += 1
                if total_checked > SAFETY_LIMIT:
                    break
        
        return best_qc, best_fid

def load_unitary(file_path):
    """Loads a unitary matrix from .npy or .txt"""
    print(f"Loading matrix from: {file_path}")
    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if ext == '.npy':
            matrix = np.load(file_path)
        elif ext == '.txt':
            # Assumes space or comma separated complex numbers or just values
            # Try loadtxt first
            try:
                matrix = np.loadtxt(file_path, dtype=complex)
            except:
                # Fallback: Read file and parse safely if it has python-style complex 'j'
                with open(file_path, 'r') as f:
                    content = f.read().replace('i', 'j') # Handle 'i' notation
                    # This is risky for arbitrary files but works for simple lists
                    # Better to stick to numpy formats
                    pass
                raise ValueError("Could not parse TXT. Ensure it is space-delimited complex numbers.")
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
            
        # Reshape to 4x4 if flattened
        if matrix.shape != (4, 4):
            # Try to reshape if it has 16 elements
            if matrix.size == 16:
                matrix = matrix.reshape((4, 4))
            else:
                raise ValueError(f"Matrix must be 4x4. Found shape {matrix.shape}")
                
        return matrix
        
    except Exception as e:
        print(f"Error loading file: {e}")
        exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum BFS Solver for Arbitrary Unitary")
    parser.add_argument("--input", type=str, help="Path to .npy or .txt file containing the 4x4 Unitary matrix")
    parser.add_argument("--max_t", type=int, default=1, help="Maximum number of T gates allowed (Default: 1)")
    
    args = parser.parse_args()

    if args.input:
        target_matrix = load_unitary(args.input)
    else:
        print("No input file specified. Using random 4x4 unitary for testing.")
        target_matrix = random_unitary(4, seed=42).data

    solver = UnitarySolver(target_matrix, max_t=args.max_t)
    qc, fid = solver.solve()
    
    print(f"\n=== RESULTS ===")
    print(f"Best Process Fidelity: {fid:.6f}")
    if qc:
        print("\nQASM:")
        print(qasm2.dumps(qc))
