import argparse
import sys
import logging
import numpy as np
import mpmath
import traceback
import copy
from collections import deque

# --- 1. SILENCE LOGGERS ---
logging.getLogger('qiskit').setLevel(logging.CRITICAL)
logging.getLogger('stevedore').setLevel(logging.CRITICAL)
logging.getLogger('matplotlib').setLevel(logging.CRITICAL)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# --- DEPENDENCY CHECKS ---
try:
    import pygridsynth as gs
except ImportError:
    print("[!] Error: pygridsynth not found. Please pip install pygridsynth")
    sys.exit(1)

try:
    import rmsynth.core as rm
    from rmsynth.optimizer import Optimizer as RMOptimizer
except ImportError:
    print("[!] Error: rmsynth not found. Please pip install rmsynth")
    sys.exit(1)

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator, process_fidelity
from qiskit.synthesis import qs_decomposition
import qiskit.qasm2
from qiskit.transpiler.passes import BasisTranslator, CommutativeCancellation
from qiskit.transpiler import PassManager
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel

# --- AUTOMATED OPTIMIZATION LIBRARY ---

class OptimizationLibrary:
    """
    Generates and stores optimal Clifford+T sequences up to a certain depth.
    """
    def __init__(self, max_depth=4):
        self.library = {} # Key: Approx Unitary Hash, Value: List of gate tuples
        self.max_depth = max_depth
        self.basis = {
            'h': np.array([[1, 1], [1, -1]]) / np.sqrt(2),
            't': np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]]),
            'tdg': np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]]),
            's': np.array([[1, 0], [0, 1j]]),
            'sdg': np.array([[1, 0], [0, -1j]]),
            'z': np.array([[1, 0], [0, -1]]),
            'x': np.array([[0, 1], [1, 0]])
        }
        self._build()

    def _hash_unitary(self, U):
        """
        Creates a hashable key for a unitary matrix, invariant to global phase.
        We normalize the top-left element to be real/positive to handle phase equivalence.
        """
        # Phase normalization
        flat = U.flatten()
        pivot = None
        for val in flat:
            if np.abs(val) > 1e-5:
                pivot = val
                break
        if pivot is None: return 0 # Should not happen for unitary
        
        # Remove global phase
        U_norm = U * (np.conj(pivot) / np.abs(pivot))
        
        # Round complex numbers to avoid float drift issues in hash
        U_round = np.round(U_norm, 4) 
        return tuple(U_round.flatten())

    def _build(self):
        logger.info(f"     [+] Building Optimization Library (Depth {self.max_depth})...")
        
        # BFS Queue: (current_unitary, gate_sequence_list)
        queue = deque([ (np.eye(2, dtype=complex), []) ])
        self.library[self._hash_unitary(np.eye(2))] = []
        
        count = 0
        while queue:
            curr_U, curr_seq = queue.popleft()
            
            if len(curr_seq) >= self.max_depth:
                continue

            # Try adding every basis gate
            for name, mat in self.basis.items():
                # Pruning: Don't add inverse of last gate (trivial identity)
                if curr_seq:
                    last = curr_seq[-1]
                    if (name == 'h' and last == 'h') or \
                       (name == 't' and last == 'tdg') or \
                       (name == 'tdg' and last == 't') or \
                       (name == 's' and last == 'sdg') or \
                       (name == 'sdg' and last == 's'):
                        continue

                next_U = mat @ curr_U
                next_seq = curr_seq + [name]
                key = self._hash_unitary(next_U)

                # Store if this is the first (shortest) time we've seen this unitary
                if key not in self.library:
                    self.library[key] = next_seq
                    queue.append((next_U, next_seq))
                    count += 1
                
                # Note: We do NOT update if key exists, because BFS guarantees 
                # we find shortest sequences first.
        
        logger.info(f"     [+] Library built: {len(self.library)} optimal sequences found.")

    def optimize_window(self, gate_names):
        """
        Input: List of gate names e.g. ['h', 't', 'h']
        Output: Shorter list if found, else None
        """
        # Compute unitary of the window
        U = np.eye(2, dtype=complex)
        for name in gate_names:
            if name not in self.basis: return None # Unknown gate (e.g. rz)
            U = self.basis[name] @ U # Apply in reverse order (standard QM notation vs circuit order)
            # Actually, standard matrix multiplication applies Left-to-Right if we treat state as column?
            # Qiskit order: U_n ... U_1 |psi>.
            # Our BFS did `mat @ curr_U`. This means the new gate is applied AFTER.
            # So here we must also multiply on the Left.
        
        key = self._hash_unitary(U)
        if key in self.library:
            optimal_seq = self.library[key]
            if len(optimal_seq) < len(gate_names):
                return optimal_seq
        return None

# --- OPTIMIZATION PASSES ---

def apply_library_optimization(qc, lib):
    """
    Scans the circuit with a sliding window and replaces sequences using the library.
    """
    # We break the circuit into chunks of single-qubit gates
    # Multi-qubit gates (CX) act as separators
    new_qc = QuantumCircuit(qc.num_qubits)
    new_qc.global_phase = qc.global_phase
    
    buffers = {i: [] for i in range(qc.num_qubits)}

    def flush_buffer(q_idx):
        if not buffers[q_idx]: return
        ops = buffers[q_idx] # List of (name, inst)
        
        # We try to match the largest window possible, starting from max_depth
        # This is a simple greedy approach
        i = 0
        while i < len(ops):
            matched = False
            # Try windows of decreasing size, max size determined by library depth
            # But practically, we want to find reductions, so we look at windows 
            # size 2 up to e.g. 6 (if we have a sequence of 6 that reduces to 1)
            # Since our library only *stores* up to depth 4, we can check windows of size 4.
            # If a window of 4 reduces to 1, we win.
            
            for width in range(lib.max_depth + 1, 1, -1): # Try window sizes
                if i + width <= len(ops):
                    window = [op[0] for op in ops[i : i+width]]
                    opt_seq = lib.optimize_window(window)
                    
                    if opt_seq is not None:
                        # Found a reduction!
                        # Emit the optimized sequence
                        target = new_qc.qubits[q_idx]
                        for gate_name in opt_seq:
                            if gate_name == 'h': new_qc.h(target)
                            elif gate_name == 't': new_qc.t(target)
                            elif gate_name == 'tdg': new_qc.tdg(target)
                            elif gate_name == 's': new_qc.s(target)
                            elif gate_name == 'sdg': new_qc.sdg(target)
                            elif gate_name == 'z': new_qc.z(target)
                            elif gate_name == 'x': new_qc.x(target)
                        
                        i += width
                        matched = True
                        break # Break width loop, continue main loop
            
            if not matched:
                # No optimization found starting at i, emit ops[i]
                inst = ops[i][1]
                new_qc.append(inst.operation, [new_qc.qubits[q_idx]])
                i += 1
                
        buffers[q_idx] = []

    for inst in qc.data:
        name = inst.operation.name
        q_indices = [qc.find_bit(q).index for q in inst.qubits]
        
        # Check if single qubit gate in our basis
        if len(q_indices) == 1 and name in ['h', 't', 'tdg', 's', 'sdg', 'z', 'x']:
            buffers[q_indices[0]].append((name, inst))
        else:
            # Separator encountered
            for q in q_indices: flush_buffer(q)
            new_qc.append(inst.operation, [new_qc.qubits[i] for i in q_indices])

    for i in range(qc.num_qubits): flush_buffer(i)
    return new_qc

# --- HELPER CLASSES (From Previous) ---

class LinearRestoration:
    @staticmethod
    def solve(target_forms, num_qubits):
        mat = copy.deepcopy(target_forms)
        n = num_qubits
        ops = []
        for col in range(n):
            pivot_row = -1
            for row in range(col, n):
                if (mat[row] >> col) & 1:
                    pivot_row = row
                    break
            if pivot_row == -1: continue
            if pivot_row != col:
                mat[col], mat[pivot_row] = mat[pivot_row], mat[col]
                ops.append(('swap', col, pivot_row))
            for row in range(n):
                if row != col:
                    if (mat[row] >> col) & 1:
                        mat[row] ^= mat[col]
                        ops.append(('cx', col, row))
        return reversed(ops)

class RMSynthOptimizerWrapper:
    def __init__(self, num_qubits, effort=3, decoder="auto"):
        self.n = num_qubits
        self.effort = effort
        self.decoder = decoder
        self.z8_map = {'t': 1, 'tdg': 7, 's': 2, 'sdg': 6, 'z': 4, 'rz': None, 'p': None, 'u1': None}

    def _is_z8(self, angle, atol=1e-12):
        angle = angle % (2 * np.pi)
        k = angle / (np.pi / 4)
        return np.isclose(k, np.round(k), atol=atol)

    def qiskit_to_rm_gate(self, name, params, qubit_indices, qubit_map):
        if name == 'cx':
            c = qubit_map[qubit_indices[0]]
            t = qubit_map[qubit_indices[1]]
            return rm.Gate("cnot", ctrl=c, tgt=t)
        k = 0
        if name in self.z8_map and self.z8_map[name] is not None:
            k = self.z8_map[name]
        elif name in ['rz', 'p', 'u1']:
            angle = float(params[0])
            if self._is_z8(angle):
                k = int(round(angle / (np.pi/4))) % 8
        if k == 0: return None
        q = qubit_map[qubit_indices[0]]
        return rm.Gate("phase", q=q, k=k)

    def optimize_block(self, block_gates, active_qubits, qc_ref):
        if not block_gates: return QuantumCircuit(self.n)
        local_n = len(active_qubits)
        sorted_qubits = sorted(list(active_qubits))
        q_map = {global_q: i for i, global_q in enumerate(sorted_qubits)}
        inv_map = {i: global_q for i, global_q in enumerate(sorted_qubits)}

        rm_circ = rm.Circuit(local_n)
        for inst in block_gates:
            q_indices = [qc_ref.find_bit(q).index for q in inst.qubits]
            g = self.qiskit_to_rm_gate(inst.operation.name.lower(), inst.operation.params, q_indices, q_map)
            if g:
                if g.kind == 'cnot': rm_circ.add_cnot(g.ctrl, g.tgt)
                elif g.kind == 'phase': rm_circ.add_phase(g.q, g.k)

        coeffs = {}
        forms = [1 << i for i in range(local_n)]
        for op in rm_circ.ops:
            if op.kind == 'cnot': forms[op.tgt] ^= forms[op.ctrl]
            elif op.kind == 'phase':
                mask = forms[op.q]
                coeffs[mask] = (coeffs.get(mask, 0) + op.k) % 8

        try:
            diag_vec = rm.coeffs_to_vec(coeffs, local_n)
            diag_circ = rm.synthesize_from_coeffs(diag_vec, local_n)
            opt = RMOptimizer(decoder=self.decoder, effort=self.effort)
            optimized_diag_circ, _ = opt.optimize(diag_circ)
        except Exception:
            diag_vec = rm.coeffs_to_vec(coeffs, local_n)
            optimized_diag_circ = rm.synthesize_from_coeffs(diag_vec, local_n)

        linear_ops = LinearRestoration.solve(forms, local_n)
        new_qc = QuantumCircuit(self.n)
        for op in optimized_diag_circ.ops:
            if op.kind == 'cnot': new_qc.cx(inv_map[op.ctrl], inv_map[op.tgt])
            elif op.kind == 'phase': self._append_phase(new_qc, op.k, inv_map[op.q])
        for op_type, a, b in linear_ops:
            if op_type == 'cx': new_qc.cx(inv_map[a], inv_map[b])
            elif op_type == 'swap': new_qc.swap(inv_map[a], inv_map[b])
        return new_qc

    def _append_phase(self, qc, k, qubit_idx):
        k = k % 8
        if k == 1: qc.t(qubit_idx)
        elif k == 2: qc.s(qubit_idx)
        elif k == 3: qc.s(qubit_idx); qc.t(qubit_idx)
        elif k == 4: qc.z(qubit_idx)
        elif k == 5: qc.z(qubit_idx); qc.t(qubit_idx)
        elif k == 6: qc.sdg(qubit_idx)
        elif k == 7: qc.tdg(qubit_idx)

    def run(self, qc):
        final_qc = QuantumCircuit(qc.num_qubits)
        final_qc.global_phase = qc.global_phase
        current_block = []
        current_active_qubits = set()
        for inst in qc.data:
            name = inst.operation.name.lower()
            is_clif_t = False
            if name == 'cx': is_clif_t = True
            elif name in self.z8_map: is_clif_t = True
            elif name in ['rz', 'p', 'u1']:
                ang = float(inst.operation.params[0])
                if self._is_z8(ang): is_clif_t = True
            if is_clif_t:
                current_block.append(inst)
                for q in inst.qubits: current_active_qubits.add(qc.find_bit(q).index)
            else:
                if current_block:
                    optimized_block = self.optimize_block(current_block, current_active_qubits, qc)
                    for opt_inst in optimized_block.data:
                         final_qc.append(opt_inst.operation, [final_qc.qubits[final_qc.find_bit(q).index] for q in opt_inst.qubits])
                    current_block = []
                    current_active_qubits = set()
                targets = [final_qc.qubits[qc.find_bit(q).index] for q in inst.qubits]
                final_qc.append(inst.operation, targets)
        if current_block:
            optimized_block = self.optimize_block(current_block, current_active_qubits, qc)
            for opt_inst in optimized_block.data:
                    final_qc.append(opt_inst.operation, [final_qc.qubits[final_qc.find_bit(q).index] for q in opt_inst.qubits])
        return final_qc

# --- SMART MERGE PASS ---

def post_process_discrete(qc):
    pm_pre = PassManager([CommutativeCancellation()])
    qc = pm_pre.run(qc)

    new_qc = QuantumCircuit(qc.num_qubits)
    new_qc.global_phase = qc.global_phase
    accumulators = {i: 0 for i in range(qc.num_qubits)}

    def flush_accumulator(q_idx):
        k = accumulators[q_idx] % 8
        accumulators[q_idx] = 0
        target = new_qc.qubits[q_idx]
        if k == 0: return 
        elif k == 1: new_qc.t(target)
        elif k == 2: new_qc.s(target)
        elif k == 3: new_qc.s(target); new_qc.t(target)
        elif k == 4: new_qc.s(target); new_qc.s(target)
        elif k == 5: new_qc.s(target); new_qc.s(target); new_qc.t(target)
        elif k == 6: new_qc.sdg(target)
        elif k == 7: new_qc.tdg(target)

    for inst in qc.data:
        name = inst.operation.name
        q_indices = [qc.find_bit(q).index for q in inst.qubits]
        k = 0; is_z_rot = False
        if name in ['t', 'tdg', 's', 'sdg', 'z']:
            is_z_rot = True
            if name == 't': k=1
            elif name == 'tdg': k=7
            elif name == 's': k=2
            elif name == 'sdg': k=6
            elif name == 'z': k=4
        elif name in ['rz', 'p', 'u1']:
            ang = float(inst.operation.params[0])
            if np.isclose(ang/(np.pi/4), np.round(ang/(np.pi/4)), atol=1e-5):
                is_z_rot=True; k=int(np.round(ang/(np.pi/4)))

        if is_z_rot: accumulators[q_indices[0]] += k
        elif name == 'cx':
            flush_accumulator(q_indices[1])
            new_qc.cx(q_indices[0], q_indices[1])
        else:
            for q in q_indices: flush_accumulator(q)
            if name == 'x':
                t = new_qc.qubits[q_indices[0]]
                new_qc.h(t); new_qc.s(t); new_qc.s(t); new_qc.h(t)
            else:
                new_qc.append(inst.operation, [new_qc.qubits[i] for i in q_indices])

    for i in range(qc.num_qubits): flush_accumulator(i)
    return new_qc

# --- MAIN LOOP ---

def count_circuit_ops(qc):
    return sum(qc.count_ops().values())

def run_pass_if_better(qc, pass_func, pass_name):
    start_ops = count_circuit_ops(qc)
    try: new_qc = pass_func(qc)
    except Exception as e:
        logger.warning(f"     [!] {pass_name} crashed: {e}")
        return qc
    end_ops = count_circuit_ops(new_qc)
    if end_ops <= start_ops:
        if end_ops < start_ops: logger.info(f"     [+] {pass_name}: Improved {start_ops} -> {end_ops}")
        return new_qc
    return qc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    parser.add_argument("--epsilon", type=float, default=1e-10)
    parser.add_argument("--loops", type=int, default=5, help="Optimization passes")
    parser.add_argument("--lib_depth", type=int, default=4, help="Identity library depth")
    args = parser.parse_args()

    logger.info(f"Loading {args.input_file}...")
    U_target = normalize_to_unitary(load_unitary(args.input_file))

    # 1. DECOMPOSE
    logger.info("1. Decomposing...")
    qc = decompose_to_base_basis(qs_decomposition(U_target))
    
    # 2. DISCRETIZE 
    logger.info(f"2. Discretizing (eps={args.epsilon})...")
    qc = apply_discretization(qc, args.epsilon)
    
    # 3. SETUP OPTIMIZERS
    logger.info(f"3. Initializing Optimizers (Library Depth {args.lib_depth})...")
    rm_opt = RMSynthOptimizerWrapper(qc.num_qubits, effort=3)
    lib = OptimizationLibrary(max_depth=args.lib_depth) # Automatically builds 4^7 identities
    
    def step_rmsynth(c): return rm_opt.run(c)
    def step_library(c): return apply_library_optimization(c, lib)
    def step_merge(c): return post_process_discrete(c)
    def step_qiskit(c): return transpile(c, basis_gates=['h','cx','t','tdg','s','sdg'], optimization_level=3)

    # 4. LOOP
    logger.info(f"4. Starting Greedy Optimization Loop ({args.loops} iterations)...")
    for i in range(args.loops):
        logger.info(f"   > Iteration {i+1}:")
        qc = run_pass_if_better(qc, step_rmsynth, "RMSynth")
        qc = run_pass_if_better(qc, step_library, "AutoLibrary")
        qc = run_pass_if_better(qc, step_merge, "SmartMerge")
        qc = run_pass_if_better(qc, step_qiskit, "QiskitLvl3")

    # 5. FINAL
    logger.info("5. Final Basis Enforcement...")
    qc_final = post_process_discrete(qc)

    # Report
    fid = process_fidelity(Operator(U_target), Operator(qc_final))
    ops = qc_final.count_ops()
    print("\n" + "="*40)
    print("            FINAL REPORT")
    print("="*40)
    print(f" Fidelity         : {fid:.10f}")
    print(f" Total Gates      : {sum(ops.values())}")
    print("-" * 40)
    print(f" Breakdown        : {dict(ops)}")
    print("="*40)

    with open(args.output_file, 'w') as f:
        f.write(qiskit.qasm2.dumps(qc_final))
    logger.info(f"Saved to {args.output_file}")

# --- BOILERPLATE HELPERS ---
def normalize_to_unitary(U):
    try: V, _, Wh = np.linalg.svd(U); return V @ Wh
    except Exception: return U
def load_unitary(filepath):
    try:
        if filepath.endswith('.npy'): return np.load(filepath)
        return np.loadtxt(filepath, dtype=complex)
    except Exception as e: sys.exit(f"Error loading file: {e}")
def decompose_to_base_basis(qc):
    try: return PassManager([BasisTranslator(sel, ['rz', 'h', 'cx', 't', 'tdg', 's', 'sdg', 'z', 'x'])]).run(qc)
    except Exception: return qc
def apply_discretization(qc, epsilon):
    final_qc = QuantumCircuit(qc.num_qubits)
    final_qc.global_phase = qc.global_phase
    eps_mp = mpmath.mpf(str(epsilon))
    for inst in qc.data:
        name = inst.operation.name.lower()
        targets = [final_qc.qubits[qc.find_bit(q).index] for q in inst.qubits]
        if name in ['rz', 'p', 'u1']:
            angle = float(inst.operation.params[0]) % (2 * np.pi)
            if np.isclose(angle, 0, atol=1e-12): continue
            if np.isclose((angle%(2*np.pi))/(np.pi/4), np.round((angle%(2*np.pi))/(np.pi/4)), atol=1e-12):
                final_qc.rz(angle, targets[0]) 
            else:
                try:
                    seq = reversed(gs.gridsynth_gates(mpmath.mpf(str(angle)), eps_mp))
                    for g in seq:
                        gs_name = str(g).upper()
                        if gs_name == 'H': final_qc.h(targets[0])
                        elif gs_name == 'T': final_qc.t(targets[0])
                        elif gs_name == 'S': final_qc.s(targets[0])
                        elif gs_name == 'X': final_qc.x(targets[0])
                        elif gs_name == 'Z': final_qc.z(targets[0])
                        elif gs_name == 'W': final_qc.global_phase += np.pi/4
                except: sys.exit(1)
        else: final_qc.append(inst.operation, targets)
    return final_qc

if __name__ == "__main__":
    main()
