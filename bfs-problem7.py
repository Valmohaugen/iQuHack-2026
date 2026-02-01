import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
from collections import deque
from qiskit import QuantumCircuit, transpile, qasm2
from qiskit.quantum_info import random_statevector, Statevector, state_fidelity
from tqdm import tqdm

# --- CONFIGURATION ---
TARGET_SEED = 42
MAX_T_GATES = 3      # Increase to 2 if you have a powerful machine (32+ cores)
BATCH_SIZE = 1000    # Larger batches reduce multiprocessing overhead
SAFETY_LIMIT = 10_000_000  # Stop if we check this many states

# --- TARGET SETUP ---
TARGET_STATE = random_statevector(4, seed=TARGET_SEED)

# --- HELPER: Hashing States ---
def state_to_key(state):
    """
    Converts a statevector to a compact bytes hash.
    Rounding is crucial to treat numerically close states as identical.
    """
    # Round to 5 decimal places to ignore floating point noise
    return np.round(state.data, 5).tobytes()

# --- HELPER: Pruning Logic ---
def is_redundant(gate_name, qubit_idx, last_ops):
    """
    Checks strictly if applying 'gate_name' on 'qubit_idx' creates a 
    known identity or inefficient sequence based on history.
    
    last_ops: Tuple (last_op_q0, last_op_q1)
    """
    prev_op = last_ops[qubit_idx] if qubit_idx is not None else None
    
    # RULE 1: Self-Inverse / Toggle (H*H=I, X*X=I)
    # If we just did H, don't do H again.
    if gate_name == 'h' and prev_op == 'h':
        return True
    
    # RULE 2: T-Gate Efficiency (T*T = S)
    # T*T costs 2 budget. S costs 0 budget. 
    # Never apply T if the last gate was T (we should have used S instead).
    if gate_name == 't' and prev_op == 't':
        return True
    
    # RULE 3: S-Gate Grouping (S*S = Z)
    # Prevent infinite S chains. S*S*S*S = I.
    # We allow S, but maybe limit it. For simplicity, we just prevent S*S*S (Sdg).
    # (Implementation complexity: keeping track of counts. 
    # Simple heuristic: Just avoid T after S to enforce T-S ordering if commuting).
    
    # RULE 4: CX Redundancy
    # CX(0,1) * CX(0,1) = I
    # We track CX as a distinct op on both qubits.
    if gate_name == 'cx_01' and prev_op == 'cx_01':
        return True
    if gate_name == 'cx_10' and prev_op == 'cx_10':
        return True

    return False

# --- WORKER FUNCTION ---
# --- WORKER FUNCTION ---
def expand_batch(batch_data):
    """
    Expands a list of states.
    Input: List of (qc, current_state, t_count, last_ops)
    Output: List of NEW candidates
    """
    local_results = []
    
    # 1. Define the fresh starting state exactly like Verification does
    # (Doing this ensures 1:1 consistency with the final check)
    ZERO_STATE = Statevector.from_label('00')

    # Operations: Name, Cost
    gate_set = [
        ('h', 0), ('s', 0), ('t', 1), 
        ('cx_01', 0), ('cx_10', 0)
    ]

    # Note: We ignore the input 'st' (the second item) to avoid the previous logic bug.
    # We rely purely on the circuit 'qc' to generate the state.
    for (qc, _, t_count, last_ops) in batch_data:

        for name, cost in gate_set:
            # 1. Budget Check
            new_t = t_count + cost
            if new_t > MAX_T_GATES:
                continue

            # 2. Identify Targets
            targets = [0, 1]
            if 'cx' in name:
                targets = [None] 

            for qubit in targets:
                # --- PRUNING ---
                check_q = qubit if qubit is not None else 0
                if is_redundant(name, check_q, last_ops):
                    continue

                # --- APPLY GATE ---
                branch_qc = qc.copy()
                new_last_ops = list(last_ops)

                if name == 'h':
                    branch_qc.h(qubit)
                    new_last_ops[qubit] = 'h'
                elif name == 's':
                    branch_qc.s(qubit)
                    new_last_ops[qubit] = 's'
                elif name == 't':
                    branch_qc.t(qubit)
                    new_last_ops[qubit] = 't'
                elif name == 'cx_01':
                    branch_qc.cx(0, 1)
                    new_last_ops[0] = 'cx_01'
                    new_last_ops[1] = 'cx_01'
                elif name == 'cx_10':
                    branch_qc.cx(1, 0)
                    new_last_ops[0] = 'cx_10'
                    new_last_ops[1] = 'cx_10'

                # --- EVOLVE STATE (THE FIX) ---
                # Previous Bug: We did st.evolve(branch_qc), which applied the whole circuit 
                # to an already evolved state (double application).
                #
                # New Logic: We evolve the fresh ZERO_STATE by the full circuit.
                # This matches the "Verification" logic 100%.
                branch_st = ZERO_STATE.evolve(branch_qc)

                # --- CALCULATE FIDELITY ---
                fid = state_fidelity(branch_st, TARGET_STATE)

                # --- PACK RESULT ---
                key = state_to_key(branch_st)
                local_results.append((fid, key, branch_st, branch_qc, new_t, tuple(new_last_ops)))

    return local_results

# --- MAIN CONTROLLER ---
def parallel_bfs_search():
    # Initial State: |00>
    initial_qc = QuantumCircuit(2)
    initial_st = Statevector.from_label('00')
    
    # State tuple: (Circuit, Statevector, T_Count, Last_Ops_Tuple)
    # Last_Ops initialized to (None, None)
    frontier = [(initial_qc, initial_st, 0, (None, None))]
    
    # The "Library of Identities" (Visited Set)
    visited = set()
    visited.add(state_to_key(initial_st))
    
    best_fidelity = -1.0
    best_qc = None
    
    total_checked = 0
    depth = 0
    workers = os.cpu_count()
    
    print(f"--- QUANTUM BFS INITIALIZED ---")
    print(f"Target: Seed {TARGET_SEED} | Max T-Gates: {MAX_T_GATES}")
    print(f"Workers: {workers} | Batch Size: {BATCH_SIZE}")
    print(f"-------------------------------")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        while frontier:
            print(f"\n[Depth {depth}] Processing {len(frontier)} states...")
            
            # Chunking for workers
            chunks = [frontier[i:i + BATCH_SIZE] for i in range(0, len(frontier), BATCH_SIZE)]
            frontier = [] # Clear for next generation
            
            # Map workers
            results_gen = executor.map(expand_batch, chunks)
            
            # Process results with progress bar
            # Using total=len(chunks) ensures tqdm knows when to finish
            for batch_results in tqdm(results_gen, total=len(chunks), unit="batch"):
                for fid, key, st, qc, t, ops in batch_results:
                    total_checked += 1
                    
                    # 1. Update Best
                    if fid > best_fidelity:
                        best_fidelity = fid
                        best_qc = qc
                    
                    # 2. Check "Identity Library" (Visited Set)
                    if key not in visited:
                        visited.add(key)
                        frontier.append((qc, st, t, ops))
            
            print(f"  -> Best Fidelity So Far: {best_fidelity:.6f}")
            print(f"  -> Unique States Found: {len(visited)}")
            
            depth += 1
            if total_checked > SAFETY_LIMIT:
                print("!!! Safety limit reached. Stopping search. !!!")
                break
                
    return best_qc, best_fidelity, total_checked

if __name__ == '__main__':
    # 1. Run the search
    best_circuit, fidelity, count = parallel_bfs_search()

    print(f"\n\n====== FINAL RESULTS ======")
    print(f"States Checked: {count}")
    print(f"Best Fidelity:  {fidelity:.6f}")

    if best_circuit:
        print("\nBest Circuit (QASM):")
        print(qasm2.dumps(best_circuit))

        # --- VERIFICATION STEP ---
        print(f"\n====== VERIFICATION ======")
        # 1. Initialize fresh |00> state
        verification_state = Statevector.from_label('00')
        
        # 2. Apply the best circuit found
        final_state = verification_state.evolve(best_circuit)
        
        # 3. Compute fidelity against the global TARGET_STATE
        check_fidelity = state_fidelity(final_state, TARGET_STATE)
        
        print(f"Target State:       {np.round(TARGET_STATE.data, 3)}")
        print(f"Result State:       {np.round(final_state.data, 3)}")
        print(f"Verified Fidelity:  {check_fidelity:.6f}")
        
        # 4. Assert correctness (floating point tolerance)
        if np.isclose(check_fidelity, fidelity):
            print(">> SUCCESS: Simulation matches search result.")
        else:
            print(">> WARNING: Discrepancy detected between search and verification.")
            
    else:
        print("\nNo circuit found (check constraints or max T-gates).")
