import numpy as np
import scipy.linalg
from qiskit.quantum_info import random_statevector
import os

def save_unitary_numpy(filename="unitary_matrix.npy"):
    # 1. Generate Target
    target_psi = random_statevector(4, seed=42).data
    
    # 2. Synthesize Unitary (QR Method)
    A = np.eye(4, dtype=complex)
    A[:, 0] = target_psi
    Q, R = scipy.linalg.qr(A)
    
    # 3. Phase Correction (Enforce Exact Match)
    nonzero_idx = np.argmax(np.abs(Q[:, 0]))
    phase_correction = target_psi[nonzero_idx] / Q[nonzero_idx, 0]
    Q[:, 0] *= phase_correction
    
    U_exact = Q

    # 4. Save as Numpy Binary (.npy)
    np.save(filename, U_exact)
    
    print(f"Successfully saved 4x4 unitary to '{filename}'")
    print(f"File size: {os.path.getsize(filename)} bytes")

    # Optional: Verify by loading it back
    loaded_U = np.load(filename)
    if np.allclose(U_exact, loaded_U):
        print("Verification: Loaded matrix matches original exactly.")

if __name__ == "__main__":
    save_unitary_numpy()
