import numpy as np
from qiskit.quantum_info import random_unitary
import os

def save_random_unitary():
    # 1. Generate the random unitary (4x4 matrix for 2 qubits)
    # Note: random_unitary returns an Operator object, we need the .data property
    u_obj = random_unitary(4, seed=42)
    u_matrix = u_obj.data

    filename = "random_unitary_4x4.npy"

    # 2. Save as Numpy Binary
    np.save(filename, u_matrix)

    print(f"Successfully saved 4x4 unitary to '{filename}'")
    print(f"File size: {os.path.getsize(filename)} bytes")

    # 3. Verification: Load it back and check shape
    loaded_U = np.load(filename)
    print(f"Loaded Shape: {loaded_U.shape}")
    print("First row sample:", loaded_U[0])

if __name__ == "__main__":
    save_random_unitary()
