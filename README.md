# iQuHack-2026

## Usage: Optimize Unitaries

### Run all unitaries in batch mode (unitary1.npy to unitary11.npy):

```bash
python optimize_unitaries.py --batch
```

Add `--verbose` for more detailed output:

```bash
python optimize_unitaries.py --batch --verbose
```

### Run a single unitary (example: unitary 5):

```bash
python optimize_unitaries.py unitary5.npy unitary5_optimized.qasm
```

You can also add `--verbose` or other flags as needed.

### Run Challenge 12 (unitary12):

```bash
python optimize_unitaries.py --challenge12 --json unitary12.json --output unitary12_optimized.qasm
```

This will optimize and output the circuit for unitary 12.
