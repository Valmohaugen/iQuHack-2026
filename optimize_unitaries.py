"""
Clifford+T Circuit Optimizer for iQuHACK 2026

This module provides tools for optimizing quantum circuits to use minimal T-gates
while maintaining high fidelity with target unitaries.

Features:
---------
- Pattern recognition for common 2-qubit unitaries (QFT, Heisenberg, XX+YY, etc.)
- Brute force search for optimal low-T circuits
- Phase polynomial optimization via rmsynth
- Gridsynth-based Rz approximation

Usage:
------
Command line:
    python optimize_unitary.py input.npy output.qasm --effort 3

Python API:
    from optimize_unitary import load_unitary, decompose_and_optimize, analyze_unitary
    
    U = load_unitary("unitary1.npy")
    qc = decompose_and_optimize(U, effort=3)
    
    # Or analyze first
    analyze_unitary(U, name="My Unitary")
    best_qc, fid = brute_force_search(U, max_t=3, max_cx=3)

Gate Set:
---------
Allowed gates: H, T, Tdg, S, Sdg, CX (CNOT)
"""

import argparse
import json
import subprocess
import tempfile
import os
import sys
import time
import numpy as np
import matplotlib as mpl
from itertools import product
from scipy.linalg import polar, logm
from multiprocessing import Pool, cpu_count
from functools import partial

import pygridsynth as gs
import mpmath
import qiskit.qasm2
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator, process_fidelity
from qiskit.synthesis import qs_decomposition

from rmsynth.optimizer import Optimizer as RMSynthOptimizer
from rmsynth.core import (
    Circuit as RMSynthCircuit,
    extract_phase_coeffs,
    synthesize_from_coeffs,
    coeffs_to_vec,
    t_count_of_coeffs,
    optimize_coefficients as rmsynth_optimize_coefficients,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

RMSYNTH_CMD = "rmsynth-optimize"


# =============================================================================
# FILE I/O
# =============================================================================

def load_unitary(filepath):
    """
    Load a unitary matrix from file and ensure unitarity via polar decomposition.
    
    Args:
        filepath: Path to .npy or .txt file containing the unitary matrix
        
    Returns:
        numpy.ndarray: The unitary matrix
        
    Supported formats:
        - .npy: NumPy binary format
        - .txt: Plain text (space-separated complex numbers)
        
    Note:
        If a .txt file is empty, automatically tries .npy with same base name.
    """
    try:
        if filepath.endswith('.npy'):
            U = np.load(filepath)
        else:
            try:
                if os.path.getsize(filepath) == 0:
                    raise ValueError("File is empty")
                U = np.loadtxt(filepath, dtype=complex)
            except (ValueError, Exception):
                npy_path = os.path.splitext(filepath)[0] + '.npy'
                if os.path.exists(npy_path):
                    print(f"Note: Loading {npy_path} instead of empty {filepath}")
                    U = np.load(npy_path)
                else:
                    raise
        
        # Ensure unitarity via polar decomposition
        U, _ = polar(U)
        return U
        
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def is_diagonal(matrix):
    """Check if a matrix is diagonal."""
    return np.allclose(matrix, np.diag(np.diag(matrix)), atol=1e-6)


def get_z8_phases(diagonal_unitary):
    """
    Extract Z8 phase polynomial coefficients from a diagonal unitary.
    
    The diagonal has 2^n phases. This function computes the Möbius inversion
    to get the polynomial coefficients c_S for the phase polynomial:
    f(x) = sum_{S⊆x, S≠∅} c_S (mod 8)
    
    Args:
        diagonal_unitary: A diagonal unitary matrix
        
    Returns:
        tuple: (coefficients, approximation_error) or (None, 999.0) on failure
    """
    try:
        diags = np.diag(diagonal_unitary)
        if len(diags) == 0 or np.any(np.abs(diags) < 1e-9):
            return None, 999.0
        
        n = int(np.log2(len(diags)))
        if (1 << n) != len(diags):
            return None, 999.0
        
        phases = np.angle(diags)
        p_z8 = np.round(phases * (4 / np.pi)).astype(int) % 8
        
        # Möbius inversion
        coeffs = []
        for S in range(1, 1 << n):
            c_S = 0
            popS = bin(S).count('1')
            for T in range(1 << n):
                if (T & S) == T:
                    popT = bin(T).count('1')
                    sign = 1 if ((popS - popT) % 2 == 0) else -1
                    c_S = (c_S + sign * int(p_z8[T])) % 8
            coeffs.append(int(c_S))
        
        approx_phases = p_z8 * (np.pi / 4)
        max_error = float(np.max(np.abs(np.exp(1j * phases) - np.exp(1j * approx_phases))))
        
        return coeffs, max_error
    except Exception:
        return None, 999.0


# =============================================================================
# RMSYNTH INTEGRATION
# =============================================================================

# Phase gate value mapping (in units of π/4)
PHASE_VALUES = {'t': 1, 'tdg': 7, 's': 2, 'sdg': 6, 'z': 4}


def emit_phase(qc, phase, qubit, allow_s=True):
    """
    Emit phase gates for a given phase value (mod 8, in units of π/4).
    
    Args:
        qc: Target QuantumCircuit
        phase: Phase value (0-7, in units of π/4)
        qubit: Qubit or list containing qubit to apply phase to
        allow_s: If True, use S gates; if False, use only T/Tdg (for Challenge 12)
    """
    phase = phase % 8
    if phase == 0:
        return
    if allow_s:
        if phase == 1: qc.t(qubit)
        elif phase == 2: qc.s(qubit)
        elif phase == 3: qc.s(qubit); qc.t(qubit)
        elif phase == 4: qc.z(qubit)
        elif phase == 5: qc.z(qubit); qc.t(qubit)
        elif phase == 6: qc.sdg(qubit)
        elif phase == 7: qc.tdg(qubit)
    else:
        # No S gates - use only T/Tdg
        if phase == 1: qc.t(qubit)
        elif phase == 2: qc.t(qubit); qc.t(qubit)
        elif phase == 3: qc.t(qubit); qc.t(qubit); qc.t(qubit)
        elif phase == 4: qc.t(qubit); qc.t(qubit); qc.t(qubit); qc.t(qubit)
        elif phase == 5: qc.tdg(qubit); qc.tdg(qubit); qc.tdg(qubit)
        elif phase == 6: qc.tdg(qubit); qc.tdg(qubit)
        elif phase == 7: qc.tdg(qubit)


def qiskit_to_rmsynth(qc):
    """Convert Qiskit circuit to rmsynth Circuit (CNOT + phase gates only)."""
    rm_circ = RMSynthCircuit(qc.num_qubits)
    
    for inst in qc.data:
        name = inst.operation.name
        qubits = [qc.find_bit(q).index for q in inst.qubits]
        
        if name == 'cx':
            rm_circ.add_cnot(qubits[0], qubits[1])
        elif name in PHASE_VALUES:
            rm_circ.add_phase(qubits[0], PHASE_VALUES[name])
        elif name not in ['id', 'barrier']:
            return None  # Contains non-phase gates
    
    return rm_circ


def rmsynth_to_qiskit(rm_circ, num_qubits):
    """Convert rmsynth Circuit back to Qiskit QuantumCircuit."""
    qc = QuantumCircuit(num_qubits)
    
    for gate in rm_circ.ops:
        if gate.kind == 'cnot':
            qc.cx(gate.ctrl, gate.tgt)
        elif gate.kind == 'phase':
            emit_phase(qc, gate.k, gate.q)
    
    return qc


def optimize_with_rmsynth(qc, effort=1, decoder="auto"):
    """
    Optimize a circuit's phase polynomial using rmsynth.
    
    Args:
        qc: Qiskit QuantumCircuit (should only contain CNOT and phase gates)
        effort: Optimization effort level (1-3)
        decoder: Decoder to use ("auto", "ml-exact", "dumer", etc.)
        
    Returns:
        Optimized QuantumCircuit, or original if optimization fails
    """
    rm_circ = qiskit_to_rmsynth(qc)
    if rm_circ is None or rm_circ.t_count() == 0:
        return qc
    
    try:
        opt = RMSynthOptimizer(decoder=decoder, effort=effort)
        new_rm_circ, report = opt.optimize(rm_circ)
        
        if report.after_t < report.before_t:
            print(f"    rmsynth: {report.before_t} -> {report.after_t} T-gates")
            return rmsynth_to_qiskit(new_rm_circ, qc.num_qubits)
    except Exception as e:
        print(f"    rmsynth failed: {e}")
    
    return qc


def synthesize_diagonal_with_rmsynth(phases, n_qubits):
    """
    Synthesize a diagonal unitary using rmsynth's phase polynomial optimization.
    
    Args:
        phases: List of 2^n phases (angles in radians)
        n_qubits: Number of qubits
        
    Returns:
        QuantumCircuit with optimized T-count
    """
    if len(phases) != (1 << n_qubits):
        raise ValueError(f"Expected {1 << n_qubits} phases, got {len(phases)}")
    
    # Quantize phases to Z8 (multiples of π/4)
    p_z8 = [round(p / (np.pi / 4)) % 8 for p in phases]
    
    # Convert to Z8 coefficient dictionary using Möbius inversion
    # f(x) = sum_{S⊆x, S≠∅} c_S gives the phase for basis state |x⟩
    coeffs = {}
    for S in range(1, 1 << n_qubits):
        c_S = 0
        popS = bin(S).count('1')
        for T in range(1 << n_qubits):
            if (T & S) == T:  # T ⊆ S
                popT = bin(T).count('1')
                sign = 1 if ((popS - popT) % 2 == 0) else -1
                c_S = (c_S + sign * p_z8[T]) % 8
        if c_S != 0:
            coeffs[S] = c_S
    
    if not coeffs:
        return QuantumCircuit(n_qubits)
    
    # Use rmsynth to optimize the coefficients
    try:
        vec = coeffs_to_vec(coeffs, n_qubits)
        vec_opt, report, selected = rmsynth_optimize_coefficients(coeffs, n_qubits)
        
        # Synthesize optimized circuit
        rm_circ = synthesize_from_coeffs(vec_opt, n_qubits)
        return rmsynth_to_qiskit(rm_circ, n_qubits)
    except Exception as e:
        # Fallback to simple synthesis
        pass
    
    # Fallback: direct synthesis without optimization
    rm_circ = RMSynthCircuit(n_qubits)
    for mask, k in coeffs.items():
        # Find target qubit (lowest bit in mask)
        tgt = (mask & -mask).bit_length() - 1
        rest = mask & ~(1 << tgt)
        
        # Add CNOTs to accumulate parity
        ctrls = []
        b = rest
        while b:
            ctrl = (b & -b).bit_length() - 1
            ctrls.append(ctrl)
            b &= b - 1
        
        for ctrl in ctrls:
            rm_circ.add_cnot(ctrl, tgt)
        rm_circ.add_phase(tgt, k)
        for ctrl in reversed(ctrls):
            rm_circ.add_cnot(ctrl, tgt)
    
    return rmsynth_to_qiskit(rm_circ, n_qubits)


def rmsynth_brute_force_2q(U_target, max_t=5, max_cx=3, verbose=True):
    """
    Brute force search over Clifford+T circuits, applying rmsynth optimization
    to reduce T-count for each candidate circuit that achieves high fidelity.
    
    The key insight is that rmsynth can reduce T-count of circuits with
    equivalent phase polynomials, so we search for high-fidelity circuits
    and then optimize their T-count.
    
    Args:
        U_target: Target 2-qubit unitary
        max_t: Maximum T-gate count to search
        max_cx: Maximum CX gates
        verbose: Print progress
        
    Returns:
        tuple: (best_circuit, best_fidelity)
    """
    from scipy.linalg import polar
    U_clean, _ = polar(U_target)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"RMSYNTH-ENHANCED SEARCH (T≤{max_t}, CX≤{max_cx})")
        print(f"{'='*60}")
    
    best_fid = 0
    best_qc = None
    best_t = float('inf')
    
    gates_1q = ['h', 's', 'sdg', 't', 'tdg', 'id']
    
    # Try each CX count
    for n_cx in range(1, max_cx + 1):
        if verbose:
            print(f"\nSearching with {n_cx} CX gates...")
        
        n_layers = n_cx + 1
        cx_patterns = list(product([(0, 1), (1, 0)], repeat=n_cx))
        
        # Pre-filter gate combinations by T-count
        all_gates = list(product(gates_1q, repeat=2*n_layers))
        valid_gates = [g for g in all_gates if _count_t_gates(g) <= max_t]
        
        for cx_pattern in cx_patterns:
            for gates in valid_gates:
                t_count = _count_t_gates(gates)
                
                try:
                    U_circ = _build_unitary_fast(gates, cx_pattern, n_layers)
                    fid = _compute_fidelity_fast(U_clean, U_circ)
                    
                    # Keep track of high-fidelity circuits
                    if fid > 0.8:  # Candidate for rmsynth optimization
                        # Build the actual circuit
                        qc = QuantumCircuit(2)
                        for layer in range(n_layers):
                            g0, g1 = gates[2*layer], gates[2*layer + 1]
                            if g0 != 'id':
                                getattr(qc, g0)(0)
                            if g1 != 'id':
                                getattr(qc, g1)(1)
                            if layer < n_cx:
                                qc.cx(*cx_pattern[layer])
                        
                        # Try rmsynth optimization
                        rm_circ = qiskit_to_rmsynth(qc)
                        if rm_circ and rm_circ.t_count() > 0:
                            try:
                                opt = RMSynthOptimizer(decoder="ml-exact", effort=3)
                                new_rm_circ, report = opt.optimize(rm_circ)
                                
                                if report.after_t <= report.before_t:
                                    opt_qc = rmsynth_to_qiskit(new_rm_circ, 2)
                                    U_opt = Operator(opt_qc).data
                                    opt_fid = _compute_fidelity_fast(U_clean, U_opt)
                                    opt_t = opt_qc.count_ops().get('t', 0) + opt_qc.count_ops().get('tdg', 0)
                                    if opt_fid > best_fid or (opt_fid >= best_fid - 1e-6 and opt_t < best_t):
                                        best_fid = opt_fid
                                        best_qc = opt_qc
                                        best_t = opt_t
                                        if verbose:
                                            print(f"  [rmsynth] T={opt_t}, fid={opt_fid:.4f}")
                            except:
                                pass
                        
                        # Also track unoptimized version
                        if fid > best_fid or (fid >= best_fid - 1e-6 and t_count < best_t):
                            best_fid = fid
                            best_qc = qc
                            best_t = t_count
                            if verbose:
                                print(f"  T={t_count}, fid={fid:.4f}")
                    
                    elif fid > best_fid + 1e-6 or (fid > best_fid - 1e-6 and t_count < best_t):
                        # Lower fidelity but still track
                        qc = QuantumCircuit(2)
                        for layer in range(n_layers):
                            g0, g1 = gates[2*layer], gates[2*layer + 1]
                            if g0 != 'id':
                                getattr(qc, g0)(0)
                            if g1 != 'id':
                                getattr(qc, g1)(1)
                            if layer < n_cx:
                                qc.cx(*cx_pattern[layer])
                        best_fid = fid
                        best_qc = qc
                        best_t = t_count
                        if verbose and fid > 0.6:
                            print(f"  T={t_count}, fid={fid:.4f}")
                except Exception:
                    continue
    
    if verbose and best_qc:
        print(f"\nBest: T={best_t}, fid={best_fid:.6f}")
        print(best_qc.draw())
    
    return best_qc, best_fid


# =============================================================================
# GRIDSYNTH INTEGRATION
# =============================================================================

def append_rz_gridsynth(qc, qubit, angle):
    """
    Append Rz(angle) to circuit using exact Z8 gate or gridsynth approximation.
    
    Z8 angles (multiples of π/4) are implemented exactly with T/S gates.
    Other angles use gridsynth for Clifford+T approximation.
    """
    angle = angle % (2 * np.pi)
    if angle > np.pi:
        angle -= 2 * np.pi
    if np.isclose(angle, 0, atol=1e-9):
        return
    
    k_f = angle / (np.pi / 4)
    k = int(round(k_f))
    
    if abs(k_f - k) < 1e-6:
        # Exact Z8 gate
        k = k % 8
        if k == 1: qc.t(qubit)
        elif k == 2: qc.s(qubit)
        elif k == 3: qc.s(qubit); qc.t(qubit)
        elif k == 4: qc.z(qubit)
        elif k == 5: qc.sdg(qubit); qc.tdg(qubit)
        elif k == 6: qc.sdg(qubit)
        elif k == 7: qc.tdg(qubit)
    else:
        # Gridsynth approximation
        try:
            for g in gs.gridsynth_gates(angle, epsilon=1e-10):
                g_upper = g.upper()
                if g_upper == 'H': qc.h(qubit)
                elif g_upper == 'T': qc.t(qubit)
                elif g_upper == 'S': qc.s(qubit)
                elif g_upper == 'X': qc.h(qubit); qc.z(qubit); qc.h(qubit)
                elif g_upper == 'Z': qc.z(qubit)
                elif g == 't': qc.tdg(qubit)
                elif g == 's': qc.sdg(qubit)
        except Exception as e:
            print(f"Warning: gridsynth failed for angle {angle}: {e}")


# =============================================================================
# PATTERN RECOGNIZERS
# =============================================================================

def recognize_controlled_gate(U):
    """
    Recognize controlled gate patterns (CX, CY, CZ, CRy, CRz).
    
    Returns:
        QuantumCircuit if recognized, None otherwise
    """
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    
    # Check control on q1 (top-left 2x2 = identity)
    if (np.allclose(U[:2, :2], I, atol=1e-6) and 
        np.allclose(U[:2, 2:], 0, atol=1e-6) and
        np.allclose(U[2:, :2], 0, atol=1e-6)):
        
        V = U[2:, 2:]
        qc = QuantumCircuit(2)
        
        # Check Pauli gates
        if np.allclose(V, X, atol=1e-6):
            qc.cx(1, 0); return qc
        if np.allclose(V, Y, atol=1e-6):
            qc.sdg(0); qc.cx(1, 0); qc.s(0); return qc
        if np.allclose(V, Z, atol=1e-6):
            qc.h(0); qc.cx(1, 0); qc.h(0); return qc
        
        # Check Ry rotation
        if np.allclose(V.imag, 0, atol=1e-6):
            V_real = V.real
            if (np.isclose(V_real[0, 0], V_real[1, 1], atol=1e-3) and
                np.isclose(V_real[1, 0], -V_real[0, 1], atol=1e-3)):
                norm = np.sqrt(V_real[0, 0]**2 + V_real[1, 0]**2)
                if norm > 0.1:
                    theta = 2 * np.arctan2(V_real[1, 0]/norm, V_real[0, 0]/norm)
                    print(f"Detected CRy({np.degrees(theta):.2f}°) ctrl=1, tgt=0")
                    return synthesize_controlled_rotation(1, 0, 'ry', theta)
        
        # Check Rz rotation
        if np.allclose(V[0, 1], 0, atol=1e-6) and np.allclose(V[1, 0], 0, atol=1e-6):
            if np.isclose(np.abs(V[0, 0]), 1, atol=1e-6) and np.isclose(np.abs(V[1, 1]), 1, atol=1e-6):
                theta = np.angle(V[1, 1]) - np.angle(V[0, 0])
                print(f"Detected CRz({np.degrees(theta):.2f}°) ctrl=1, tgt=0")
                return synthesize_controlled_rotation(1, 0, 'rz', theta)
    
    # Check control on q0
    ctrl_q0_unchanged = np.array([[U[0, 0], U[0, 2]], [U[2, 0], U[2, 2]]])
    ctrl_q0_zero = np.array([[U[0, 1], U[0, 3]], [U[2, 1], U[2, 3]]])
    
    if (np.allclose(ctrl_q0_unchanged, I, atol=1e-6) and 
        np.allclose(ctrl_q0_zero, 0, atol=1e-6)):
        
        V = np.array([[U[1, 1], U[1, 3]], [U[3, 1], U[3, 3]]])
        qc = QuantumCircuit(2)
        
        if np.allclose(V, X, atol=1e-6):
            qc.cx(0, 1); return qc
        if np.allclose(V, Y, atol=1e-6):
            qc.sdg(1); qc.cx(0, 1); qc.s(1); return qc
        if np.allclose(V, Z, atol=1e-6):
            qc.h(1); qc.cx(0, 1); qc.h(1); return qc
    
    return None


def synthesize_controlled_rotation(ctrl, tgt, rot_type, angle):
    """
    Synthesize a controlled rotation using Clifford+T approximation.
    
    For small angles, uses identity approximation for minimal T-count.
    """
    qc = QuantumCircuit(2)
    
    # Small angle approximation
    if rot_type == 'ry' and abs(angle) < 0.5:
        fid = np.cos(abs(angle)/4)**4 * (1 + np.cos(abs(angle)/2)**2) / 2
        print(f"  Small CRy: Identity approx, fidelity ~{fid:.4f}")
        qc.cx(ctrl, tgt); qc.cx(ctrl, tgt)
        return qc
    
    if rot_type == 'ry':
        half = angle / 2
        qc.ry(half, tgt)
        qc.cx(ctrl, tgt)
        qc.ry(-half, tgt)
        qc.cx(ctrl, tgt)
    elif rot_type == 'rz':
        half = angle / 2
        qc.rz(half, tgt)
        qc.cx(ctrl, tgt)
        qc.rz(-half, tgt)
        qc.cx(ctrl, tgt)
        qc.rz(half, ctrl)
    
    return qc


def recognize_heisenberg_xxyyzz(U):
    """
    Recognize exp(iθ*(XX+YY+ZZ)) - the isotropic Heisenberg interaction.
    
    Special case θ=π/4 is pure Clifford (T=0).
    
    Returns:
        tuple: (recognized, theta, circuit) or (False, None, None)
    """
    # Check structure: U[1,1]=U[2,2]=0, U[0,0]=U[3,3]=U[1,2]=U[2,1]
    zeros = [(0,1), (0,2), (0,3), (1,0), (1,3), (2,0), (2,3), (3,0), (3,1), (3,2)]
    if not all(np.isclose(U[i, j], 0, atol=1e-6) for i, j in zeros):
        return False, None, None
    
    if not (np.isclose(U[1, 1], 0, atol=1e-6) and np.isclose(U[2, 2], 0, atol=1e-6)):
        return False, None, None
    
    phase = U[0, 0]
    if np.abs(phase) < 0.98:
        return False, None, None
    
    if not (np.isclose(U[3, 3], phase, atol=1e-4) and 
            np.isclose(U[1, 2], phase, atol=1e-4) and
            np.isclose(U[2, 1], phase, atol=1e-4)):
        return False, None, None
    
    theta = np.angle(phase)
    print(f"Detected Heisenberg exp(i*{theta:.4f}*(XX+YY+ZZ))")
    
    # Special case: θ = π/4 is pure Clifford
    if np.isclose(theta, np.pi/4, atol=1e-4) or np.isclose(theta, -3*np.pi/4, atol=1e-4):
        print("  Special case: Pure Clifford (T=0)")
        qc = QuantumCircuit(2, global_phase=np.pi/4)
        qc.cx(0, 1); qc.h(0); qc.h(1); qc.cx(0, 1); qc.h(0); qc.h(1); qc.cx(0, 1)
        return True, theta, qc
    
    # General case with controlled phase
    phi = -4 * theta
    qc = QuantumCircuit(2, global_phase=theta)
    qc.cx(0, 1); qc.h(0); qc.h(1)
    append_rz_gridsynth(qc, 0, phi/2)
    append_rz_gridsynth(qc, 1, phi/2)
    qc.cx(0, 1)
    append_rz_gridsynth(qc, 1, -phi/2)
    qc.cx(0, 1)
    qc.h(0); qc.h(1); qc.cx(0, 1)
    
    return True, theta, qc


def recognize_qft2(U):
    """
    Recognize 2-qubit QFT (Quantum Fourier Transform).
    Exact implementation with T=3.
    
    Returns:
        tuple: (recognized, circuit) or (False, None)
    """
    if not np.allclose(np.abs(U), 0.5, atol=1e-6):
        return False, None
    
    omega = np.exp(2j * np.pi / 4)
    QFT2 = np.array([[1, 1, 1, 1], [1, omega, omega**2, omega**3],
                     [1, omega**2, omega**4, omega**6], [1, omega**3, omega**6, omega**9]]) / 2
    
    fid = np.abs(np.trace(U.conj().T @ QFT2))**2 / 16
    fid_dag = np.abs(np.trace(U.conj().T @ QFT2.conj().T))**2 / 16
    
    SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
    fid_swap = np.abs(np.trace(U.conj().T @ SWAP @ QFT2 @ SWAP))**2 / 16
    
    best_fid = max(fid, fid_dag, fid_swap)
    if best_fid < 0.99:
        return False, None
    
    print(f"Detected 2-qubit QFT (T=3, fidelity={best_fid:.6f})")
    qc = QuantumCircuit(2)
    
    if fid >= 0.99:
        qc.h(1); qc.cx(0,1); qc.tdg(1); qc.cx(0,1); qc.t(1); qc.t(0); qc.h(0)
        qc.cx(0,1); qc.cx(1,0); qc.cx(0,1)
    elif fid_dag >= 0.99:
        qc.cx(0,1); qc.cx(1,0); qc.cx(0,1); qc.h(0)
        qc.cx(0,1); qc.t(1); qc.cx(0,1); qc.tdg(1); qc.tdg(0); qc.h(1)
    else:
        qc.h(0); qc.cx(1,0); qc.tdg(0); qc.cx(1,0); qc.t(0); qc.t(1); qc.h(1)
        qc.cx(1,0); qc.cx(0,1); qc.cx(1,0)
    
    return True, qc


def recognize_xxyy_rotation(U):
    """
    Recognize exp(iθ*(XX+YY)) rotation pattern.
    
    Returns:
        tuple: (recognized, theta, circuit) or (False, None, None)
    """
    if not (np.isclose(np.abs(U[0, 0]), 1, atol=1e-4) and 
            np.isclose(np.abs(U[3, 3]), 1, atol=1e-4)):
        return False, None, None
    
    if not (np.allclose(U[0, 1:], 0, atol=1e-4) and np.allclose(U[3, :3], 0, atol=1e-4)):
        return False, None, None
    
    V = U[1:3, 1:3]
    if not (np.isclose(V[0, 0], V[1, 1], atol=1e-4) and 
            np.isclose(V[0, 1].real, 0, atol=1e-4) and
            np.isclose(V[0, 1], V[1, 0], atol=1e-4)):
        return False, None, None
    
    cos_2theta = V[0, 0].real
    sin_2theta = V[0, 1].imag
    if not np.isclose(cos_2theta**2 + sin_2theta**2, 1, atol=1e-3):
        return False, None, None
    
    theta = np.arctan2(sin_2theta, cos_2theta) / 2
    print(f"Detected XX+YY rotation: θ={np.degrees(2*theta):.2f}°")
    
    abs_2theta = abs(2 * theta)
    qc = QuantumCircuit(2)
    
    if abs_2theta < 0.3:
        print("  Using identity approximation (T=0)")
        qc.cx(0, 1); qc.cx(0, 1)
    elif abs(abs_2theta - np.pi/2) < 0.1:
        print("  Using iSWAP approximation (T=0)")
        qc.iswap(0, 1)
    else:
        qc.rz(np.pi/2, 0); qc.ry(np.pi/2, 1)
        qc.cx(1, 0)
        qc.ry(2*theta, 0); qc.ry(2*theta, 1)
        qc.cx(1, 0)
        qc.rz(-np.pi/2, 0); qc.ry(-np.pi/2, 1)
        if not np.isclose(np.angle(U[0, 0]), 0, atol=1e-6):
            qc.global_phase = np.angle(U[0, 0])
    
    return True, theta, qc


def recognize_zi_iz_xx(U):
    """
    Recognize exp(i*θ*(ZI + IZ + XX)) pattern (like unitary6).
    Uses T=2 approximate circuit with ~82% fidelity.
    
    Returns:
        tuple: (recognized, theta, circuit) or (False, None, None)
    """
    eigenvalues = np.linalg.eigvals(U)
    phases = np.sort(np.angle(eigenvalues))
    
    if len(phases) != 4:
        return False, None, None
    
    # Find ±θ and ±√5*θ pairs
    sqrt5 = np.sqrt(5)
    theta_candidates = []
    for i in range(4):
        for j in range(i+1, 4):
            if np.isclose(phases[i], -phases[j], atol=1e-3):
                theta_candidates.append(abs(phases[j]))
    
    if len(theta_candidates) < 2:
        return False, None, None
    
    theta_candidates.sort()
    if not np.isclose(theta_candidates[-1] / theta_candidates[0], sqrt5, atol=0.1):
        return False, None, None
    
    theta = theta_candidates[0]
    print(f"Detected ZI+IZ+XX: θ={theta:.4f} (T=2, fidelity ~0.82)")
    
    qc = QuantumCircuit(2)
    qc.tdg(0); qc.tdg(1); qc.h(0); qc.h(1)
    qc.cx(0, 1); qc.cx(0, 1)
    qc.h(0); qc.h(1)
    
    return True, theta, qc


# Known unitary patterns with their expected matrices and optimal circuits
# Format: (name, expected_matrix, circuit_builder, t_count, approx_fidelity)
# Only include patterns where the stored circuit achieves good fidelity (>0.65)
KNOWN_UNITARIES = {
    'unitary7': {
        'matrix': np.array([
            [0.10614794-0.67964147j, 0.33139935+0.36152515j, -0.00401607-0.29097921j, 0.20705723-0.39841737j],
            [-0.36227759-0.45361314j, -0.81424202+0j, 0j, 0j],
            [0.26141904+0.04453310j, -0.14112150+0.12582226j, -0.94547891+0j, 0j],
            [0.32764493-0.11016284j, -0.08440623+0.23154488j, 0.12881489-0.06921940j, -0.89352723+0j]
        ], dtype=complex),
        'gates': [('s', 0), ('sdg', 1), ('cx', 0, 1), ('h', 0), ('s', 1),
                  ('cx', 1, 0), ('h', 0), ('tdg', 1), ('cx', 0, 1), ('tdg', 0), ('tdg', 1)],
        't_count': 3, 'fidelity': 0.695
    },
    # Note: unitary9 and unitary10 removed - stored circuits had poor fidelity (<0.65)
    # QSD decomposition achieves better results for these unitaries
}


def _build_circuit_from_gates(gates):
    """Build a QuantumCircuit from a list of gate tuples."""
    qc = QuantumCircuit(2)
    for gate in gates:
        name = gate[0]
        if name == 'cx':
            qc.cx(gate[1], gate[2])
        else:
            getattr(qc, name)(gate[1])
    return qc


def recognize_known_unitary(U, name):
    """Recognize a known unitary pattern by name."""
    if name not in KNOWN_UNITARIES:
        return False, None
    info = KNOWN_UNITARIES[name]
    fid = np.abs(np.trace(U.conj().T @ info['matrix']))**2 / 16
    if fid < 0.98:
        return False, None
    print(f"Detected {name} (T={info['t_count']}, fidelity ~{info['fidelity']})")
    return True, _build_circuit_from_gates(info['gates'])


def recognize_unitary4(U):
    """
    Recognize unitary4 - XX+YY rotation with θ=2π/7 (irrational).
    Uses T=1 approximation with ~81% fidelity.
    """
    # U4 is an XX+YY rotation: |00⟩ and |11⟩ unchanged, middle block rotates
    if not (np.isclose(np.abs(U[0, 0]), 1, atol=1e-3) and 
            np.isclose(np.abs(U[3, 3]), 1, atol=1e-3)):
        return False, None
    if not (np.allclose(U[0, 1:], 0, atol=1e-3) and np.allclose(U[3, :3], 0, atol=1e-3)):
        return False, None
    V = U[1:3, 1:3]
    cos_2theta = V[0, 0].real
    sin_2theta = V[0, 1].imag if np.isclose(V[0, 1].real, 0, atol=1e-3) else 0
    if not (np.isclose(cos_2theta, 0.6235, atol=0.01) and np.isclose(abs(sin_2theta), 0.7818, atol=0.01)):
        return False, None
    print(f"Detected unitary4 (XX+YY θ=2π/7, T=1, fidelity ~0.81)")
    qc = QuantumCircuit(2)
    qc.h(0); qc.h(1); qc.cx(0, 1); qc.tdg(1); qc.cx(0, 1); qc.h(0); qc.h(1)
    return True, qc


# Convenience wrappers for backward compatibility
def recognize_unitary7(U): return recognize_known_unitary(U, 'unitary7')
def recognize_unitary9(U): return recognize_known_unitary(U, 'unitary9')
def recognize_unitary10(U): return recognize_known_unitary(U, 'unitary10')


# =============================================================================
# CIRCUIT OPTIMIZATION
# =============================================================================

def simplify_circuit(qc):
    """
    Simplify circuit by accumulating phase gates (mod 8).
    T=1, S=2, Sdg=6, Tdg=7 in units of π/4.
    """
    new_qc = QuantumCircuit(qc.num_qubits)
    pending = {}
    
    for inst in qc.data:
        name = inst.operation.name
        qubits = inst.qubits
        
        if len(qubits) == 1:
            q_idx = qc.find_bit(qubits[0]).index
            if name in PHASE_VALUES:
                pending[q_idx] = (pending.get(q_idx, 0) + PHASE_VALUES[name]) % 8
            elif name == 'h':
                emit_phase(new_qc, pending.get(q_idx, 0), qubits)
                pending[q_idx] = 0
                new_qc.h(qubits)
            else:
                emit_phase(new_qc, pending.get(q_idx, 0), qubits)
                pending[q_idx] = 0
                new_qc.append(inst.operation, qubits)
        else:
            for q in qubits:
                q_idx = qc.find_bit(q).index
                emit_phase(new_qc, pending.get(q_idx, 0), [q])
                pending[q_idx] = 0
            new_qc.append(inst.operation, qubits)
    
    for q_idx in range(qc.num_qubits):
        emit_phase(new_qc, pending.get(q_idx, 0), [qc.qubits[q_idx]])
    
    return new_qc


def cancel_adjacent_h(qc):
    """Cancel adjacent H gates on the same qubit (H·H = I)."""
    new_qc = QuantumCircuit(qc.num_qubits)
    instructions = list(qc.data)
    last_h_idx = {}
    skip = set()
    
    for i, inst in enumerate(instructions):
        if len(inst.qubits) == 1 and inst.operation.name == 'h':
            q_idx = qc.find_bit(inst.qubits[0]).index
            if q_idx in last_h_idx and last_h_idx[q_idx] is not None:
                prev_i = last_h_idx[q_idx]
                can_cancel = all(
                    qc.find_bit(instructions[j].qubits[0]).index != q_idx
                    for j in range(prev_i + 1, i)
                    if j not in skip and len(instructions[j].qubits) == 1
                )
                if can_cancel:
                    skip.add(prev_i); skip.add(i)
                    last_h_idx[q_idx] = None
                else:
                    last_h_idx[q_idx] = i
            else:
                last_h_idx[q_idx] = i
        elif len(inst.qubits) == 1:
            last_h_idx[qc.find_bit(inst.qubits[0]).index] = None
        else:
            for q in inst.qubits:
                last_h_idx[qc.find_bit(q).index] = None
    
    for i, inst in enumerate(instructions):
        if i not in skip:
            new_qc.append(inst.operation, inst.qubits)
    
    return new_qc


def optimize_circuit(qc):
    """Apply optimization passes until no more improvements."""
    prev_count = -1
    curr_count = sum(qc.count_ops().values())
    
    while curr_count < prev_count or prev_count == -1:
        prev_count = curr_count
        qc = simplify_circuit(qc)
        qc = cancel_adjacent_h(qc)
        curr_count = sum(qc.count_ops().values())
    
    return qc


def apply_manual_identities(qc, epsilon=1e-6):
    """
    Convert non-Clifford+T gates to Clifford+T using gridsynth.
    
    Handles: X, Y, Z, Rx, Ry, Rz gates
    """
    def apply_gridsynth_seq(target, qubit, seq):
        for g in seq:
            g_up = g.upper()
            if g_up == 'H': target.h(qubit)
            elif g_up == 'T': target.t(qubit)
            elif g_up == 'S': target.s(qubit)
            elif g_up == 'X': target.h(qubit); target.z(qubit); target.h(qubit)
            elif g_up == 'Z': target.z(qubit)
            elif g == 't': target.tdg(qubit)
            elif g == 's': target.sdg(qubit)
    
    new_qc = QuantumCircuit(qc.num_qubits)
    
    for inst in qc.data:
        name = inst.operation.name
        qbs_idx = [qc.find_bit(q).index for q in inst.qubits]
        qbs = qbs_idx[0] if len(qbs_idx) == 1 else qbs_idx
        
        if name == 'x':
            new_qc.h(qbs); new_qc.z(qbs); new_qc.h(qbs)
        elif name == 'z':
            new_qc.s(qbs); new_qc.s(qbs)
        elif name == 'y':
            new_qc.s(qbs); new_qc.h(qbs); new_qc.z(qbs); new_qc.h(qbs); new_qc.sdg(qbs)
        elif name in ['rx', 'ry', 'rz']:
            angle = float(inst.operation.params[0]) % (2 * np.pi)
            if angle > np.pi: angle -= 2 * np.pi
            if np.isclose(angle, 0, atol=1e-9): continue
            
            if name == 'rx':
                new_qc.h(qbs)
                name = 'rz'  # Rx = H·Rz·H
            if name == 'ry':
                new_qc.sdg(qbs); new_qc.h(qbs)  # Ry = Sdg·H·Rz·H·S
            
            # Check Z8 angle
            k_f = angle / (np.pi / 4)
            k = int(round(k_f))
            if abs(k_f - k) < 1e-6:
                k = k % 8
                if k == 1: new_qc.t(qbs)
                elif k == 2: new_qc.s(qbs)
                elif k == 3: new_qc.s(qbs); new_qc.t(qbs)
                elif k == 4: new_qc.z(qbs)
                elif k == 5: new_qc.sdg(qbs); new_qc.tdg(qbs)
                elif k == 6: new_qc.sdg(qbs)
                elif k == 7: new_qc.tdg(qbs)
            else:
                try:
                    seq = gs.gridsynth_gates(mpmath.mpf(angle), epsilon=mpmath.mpf(epsilon))
                    apply_gridsynth_seq(new_qc, qbs, seq)
                except Exception as e:
                    print(f"Warning: gridsynth failed: {e}")
            
            if name == 'ry':
                new_qc.h(qbs); new_qc.s(qbs)
            if inst.operation.name == 'rx':
                new_qc.h(qbs)
        elif name == 'cx':
            new_qc.cx(qbs_idx[0], qbs_idx[1])
        elif name == 'h':
            new_qc.h(qbs)
        else:
            new_qc.append(inst.operation, qbs_idx)
    
    return new_qc


def adaptive_clifford_t_approximation(qc_rz, U_target, min_fidelity=0.98, max_t=250, max_total_gates=6000,
                                     epsilons=(1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2),
                                     effort=3, decoder="ml-exact", verbose=True):
    """
    Convert an {rz,cx,h}-basis circuit into Clifford+T using gridsynth, but adapt the
    gridsynth approximation tolerance to avoid absurd T-count blow-ups.

    Strategy:
      - Try a ladder of epsilons (coarse -> fewer T gates).
      - For each epsilon: apply_manual_identities -> optimize passes -> optional rmsynth on diagonal chunks.
      - Pick the smallest T-count circuit that still meets min_fidelity.
      - If none meet min_fidelity, return the best-fidelity candidate (tie-break by T-count).
    """
    # Ensure target is a proper unitary (numerically)
    try:
        U_clean, _ = polar(U_target)
    except Exception:
        U_clean = U_target

    best_meeting = None
    best_meeting_metrics = None  # (t_count, total_gates, -fid)

    best_overall = None
    best_overall_metrics = None  # (-fid, t_count, total_gates)

    for eps in epsilons:
        cand = apply_manual_identities(qc_rz, epsilon=eps)
        cand = optimize_circuit(cand)

        # rmsynth only helps on diagonal (CNOT+phase) chunks; safe even if it does nothing
        try:
            cand = optimize_diagonal_chunks_with_rmsynth(cand, effort=effort, decoder=decoder)
        except Exception:
            pass

        try:
            fid = float(process_fidelity(Operator(U_clean), Operator(cand)))
        except Exception:
            # Fallback: compute via trace overlap
            U_c = Operator(cand).data
            d = U_c.shape[0]
            fid = float(np.abs(np.trace(U_clean.conj().T @ U_c))**2 / (d * d))

        counts = cand.count_ops()
        t_count = int(counts.get('t', 0) + counts.get('tdg', 0))
        total_gates = int(sum(counts.values()))

        if verbose:
            print(f"    [approx eps={eps:g}] fid={fid:.6f}, T={t_count}, total={total_gates}")

        # Track best overall (highest fidelity, then fewer T, then fewer gates)
        overall_key = (-fid, t_count, total_gates)
        if best_overall is None or overall_key < best_overall_metrics:
            best_overall = cand
            best_overall_metrics = overall_key

        # Track best meeting threshold with sane sizes
        if fid >= min_fidelity and t_count <= max_t and total_gates <= max_total_gates:
            meet_key = (t_count, total_gates, -fid)
            if best_meeting is None or meet_key < best_meeting_metrics:
                best_meeting = cand
                best_meeting_metrics = meet_key

    if best_meeting is not None:
        return best_meeting

    # If nothing meets threshold, return the highest-fidelity candidate (still usually much shorter than eps=1e-10)
    return best_overall


def optimize_diagonal_chunks_with_rmsynth(qc, effort=1, decoder="auto"):
    """
    Split circuit at H gates and optimize each diagonal chunk with rmsynth.
    """
    num_qubits = qc.num_qubits
    result_qc = QuantumCircuit(num_qubits)
    current_chunk = QuantumCircuit(num_qubits)
    
    for inst in qc.data:
        name = inst.operation.name
        q_indices = [qc.find_bit(q).index for q in inst.qubits]
        
        if name == 'h':
            if current_chunk.size() > 0:
                result_qc.compose(optimize_with_rmsynth(current_chunk, effort, decoder), inplace=True)
                current_chunk = QuantumCircuit(num_qubits)
            result_qc.h(q_indices[0])
        elif name in ['t', 'tdg', 's', 'sdg', 'z', 'cx']:
            getattr(current_chunk, name)(*q_indices)
        else:
            if current_chunk.size() > 0:
                result_qc.compose(optimize_with_rmsynth(current_chunk, effort, decoder), inplace=True)
                current_chunk = QuantumCircuit(num_qubits)
            result_qc.append(inst.operation, q_indices)
    
    if current_chunk.size() > 0:
        result_qc.compose(optimize_with_rmsynth(current_chunk, effort, decoder), inplace=True)
    
    return result_qc


# =============================================================================
# MAIN DECOMPOSITION
# =============================================================================

def decompose_and_optimize(U, effort=3):
    """
    Decompose a unitary matrix into an optimized Clifford+T circuit.
    
    Args:
        U: Unitary matrix (numpy.ndarray)
        effort: Optimization effort level (1-3)
        
    Returns:
        QuantumCircuit: Optimized Clifford+T circuit
        
    The function tries pattern recognition first, then falls back to
    QSD decomposition with rmsynth optimization.
    """
    num_qubits = int(np.log2(U.shape[0]))
    
    if num_qubits == 2:
        # Try pattern recognizers in order of efficiency
        recognizers = [
            ('controlled gate', lambda: recognize_controlled_gate(U)),
            ('unitary4', lambda: recognize_unitary4(U)),
            ('unitary7', lambda: recognize_unitary7(U)),
            ('QFT', lambda: recognize_qft2(U)),
            ('Heisenberg', lambda: recognize_heisenberg_xxyyzz(U)),
            ('XX+YY', lambda: recognize_xxyy_rotation(U)),
            ('ZI+IZ+XX', lambda: recognize_zi_iz_xx(U)),
        ]
        
        for name, recognizer in recognizers:
            result = recognizer()
            if result is not None:
                if isinstance(result, QuantumCircuit):
                    print(f"Recognized {name} pattern")
                    return result
                elif result[0]:  # (True, qc) or (True, theta, qc)
                    print(f"Recognized {name} pattern")
                    return result[-1]  # Return the circuit
    
    # Diagonal unitary
    if is_diagonal(U):
        print("Detected diagonal unitary")
        z8, err = get_z8_phases(U)
        if err < 1e-4 and z8 and any(v != 0 for v in z8):
            print(f"  Z8 coefficients: {z8}, error: {err:.2e}")
        return synthesize_diagonal_unitary(U, num_qubits)
    
    # General QSD decomposition
    qc_qsd = qs_decomposition(U)
    qc_rz = transpile(qc_qsd, basis_gates=['rz', 'cx', 'h'], optimization_level=3)

    # Convert to Clifford+T with adaptive gridsynth tolerance to avoid huge T blow-ups (notably U9/U10).
    qc_ct = adaptive_clifford_t_approximation(
        qc_rz,
        U_target=U,
        min_fidelity=0.98,
        max_t=250,
        max_total_gates=6000,
        epsilons=(1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2),
        effort=effort,
        decoder="ml-exact",
        verbose=True,
    )
    return qc_ct


def synthesize_diagonal_unitary(U, num_qubits):
    """Synthesize a diagonal unitary using Z8 gates and CNOTs."""
    diags = np.diag(U)
    phases = np.angle(diags)
    qc = QuantumCircuit(num_qubits)
    
    if num_qubits == 2:
        p00, p01, p10, p11 = phases
        
        # Check for pure ZZ rotation
        if (np.isclose(p00, p11, atol=1e-4) and np.isclose(p01, p10, atol=1e-4) and
            np.isclose(p00, -p01, atol=1e-4)):
            rz_angle = -2 * p00
            k = round(rz_angle / (np.pi / 4)) % 8
            qc.cx(0, 1)
            if k == 1: qc.t(1)
            elif k == 2: qc.s(1)
            elif k == 3: qc.s(1); qc.t(1)
            elif k == 4: qc.z(1)
            elif k == 5: qc.z(1); qc.t(1)
            elif k == 6: qc.sdg(1)
            elif k == 7: qc.tdg(1)
            qc.cx(0, 1)
        else:
            # General diagonal decomposition
            global_phase = (p00 + p01 + p10 + p11) / 4
            p00 -= global_phase; p01 -= global_phase; p10 -= global_phase; p11 -= global_phase
            
            for phi, q in [(-p00 - p10, 0), (-p00 - p01, 1)]:
                if not np.isclose(phi, 0, atol=1e-9):
                    k = round(phi / (np.pi / 4)) % 8
                    if k == 1: qc.t(q)
                    elif k == 2: qc.s(q)
                    elif k == 3: qc.s(q); qc.t(q)
                    elif k == 4: qc.z(q)
                    elif k == 5: qc.z(q); qc.t(q)
                    elif k == 6: qc.sdg(q)
                    elif k == 7: qc.tdg(q)
            
            phi2 = p01 + p10
            if not np.isclose(phi2, 0, atol=1e-9):
                k = round(phi2 / (np.pi / 4)) % 8
                qc.cx(0, 1)
                if k == 1: qc.t(1)
                elif k == 2: qc.s(1)
                elif k == 3: qc.s(1); qc.t(1)
                elif k == 4: qc.z(1)
                elif k == 5: qc.z(1); qc.t(1)
                elif k == 6: qc.sdg(1)
                elif k == 7: qc.tdg(1)
                qc.cx(0, 1)
            
            qc.global_phase = global_phase
    else:
        qc = transpile(qs_decomposition(U), basis_gates=['rz', 'cx', 'h'], optimization_level=3)
    
    return qc


# =============================================================================
# BRUTE FORCE ANALYZER
# =============================================================================

def check_clifford_2q(U):
    """Check if a 2-qubit unitary is Clifford."""
    magnitudes = np.abs(U).flatten()
    magnitudes = magnitudes[magnitudes > 1e-6]
    
    for m in magnitudes:
        if not any(np.isclose(m, v, atol=0.01) for v in [0.5, 1/np.sqrt(2), 1.0]):
            return False
    
    phases = np.angle(U.flatten())
    phases = phases[np.abs(U.flatten()) > 1e-6]
    for p in phases:
        if not np.isclose(p / (np.pi / 4), round(p / (np.pi / 4)), atol=0.01):
            return False
    
    return True


def analyze_block_structure(U):
    """Analyze block structure of 2-qubit unitary."""
    blocks = [U[0:2, 0:2], U[0:2, 2:4], U[2:4, 0:2], U[2:4, 2:4]]
    zero_blocks = sum(1 for b in blocks if np.allclose(b, 0, atol=1e-6))
    
    if zero_blocks == 2:
        return "2x2 block diagonal/anti-diagonal"
    elif zero_blocks == 0:
        return "full (no block structure)"
    return f"{zero_blocks} zero blocks"


def detect_known_patterns(U, n_qubits):
    """Detect known unitary patterns."""
    patterns = {}
    if n_qubits == 2:
        patterns['QFT'] = np.allclose(np.abs(U), 0.5, atol=1e-4)
        SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
        patterns['SWAP'] = np.allclose(np.abs(U), np.abs(SWAP), atol=1e-4)
        patterns['Controlled-U'] = (np.isclose(np.abs(U[0,0]), 1, atol=1e-4) and
                                     np.allclose(U[0,1:], 0, atol=1e-4) and
                                     np.allclose(U[1:,0], 0, atol=1e-4))
        patterns['XX+YY'] = (np.isclose(np.abs(U[0,0]), 1, atol=1e-4) and
                            np.isclose(np.abs(U[3,3]), 1, atol=1e-4) and
                            np.allclose(U[0,1:], 0, atol=1e-4) and
                            np.allclose(U[3,:3], 0, atol=1e-4))
        patterns['Heisenberg'] = (np.isclose(U[1,1], 0, atol=1e-4) and
                                   np.isclose(U[2,2], 0, atol=1e-4) and
                                   np.isclose(U[1,2], U[0,0], atol=1e-4))
    return patterns


def analyze_unitary(U, name="Unitary"):
    """
    Analyze the structure of a unitary matrix.
    
    Prints eigenvalue analysis, structure detection, and pattern recognition.
    Returns a dict with analysis results.
    """
    print(f"\n{'='*60}")
    print(f"ANALYSIS: {name}")
    print(f"{'='*60}")
    
    U_clean, _ = polar(U)
    n = int(np.log2(U.shape[0]))
    print(f"\nDimension: {U.shape[0]}x{U.shape[0]} ({n} qubits)")
    
    eigenvalues = np.linalg.eigvals(U_clean)
    phases = np.angle(eigenvalues) / np.pi
    
    print(f"\nEigenvalue phases (×π):")
    for i, p in enumerate(sorted(phases)):
        z8_k = p * 4
        is_z8 = np.isclose(z8_k, round(z8_k), atol=0.01)
        z8_str = f" [Z8]" if is_z8 else " [irrational]"
        print(f"  λ{i}: {p:+.6f}π{z8_str}")
    
    all_z8 = all(np.isclose(p * 4, round(p * 4), atol=0.01) for p in phases)
    print(f"\nAll Z8 (exact Clifford+T): {all_z8}")
    
    print(f"\nStructure:")
    is_diag = is_diagonal(U_clean)
    print(f"  Diagonal: {is_diag}")
    
    if n == 2:
        print(f"  Clifford: {check_clifford_2q(U_clean)}")
        print(f"  Block: {analyze_block_structure(U_clean)}")
    
    print(f"\nPatterns:")
    patterns = detect_known_patterns(U_clean, n)
    for pattern, detected in patterns.items():
        if detected:
            print(f"  ✓ {pattern}")
    
    print(f"\nRecommendation:")
    if all_z8:
        print("  → Exact Clifford+T synthesis possible")
    elif is_diag:
        print("  → Diagonal synthesis with gridsynth")
    elif any(patterns.values()):
        print("  → Use specialized circuit")
    else:
        print("  → Try brute_force_search(U, max_t=3, max_cx=3)")
    
    return {'eigenvalues': eigenvalues, 'phases': phases, 'all_z8': all_z8,
            'is_diagonal': is_diag, 'patterns': patterns}


def generate_cx_patterns(n_cx):
    """Generate all CX patterns for n CX gates."""
    return [[(0, 1) if b == 0 else (1, 0) for b in bits]
            for bits in product([0, 1], repeat=n_cx)]


# Precomputed gate matrices for fast brute force
_GATE_MATRICES = {
    'id': np.eye(2, dtype=complex),
    'h': np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
    's': np.array([[1, 0], [0, 1j]], dtype=complex),
    'sdg': np.array([[1, 0], [0, -1j]], dtype=complex),
    't': np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex),
    'tdg': np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex),
}
_CX_01 = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
_CX_10 = np.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]], dtype=complex)


def _build_unitary_fast(gates, cx_pattern, n_layers):
    """Build unitary matrix directly using precomputed matrices (much faster)."""
    U = np.eye(4, dtype=complex)
    
    for layer in range(n_layers):
        g0, g1 = gates[2*layer], gates[2*layer + 1]
        # Kronecker product for 2-qubit gate from 1-qubit gates
        layer_U = np.kron(_GATE_MATRICES[g0], _GATE_MATRICES[g1])
        U = layer_U @ U
        
        if layer < len(cx_pattern):
            cx_matrix = _CX_01 if cx_pattern[layer] == (0, 1) else _CX_10
            U = cx_matrix @ U
    
    return U


def _compute_fidelity_fast(U_target, U_circuit):
    """Compute process fidelity directly (faster than Qiskit)."""
    # Process fidelity = |Tr(U†V)|² / d²
    d = U_target.shape[0]
    overlap = np.trace(U_target.conj().T @ U_circuit)
    return (np.abs(overlap) ** 2) / (d * d)


def _count_t_gates(gates):
    """Count T and Tdg gates in gate list."""
    return sum(1 for g in gates if g in ('t', 'tdg'))


def _eval_circuit_batch(args):
    """Worker function for parallel evaluation. Returns (gates, cx_pattern, t_count, fidelity)."""
    gates, cx_pattern, n_layers, U_target = args
    t_count = _count_t_gates(gates)
    try:
        U_circ = _build_unitary_fast(gates, cx_pattern, n_layers)
        fid = _compute_fidelity_fast(U_target, U_circ)
        return (gates, cx_pattern, t_count, fid)
    except:
        return (gates, cx_pattern, t_count, 0.0)


def _reconstruct_circuit(best_gates, best_cx_pat):
    """Reconstruct QuantumCircuit from gate list and CX pattern."""
    qc = QuantumCircuit(2)
    n_layers = len(best_cx_pat) + 1
    for layer in range(n_layers):
        g0, g1 = best_gates[2*layer], best_gates[2*layer + 1]
        if g0 != 'id': getattr(qc, g0)(0)
        if g1 != 'id': getattr(qc, g1)(1)
        if layer < len(best_cx_pat):
            qc.cx(*best_cx_pat[layer])
    return qc


def brute_force_search(U, max_t=3, max_cx=3, verbose=True, timeout=300, parallel=False, n_workers=None):
    """
    Brute force search for optimal Clifford+T circuit.
    
    Args:
        U: Target unitary matrix
        max_t: Maximum T-gate count
        max_cx: Maximum CX gates
        verbose: Print progress
        timeout: Maximum search time (seconds, ignored if parallel=True)
        parallel: Use multiprocessing for faster search
        n_workers: Number of parallel workers (default: all CPU cores)
        
    Returns:
        tuple: (best_circuit, best_fidelity)
    """
    U_clean, _ = polar(U)
    n_qubits = int(np.log2(U.shape[0]))
    
    if n_qubits != 2:
        print("Brute force only supports 2-qubit unitaries")
        return None, 0
    
    if parallel and n_workers is None:
        n_workers = cpu_count()
    
    mode = f"{n_workers} workers" if parallel else f"timeout={timeout}s"
    print(f"\n{'='*60}")
    print(f"BRUTE FORCE SEARCH (T≤{max_t}, CX≤{max_cx}, {mode})")
    print(f"{'='*60}")
    
    gates_1q = ['h', 's', 'sdg', 't', 'tdg', 'id']
    best_fid, best_gates, best_cx_pat, best_t = 0, None, None, float('inf')
    start = time.time()
    total_count = 0
    
    for n_cx in range(0, max_cx + 1):
        if not parallel and time.time() - start > timeout:
            print(f"\nTimeout after {total_count} circuits")
            break
        if verbose:
            print(f"\nSearching with {n_cx} CX...")
        
        n_layers = n_cx + 1
        cx_patterns = generate_cx_patterns(n_cx)
        all_gates = list(product(gates_1q, repeat=2*n_layers))
        valid_gates = [g for g in all_gates if _count_t_gates(g) <= max_t]
        
        if verbose:
            print(f"  {len(valid_gates)} valid gate combos × {len(cx_patterns)} CX patterns")
        
        if parallel:
            # Parallel evaluation
            work_items = [(gates, cx_pat, n_layers, U_clean)
                          for cx_pat in cx_patterns for gates in valid_gates]
            total_count += len(work_items)
            
            with Pool(n_workers) as pool:
                results = pool.map(_eval_circuit_batch, work_items, chunksize=1000)
            
            for gates, cx_pat, t_count, fid in results:
                if fid > best_fid + 1e-6 or (fid > best_fid - 1e-6 and t_count < best_t):
                    best_fid, best_gates, best_cx_pat, best_t = fid, gates, cx_pat, t_count
                    if verbose:
                        print(f"  T={t_count}, CX={n_cx}, fid={fid:.4f}")
        else:
            # Sequential evaluation with timeout
            for cx_pattern in cx_patterns:
                if time.time() - start > timeout:
                    break
                for gates in valid_gates:
                    if time.time() - start > timeout:
                        break
                    total_count += 1
                    t_count = _count_t_gates(gates)
                    try:
                        U_circ = _build_unitary_fast(gates, cx_pattern, n_layers)
                        fid = _compute_fidelity_fast(U_clean, U_circ)
                        if fid > best_fid + 1e-6 or (fid > best_fid - 1e-6 and t_count < best_t):
                            best_fid, best_gates, best_cx_pat, best_t = fid, gates, cx_pattern, t_count
                            if verbose:
                                print(f"  T={t_count}, CX={n_cx}, fid={fid:.4f}")
                    except:
                        pass
    
    elapsed = time.time() - start
    rate = total_count / elapsed if elapsed > 0 else 0
    print(f"\nDone: {total_count} circuits in {elapsed:.1f}s ({rate:.0f}/sec)")
    
    if best_gates:
        qc = _reconstruct_circuit(best_gates, best_cx_pat)
        print(f"\nBest: T={best_t}, fid={best_fid:.6f}")
        print(qc.draw('mpl'))
        return qc, best_fid
    
    return None, 0


# Backward compatibility alias
def brute_force_parallel(U, max_t=3, max_cx=3, verbose=True, n_workers=None):
    """Parallel brute force search. Alias for brute_force_search(parallel=True)."""
    return brute_force_search(U, max_t=max_t, max_cx=max_cx, verbose=verbose,
                              parallel=True, n_workers=n_workers)


# Progressive low-T search helper for U9/U10 etc.
def progressive_low_t_search(U, t_cap=8, cx_cap=3, fid_target=1.0 - 1e-12, n_workers=None, verbose=False):
    """Progressively search for the lowest-T 2-qubit Clifford+T circuit.

    This is designed for cases like U9/U10 where generic decomposition + gridsynth
    can explode the T-count even when an exact low-T circuit exists.

    Returns:
        (qc, fid, t_used) where t_used is the T budget that first achieved fid_target.
    """
    U_clean, _ = polar(U)

    best_qc = None
    best_fid = 0.0
    best_t = None

    for t in range(0, t_cap + 1):
        qc, fid = brute_force_search(
            U_clean,
            max_t=t,
            max_cx=cx_cap,
            verbose=verbose,
            parallel=True,
            n_workers=n_workers,
        )

        if qc is not None and fid > best_fid:
            best_qc, best_fid, best_t = qc, fid, t

        if qc is not None and fid >= fid_target:
            return qc, fid, t

    if best_qc is not None:
        return best_qc, best_fid, best_t

    return None, 0.0, None


def quick_search(U, max_t=2):
    """Quick search with limited parameters."""
    return brute_force_search(U, max_t=max_t, max_cx=2, verbose=True, timeout=60)

# =============================================================================
# MAIN
# =============================================================================

def optimize_and_display(name, filepath, effort=3, verbose=False, use_brute_force=True, fidelity_threshold=0.95):
    """Load unitary, optimize it, and display results.
    
    Uses the standard optimizer pipeline, then optionally runs brute force search
    if fidelity is below threshold to find potentially better circuits.
    
    Args:
        name: Display name for the unitary
        filepath: Path to the .npy file
        effort: Optimization effort level (1-3)
        verbose: If True, show detailed synthesis info (Z8 coefficients, etc.)
        use_brute_force: If True, try brute force search when fidelity is below threshold
        fidelity_threshold: Fidelity threshold below which to try brute force
        
    Returns:
        tuple: (best_circuit, target_unitary)
    """
    import io
    from contextlib import redirect_stdout
    
    # Load the unitary
    U_target = load_unitary(filepath)
    n_qubits = int(np.log2(U_target.shape[0]))
    
    print(f"{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    
    # Redirect stdout to capture verbose output if not verbose
    if not verbose:
        f = io.StringIO()
        with redirect_stdout(f):
            qc = decompose_and_optimize(U_target, effort)
            # decompose_and_optimize() already returns Clifford+T after adaptive synthesis.
            # Only run manual identities if any non-basis rotations slipped through.
            if any(inst.operation.name in ("rx", "ry", "rz", "x", "y") for inst in qc.data):
                qc = apply_manual_identities(qc, epsilon=1e-6)
            qc = optimize_circuit(qc)
            qc = optimize_diagonal_chunks_with_rmsynth(qc, effort=effort, decoder="auto")
            qc = optimize_circuit(qc)
    else:
        qc = decompose_and_optimize(U_target, effort)
        # decompose_and_optimize() already returns Clifford+T after adaptive synthesis.
        # Only run manual identities if any non-basis rotations slipped through.
        if any(inst.operation.name in ("rx", "ry", "rz", "x", "y") for inst in qc.data):
            qc = apply_manual_identities(qc, epsilon=1e-6)
        qc = optimize_circuit(qc)
        qc = optimize_diagonal_chunks_with_rmsynth(qc, effort=effort, decoder="auto")
        qc = optimize_circuit(qc)
    
    # Phase alignment
    U_actual = Operator(qc).data
    phase_diff = np.angle(np.trace(np.conj(U_target.T) @ U_actual))
    qc.global_phase = -phase_diff
    
    # Compute metrics
    U_final = Operator(qc).data
    dist = np.linalg.norm(U_target - U_final, ord=2)
    fid = process_fidelity(Operator(U_target), Operator(qc))
    ops = qc.count_ops()
    t_count = ops.get('t', 0) + ops.get('tdg', 0)
    
    best_qc = qc
    best_fid = fid
    method_used = "Standard optimizer"

    # If we got an absurd T-count (common for U9/U10), try a progressive low-T exact search.
    # This targets the *minimum T* solution that hits essentially-unit fidelity.
    if n_qubits == 2 and (t_count > 50 or "Unitary 9" in name or "Unitary 10" in name or "unitary9" in filepath or "unitary10" in filepath):
        print("\n  [Trying progressive low-T search to reduce T-count]")
        try:
            # Suppress brute-force logs unless user explicitly requested verbose.
            if not verbose:
                import io
                from contextlib import redirect_stdout
                with redirect_stdout(io.StringIO()):
                    bf_qc, bf_fid, bf_t = progressive_low_t_search(
                        U_target,
                        t_cap=8,
                        cx_cap=3,
                        fid_target=1.0 - 1e-12,
                        n_workers=os.cpu_count(),
                        verbose=False,
                    )
            else:
                bf_qc, bf_fid, bf_t = progressive_low_t_search(
                    U_target,
                    t_cap=8,
                    cx_cap=3,
                    fid_target=1.0 - 1e-12,
                    n_workers=os.cpu_count(),
                    verbose=True,
                )

            if bf_qc is not None:
                # Phase align brute-force result
                bf_U = Operator(bf_qc).data
                bf_phase = np.angle(np.trace(np.conj(U_target.T) @ bf_U))
                bf_qc.global_phase = -bf_phase

                bf_ops = bf_qc.count_ops()
                bf_t_count = bf_ops.get('t', 0) + bf_ops.get('tdg', 0)
                bf_dist = np.linalg.norm(U_target - Operator(bf_qc).data, ord=2)
                bf_fid2 = process_fidelity(Operator(U_target), Operator(bf_qc))

                # Prefer strictly fewer T; tie-break on smaller distance.
                if bf_t_count < t_count or (bf_t_count == t_count and bf_dist < dist):
                    best_qc = bf_qc
                    best_fid = bf_fid2
                    method_used = f"Progressive brute force (T={bf_t_count})"
                    # Update the local metrics used by later heuristics
                    ops = bf_ops
                    t_count = bf_t_count
                    dist = bf_dist
                    fid = bf_fid2
        except Exception as e:
            print(f"  [Progressive low-T search failed: {e}]")

    # Try brute force if fidelity is below threshold (only for 2-qubit unitaries)
    if use_brute_force and fid < fidelity_threshold and n_qubits == 2:
        print(f"\n  [Trying brute force search - fidelity {fid:.4f} < {fidelity_threshold}]")
        try:
            bf_qc, bf_fid = quick_search(U_target, max_t=t_count + 2)
            if bf_qc is not None and bf_fid > best_fid:
                # Phase align brute force result
                bf_U = Operator(bf_qc).data
                bf_phase = np.angle(np.trace(np.conj(U_target.T) @ bf_U))
                bf_qc.global_phase = -bf_phase
                
                best_qc = bf_qc
                best_fid = bf_fid
                method_used = "Brute force search"
                print(f"  [Brute force found better: F={bf_fid:.4f}]")
        except Exception as e:
            print(f"  [Brute force failed: {e}]")
    
    # Recompute final metrics with best circuit
    U_final = Operator(best_qc).data
    dist = np.linalg.norm(U_target - U_final, ord=2)
    fid = process_fidelity(Operator(U_target), Operator(best_qc))
    ops = best_qc.count_ops()
    t_count = ops.get('t', 0) + ops.get('tdg', 0)
    cx_count = ops.get('cx', 0)
    total_gates = sum(ops.values())
    
    # Display results
    print(f"\nMethod: {method_used}")
    print(f"TOTAL T-GATES: {t_count}")
    print(f"CX Count: {cx_count}")
    print(f"Total Gates: {total_gates}")
    print(f"Gate counts: {dict(ops)}")
    print(f"Operator Norm Distance: {dist:.6e}")
    print(f"Process Fidelity: {fid:.6f}")
    
    # Print the circuit
    print(f"\nCircuit:")
    print(best_qc.draw(output='text', fold=120))
    print()
    
    return best_qc, U_target


def check_existing_solution(i):
    """Check if we already have a good solution for unitary i.
    
    Args:
        i: Unitary index (1-11)
        
    Returns:
        tuple: (circuit, fidelity, t_count) or (None, 0, 0) if not found
    """
    qasm_file = f'unitary{i}_optimized.qasm'
    if not os.path.exists(qasm_file):
        return None, 0, 0
    
    try:
        with open(qasm_file) as f:
            qasm_str = f.read()
        qc = QuantumCircuit.from_qasm_str(qasm_str)
        
        U = np.load(f'unitary{i}.npy')
        U_clean, _ = polar(U)
        
        fid = process_fidelity(Operator(U_clean), Operator(qc))
        t_count = qc.count_ops().get('t', 0) + qc.count_ops().get('tdg', 0)
        
        return qc, fid, t_count
    except Exception as e:
        print(f"  Error loading existing solution: {e}")
        return None, 0, 0


def optimize_hard_unitary(i, max_t=7, max_cx=3, n_workers=4):
    """
    Optimize a hard unitary using multiple strategies (brute force + rmsynth).
    
    This is for unitaries that don't respond well to standard decomposition
    and need exhaustive search to find good Clifford+T approximations.
    
    Args:
        i: Unitary index
        max_t: Maximum T-gate count to search
        max_cx: Maximum CX gates
        n_workers: Number of parallel workers
        
    Returns:
        tuple: (best_circuit, best_fidelity, best_t_count)
    """
    print(f"\n{'='*60}")
    print(f"Optimizing Unitary {i} (hard case)")
    print(f"{'='*60}")
    
    U = np.load(f'unitary{i}.npy')
    U_clean, _ = polar(U)
    
    results = []
    
    # Strategy 1: Brute force parallel search
    print("\n--- Strategy 1: Brute Force Parallel ---")
    try:
        qc1, fid1 = brute_force_parallel(U_clean, max_t=max_t, max_cx=max_cx, n_workers=n_workers)
        if qc1:
            t1 = qc1.count_ops().get('t', 0) + qc1.count_ops().get('tdg', 0)
            results.append(('brute_force', qc1, fid1, t1))
            print(f"  Result: T={t1}, fid={fid1:.4f}")
    except Exception as e:
        print(f"  Failed: {e}")
    
    # Strategy 2: rmsynth-enhanced search (only if brute force didn't find good solution)
    best_so_far = max([r[2] for r in results]) if results else 0
    if best_so_far < 0.98:
        print("\n--- Strategy 2: rmsynth Structure Search ---")
        try:
            qc2, fid2 = rmsynth_brute_force_2q(U_clean, max_t=max_t, max_cx=max_cx)
            if qc2:
                t2 = qc2.count_ops().get('t', 0) + qc2.count_ops().get('tdg', 0)
                results.append(('rmsynth', qc2, fid2, t2))
                print(f"  Result: T={t2}, fid={fid2:.4f}")
        except Exception as e:
            print(f"  Failed: {e}")
    
    # Select best result by fidelity, then T-count
    if not results:
        return None, 0, float('inf')
    
    best = max(results, key=lambda x: (x[2], -x[3]))  # max fidelity, min T
    best_name, best_qc, best_fid, best_t = best
    
    print(f"\nBest for U{i}: {best_name}")
    print(f"  T={best_t}, fid={best_fid:.4f}")
    
    return best_qc, best_fid, best_t


def run_batch(effort=3, verbose=False, use_brute_force=True, fidelity_threshold=0.95):
    """
    Process all 11 unitaries and display summary table.
    
    This produces the same output as the notebook.
    
    Args:
        effort: Optimization effort level (1-3)
        verbose: If True, show detailed synthesis info
        use_brute_force: If True, try brute force search when fidelity is below threshold
        fidelity_threshold: Fidelity threshold below which to try brute force
    """
    # Process all unitaries
    circuits = {}
    unitaries = {}
    
    for i in range(1, 12):
        name = f"Unitary {i}"
        filepath = f"unitary{i}.npy"
        if os.path.exists(filepath):
            qc, U = optimize_and_display(name, filepath, effort=effort, verbose=verbose,
                                         use_brute_force=use_brute_force, 
                                         fidelity_threshold=fidelity_threshold)
            circuits[name] = qc
            unitaries[name] = U
        else:
            print(f"Warning: {filepath} not found, skipping {name}")
    
    # Generate summary table
    print("\n" + "=" * 100)
    print("COMPLETE VERIFICATION SUMMARY - All 11 Unitaries")
    print("=" * 100)
    
    results = []
    for name in sorted(circuits.keys(), key=lambda x: int(x.split()[-1])):
        qc = circuits[name]
        U = unitaries[name]
        
        ops = qc.count_ops()
        t_count = ops.get('t', 0) + ops.get('tdg', 0)
        cx_count = ops.get('cx', 0)
        total_gates = sum(ops.values())
        U_final = Operator(qc).data
        dist = np.linalg.norm(U - U_final, ord=2)
        fid = process_fidelity(Operator(U), Operator(qc))
        n_qubits = int(np.log2(U.shape[0]))
        
        results.append({
            "Unitary": name,
            "Qubits": n_qubits,
            "T-Count": t_count,
            "CX": cx_count,
            "Total": total_gates,
            "Distance": f"{dist:.2e}",
            "Fidelity": f"{fid:.6f}",
        })
    
    # Print table using pandas for consistent formatting with notebook
    try:
        import pandas as pd
        df = pd.DataFrame(results)
        print(df.to_string(index=False))
    except ImportError:
        # Fallback to manual formatting if pandas not available
        header = f"{'Unitary':<12} {'Qubits':<7} {'T-Count':<8} {'CX':<5} {'Total':<7} {'Distance':<12} {'Fidelity':<12}"
        print(header)
        print("-" * len(header))
        for r in results:
            print(f"{r['Unitary']:<12} {r['Qubits']:<7} {r['T-Count']:<8} {r['CX']:<5} {r['Total']:<7} {r['Distance']:<12} {r['Fidelity']:<12}")
    print("=" * 100)
    
    # Summary statistics
    total_t = sum(r['T-Count'] for r in results)
    total_cx = sum(r['CX'] for r in results)
    avg_fid = sum(float(r['Fidelity']) for r in results) / len(results)
    
    # Flag circuits that might need attention
    low_fid = [(r['Unitary'], float(r['Fidelity'])) for r in results if float(r['Fidelity']) < 0.98]
    if low_fid:
        print("⚠️  Circuits with fidelity < 0.98:")
        for name, fid in low_fid:
            print(f"   - {name}: F={fid:.4f}")
    else:
        print("✅ All circuits have fidelity ≥ 0.98")
    
    return circuits, unitaries, results


def run_hard_batch(hard_unitaries=None, max_t=7, max_cx=3, n_workers=4, fidelity_threshold=0.80):
    """
    Run heavy optimization on hard unitaries only.
    
    This uses brute force parallel search + rmsynth for unitaries that
    don't respond well to standard optimization.
    
    Args:
        hard_unitaries: List of unitary indices to optimize (default: [6, 7, 10])
        max_t: Maximum T-gate count to search
        max_cx: Maximum CX gates
        n_workers: Number of parallel workers
        fidelity_threshold: Minimum fidelity to save (default 0.80 for hard cases)
        
    Returns:
        list: Summary of results
    """
    import multiprocessing
    multiprocessing.set_start_method('fork', force=True)
    
    if hard_unitaries is None:
        hard_unitaries = [6, 7, 10]  # Default hard cases
    
    summary = []
    
    # Check existing solutions first
    print("Checking existing solutions...")
    for i in hard_unitaries[:]:  # Copy list to allow modification
        result = check_existing_solution(i)
        if result[0] is not None:
            qc, fid, t_count = result
            print(f"  U{i}: existing T={t_count}, fid={fid:.4f}")
            if fid >= 0.85:
                summary.append((i, t_count, fid, 'EXISTING_GOOD'))
                hard_unitaries.remove(i)
            else:
                summary.append((i, t_count, fid, 'NEEDS_IMPROVEMENT'))
        else:
            print(f"  U{i}: no existing solution")
    
    print(f"\nHard unitaries to optimize: {hard_unitaries}")
    
    for i in hard_unitaries:
        best_qc, best_fid, best_t = optimize_hard_unitary(i, max_t=max_t, max_cx=max_cx, n_workers=n_workers)
        
        if best_qc and best_fid > fidelity_threshold:
            qasm = qiskit.qasm2.dumps(best_qc)
            with open(f'unitary{i}_optimized.qasm', 'w') as f:
                f.write(qasm)
            print(f"\n✅ U{i}: Saved with T={best_t}, Fid={best_fid:.4f}")
            summary.append((i, best_t, best_fid, 'SAVED'))
        else:
            fid_str = f"{best_fid:.4f}" if best_fid else "N/A"
            print(f"\n⚠️ U{i}: Best found fid={fid_str} (keeping as approximate)")
            if best_qc:
                qasm = qiskit.qasm2.dumps(best_qc)
                with open(f'unitary{i}_optimized.qasm', 'w') as f:
                    f.write(qasm)
                summary.append((i, best_t, best_fid, 'APPROXIMATE'))
            else:
                summary.append((i, 'N/A', 0, 'FAILED'))
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Unitary':<10} {'T-count':<10} {'Fidelity':<12} {'Status':<15}")
    print("-"*47)
    for i, t, fid, status in sorted(summary, key=lambda x: x[0]):
        t_str = str(t) if isinstance(t, int) else t
        fid_str = f"{fid:.4f}" if isinstance(fid, float) else str(fid)
        print(f"U{i:<9} {t_str:<10} {fid_str:<12} {status}")
    print("="*60)
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Optimize quantum circuits for minimal T-gate count",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file optimization:
  python optimize_unitaries.py unitary1.npy output.qasm --effort 3
  python optimize_unitaries.py unitary7.npy output.qasm --effort 3 --decoder ml-exact
  
  # Batch mode (process all 11 unitaries):
  python optimize_unitaries.py --batch
  python optimize_unitaries.py --batch --effort 3 --no-brute-force
  
  # Hard mode (brute force search on specific hard unitaries):
  python optimize_unitaries.py --hard                    # Default: U6, U7, U10
  python optimize_unitaries.py --hard 6 7 9 10          # Specify which unitaries
  python optimize_unitaries.py --hard --max-t 5 --max-cx 2
  
  # Challenge 12 (commuting Pauli terms, NO S gates!):
  python optimize_unitaries.py --challenge12
  python optimize_unitaries.py --challenge12 --json custom.json --output custom.qasm
        """
    )
    parser.add_argument("input_file", nargs='?', help="Input unitary file (.npy or .txt)")
    parser.add_argument("output_file", nargs='?', help="Output QASM file")
    parser.add_argument("--batch", action="store_true", 
                        help="Process all 11 unitaries and show summary")
    parser.add_argument("--hard", nargs='*', type=int, metavar='N',
                        help="Run heavy brute force optimization on hard unitaries (default: 6 7 10)")
    parser.add_argument("--challenge12", action="store_true",
                        help="Run Challenge 12 optimization (commuting Paulis, NO S gates!)")
    parser.add_argument("--json", type=str, default="challenge12.json",
                        help="JSON file for Challenge 12 (default: challenge12.json)")
    parser.add_argument("--output", type=str, default="challenge12_optimized.qasm",
                        help="Output file for Challenge 12 (default: challenge12_optimized.qasm)")
    parser.add_argument("--max-t", type=int, default=7, 
                        help="Maximum T-gate count for hard search (default=7)")
    parser.add_argument("--max-cx", type=int, default=3, 
                        help="Maximum CX gates for hard search (default=3)")
    parser.add_argument("--workers", type=int, default=4, 
                        help="Number of parallel workers for hard search (default=4)")
    parser.add_argument("--effort", type=int, default=3, help="Optimization effort (1-3, default=3)")
    parser.add_argument("--epsilon", type=float, default=1e-10, help="Gridsynth precision")
    parser.add_argument("--decoder", default="auto", 
                        choices=["auto", "ml-exact", "dumer", "dumer-list", "rpa"],
                        help="rmsynth decoder")
    parser.add_argument("--no-rmsynth", action="store_true", help="Disable rmsynth")
    parser.add_argument("--no-brute-force", action="store_true", 
                        help="Disable brute force fallback for low-fidelity circuits")
    parser.add_argument("--fidelity-threshold", type=float, default=0.95,
                        help="Fidelity threshold for brute force fallback (default=0.95)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed synthesis info")
    args = parser.parse_args()

    # Challenge 12 mode - commuting Pauli exponentials, NO S gates!
    if args.challenge12:
        run_challenge12(json_file=args.json, output_file=args.output)
        return

    # Hard mode - brute force on specific unitaries
    if args.hard is not None:
        hard_list = args.hard if args.hard else [6, 7, 10]  # Default hard cases
        run_hard_batch(
            hard_unitaries=hard_list,
            max_t=args.max_t,
            max_cx=args.max_cx,
            n_workers=args.workers,
            fidelity_threshold=0.80  # Lower threshold for hard cases
        )
        return

    # Batch mode
    if args.batch:
        run_batch(
            effort=args.effort,
            verbose=args.verbose,
            use_brute_force=not args.no_brute_force,
            fidelity_threshold=args.fidelity_threshold
        )
        return
    
    # Single file mode - require input and output files
    if not args.input_file or not args.output_file:
        parser.error("input_file and output_file are required unless using --batch mode")

    try:
        # Load and decompose
        U_target = load_unitary(args.input_file)
        
        qc = decompose_and_optimize(U_target, args.effort)
        qc = apply_manual_identities(qc, epsilon=args.epsilon)
        qc = optimize_circuit(qc)
        
        # rmsynth optimization
        if not args.no_rmsynth:
            print("\n--- rmsynth optimization ---")
            before_t = qc.count_ops().get('t', 0) + qc.count_ops().get('tdg', 0)
            qc = optimize_diagonal_chunks_with_rmsynth(qc, effort=args.effort, decoder=args.decoder)
            qc = optimize_circuit(qc)
            after_t = qc.count_ops().get('t', 0) + qc.count_ops().get('tdg', 0)
            print(f"  T-count: {before_t} -> {after_t}")

        # Phase alignment
        U_actual = Operator(qc).data
        phase_diff = np.angle(np.trace(np.conj(U_target.T) @ U_actual))
        qc.global_phase = -phase_diff
        
        # Brute force fallback for 2-qubit unitaries
        n_qubits = int(np.log2(U_target.shape[0]))
        fid = process_fidelity(Operator(U_target), Operator(qc))
        if not args.no_brute_force and fid < args.fidelity_threshold and n_qubits == 2:
            print(f"\n--- Trying brute force search (fidelity {fid:.4f} < {args.fidelity_threshold}) ---")
            t_count = qc.count_ops().get('t', 0) + qc.count_ops().get('tdg', 0)
            try:
                bf_qc, bf_fid = quick_search(U_target, max_t=t_count + 2)
                if bf_qc is not None and bf_fid > fid:
                    bf_U = Operator(bf_qc).data
                    bf_phase = np.angle(np.trace(np.conj(U_target.T) @ bf_U))
                    bf_qc.global_phase = -bf_phase
                    qc = bf_qc
                    print(f"  Brute force found better: F={bf_fid:.4f}")
            except Exception as e:
                print(f"  Brute force failed: {e}")

        # Results
        U_final = Operator(qc).data
        dist = np.linalg.norm(U_target - U_final, ord=2)
        fid = process_fidelity(Operator(U_target), Operator(qc))
        ops = qc.count_ops()

        print("\n--- Results ---")
        print(f"T-GATES: {ops.get('t', 0) + ops.get('tdg', 0)}")
        print(f"Gates: {dict(ops)}")
        print(f"Distance: {dist:.6e}")
        print(f"Fidelity: {fid:.6f}")
        print(f"\n{qc.draw(output='text', fold=120)}")

        # Save
        with open(args.output_file, 'w') as f:
            f.write(qiskit.qasm2.dumps(qc))
        print(f"\nSaved to {args.output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# =============================================================================
# CHALLENGE 12: COMMUTING PAULI PHASE PROGRAM
# =============================================================================
# Gate set: {H, T, T†, CNOT} only - NO S GATES ALLOWED!
# No scoring metric - just T-count and fidelity verification
#
# Two synthesis strategies:
# 1. Direct Synthesis: Synthesize each Pauli exponential term independently
# 2. Diagonalization + rmsynth: Find Clifford C that diagonalizes all Paulis,
#    optimize the diagonal phase polynomial with rmsynth, then wrap with C
# =============================================================================

def pauli_to_symplectic(pauli_str):
    """
    Convert Pauli string to symplectic representation (x_vec, z_vec).
    
    For a Pauli P = i^p * X^{x} * Z^{z}, we store (x, z) as binary vectors.
    X -> (1, 0), Y -> (1, 1), Z -> (0, 1), I -> (0, 0)
    
    Args:
        pauli_str: String like "XIZY"
        
    Returns:
        tuple: (x_vec, z_vec) as lists of 0/1
    """
    x_vec = []
    z_vec = []
    for c in pauli_str:
        if c == 'I':
            x_vec.append(0)
            z_vec.append(0)
        elif c == 'X':
            x_vec.append(1)
            z_vec.append(0)
        elif c == 'Y':
            x_vec.append(1)
            z_vec.append(1)
        elif c == 'Z':
            x_vec.append(0)
            z_vec.append(1)
    return x_vec, z_vec


def symplectic_to_pauli(x_vec, z_vec):
    """
    Convert symplectic representation back to Pauli string.
    
    Args:
        x_vec, z_vec: Binary vectors
        
    Returns:
        str: Pauli string like "XIZY"
    """
    result = []
    for x, z in zip(x_vec, z_vec):
        if x == 0 and z == 0:
            result.append('I')
        elif x == 1 and z == 0:
            result.append('X')
        elif x == 1 and z == 1:
            result.append('Y')
        elif x == 0 and z == 1:
            result.append('Z')
    return ''.join(result)


def diagonalize_commuting_paulis_proper(pauli_strings, n_qubits):
    """
    Find a Clifford circuit C that maps all Paulis to Z-strings.
    
    For commuting Paulis that have mixed X/Y/Z on same qubit across different terms,
    we need entangling gates (CNOTs) in addition to single-qubit Cliffords.
    
    The algorithm uses symplectic Gaussian elimination:
    1. Build the (X|Z) symplectic matrix
    2. Row-reduce the X part to eliminate X components
    3. Track the Clifford operations needed (H, S, CNOT)
    
    Key insight: CNOT in Heisenberg picture:
    - CNOT(c,t): X_c -> X_c X_t, X_t -> X_t, Z_c -> Z_c, Z_t -> Z_c Z_t
    - In symplectic: x_t' = x_t, x_c' = x_c XOR x_t (when used for X elimination)
                   z_c' = z_c, z_t' = z_t XOR z_c
    
    Args:
        pauli_strings: List of Pauli strings
        n_qubits: Number of qubits
        
    Returns:
        tuple: (diagonalizing_circuit, z_strings)
    """
    # Convert to symplectic form
    paulis = [pauli_to_symplectic(p) for p in pauli_strings]
    
    # X[i,j] = 1 if Pauli i has X component on qubit j
    X = np.array([p[0] for p in paulis], dtype=np.int8)
    Z = np.array([p[1] for p in paulis], dtype=np.int8)
    num_paulis = len(pauli_strings)
    
    qc = QuantumCircuit(n_qubits)
    
    # Process each qubit to eliminate X components
    for col in range(n_qubits):
        # Check if any Pauli has X on this qubit
        x_rows = np.where(X[:, col] == 1)[0]
        
        if len(x_rows) == 0:
            continue  # Already diagonal on this qubit
        
        # Check if there's also Y (x=1, z=1) - handle Y first
        y_mask = (X[:, col] == 1) & (Z[:, col] == 1)
        if np.any(y_mask):
            # Apply S (= TT) to convert Y -> X
            # S: (x, z) -> (x, x XOR z)
            qc.t(col)
            qc.t(col)
            Z[:, col] = (Z[:, col] + X[:, col]) % 2
        
        # Now apply H to convert X -> Z
        qc.h(col)
        X[:, col], Z[:, col] = Z[:, col].copy(), X[:, col].copy()
        
        # Note: This simple approach doesn't fully handle the case where
        # different Paulis have X, Y, Z on the same qubit.
        # For full generality, we'd need CNOT-based row reduction.
        # However, the per-term approach handles this correctly.
    
    # Build resulting strings
    z_strings = []
    for row in range(num_paulis):
        z_str = symplectic_to_pauli(X[row].tolist(), Z[row].tolist())
        z_strings.append(z_str)
    
    return qc, z_strings


def diagonalize_single_pauli(pauli_str, n_qubits):
    """
    Create a circuit that diagonalizes a single Pauli to Z-form.
    
    For each X: apply H (X -> Z)
    For each Y: apply Sdg H (Y -> Z)
    
    Returns both the circuit and the resulting Z-string.
    """
    qc = QuantumCircuit(n_qubits)
    z_str = []
    
    for i, c in enumerate(pauli_str):
        if c == 'I':
            z_str.append('I')
        elif c == 'X':
            qc.h(i)
            z_str.append('Z')
        elif c == 'Y':
            # Sdg H: Y -> -Z (up to sign)
            # Using Tdg Tdg instead of Sdg
            qc.tdg(i)
            qc.tdg(i)
            qc.h(i)
            z_str.append('Z')
        elif c == 'Z':
            z_str.append('Z')
    
    return qc, ''.join(z_str)


def emit_t_power(qc, qubit, k):
    """Emit T^k (mod 8) gates on a qubit. Used for Challenge 12 (no S gates)."""
    k_mod = k % 8
    if k_mod == 0: return
    elif k_mod == 1: qc.t(qubit)
    elif k_mod == 2: qc.t(qubit); qc.t(qubit)
    elif k_mod == 3: qc.t(qubit); qc.t(qubit); qc.t(qubit)
    elif k_mod == 4: qc.t(qubit); qc.t(qubit); qc.t(qubit); qc.t(qubit)
    elif k_mod == 5: qc.tdg(qubit); qc.tdg(qubit); qc.tdg(qubit)
    elif k_mod == 6: qc.tdg(qubit); qc.tdg(qubit)
    elif k_mod == 7: qc.tdg(qubit)


def synthesize_diagonal_z_rotation(z_str, k, n_qubits):
    """Synthesize exp(-i * pi/8 * k * Z-string) using CNOT ladder + T^k."""
    qc = QuantumCircuit(n_qubits)
    z_qubits = [i for i, c in enumerate(z_str) if c == 'Z']
    if not z_qubits:
        return qc
    target = z_qubits[-1]
    for ctrl in z_qubits[:-1]:
        qc.cx(ctrl, target)
    emit_t_power(qc, target, k)
    for ctrl in reversed(z_qubits[:-1]):
        qc.cx(ctrl, target)
    return qc


def optimize_challenge12_per_term_diag(data, output_file="challenge12_perterm_diag.qasm", verbose=True):
    """
    Optimize Challenge 12 using per-term diagonalization.
    
    For each Pauli term:
    1. Apply basis change circuit C to diagonalize (X->Z via H, Y->Z via SdgH)
    2. Apply diagonal Z rotation
    3. Apply inverse basis change C†
    
    This is equivalent to the direct synthesis but structured differently.
    """
    n = data['n']
    terms = data['terms']
    
    if verbose:
        print(f"\n{'='*60}")
        print("CHALLENGE 12: PER-TERM DIAGONALIZATION")
        print(f"{'='*60}")
        print(f"Qubits: {n}, Terms: {len(terms)}")
    
    U_target = compute_challenge12_unitary(data)
    
    qc = QuantumCircuit(n)
    
    for term in terms:
        pauli = term['pauli']
        k = term['k']
        
        # Diagonalize
        diag_circ, z_str = diagonalize_single_pauli(pauli, n)
        
        # Apply basis change
        qc = qc.compose(diag_circ)
        
        # Apply diagonal rotation
        rot_circ = synthesize_diagonal_z_rotation(z_str, k, n)
        qc = qc.compose(rot_circ)
        
        # Undo basis change
        qc = qc.compose(diag_circ.inverse())
    
    # Simplify
    qc = simplify_circuit_no_s(qc)
    qc = cancel_adjacent_h(qc)
    
    # Validate
    is_valid, msg = validate_circuit_u12(qc)
    if not is_valid:
        if verbose:
            print(f"ERROR: {msg}")
        return None, None, None
    
    ops = qc.count_ops()
    t_count = ops.get('t', 0) + ops.get('tdg', 0)
    
    if verbose:
        print(f"T-count: {t_count}")
    
    # Fidelity
    try:
        U_circuit = Operator(qc).data
        fid = np.abs(np.trace(U_target.conj().T @ U_circuit))**2 / (2**(2*n))
        if verbose:
            print(f"Fidelity: {fid:.6f}")
    except Exception as e:
        fid = None
        if verbose:
            print(f"Could not compute fidelity: {e}")
    
    qasm = qiskit.qasm2.dumps(qc)
    with open(output_file, 'w') as f:
        f.write(qasm)
    
    if verbose:
        print(f"Saved to {output_file}")
    
    return qc, t_count, fid


def build_diagonal_phase_circuit(z_strings, k_values, n_qubits):
    """
    Build a diagonal phase circuit for Z-string exponentials.
    
    For each exp(-i * pi/8 * k * Z_string), we compute the parity
    and apply the appropriate phase.
    
    Args:
        z_strings: List of Z-strings (only 'Z' and 'I')
        k_values: List of k coefficients
        n_qubits: Number of qubits
        
    Returns:
        QuantumCircuit with diagonal phase gates
    """
    qc = QuantumCircuit(n_qubits)
    
    for z_str, k in zip(z_strings, k_values):
        z_qubits = [i for i, c in enumerate(z_str) if c == 'Z']
        if not z_qubits:
            continue
        target = z_qubits[-1]
        for ctrl in z_qubits[:-1]:
            qc.cx(ctrl, target)
        emit_t_power(qc, target, k)
        for ctrl in reversed(z_qubits[:-1]):
            qc.cx(ctrl, target)
    
    return qc


def optimize_diagonal_with_rmsynth_no_s(qc, n_qubits):
    """
    Optimize a diagonal circuit using rmsynth, then convert S gates to T gates.
    
    rmsynth optimizes phase polynomials but may output S gates.
    We convert S -> T T and Sdg -> Tdg Tdg afterward.
    
    Args:
        qc: Diagonal QuantumCircuit (CNOT + phase gates)
        n_qubits: Number of qubits
        
    Returns:
        Optimized QuantumCircuit with no S gates
    """
    # Try rmsynth optimization
    try:
        rm_circ = qiskit_to_rmsynth(qc)
        if rm_circ is not None and rm_circ.t_count() > 0:
            optimizer = RMSynthOptimizer(decoder="auto", effort=2)
            rm_optimized = optimizer.optimize(rm_circ)
            qc_opt = rmsynth_to_qiskit(rm_optimized, n_qubits)
        else:
            qc_opt = qc
    except Exception:
        qc_opt = qc
    
    # Convert any S gates to T gates
    new_qc = QuantumCircuit(n_qubits)
    for inst in qc_opt.data:
        name = inst.operation.name.lower()
        qubits = inst.qubits
        
        if name == 's':
            new_qc.t(qubits)
            new_qc.t(qubits)
        elif name == 'sdg':
            new_qc.tdg(qubits)
            new_qc.tdg(qubits)
        else:
            new_qc.append(inst.operation, qubits)
    
    return new_qc


def optimize_challenge12_diagonalization(data, output_file="challenge12_diag_optimized.qasm", verbose=True):
    """
    Optimize Challenge 12 using Clifford Diagonalization + rmsynth.
    
    Strategy:
    1. Find Clifford C that diagonalizes all Paulis to Z-strings
    2. Build diagonal phase polynomial circuit
    3. Optimize with rmsynth
    4. Final circuit: C† @ optimized_diagonal @ C
    
    NO S GATES ALLOWED - only {H, T, Tdg, CX}!
    
    Args:
        data: Challenge 12 data dict
        output_file: Output QASM file path
        verbose: Print progress
        
    Returns:
        tuple: (circuit, t_count, fidelity)
    """
    n = data['n']
    terms = data['terms']
    
    if verbose:
        print(f"\n{'='*60}")
        print("CHALLENGE 12: DIAGONALIZATION METHOD")
        print(f"{'='*60}")
        print(f"Qubits: {n}")
        print(f"Terms: {len(terms)}")
        print(f"Gate set: {{H, T, Tdg, CX}} - NO S GATES!")
        print(f"{'='*60}")
    
    # Compute target unitary
    if verbose:
        print("\nComputing target unitary...")
    U_target = compute_challenge12_unitary(data)
    
    # Extract Pauli strings and k values
    pauli_strings = [t['pauli'] for t in terms]
    k_values = [t['k'] for t in terms]
    
    # Step 1: Diagonalize using proper symplectic method
    if verbose:
        print("Finding diagonalizing Clifford circuit...")
    diag_qc, z_strings = diagonalize_commuting_paulis_proper(pauli_strings, n)
    
    # Check how many are actually diagonalized
    n_diag = sum(1 for z in z_strings if all(c in 'IZ' for c in z))
    if verbose:
        print(f"  Diagonalized: {n_diag}/{len(z_strings)} terms")
    
    # Step 2: Build diagonal phase circuit
    if verbose:
        print("Building diagonal phase circuit...")
    diag_phase_qc = build_diagonal_phase_circuit(z_strings, k_values, n)
    
    # Step 3: Optimize with rmsynth
    if verbose:
        print("Optimizing with rmsynth (converting S to T)...")
    diag_opt_qc = optimize_diagonal_with_rmsynth_no_s(diag_phase_qc, n)
    
    # Step 4: Compose: C† @ diagonal @ C
    if verbose:
        print("Composing final circuit...")
    
    # C† (inverse of diagonalization)
    c_inv = diag_qc.inverse()
    
    # Final: C, diagonal, C†
    final_qc = QuantumCircuit(n)
    final_qc = final_qc.compose(diag_qc)
    final_qc = final_qc.compose(diag_opt_qc)
    final_qc = final_qc.compose(c_inv)
    
    # Simplify
    final_qc = simplify_circuit_no_s(final_qc)
    final_qc = cancel_adjacent_h(final_qc)
    
    # Validate
    is_valid, msg = validate_circuit_u12(final_qc)
    if not is_valid:
        if verbose:
            print(f"ERROR: {msg}")
        return None, None, None
    
    # Count gates
    ops = final_qc.count_ops()
    t_count = ops.get('t', 0) + ops.get('tdg', 0)
    cx_count = ops.get('cx', 0)
    h_count = ops.get('h', 0)
    
    if verbose:
        print(f"\nCircuit stats:")
        print(f"  T-count: {t_count}")
        print(f"  CX-count: {cx_count}")
        print(f"  H-count: {h_count}")
    
    # Compute fidelity
    if verbose:
        print("\nComputing fidelity...")
    try:
        U_circuit = Operator(final_qc).data
        fid = np.abs(np.trace(U_target.conj().T @ U_circuit))**2 / (2**(2*n))
        dist = np.linalg.norm(U_target - U_circuit * np.exp(-1j * np.angle(np.trace(U_target.conj().T @ U_circuit))), ord=2)
        
        if verbose:
            print(f"  Process fidelity: {fid:.6f}")
            print(f"  Distance: {dist:.6e}")
    except Exception as e:
        if verbose:
            print(f"  Could not compute fidelity: {e}")
        fid = None
    
    # Save
    qasm = qiskit.qasm2.dumps(final_qc)
    with open(output_file, 'w') as f:
        f.write(qasm)
    
    if verbose:
        print(f"\n✅ Saved to {output_file}")
    
    return final_qc, t_count, fid


def load_challenge12(filepath="challenge12.json"):
    """
    Load Challenge 12 JSON file containing commuting Pauli terms.
    
    Args:
        filepath: Path to challenge12.json
        
    Returns:
        dict with keys: n (qubits), terms (list of {pauli, k})
    """
    with open(filepath) as f:
        data = json.load(f)
    return data


def pauli_string_to_matrix(pauli_str):
    """
    Convert a Pauli string like "XIZY" to the full tensor product matrix.
    
    Uses Qiskit qubit ordering: for string "XY", q0=X, q1=Y,
    the tensor product is Y ⊗ X (reversed from index order).
    
    Args:
        pauli_str: String of I, X, Y, Z characters
        
    Returns:
        numpy.ndarray: The tensor product Pauli operator (Qiskit ordering)
    """
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    pauli_map = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
    
    # Reversed order to match Qiskit convention
    result = pauli_map[pauli_str[-1]]
    for c in reversed(pauli_str[:-1]):
        result = np.kron(result, pauli_map[c])
    
    return result


def compute_challenge12_unitary(data):
    """
    Compute the target unitary U = prod_j exp(-i*pi/8 * k_j * P_j).
    
    Since all terms commute, order doesn't matter.
    
    Args:
        data: Challenge 12 data dict with n and terms
        
    Returns:
        numpy.ndarray: The target unitary matrix
    """
    n = data['n']
    dim = 2 ** n
    U = np.eye(dim, dtype=complex)
    
    for term in data['terms']:
        pauli_str = term['pauli']
        k = term['k']
        
        P = pauli_string_to_matrix(pauli_str)
        
        # exp(-i * pi/8 * k * P)
        # Since P^2 = I for Pauli, exp(-i*theta*P) = cos(theta)*I - i*sin(theta)*P
        theta = np.pi / 8 * k
        expP = np.cos(theta) * np.eye(dim, dtype=complex) - 1j * np.sin(theta) * P
        
        U = U @ expP
    
    return U


def validate_circuit_u12(qc):
    """
    Validate that circuit uses only {H, T, Tdg, CX} - NO S gates!
    
    Args:
        qc: QuantumCircuit to validate
        
    Returns:
        tuple: (is_valid, error_message)
    """
    allowed = {'h', 't', 'tdg', 'cx', 'barrier', 'measure', 'id'}
    ops = qc.count_ops()
    
    for gate in ops:
        if gate.lower() not in allowed:
            return False, f"Invalid gate '{gate}' - only {{H, T, Tdg, CX}} allowed for U12!"
    
    # Check for S gates specifically
    if 's' in ops or 'sdg' in ops:
        return False, "S/Sdg gates NOT allowed for Challenge 12!"
    
    return True, "Valid"


def synthesize_pauli_exponential_no_s(pauli_str, k, n_qubits):
    """
    Synthesize exp(-i * pi/8 * k * P) using only {H, T, Tdg, CX}.
    NO S GATES ALLOWED!
    
    Strategy:
    1. Use Cliffords (H, CX) to diagonalize the Pauli to Z...Z form
    2. Apply T rotations for the diagonal phase
    3. Undo the Clifford transformation
    
    Args:
        pauli_str: Pauli string like "XIZY"
        k: Coefficient (typically 1 or 7)
        n_qubits: Number of qubits
        
    Returns:
        QuantumCircuit
    """
    qc = QuantumCircuit(n_qubits)
    
    # Step 1: Basis change to diagonalize - transform X->Z, Y->Z
    # For Y: H*S†*Y*S*H = H*S†*Y*S*H... but we can't use S!
    # Instead: Y = i*X*Z, and we can use HXH=Z
    # Actually, to diagonalize Y, we need: Y -> Z via appropriate rotation
    # Without S: we can use H and T gates to approximate
    # But exact: HYH ≠ Z. We need basis that takes Y to Z.
    # Y eigenbasis: |+y⟩, |-y⟩. To get there: apply S†H or HS
    # Without S: Use H and multiple T gates (T^4 = Z, T^2 = S)
    
    # Track which qubits have non-I Paulis
    active_qubits = []
    for i, p in enumerate(pauli_str):
        if p != 'I':
            active_qubits.append((i, p))
    
    # Apply basis change
    for i, p in active_qubits:
        if p == 'X':
            qc.h(i)  # X -> Z
        elif p == 'Y':
            # Y -> Z requires: S†H or equivalent
            # Without S: T†T†H (since T† T† = S†)
            qc.tdg(i)
            qc.tdg(i)
            qc.h(i)
    
    # Step 2: Now we have product of Z's on active qubits
    # For a product Z_i1 ⊗ Z_i2 ⊗ ... , the exponential is a diagonal
    # We need to compute the parity and apply phase based on it
    
    # Use CNOT ladder to compute parity into one qubit
    if len(active_qubits) > 1:
        # CNOT cascade to accumulate parity
        target = active_qubits[-1][0]  # Last active qubit gets the parity
        for i, p in active_qubits[:-1]:
            qc.cx(i, target)
    
    # Step 3: Apply the phase rotation on the target qubit
    # exp(-i * pi/8 * k * Z) = diag(exp(-i*pi*k/8), exp(i*pi*k/8))
    # This is an Rz(pi*k/4) = T^k rotation (since T = Rz(pi/4))
    if len(active_qubits) > 0:
        target = active_qubits[-1][0] if len(active_qubits) > 1 else active_qubits[0][0]
        
        # Apply T^k (mod 8): k=1 -> T, k=7 -> T† (since T^7 = T†)
        k_mod = k % 8
        if k_mod == 1:
            qc.t(target)
        elif k_mod == 7:
            qc.tdg(target)
        elif k_mod == 2:
            qc.t(target)
            qc.t(target)
        elif k_mod == 3:
            qc.t(target)
            qc.t(target)
            qc.t(target)
        elif k_mod == 4:
            # T^4 = Z, but Z = T^4, we don't have Z directly
            # Use H T^2 H since HZH = X and we don't need X here
            # Actually Z can be decomposed as T T T T
            for _ in range(4):
                qc.t(target)
        elif k_mod == 5:
            qc.tdg(target)
            qc.tdg(target)
            qc.tdg(target)
        elif k_mod == 6:
            qc.tdg(target)
            qc.tdg(target)
    
    # Step 4: Undo CNOT cascade (reverse order)
    if len(active_qubits) > 1:
        target = active_qubits[-1][0]
        for i, p in reversed(active_qubits[:-1]):
            qc.cx(i, target)
    
    # Step 5: Undo basis change (reverse order)
    # For Y: basis change was Sdg then H (matrix: H @ Sdg)
    # Undo is (H @ Sdg)† = S @ H, applied in circuit as: first H, then S
    for i, p in reversed(active_qubits):
        if p == 'X':
            qc.h(i)
        elif p == 'Y':
            # Undo: S @ H in matrix = first H, then S in circuit = H T T
            qc.h(i)
            qc.t(i)
            qc.t(i)
    
    return qc


def simplify_circuit_no_s(qc):
    """
    Simplify circuit by accumulating phase gates (mod 8).
    ONLY uses T and Tdg - NO S GATES for Challenge 12!
    """
    new_qc = QuantumCircuit(qc.num_qubits)
    pending = {}
    phase_gates = {'t': 1, 'tdg': 7}  # Only T gates allowed
    
    for inst in qc.data:
        name = inst.operation.name
        qubits = inst.qubits
        
        if len(qubits) == 1:
            q_idx = qc.find_bit(qubits[0]).index
            if name in phase_gates:
                pending[q_idx] = (pending.get(q_idx, 0) + phase_gates[name]) % 8
            elif name == 'h':
                emit_phase(new_qc, pending.get(q_idx, 0), qubits, allow_s=False)
                pending[q_idx] = 0
                new_qc.h(qubits)
            else:
                emit_phase(new_qc, pending.get(q_idx, 0), qubits, allow_s=False)
                pending[q_idx] = 0
                new_qc.append(inst.operation, qubits)
        else:
            for q in qubits:
                q_idx = qc.find_bit(q).index
                emit_phase(new_qc, pending.get(q_idx, 0), [q], allow_s=False)
                pending[q_idx] = 0
            new_qc.append(inst.operation, qubits)
    
    for q_idx in range(qc.num_qubits):
        emit_phase(new_qc, pending.get(q_idx, 0), [qc.qubits[q_idx]], allow_s=False)
    
    return new_qc


def optimize_challenge12(data, output_file="challenge12_optimized.qasm", verbose=True):
    """
    Optimize Challenge 12: Commuting Pauli Phase Program.
    
    NO S GATES ALLOWED - only {H, T, Tdg, CX}!
    
    Args:
        data: Challenge 12 data dict
        output_file: Output QASM file path
        verbose: Print progress
        
    Returns:
        tuple: (circuit, t_count, fidelity)
    """
    n = data['n']
    terms = data['terms']
    
    if verbose:
        print(f"\n{'='*60}")
        print("CHALLENGE 12: COMMUTING PAULI PHASE PROGRAM")
        print(f"{'='*60}")
        print(f"Qubits: {n}")
        print(f"Terms: {len(terms)}")
        print(f"Gate set: {{H, T, Tdg, CX}} - NO S GATES!")
        print(f"{'='*60}")
    
    # Compute target unitary
    if verbose:
        print("\nComputing target unitary...")
    U_target = compute_challenge12_unitary(data)
    
    # Build circuit by synthesizing each term
    if verbose:
        print("Synthesizing circuit...")
    
    qc = QuantumCircuit(n)
    
    # Since terms commute, we can potentially optimize by grouping
    # For now, straightforward synthesis
    for term in terms:
        term_qc = synthesize_pauli_exponential_no_s(term['pauli'], term['k'], n)
        qc = qc.compose(term_qc)
    
    # Validate - no S gates!
    is_valid, msg = validate_circuit_u12(qc)
    if not is_valid:
        print(f"ERROR: {msg}")
        return None, None, None
    
    # Count gates
    ops = qc.count_ops()
    t_count = ops.get('t', 0) + ops.get('tdg', 0)
    cx_count = ops.get('cx', 0)
    h_count = ops.get('h', 0)
    total_gates = sum(ops.values())
    
    if verbose:
        print(f"\nCircuit stats (before optimization):")
        print(f"  T-count: {t_count}")
        print(f"  CX-count: {cx_count}")
        print(f"  H-count: {h_count}")
        print(f"  Total gates: {total_gates}")
    
    # Simplify circuit - NO S GATES!
    if verbose:
        print("\nSimplifying circuit (no S gates)...")
    qc = simplify_circuit_no_s(qc)
    qc = cancel_adjacent_h(qc)
    
    # Re-validate
    is_valid, msg = validate_circuit_u12(qc)
    if not is_valid:
        print(f"ERROR after optimization: {msg}")
        return None, None, None
    
    # Recount gates
    ops = qc.count_ops()
    t_count = ops.get('t', 0) + ops.get('tdg', 0)
    cx_count = ops.get('cx', 0)
    h_count = ops.get('h', 0)
    total_gates = sum(ops.values())
    
    if verbose:
        print(f"\nCircuit stats (after optimization):")
        print(f"  T-count: {t_count}")
        print(f"  CX-count: {cx_count}")
        print(f"  H-count: {h_count}")
        print(f"  Total gates: {total_gates}")
    
    # Compute fidelity (may be slow for 9 qubits = 512x512 matrix)
    if verbose:
        print("\nComputing fidelity (this may take a moment for 9 qubits)...")
    
    try:
        U_circuit = Operator(qc).data
        
        # Phase-align
        phase_diff = np.angle(np.trace(U_target.conj().T @ U_circuit))
        U_circuit_aligned = U_circuit * np.exp(-1j * phase_diff)
        
        # Operator norm distance
        dist = np.linalg.norm(U_target - U_circuit_aligned, ord=2)
        
        # Process fidelity
        fid = np.abs(np.trace(U_target.conj().T @ U_circuit))**2 / (2**(2*n))
        
        if verbose:
            print(f"\n  Operator norm distance: {dist:.6e}")
            print(f"  Process fidelity: {fid:.6f}")
    except Exception as e:
        if verbose:
            print(f"\n  Could not compute fidelity: {e}")
        fid = None
        dist = None
    
    # Save circuit
    qasm = qiskit.qasm2.dumps(qc)
    with open(output_file, 'w') as f:
        f.write(qasm)
    
    if verbose:
        print(f"\n✅ Saved to {output_file}")
        print(f"\nFinal Results:")
        print(f"  T-count: {t_count}")
        if fid is not None:
            print(f"  Fidelity: {fid:.6f}")
        print(f"  Distance: {dist:.6e}" if dist else "  Distance: N/A")
    
    return qc, t_count, fid


def run_challenge12(json_file="unitary12.json", output_file="unitary12_optimized.qasm", method="direct"):
    """
    Run Challenge 12 optimization from command line.
    
    Args:
        json_file: Input JSON file with Pauli terms (default: unitary12.json)
        output_file: Output QASM file path
        method: Synthesis method - "direct" (recommended), "diag", or "best" (try both)
    
    Usage:
        python optimize_unitaries.py --challenge12
        python optimize_unitaries.py --challenge12 --method diag
        python optimize_unitaries.py --challenge12 --json custom.json --output custom.qasm
    """
    if not os.path.exists(json_file):
        print(f"Error: {json_file} not found!")
        print("Please provide the unitary12.json file.")
        return None
    
    data = load_challenge12(json_file)
    
    if method == "direct":
        return optimize_challenge12(data, output_file)
    elif method == "diag":
        return optimize_challenge12_diagonalization(data, output_file)
    elif method == "perterm":
        return optimize_challenge12_per_term_diag(data, output_file)
    else:
        # Try both methods and pick the best
        print("\n" + "="*70)
        print("CHALLENGE 12: COMPARING SYNTHESIS METHODS")
        print("="*70)
        
        # Method 1: Direct synthesis
        print("\n[Method 1: Direct Synthesis]")
        qc1, t1, fid1 = optimize_challenge12(data, output_file.replace(".qasm", "_direct.qasm"), verbose=True)
        
        # Method 2: Diagonalization + rmsynth
        print("\n[Method 2: Diagonalization + rmsynth]")
        qc2, t2, fid2 = optimize_challenge12_diagonalization(data, output_file.replace(".qasm", "_diag.qasm"), verbose=True)
        
        # Compare
        print("\n" + "="*70)
        print("COMPARISON")
        print("="*70)
        print(f"{'Method':<30} {'T-count':<15} {'Fidelity':<15}")
        print("-"*60)
        print(f"{'Direct Synthesis':<30} {t1 if t1 else 'N/A':<15} {f'{fid1:.6f}' if fid1 else 'N/A':<15}")
        print(f"{'Diagonalization + rmsynth':<30} {t2 if t2 else 'N/A':<15} {f'{fid2:.6f}' if fid2 else 'N/A':<15}")
        
        # Pick the best (lowest T-count with fidelity > 0.99)
        best_qc, best_t, best_fid = None, float('inf'), None
        if t1 is not None and fid1 is not None and fid1 > 0.99:
            best_qc, best_t, best_fid = qc1, t1, fid1
        if t2 is not None and fid2 is not None and fid2 > 0.99 and t2 < best_t:
            best_qc, best_t, best_fid = qc2, t2, fid2
        
        if best_qc is not None:
            print(f"\n✅ Best method: T-count = {best_t}, Fidelity = {best_fid:.6f}")
            qasm = qiskit.qasm2.dumps(best_qc)
            with open(output_file, 'w') as f:
                f.write(qasm)
            print(f"   Saved to {output_file}")
            return best_qc, best_t, best_fid
        else:
            print("\n❌ No valid method found!")
            return None, None, None


if __name__ == "__main__":
    main()
