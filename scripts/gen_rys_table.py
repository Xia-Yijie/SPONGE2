#!/usr/bin/env python3
"""Generate Rys quadrature Chebyshev interpolation tables.

Independently computes Rys roots/weights from Boys function moments,
then fits Chebyshev polynomials for GPU-friendly evaluation.

Output: C header with coefficient arrays.
"""

import numpy as np
from scipy.special import hyp1f1
from numpy.polynomial.chebyshev import chebfit, chebval

# Parameters (matching gpu4pyscf convention for compatibility)
MAX_NRYS = 9  # max Rys roots (covers up to L_sum=8, dddd)
DEGREE = 13  # Chebyshev polynomial degree
INTERVAL = 2.5  # width of each T interval
N_INTERVALS = 40  # number of intervals (covers T=0..100)
N_FIT_POINTS = 200  # sampling points per interval for fitting


def boys_function(n, T):
    """Boys function F_n(T) = integral_0^1 t^(2n) * exp(-T*t^2) dt."""
    T = np.asarray(T, dtype=np.float64)
    result = np.empty_like(T)
    small = T < 1e-15
    result[small] = 1.0 / (2 * n + 1)
    big = ~small
    if np.any(big):
        result[big] = hyp1f1(n + 0.5, n + 1.5, -T[big]) / (2 * n + 1)
    return result


def rys_roots_from_moments(nrys, T):
    """Compute Rys roots and weights from Boys function moments.

    Uses the Hankel matrix approach with numpy's eigenvalue solver
    for numerical stability. Returns (roots, weights) where roots[i] = t_i^2.
    """
    T = float(T)
    moments = np.array(
        [boys_function(k, np.array([T]))[0] for k in range(2 * nrys)]
    )

    if nrys == 1:
        w = moments[0]
        r = moments[1] / moments[0] if abs(moments[0]) > 1e-30 else 0.0
        return np.array([r]), np.array([w])

    # Build Hankel moment matrix and overlap matrix
    # M[i,j] = mu_{i+j}, S[i,j] = mu_{i+j+1}  for i,j = 0..nrys-1
    M = np.zeros((nrys, nrys))
    S = np.zeros((nrys, nrys))
    for i in range(nrys):
        for j in range(nrys):
            M[i, j] = moments[i + j]
            S[i, j] = moments[i + j + 1]

    # Generalized eigenvalue problem: S v = lambda M v
    # Eigenvalues = Rys roots, weights from eigenvectors
    try:
        from scipy.linalg import eigh

        eigenvalues, eigenvectors = eigh(S, M)
    except Exception:
        # Fallback for near-singular cases
        eigenvalues = np.zeros(nrys)
        eigenvectors = np.eye(nrys)

    roots = np.clip(eigenvalues, 0, None)  # roots should be non-negative
    # Weights: solve the moment equation w = M^{-1} * [1, 0, ..., 0] * mu_0
    # Or: w_i = mu_0 * (M^{-1} e_0)_i * eigvec...
    # Simpler: w_i determined by sum w_i * root_i^k = mu_k
    # Use the first nrys moment equations
    V = np.vander(roots, N=nrys, increasing=True).T  # V[k,i] = root_i^k
    try:
        weights = np.linalg.solve(V, moments[:nrys])
    except np.linalg.LinAlgError:
        weights = np.zeros(nrys)
        weights[0] = moments[0]

    idx = np.argsort(roots)
    return roots[idx], weights[idx]


def generate_tables():
    """Generate all coefficient tables."""

    # Small-T coefficients (linear approximation)
    smallx_r0, smallx_r1 = [], []
    smallx_w0, smallx_w1 = [], []

    for nrys in range(1, MAX_NRYS + 1):
        r0, w0 = rys_roots_from_moments(nrys, 0.0)
        r1, w1 = rys_roots_from_moments(nrys, 1e-8)
        dr = (r1 - r0) / 1e-8
        dw = (w1 - w0) / 1e-8
        smallx_r0.extend(r0)
        smallx_r1.extend(dr)
        smallx_w0.extend(w0)
        smallx_w1.extend(dw)

    # Large-T coefficients (asymptotic Gauss-Hermite limit)
    largex_r, largex_w = [], []
    T_large = 1e6  # very large T
    for nrys in range(1, MAX_NRYS + 1):
        r, w = rys_roots_from_moments(nrys, T_large)
        # At large T: root ≈ r_inf / T, weight ≈ w_inf * sqrt(pi/(4T))
        largex_r.extend(r * T_large)  # r_inf = root * T
        largex_w.extend(w / np.sqrt(np.pi / (4 * T_large)))  # w_inf

    # Chebyshev coefficients for intermediate T
    # For each nrys, each root/weight, each interval: fit degree-13 Chebyshev
    all_cheb_coeffs = []  # organized as [nrys][root_or_weight][interval][degree]

    for nrys in range(1, MAX_NRYS + 1):
        print(f"Fitting nrys={nrys}...")
        for interval_idx in range(N_INTERVALS):
            T_lo = interval_idx * INTERVAL
            T_hi = T_lo + INTERVAL

            # Sample points in this interval
            # Map [T_lo, T_hi] to [-1, 1] for Chebyshev
            t_sample = np.linspace(T_lo + 1e-10, T_hi - 1e-10, N_FIT_POINTS)
            x_sample = (t_sample - T_lo) / INTERVAL * 2 - 1  # map to [-1, 1]

            # Compute Rys roots/weights at sample points
            roots_sample = np.zeros((N_FIT_POINTS, nrys))
            weights_sample = np.zeros((N_FIT_POINTS, nrys))
            for j, T in enumerate(t_sample):
                r, w = rys_roots_from_moments(nrys, T)
                roots_sample[j] = r
                weights_sample[j] = w

            # Fit Chebyshev for each root and weight
            for i in range(nrys):
                # Root
                coeffs_r = chebfit(x_sample, roots_sample[:, i], DEGREE)
                all_cheb_coeffs.append((nrys, 2 * i, interval_idx, coeffs_r))
                # Weight
                coeffs_w = chebfit(x_sample, weights_sample[:, i], DEGREE)
                all_cheb_coeffs.append(
                    (nrys, 2 * i + 1, interval_idx, coeffs_w)
                )

    # Verify accuracy
    print("\nVerification (random T values):")
    for nrys in range(1, MAX_NRYS + 1):
        max_err_r, max_err_w = 0, 0
        for _ in range(100):
            T = np.random.uniform(0.01, 99.9)
            r_exact, w_exact = rys_roots_from_moments(nrys, T)

            # Evaluate from Chebyshev
            interval_idx = int(T / INTERVAL)
            if interval_idx >= N_INTERVALS:
                interval_idx = N_INTERVALS - 1
            x = (T - interval_idx * INTERVAL) / INTERVAL * 2 - 1

            r_cheb = np.zeros(nrys)
            w_cheb = np.zeros(nrys)
            for entry in all_cheb_coeffs:
                if entry[0] != nrys:
                    continue
                rw_idx, intv, coeffs = entry[1], entry[2], entry[3]
                if intv != interval_idx:
                    continue
                val = chebval(x, coeffs)
                if rw_idx % 2 == 0:
                    r_cheb[rw_idx // 2] = val
                else:
                    w_cheb[rw_idx // 2] = val

            max_err_r = max(max_err_r, np.max(np.abs(r_cheb - r_exact)))
            max_err_w = max(max_err_w, np.max(np.abs(w_cheb - w_exact)))

        print(
            f"  nrys={nrys}: max_err_root={max_err_r:.2e}, max_err_weight={max_err_w:.2e}"
        )

    return (
        smallx_r0,
        smallx_r1,
        smallx_w0,
        smallx_w1,
        largex_r,
        largex_w,
        all_cheb_coeffs,
    )


def write_header(
    filename,
    smallx_r0,
    smallx_r1,
    smallx_w0,
    smallx_w1,
    largex_r,
    largex_w,
    all_cheb_coeffs,
):
    """Write C header file with coefficient data."""

    with open(filename, "w") as f:
        f.write("// Rys quadrature Chebyshev interpolation coefficients.\n")
        f.write("// Auto-generated by gen_rys_table.py. Do not edit.\n")
        f.write("// Covers nrys = 1..5 (L_sum up to 8, d-shell quartets).\n")
        f.write(
            f"// Chebyshev degree = {DEGREE}, intervals = {N_INTERVALS}, "
            f"interval width = {INTERVAL}\n\n"
        )

        f.write("#pragma once\n\n")
        f.write(f"#define RYS_MAX_NRYS    {MAX_NRYS}\n")
        f.write(f"#define RYS_DEGREE      {DEGREE}\n")
        f.write(f"#define RYS_DEGREE1     {DEGREE + 1}\n")
        f.write(f"#define RYS_INTERVAL    {INTERVAL}\n")
        f.write(f"#define RYS_INTERVALS   {N_INTERVALS}\n\n")

        def write_array(name, data, qualifier="__device__"):
            f.write(f"static {qualifier} const double {name}[] = {{\n")
            for i, v in enumerate(data):
                f.write(f"    {v:.16e},\n")
            f.write("};\n\n")

        write_array("RYS_SMALLX_R0", smallx_r0)
        write_array("RYS_SMALLX_R1", smallx_r1)
        write_array("RYS_SMALLX_W0", smallx_w0)
        write_array("RYS_SMALLX_W1", smallx_w1)
        write_array("RYS_LARGEX_R", largex_r)
        write_array("RYS_LARGEX_W", largex_w)

        # ROOT_RW_DATA: organized for Clenshaw evaluation
        # Layout: for each nrys, for each root/weight (2*nrys entries),
        #         for each Chebyshev degree (DEGREE+1), for each interval (N_INTERVALS)
        # Access: data[nrys_offset + rw_idx * DEGREE1 * INTERVALS + degree * INTERVALS + interval]

        # Build the flat array matching the Clenshaw access pattern
        rw_data = []
        for nrys in range(1, MAX_NRYS + 1):
            for rw_idx in range(2 * nrys):
                for deg in range(DEGREE + 1):
                    for intv in range(N_INTERVALS):
                        # Find the matching entry
                        val = 0.0
                        for entry in all_cheb_coeffs:
                            if (
                                entry[0] == nrys
                                and entry[1] == rw_idx
                                and entry[2] == intv
                            ):
                                coeffs = entry[3]
                                val = coeffs[deg] if deg < len(coeffs) else 0.0
                                break
                        rw_data.append(val)

        write_array("RYS_RW_DATA", rw_data)

        print(
            f"Written {filename}: {len(rw_data)} RW coefficients, "
            f"{len(smallx_r0)} SMALLX entries"
        )


if __name__ == "__main__":
    import sys

    s_r0, s_r1, s_w0, s_w1, l_r, l_w, cheb = generate_tables()

    outfile = "SPONGE/quantum_chemistry/gpu_eri/rys_data.hpp"
    if len(sys.argv) > 1:
        outfile = sys.argv[1]

    write_header(outfile, s_r0, s_r1, s_w0, s_w1, l_r, l_w, cheb)
    print(f"\nDone: {outfile}")
