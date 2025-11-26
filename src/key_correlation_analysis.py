#!/usr/bin/env python3
"""
Key correlation analysis for fish TRNG.

Reads hex keys from generated_keys.txt and computes:
- Pairwise Hamming distance distribution
- Bitwise bias per bit position
- Pearson correlation between bit columns (subset for efficiency)
- Monte Carlo comparison versus random expectation
- Collision and near-duplicate detection
- Summary report with thresholds and recommendations

Author: You
"""

import os
import sys
import argparse
import logging
import hashlib
from statistics import mean, stdev
from typing import List, Tuple, Optional

import numpy as np

# ---------- Configuration thresholds ----------
DEFAULT_MIN_KEYS = 20   # Minimum keys to analyze
NEAR_DUPLICATE_HD_FRAC = 0.05  # Hamming distance fraction threshold for near-duplicates (e.g., < 5% of bits)
WARN_HD_MEAN_LOW = 0.45        # Mean Hamming distance fraction lower bound indicative of correlation
WARN_BIT_AVG_BIAS = 0.05       # Average absolute bit bias threshold
WARN_BIT_MAX_BIAS = 0.15       # Maximum absolute bit bias threshold
PEARSON_SUBSET_SIZE = 256      # Number of bit columns to sample for Pearson correlation heatmap
MC_PERMUTATIONS = 500          # Monte Carlo permutations to estimate random baseline
CSV_DEFAULT_DIR = "analysis_out"

# ---------- Helpers ----------

def hex_to_bits(hex_str: str) -> np.ndarray:
    """Convert a hex-encoded key to a numpy array of bits (uint8)."""
    raw = bytes.fromhex(hex_str.strip())
    return np.array([int(b) for byte in raw for b in f"{byte:08b}"], dtype=np.uint8)

def load_keys(keys_file: str) -> List[str]:
    """Load hex-encoded keys from a file (one per line)."""
    keys = []
    with open(keys_file, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # Validate hex format
            try:
                _ = bytes.fromhex(s)
                keys.append(s)
            except ValueError:
                logging.warning(f"Skipping non-hex line: {s[:64]}...")
    return keys

def pairwise_hamming(bits_matrix: np.ndarray) -> np.ndarray:
    """Compute pairwise Hamming distances for rows in bits_matrix."""
    n = bits_matrix.shape[0]
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sum(bits_matrix[i] ^ bits_matrix[j])
            dists.append(d)
    return np.array(dists, dtype=np.int32)

def bitwise_bias(bits_matrix: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """Compute average and max absolute bias across columns; return per-column biases."""
    p = bits_matrix.mean(axis=0)  # fraction of ones in each bit position
    bias = np.abs(p - 0.5)
    return float(np.mean(bias)), float(np.max(bias)), bias

def pearson_correlation_subset(bits_matrix: np.ndarray, subset_size: int = PEARSON_SUBSET_SIZE) -> np.ndarray:
    """
    Compute Pearson correlation for a subset of bit columns to avoid O(m^2) blowup.
    Returns correlation matrix (subset_size x subset_size).
    """
    m = bits_matrix.shape[1]
    if subset_size > m:
        subset_size = m
    idx = np.random.choice(m, size=subset_size, replace=False)
    sub = bits_matrix[:, idx].astype(np.float64)
    # Center columns
    sub = sub - sub.mean(axis=0, keepdims=True)
    # Normalize columns
    denom = sub.std(axis=0, ddof=1, keepdims=True)
    denom[denom == 0] = 1.0  # avoid zero division
    sub = sub / denom
    # Pearson correlation = (sub^T * sub) / (n-1)
    n = sub.shape[0]
    corr = (sub.T @ sub) / (n - 1 if n > 1 else 1)
    # Clamp numerical noise
    corr = np.clip(corr, -1.0, 1.0)
    return corr

def detect_collisions(keys_hex: List[str]) -> bool:
    """Detect collisions using SHA-256 hashes of raw key bytes."""
    hashes = [hashlib.sha256(bytes.fromhex(k)).hexdigest() for k in keys_hex]
    return len(hashes) != len(set(hashes))

def detect_near_duplicates(bits_matrix: np.ndarray, threshold_frac: float) -> List[Tuple[int, int, float]]:
    """
    Find pairs of keys whose Hamming distance fraction is below threshold_frac.
    Returns list of (i, j, frac).
    """
    n, m = bits_matrix.shape
    suspicious = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sum(bits_matrix[i] ^ bits_matrix[j])
            frac = d / m
            if frac < threshold_frac:
                suspicious.append((i, j, frac))
    return suspicious

def monte_carlo_baseline(num_keys: int, key_bits: int, permutations: int = MC_PERMUTATIONS) -> Tuple[float, float]:
    """
    Estimate expected mean Hamming distance fraction under randomness by sampling
    random keys of the same shape. Returns (mean, std) of mean pairwise HD fraction.
    """
    means = []
    for _ in range(permutations):
        rnd = np.random.randint(0, 2, size=(num_keys, key_bits), dtype=np.uint8)
        dists = pairwise_hamming(rnd)
        frac = dists / key_bits
        means.append(frac.mean())
    return float(mean(means)), float(stdev(means)) if len(means) > 1 else 0.0

def save_csv(path: str, header: List[str], rows: List[List[float]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")

# ---------- Main analysis ----------

def analyze_keys(
    keys_file: str,
    min_keys: int = DEFAULT_MIN_KEYS,
    csv_dir: Optional[str] = CSV_DEFAULT_DIR,
) -> None:
    keys_hex = load_keys(keys_file)
    n = len(keys_hex)
    if n < min_keys:
        print(f"[INFO] Not enough keys to analyze. Found {n}, need at least {min_keys}.")
        return

    # Convert to bits matrix (n x m)
    bits = np.stack([hex_to_bits(k) for k in keys_hex], axis=0)
    m = bits.shape[1]
    print(f"[INFO] Loaded {n} keys, each {m} bits.")

    # Pairwise Hamming distance analysis
    dists = pairwise_hamming(bits)
    frac = dists / m
    hd_mean = float(frac.mean())
    hd_std = float(frac.std())
    hd_min = float(frac.min())
    hd_max = float(frac.max())

    # Bitwise bias
    avg_bias, max_bias, bias_vec = bitwise_bias(bits)

    # Pearson correlation (subset)
    corr = pearson_correlation_subset(bits, subset_size=min(PEARSON_SUBSET_SIZE, m))
    corr_abs_mean = float(np.mean(np.abs(corr - np.eye(corr.shape[0]))))  # exclude diagonal
    corr_abs_max = float(np.max(np.abs(corr - np.eye(corr.shape[0]))))

    # Monte Carlo baseline
    mc_mean, mc_std = monte_carlo_baseline(n, m, permutations=MC_PERMUTATIONS)

    # Collisions and near-duplicates
    collisions = detect_collisions(keys_hex)
    near_dups = detect_near_duplicates(bits, NEAR_DUPLICATE_HD_FRAC)

    # Report
    print("\n=== Key Correlation Report ===")
    print(f"- Keys analyzed: {n}")
    print(f"- Bits per key: {m}")
    print(f"- Pairwise Hamming distance fraction: mean={hd_mean:.4f}, std={hd_std:.4f}, min={hd_min:.4f}, max={hd_max:.4f}")
    print(f"- Monte Carlo baseline (random): mean={mc_mean:.4f} ± {mc_std:.4f}")
    print(f"- Bitwise bias: avg={avg_bias:.4f}, max={max_bias:.4f}")
    print(f"- Pearson correlation (subset): mean |corr|={corr_abs_mean:.4f}, max |corr|={corr_abs_max:.4f}")
    print(f"- Collisions: {collisions}")
    if near_dups:
        print(f"- Near-duplicates (< {NEAR_DUPLICATE_HD_FRAC:.2f} HD frac): {len(near_dups)} pairs")
        # Print top 5 lowest distances
        near_dups_sorted = sorted(near_dups, key=lambda x: x[2])[:5]
        for i, j, f in near_dups_sorted:
            print(f"    pair ({i}, {j}) -> HD frac={f:.4f}")
    else:
        print(f"- Near-duplicates: none")

    # Warnings and recommendations
    print("\n=== Assessment ===")
    issues = []
    if collisions:
        issues.append("Collisions detected (identical keys).")
    if hd_mean < WARN_HD_MEAN_LOW:
        issues.append(f"Mean Hamming distance fraction {hd_mean:.4f} is below {WARN_HD_MEAN_LOW:.2f} (possible correlation).")
    if avg_bias > WARN_BIT_AVG_BIAS:
        issues.append(f"Average bit bias {avg_bias:.4f} exceeds {WARN_BIT_AVG_BIAS:.2f}.")
    if max_bias > WARN_BIT_MAX_BIAS:
        issues.append(f"Max bit bias {max_bias:.4f} exceeds {WARN_BIT_MAX_BIAS:.2f}.")
    if corr_abs_max > 0.25:  # empirical threshold for concern
        issues.append(f"Max |Pearson correlation| {corr_abs_max:.4f} suggests inter-bit dependency.")

    if not issues:
        print("No significant correlation or bias detected under current thresholds.")
    else:
        for s in issues:
            print(f"- WARNING: {s}")

    print("\n=== Recommendations ===")
    if collisions or near_dups:
        print("* Increase source diversity (more concurrent streams) and enforce minimum source count per pool.")
        print("* Strengthen per-sample hashing before pooling and ensure proper salt rotation in HKDF.")
    if hd_mean < WARN_HD_MEAN_LOW or avg_bias > WARN_BIT_AVG_BIAS or max_bias > WARN_BIT_MAX_BIAS:
        print("* Review entropy extraction weights; favor higher motion energy and timing jitter.")
        print("* Verify that entropy pool is cleared and sources reset between keys.")
    if corr_abs_max > 0.25:
        print("* Inspect bit columns with high correlation; ensure conditioning combines metadata (stream/time) in HKDF context.")
    print("* Continue collecting more keys and re-run analysis to confirm stability.")

    # Save CSV summaries
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
        save_csv(os.path.join(csv_dir, "pairwise_hamming.csv"), ["hamming_frac"], [[float(x)] for x in frac.tolist()])
        save_csv(os.path.join(csv_dir, "bit_bias.csv"), ["bit_index", "bias"], [[int(i), float(b)] for i, b in enumerate(bias_vec.tolist())])
        # For Pearson subset, save as matrix
        corr_rows = corr.tolist()
        header = [f"bit_{i}" for i in range(len(corr_rows[0]))]
        save_csv(os.path.join(csv_dir, "pearson_subset.csv"), header, corr_rows)
        print(f"\n[INFO] CSV summaries written to: {csv_dir}")

def parse_args():
    p = argparse.ArgumentParser(description="Analyze correlation between generated keys.")
    p.add_argument("--keys-file", type=str, default=os.path.join(os.path.dirname(__file__), "..", "generated_keys.txt"),
                   help="Path to file containing hex-encoded keys.")
    p.add_argument("--min-keys", type=int, default=DEFAULT_MIN_KEYS, help="Minimum number of keys required to analyze.")
    p.add_argument("--csv-dir", type=str, default=CSV_DEFAULT_DIR, help="Directory to write CSV summaries.")
    return p.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    analyze_keys(keys_file=args.keys_file, min_keys=args.min_keys, csv_dir=args.csv_dir)

if __name__ == "__main__":
    main()