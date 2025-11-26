import numpy as np
import hashlib
import logging

class KeyAuditor:
    """
    Batch correlation analysis: Hamming distances, bitwise bias, collisions.
    Logs warnings when suspicious patterns are detected.
    """

    def __init__(self, min_batch=20, max_batch=50):
        self.keys = []
        self.min_batch = min_batch
        self.max_batch = max_batch

    def add_key(self, key_bytes):
        self.keys.append(key_bytes)
        if len(self.keys) >= self.min_batch:
            self.audit()
        if len(self.keys) > self.max_batch:
            self.keys = self.keys[-self.max_batch:]

    def audit(self):
        n = len(self.keys)
        if n < self.min_batch:
            return
        bits = np.array([[int(b) for byte in k for b in f"{byte:08b}"] for k in self.keys], dtype=np.uint8)
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sum(bits[i] ^ bits[j])
                dists.append(d)
        dists = np.array(dists)
        mean_hd = float(dists.mean() / bits.shape[1])
        std_hd = float(dists.std() / bits.shape[1])

        corr = []
        for col in range(bits.shape[1]):
            p1 = float(bits[:, col].mean())
            corr.append(abs(p1 - 0.5))
        avg_bias = float(np.mean(corr))
        max_bias = float(np.max(corr))

        hashes = [hashlib.sha256(k).hexdigest() for k in self.keys]
        collisions = len(hashes) != len(set(hashes))

        logging.info(f"[KeyAuditor] Hamming mean={mean_hd:.4f}, std={std_hd:.4f}, "
                     f"avg_bit_bias={avg_bias:.4f}, max_bit_bias={max_bias:.4f}, collisions={collisions}")

        if collisions or mean_hd < 0.45 or avg_bias > 0.05 or max_bias > 0.15:
            logging.warning("[KeyAuditor] Potential correlation or bias detected in recent key batch.")
