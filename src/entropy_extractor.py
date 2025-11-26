import time
from Crypto.Hash import SHA256

class EntropyExtractor:
    """
    Extract entropy from tracked objects using positional quantization, timing jitter,
    source ID mixing, and per-sample hashing to whiten before pooling.
    """

    def __init__(self):
        self.entropy_pool = bytearray()
        self.last_ts_ns = None
        self.sources_seen = set()

    def extract_entropy(self, tracked_objects, frame_width, frame_height, timestamp_ns, source_id=None):
        if not tracked_objects:
            return

        delta_ts = 0 if self.last_ts_ns is None else (timestamp_ns - self.last_ts_ns) & 0xFFFF
        self.last_ts_ns = timestamp_ns

        src_tag = (hash(source_id) if source_id else 0) & 0xFF

        for obj in tracked_objects:
            obj_id = obj['id'] & 0xFF
            cx, cy = obj['centroid']
            area = int(obj['area']) & 0xFF

            ts_bits = timestamp_ns & 0xFFFF
            jitter_bits = delta_ts & 0xFFFF
            x_bits = int((cx / frame_width) * 1023) & 0x3FF
            y_bits = int((cy / frame_height) * 1023) & 0x3FF

            combined_data = (
                (ts_bits << 40) |
                (jitter_bits << 24) |
                (obj_id << 16) |
                (src_tag << 8) |
                (x_bits ^ y_bits)
            )

            record = combined_data.to_bytes(7, 'big') + area.to_bytes(1, 'big')
            h = SHA256.new(record).digest()  # 32 bytes
            self.entropy_pool.extend(h[:12])  # truncate to control throughput; tune 8–16 bytes

    def get_entropy_pool(self):
        return self.entropy_pool

    def clear_entropy_pool(self):
        self.entropy_pool = bytearray()
        self.last_ts_ns = None