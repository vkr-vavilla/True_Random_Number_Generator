import time
from Crypto.Hash import SHA256
from Crypto.Protocol.KDF import HKDF
from Crypto.Random import get_random_bytes

class Conditioner:
    """
    Conditions the entropy with SHA-256, rotates salt with system randomness and digest,
    and derives the final key via HKDF bound to context meta.
    """

    def __init__(self, key_size=32, salt=None):
        self.key_size = key_size
        self.salt = salt or get_random_bytes(16)
        self.context_epoch = int(time.time()) & 0xFFFFFFFF

    def condition_data(self, entropy_pool, context_meta=b''):
        hashed_entropy = SHA256.new(entropy_pool).digest()
        self.salt = SHA256.new(self.salt + get_random_bytes(16) + hashed_entropy[:8]).digest()
        ctx = b'fish-trng-key-v1|' + context_meta + self.context_epoch.to_bytes(4, 'big')
        key = HKDF(master=hashed_entropy, key_len=self.key_size, salt=self.salt, hashmod=SHA256, num_keys=1, context=ctx)
        return key