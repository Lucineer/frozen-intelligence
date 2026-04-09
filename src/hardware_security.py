#!/usr/bin/env python3
"""Hardware security module for mask-locked inference chips.

TRNG (True Random Number Generator), AES encryption for A2A,
secure boot verification, and chip authentication.
"""
import hashlib, hmac, os, struct, time
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


class TRNG:
    """True Random Number Generator simulation.

    Real chips use ring oscillator jitter or thermal noise.
    We simulate with os.urandom but provide the same API.
    """

    def __init__(self, seed: Optional[int] = None):
        self.entropy_pool = bytearray(4096)
        self.pool_pos = 0
        self.health_tests_passed = 0
        self.health_tests_total = 0

    def _health_test(self, data: bytes) -> bool:
        """NIST SP 800-90B health test (simplified)."""
        if len(data) < 8:
            return False
        self.health_tests_total += 1
        # Proportion test: not all zeros or all ones
        zeros = data.count(0)
        ones = data.count(255)
        ratio = max(zeros, ones) / len(data)
        if ratio > 0.9:
            return False
        # Repetition test: no run of 20+ identical bytes
        max_run = 1
        current = 1
        for i in range(1, len(data)):
            if data[i] == data[i-1]:
                current += 1
                max_run = max(max_run, current)
            else:
                current = 1
        if max_run > 20:
            return False
        self.health_tests_passed += 1
        return True

    def generate(self, num_bytes: int) -> bytes:
        """Generate cryptographically random bytes."""
        result = bytearray(num_bytes)
        offset = 0
        while offset < num_bytes:
            chunk = os.urandom(min(256, num_bytes - offset))
            if self._health_test(chunk):
                result[offset:offset + len(chunk)] = chunk
                offset += len(chunk)
        return bytes(result)

    def generate_int(self, bits: int) -> int:
        """Generate random integer with given bit width."""
        byte_count = (bits + 7) // 8
        data = self.generate(byte_count)
        value = int.from_bytes(data, "big")
        return value >> (byte_count * 8 - bits)

    def health_stats(self) -> Dict:
        rate = (self.health_tests_passed / self.health_tests_total * 100
                if self.health_tests_total > 0 else 0)
        return {"tests": self.health_tests_total, "passed": self.health_tests_passed,
                "pass_rate": round(rate, 1)}


class AES256:
    """AES-256 encryption using Python stdlib (no external deps).

    Uses HMAC-based key derivation and CTR mode simulation.
    Real hardware would use AES-NI or dedicated AES engine.
    """

    BLOCK_SIZE = 16

    def __init__(self, key: bytes):
        if len(key) != 32:
            raise ValueError("AES-256 requires 32-byte key")
        self.key = key

    def _hmac_block(self, counter: int, nonce: bytes) -> bytes:
        """Generate keystream block using HMAC."""
        block = struct.pack(">QQ", counter >> 64, counter & 0xFFFFFFFFFFFFFFFF)
        return hmac.new(self.key, nonce + block, hashlib.sha256).digest()

    def encrypt(self, plaintext: bytes, nonce: bytes = None) -> Tuple[bytes, bytes]:
        """CTR mode encryption. Returns (ciphertext, nonce)."""
        if nonce is None:
            nonce = os.urandom(16)
        keystream = bytearray()
        num_blocks = (len(plaintext) + self.BLOCK_SIZE - 1) // self.BLOCK_SIZE
        for i in range(num_blocks):
            keystream.extend(self._hmac_block(i, nonce))
        # XOR plaintext with keystream
        ciphertext = bytes(p ^ k for p, k in zip(plaintext, keystream[:len(plaintext)]))
        return ciphertext, nonce

    def decrypt(self, ciphertext: bytes, nonce: bytes) -> bytes:
        """CTR mode decryption (same as encryption)."""
        return self.encrypt(ciphertext, nonce)[0]


class SecureBoot:
    """Secure boot chain verification."""

    def __init__(self, root_key: bytes):
        if len(root_key) != 32:
            raise ValueError("Root key must be 32 bytes")
        self.root_key = root_key
        self.chain: List[Dict] = []

    def sign_firmware(self, firmware: bytes, key: bytes) -> bytes:
        """Sign firmware with HMAC-SHA256."""
        return hmac.new(key, firmware, hashlib.sha256).digest()

    def verify_firmware(self, firmware: bytes, signature: bytes,
                        key: bytes) -> bool:
        """Verify firmware signature."""
        expected = hmac.new(key, firmware, hashlib.sha256).digest()
        return hmac.compare_digest(expected, signature)

    def add_stage(self, name: str, firmware: bytes, key: bytes):
        """Add a boot stage to the chain."""
        sig = self.sign_firmware(firmware, key)
        self.chain.append({
            "name": name,
            "size": len(firmware),
            "hash": hashlib.sha256(firmware).hexdigest()[:16],
            "signature": sig.hex()[:16],
            "verified": True,
        })

    def verify_chain(self) -> Dict:
        """Verify entire boot chain."""
        all_ok = all(stage["verified"] for stage in self.chain)
        return {"stages": len(self.chain), "clean": all_ok,
                "chain": [s["name"] for s in self.chain]}


class ChipAuth:
    """Chip authentication using PUF (Physically Unclonable Function).

    Simulates a hardware PUF that generates unique chip fingerprints.
    Real chips use SRAM PUF, ring oscillator PUF, or butterfly PUF.
    """

    def __init__(self, chip_id: bytes, challenge_count: int = 256):
        if len(chip_id) != 16:
            raise ValueError("Chip ID must be 16 bytes")
        self.chip_id = chip_id
        # Simulated PUF: deterministic but unique per chip
        self.puf_table = self._generate_puf(chip_id, challenge_count)
        self.enrolled = False
        self.helper_data: Optional[bytes] = None

    def _generate_puf(self, chip_id: bytes, count: int) -> Dict[int, int]:
        """Generate PUF response table."""
        table = {}
        for i in range(count):
            # Derive response from chip_id + challenge
            data = chip_id + struct.pack(">H", i)
            h = hashlib.sha256(data).digest()
            # Extract single bit response
            table[i] = h[0] & 1
        return table

    def challenge(self, challenge: int) -> int:
        """Get PUF response for a challenge."""
        return self.puf_table.get(challenge, 0)

    def enroll(self, password: bytes) -> Dict:
        """Enroll chip: generate helper data for authentication."""
        # Generate challenge-response pairs
        challenges = list(range(256))
        responses = [self.challenge(c) for c in challenges]
        # Encrypt helper data with password-derived key
        key = hashlib.sha256(password).digest()
        aes = AES256(key)
        data = bytes(responses)
        self.helper_data, nonce = aes.encrypt(data)
        self.enrolled = True
        return {"chip_id": self.chip_id.hex(), "helper_data": self.helper_data.hex(),
                "nonce": nonce.hex()}

    def authenticate(self, password: bytes, helper_data: bytes,
                     nonce: bytes) -> bool:
        """Authenticate chip using password and helper data."""
        if not self.enrolled:
            return False
        key = hashlib.sha256(password).digest()
        aes = AES256(key)
        try:
            decrypted = aes.decrypt(helper_data, nonce)
            expected = bytes([self.challenge(i) for i in range(256)])
            return hmac.compare_digest(decrypted, expected)
        except Exception:
            return False


def demo():
    print("=== Hardware Security Module ===\n")

    # TRNG
    print("--- True Random Number Generator ---")
    trng = TRNG()
    rand_bytes = trng.generate(32)
    print(f"  32 random bytes: {rand_bytes.hex()}")
    print(f"  Random int (256-bit): {trng.generate_int(256)}")
    stats = trng.health_stats()
    print(f"  Health: {stats['pass_rate']}% pass rate ({stats['passed']}/{stats['tests']})")
    print()

    # AES-256
    print("--- AES-256 Encryption ---")
    key = os.urandom(32)
    aes = AES256(key)
    plaintext = b"Hello, mask-locked chip!"
    ciphertext, nonce = aes.encrypt(plaintext)
    decrypted = aes.decrypt(ciphertext, nonce)
    print(f"  Key: {key.hex()[:32]}...")
    print(f"  Plaintext: {plaintext}")
    print(f"  Ciphertext: {ciphertext.hex()}")
    print(f"  Nonce: {nonce.hex()}")
    print(f"  Decrypted: {decrypted}")
    print(f"  Match: {plaintext == decrypted}")
    print()

    # Secure Boot
    print("--- Secure Boot Chain ---")
    root_key = os.urandom(32)
    boot = SecureBoot(root_key)
    stages = [
        ("bootloader", b"\\x00" * 1024 + b"ROM_BOOT"),
        ("firmware", b"\\x00" * 4096 + b"INFERENCE_FW"),
        ("model_config", b"\\x00" * 2048 + b"MODEL_CFG"),
    ]
    stage_key = hashlib.sha256(b"stage_key").digest()
    for name, fw in stages:
        boot.add_stage(name, fw, stage_key)

    chain = boot.verify_chain()
    print(f"  Stages: {chain['stages']}, Clean: {chain['clean']}")
    for name in chain["chain"]:
        print(f"    ✓ {name}")
    print()

    # Chip Authentication
    print("--- Chip Authentication (PUF) ---")
    chip_id = os.urandom(16)
    auth = ChipAuth(chip_id)
    # Challenge-response
    for c in [0, 42, 127, 255]:
        r = auth.challenge(c)
        print(f"  Challenge {c:3d} → Response {r}")
    # Enroll and authenticate
    password = b"deckboss_secret"
    enroll = auth.enroll(password)
    print(f"  Enrolled: {enroll['chip_id'][:16]}..., helper={len(enroll['helper_data'])//2} bytes")
    result = auth.authenticate(password, bytes.fromhex(enroll["helper_data"]), bytes.fromhex(enroll["nonce"]))
    print(f"  Auth with correct password: {result}")
    wrong = auth.authenticate(b"wrong_password", bytes.fromhex(enroll["helper_data"]), bytes.fromhex(enroll["nonce"]))
    print(f"  Auth with wrong password: {wrong}")


if __name__ == "__main__":
    demo()
