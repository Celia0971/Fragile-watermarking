"""
iblt.py — Invertible Bloom Lookup Table (IBLT) for primary-key set reconciliation.

Implements the IBLT primitive described in paper §3.7 (Theorem 3.2).
Algorithm follows Eppstein et al. (2011) [reference 7 in the paper].

Structure  [§3.7]:
  An IBLT with ℓ cells and k hash functions is an array T[1..ℓ].
  Each cell stores a triple (count, keySum, hashSum):
    count   : signed integer — net number of keys mapped to this cell
    keySum  : XOR of all encoded primary keys mapped to this cell  [Eq. 46]
    hashSum : XOR of 64-bit fingerprints f_K(pk_i) for all keys mapped here  [Eq. 47]

Difference decoding  [§3.7, Eq. 9]:
  Given two IBLTs built with the same parameters:
    IBLT_K(X) − IBLT_K(Y) = IBLT_K(X △ Y)
  The cellwise subtraction encodes the symmetric difference X △ Y.
  The peeling decoder recovers X \ Y and Y \ X by iteratively extracting
  singleton cells (|count| = 1 with valid fingerprint).

Theorem 3.2 (IBLT Decoding):
  For ℓ = ⌈α_IBLT × d⌉ cells with d = |X △ Y| and load factor α_IBLT above the
  peeling threshold, decoding succeeds with high probability.
  Residual failure probability ≤ η_IBLT + d × k × 2^{−λ_fp}.

In the proposed scheme:
  X = PK(R),  Y = PK(R′),  d ≤ d_max = s_ins + s_del,  ℓ = ⌈α_IBLT × d_max⌉  [Eq. 44].
  Keys are encoded via HMAC: pk_to_iblt_key(K, pk_str) = HMAC_K("iblt_pk" || pk_str)  [§6.1].
"""

import hashlib
import struct
from collections import deque
from copy import deepcopy
from typing import Tuple, Set


# ─────────────────────────────────────────────────────────────────────────────
# Fast, stable 64-bit hash used internally by the IBLT
# (MD5 is deterministic, fast, and sufficient for prototype purposes)
# ─────────────────────────────────────────────────────────────────────────────
def _h64(data: bytes) -> int:
    """Return a 64-bit unsigned integer derived from data via MD5."""
    digest = hashlib.md5(data).digest()
    return struct.unpack_from("<Q", digest)[0]   # little-endian uint64


class IBLT:
    """
    Invertible Bloom Lookup Table.

    Parameters
    ----------
    m : int   — number of cells (choose ≥ 2 × expected_max_differences)
    k : int   — number of hash functions per key (default 3)
    """

    def __init__(self, m: int, k: int = 3):
        assert m > 0 and k > 0
        self.m = m
        self.k = k
        # Each cell: [count (signed), keySum (xor), hashSum (xor of fingerprints)]
        self._count   = [0] * m    # int
        self._keySum  = [0] * m    # int (XOR of raw key values)
        self._hashSum = [0] * m    # int (XOR of 64-bit key fingerprints)

    # ── internal helpers ──────────────────────────────────────────────────────
    def _cell_indices(self, key: int) -> list:
        """Return k cell indices for *key*."""
        key_bytes = key.to_bytes(8, "little", signed=False)
        indices = []
        for i in range(self.k):
            h = _h64(bytes([i]) + key_bytes)
            indices.append(int(h) % self.m)
        return indices

    def _fingerprint(self, key: int) -> int:
        """64-bit fingerprint used as per-cell consistency check."""
        key_bytes = key.to_bytes(8, "little", signed=False)
        return _h64(b"fp" + key_bytes)

    def _is_pure(self, idx: int) -> bool:
        """True iff cell idx contains exactly one (or minus-one) key."""
        c = self._count[idx]
        if c != 1 and c != -1:
            return False
        return self._hashSum[idx] == self._fingerprint(self._keySum[idx])

    # ── public API ────────────────────────────────────────────────────────────
    def insert(self, key: int) -> None:
        """Add *key* to the IBLT."""
        fp = self._fingerprint(key)
        for idx in self._cell_indices(key):
            self._count[idx]   += 1
            self._keySum[idx]  ^= key
            self._hashSum[idx] ^= fp

    def delete(self, key: int) -> None:
        """Remove *key* from the IBLT (reverses insert)."""
        fp = self._fingerprint(key)
        for idx in self._cell_indices(key):
            self._count[idx]   -= 1
            self._keySum[idx]  ^= key
            self._hashSum[idx] ^= fp

    def subtract(self, other: "IBLT") -> "IBLT":
        """
        Return a new IBLT representing (self − other), i.e.,
        cells are cell-wise differences of count / keySum / hashSum.
        """
        assert self.m == other.m and self.k == other.k
        diff = IBLT(self.m, self.k)
        for i in range(self.m):
            diff._count[i]   = self._count[i]   - other._count[i]
            diff._keySum[i]  = self._keySum[i]   ^ other._keySum[i]
            diff._hashSum[i] = self._hashSum[i]  ^ other._hashSum[i]
        return diff

    def decode(self) -> Tuple[Set[int], Set[int], bool]:
        """
        Decode this IBLT (typically a diff = original − suspect).

        Returns
        -------
        deleted  : keys with positive count  (in original, absent in suspect)
        inserted : keys with negative count  (in suspect, absent in original)
        success  : True if all cells were fully peeled (complete decode)
        """
        # Work on mutable copies so the original IBLT is unchanged
        count   = list(self._count)
        keySum  = list(self._keySum)
        hashSum = list(self._hashSum)

        deleted:  Set[int] = set()
        inserted: Set[int] = set()

        def fp(key):
            return self._fingerprint(key)

        def indices(key):
            return self._cell_indices(key)

        def is_pure(idx):
            c = count[idx]
            if c != 1 and c != -1:
                return False
            return hashSum[idx] == fp(keySum[idx])

        # Seed the queue with currently-pure cells
        queue = deque(i for i in range(self.m) if is_pure(i))

        while queue:
            idx = queue.popleft()
            if not is_pure(idx):        # might have been dirtied in the meantime
                continue
            c   = count[idx]
            key = keySum[idx]

            if c == 1:
                deleted.add(key)
            else:                       # c == -1
                inserted.add(key)

            # Peel key from every cell it touches
            f = fp(key)
            for cell_idx in indices(key):
                count[cell_idx]   -= c
                keySum[cell_idx]  ^= key
                hashSum[cell_idx] ^= f
                if is_pure(cell_idx):
                    queue.append(cell_idx)

        success = all(c == 0 for c in count)
        return deleted, inserted, success

    def copy(self) -> "IBLT":
        return deepcopy(self)

    def __repr__(self) -> str:
        nonempty = sum(1 for c in self._count if c != 0)
        return f"IBLT(m={self.m}, k={self.k}, non-zero-cells={nonempty})"


# ─────────────────────────────────────────────────────────────────────────────
def build_iblt(pk_set: Set[int], m: int, k: int = 3) -> IBLT:
    """Convenience: create and populate an IBLT from a set of primary keys."""
    iblt = IBLT(m, k)
    for pk in pk_set:
        iblt.insert(pk)
    return iblt


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Quick self-test
    original = set(range(1, 101))
    attacked = (original - {5, 17, 42}) | {200, 201}  # del 3, ins 2

    m_iblt = 400
    A = build_iblt(original, m_iblt)
    B = build_iblt(attacked, m_iblt)
    diff = A.subtract(B)
    deleted, inserted, ok = diff.decode()

    print(f"Decode success : {ok}")
    print(f"Deleted  (expected {{5,17,42}})  : {sorted(deleted)}")
    print(f"Inserted (expected {{200,201}})  : {sorted(inserted)}")
    assert ok
    assert deleted  == {5, 17, 42}
    assert inserted == {200, 201}
    print("IBLT self-test PASSED.")
