"""Commitments for Proof-of-Denoising: a SHA-256 Merkle tree over the job header and
every latent state of the trajectory.

Leaf 0 binds the job (model hash, prompt, seed, sampler spec, safety config); leaf j+1
binds state j. The worker publishes the root before any challenge is drawn.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np


def _h(*parts: bytes) -> bytes:
    m = hashlib.sha256()
    for p in parts:
        m.update(p)
    return m.digest()


def header_leaf(header: dict) -> bytes:
    return _h(b"pod/header", json.dumps(header, sort_keys=True, separators=(",", ":")).encode())


def state_leaf(j: int, state: np.ndarray) -> bytes:
    arr = np.ascontiguousarray(state, dtype="<f4")
    return _h(b"pod/state", j.to_bytes(4, "little"), str(arr.shape).encode(), arr.tobytes())


def _parent(a: bytes, b: bytes) -> bytes:
    return _h(b"pod/node", a, b)


@dataclass
class MerkleTree:
    levels: list[list[bytes]]

    @classmethod
    def build(cls, leaves: list[bytes]) -> "MerkleTree":
        if not leaves:
            raise ValueError("a Merkle tree requires at least one leaf")
        levels = [list(leaves)]
        while len(levels[-1]) > 1:
            cur = levels[-1]
            if len(cur) % 2:
                cur = cur + [cur[-1]]
            levels.append([_parent(cur[k], cur[k + 1]) for k in range(0, len(cur), 2)])
        return cls(levels)

    @property
    def root(self) -> bytes:
        return self.levels[-1][0]

    def path(self, idx: int) -> list[tuple[bytes, bool]]:
        """Sibling hashes from leaf to root; the flag is True when the sibling is on the right."""
        if not isinstance(idx, int) or isinstance(idx, bool) or not 0 <= idx < len(self.levels[0]):
            raise ValueError("leaf index is outside the committed tree")
        out = []
        for level in self.levels[:-1]:
            lvl = level + [level[-1]] if len(level) % 2 else level
            sib = idx ^ 1
            out.append((lvl[sib], sib > idx))
            idx //= 2
        return out


def verify_path(leaf: bytes, path: list[tuple[bytes, bool]], root: bytes) -> bool:
    node = leaf
    for sib, right in path:
        node = _parent(node, sib) if right else _parent(sib, node)
    return node == root


def verify_indexed_path(leaf: bytes, path: list[tuple[bytes, bool]], root: bytes,
                        index: int, leaf_count: int) -> bool:
    """Verify an opening at its expected position in a fixed-size tree.

    The protocol pins both the leaf index and total number of leaves. An arbitrary
    inclusion proof alone does not enforce either. Duplicate-last padding must also
    match the current node at odd-width levels.
    """
    if not (isinstance(leaf, bytes) and len(leaf) == 32
            and isinstance(root, bytes) and len(root) == 32
            and isinstance(index, int) and not isinstance(index, bool)
            and isinstance(leaf_count, int) and not isinstance(leaf_count, bool)
            and 0 <= index < leaf_count and isinstance(path, (list, tuple))):
        return False
    node, width, pos = leaf, leaf_count, index
    for entry in path:
        if width <= 1 or not isinstance(entry, (list, tuple)) or len(entry) != 2:
            return False
        sibling, right = entry
        if not (isinstance(sibling, bytes) and len(sibling) == 32
                and type(right) is bool and right == (pos % 2 == 0)):
            return False
        if pos == width - 1 and width % 2 and sibling != node:
            return False
        node = _parent(node, sibling) if right else _parent(sibling, node)
        pos //= 2
        width = (width + 1) // 2
    return width == 1 and node == root


def commit(header: dict, states: np.ndarray) -> MerkleTree:
    leaves = [header_leaf(header)] + [state_leaf(j, s) for j, s in enumerate(states)]
    return MerkleTree.build(leaves)
