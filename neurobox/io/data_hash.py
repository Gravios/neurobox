"""
neurobox.io.data_hash
======================
Content-addressed hash for arbitrary Python objects.

Port of :file:`MTA/utilities/DataHash.m` (Jan Simon's MIT-licensed
MATLAB MD5 hasher used throughout the MTA pipeline as a cache-key
generator).

Usage
-----
::

    from neurobox.io.data_hash import data_hash

    params = {"sample_rate": 250.0, "channels": [4, 5, 6]}
    tag = data_hash(params)         # → 'a3c9...'
    cache_path = f"results-{tag}.pkl"

The MATLAB version returned a 32-character hex MD5; this port uses
SHA-1 (40 chars) by default — collisions on cache keys are not a
security issue and SHA-1 is built in to the Python stdlib.  Pass
``algorithm='md5'`` to match the MATLAB output character count.

Supported types
---------------
* numpy arrays — hashed via raw byte buffer
* dicts — keys are sorted alphabetically before hashing (so insert
  order doesn't affect the result)
* lists / tuples — order *is* significant
* sets / frozensets — sorted before hashing (order *not* significant)
* str, bytes, bytearray
* int, float, bool
* None

Anything else falls through to :func:`pickle.dumps` (with
``protocol=4`` for cross-version stability).  Two different objects
that pickle to the same bytes will hash equal — be cautious when
hashing classes with custom ``__reduce__``.
"""

from __future__ import annotations

import hashlib
import pickle
from typing import Any

import numpy as np


__all__ = ["data_hash"]


def _update_with(h, data: Any) -> None:
    """Recursive content-feed into a hashlib object."""
    if data is None:
        h.update(b"\x00None")
        return
    if isinstance(data, bool):
        h.update(b"\x01b")
        h.update(b"\x01" if data else b"\x00")
        return
    if isinstance(data, int):
        h.update(b"\x02i")
        # Use a fixed-width representation to avoid platform variability
        h.update(str(data).encode())
        return
    if isinstance(data, float):
        h.update(b"\x03f")
        h.update(np.float64(data).tobytes())
        return
    if isinstance(data, (bytes, bytearray)):
        h.update(b"\x04B")
        h.update(bytes(data))
        return
    if isinstance(data, str):
        h.update(b"\x05s")
        h.update(data.encode("utf-8"))
        return
    if isinstance(data, np.ndarray):
        h.update(b"\x06a")
        # Include shape + dtype so different layouts of equal bytes hash differently
        h.update(str(data.shape).encode())
        h.update(str(data.dtype).encode())
        # Use ascontiguousarray so non-contiguous slices still produce
        # a deterministic byte stream
        h.update(np.ascontiguousarray(data).tobytes())
        return
    if isinstance(data, dict):
        h.update(b"\x07d")
        for k in sorted(data.keys(), key=lambda x: repr(x)):
            _update_with(h, k)
            _update_with(h, data[k])
        return
    if isinstance(data, (list, tuple)):
        h.update(b"\x08l" if isinstance(data, list) else b"\x09t")
        for item in data:
            _update_with(h, item)
        return
    if isinstance(data, (set, frozenset)):
        h.update(b"\x0aS")
        for item in sorted(data, key=lambda x: repr(x)):
            _update_with(h, item)
        return
    # Fallback: pickle
    h.update(b"\x0bP")
    h.update(pickle.dumps(data, protocol=4))


def data_hash(data: Any, algorithm: str = "sha1") -> str:
    """Compute a deterministic hex hash of *data*.

    Port of :file:`MTA/utilities/DataHash.m`.

    Parameters
    ----------
    data:
        Any Python object.  See module docstring for supported types.
    algorithm:
        ``'sha1'`` (default, 40-char output), ``'md5'`` (32-char,
        matches MATLAB), ``'sha256'`` (64-char), or any name accepted
        by :func:`hashlib.new`.

    Returns
    -------
    str
        Hex digest.
    """
    h = hashlib.new(algorithm)
    _update_with(h, data)
    return h.hexdigest()
