"""
neurobox.io.cached_compute
===========================
Disk-cache decorator using :func:`data_hash` for tag generation.

Mirrors the disk-cache pattern that the MTA ``compute_*`` family
relies on through ``MTAApfs`` / ``MTAApfs.exist`` — hash the
relevant inputs, derive a tag, look up
``<cache_dir>/<prefix>-<tag>.<ext>`` and either load it or compute
and save.

The user explicitly highlighted that ``data_hash`` is the keystone
of this pattern; this module is what makes it useful.

Quick reference
---------------
::

    from neurobox.io.cached_compute import cached_compute

    @cached_compute(
        cache_dir = lambda session, *a, **kw: session.paths.figures_dir,
        prefix    = "my_ratemap",
        hash_args = ["units", "states", "bin_dims"],
    )
    def my_ratemap(session, *, units=None, states="theta", bin_dims=(20, 20)):
        ...
        return result

Then::

    out = my_ratemap(sess, units=[1, 2, 3])              # computes + caches
    out = my_ratemap(sess, units=[1, 2, 3])              # loads from cache
    out = my_ratemap(sess, units=[1, 2, 3], overwrite=True)   # forces recompute

Key design choices
------------------
* **Hash inputs are explicit, not auto-detected.**  Hashing every
  bound argument silently leads to spurious cache misses (different
  ``Generator`` seeds, different ``NBSession`` repr objects).  The
  decorator requires *hash_args* — either a list of parameter names
  or a callable returning a hashable summary.
* **cache_dir is a callable, not a fixed path.**  Most useful caches
  are session-local; the decorator passes the call's positional
  arguments to *cache_dir* so it can derive a path from the first
  argument (typically a session).
* **overwrite + purge are auto-injected kwargs.**  Decorated
  functions get two extra keyword-only arguments: ``overwrite`` and
  ``cache``.  ``overwrite=True`` ignores the cache and recomputes;
  ``cache=False`` skips both load and save.
* **Pickle for serialisation.**  Most function results are not
  primitive types, so pickle is the only correct default.  Pass
  ``loader=...`` / ``dumper=...`` to override (e.g. for
  scipy.io.savemat compatibility).
"""

from __future__ import annotations

import functools
import inspect
import pickle
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .data_hash import data_hash


__all__ = ["cached_compute", "cache_path_for"]


# ─────────────────────────────────────────────────────────────────────── #
# Public helpers                                                            #
# ─────────────────────────────────────────────────────────────────────── #

def cache_path_for(
    cache_dir:   Path | str,
    prefix:      str,
    tag:         str,
    extension:   str = "pkl",
) -> Path:
    """Compose a cache file path from its components.

    Layout::

        <cache_dir>/<prefix>-<tag>.<extension>

    The tag is the :func:`data_hash` digest of the function's
    relevant arguments; *prefix* is the function's name (or any
    user-supplied identifier) and helps humans recognise what's in
    the directory.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{prefix}-{tag}.{extension}"


# ─────────────────────────────────────────────────────────────────────── #
# Decorator                                                                 #
# ─────────────────────────────────────────────────────────────────────── #

def cached_compute(
    *,
    cache_dir:    Callable[..., Path | str] | Path | str,
    prefix:       str | None = None,
    hash_args:    Sequence[str] | Callable[..., Any] = (),
    extension:    str = "pkl",
    loader:       Callable[[Path], Any] | None = None,
    dumper:       Callable[[Any, Path], None] | None = None,
    hash_algorithm: str = "sha1",
) -> Callable[[Callable], Callable]:
    """Disk-cache decorator built on :func:`data_hash`.

    Parameters
    ----------
    cache_dir:
        Directory in which to store cached results.  Either a fixed
        :class:`Path`/str or a callable ``f(*positional_args) -> Path``
        evaluated at every call.  Callable form lets you derive the
        directory from the call's positional arguments (e.g. the
        first positional arg is typically a session, and
        ``cache_dir=lambda s: s.paths.figures_dir`` is a common
        pattern).  Keyword arguments are **not** passed to this
        callable — they are reserved for the function's own
        configuration.
    prefix:
        File-name prefix.  Default = decorated function's ``__name__``.
    hash_args:
        Either:

        * A sequence of parameter names — those bound arguments are
          fed into :func:`data_hash` to derive the cache tag.  Argument
          binding follows the function's signature (positional or
          keyword arguments both supported).
        * A callable ``f(*args, **kwargs) -> hashable`` that returns
          the value to hash.  Use this when the relevant inputs are
          derived (e.g. ``session.session_name`` rather than the
          ``session`` object itself).
    extension:
        Cache-file extension (without the leading dot).  Default
        ``'pkl'``.
    loader, dumper:
        Custom serialisers.  Default uses :mod:`pickle` with
        ``protocol=4``.  Use, for example,
        ``loader=lambda p: scipy.io.loadmat(p, squeeze_me=True)`` for
        ``.mat`` outputs.
    hash_algorithm:
        Passed through to :func:`data_hash`.  Default ``'sha1'``.

    Returns
    -------
    The decorated function gains two extra keyword-only arguments::

        * overwrite : bool, default False
            Ignore the cache and force recomputation; the new result
            is written to disk on success.
        * cache : bool, default True
            If False, skip both cache lookup and save (useful for
            tests).
    """
    if loader is None:
        loader = _pickle_loader
    if dumper is None:
        dumper = _pickle_dumper

    def decorator(fn: Callable) -> Callable:
        nonlocal prefix
        prefix_used = prefix if prefix is not None else fn.__name__
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            overwrite = kwargs.pop("overwrite", False)
            use_cache = kwargs.pop("cache",     True)

            # Resolve cache directory.  Callable form receives the
            # call's positional arguments only — kwargs are reserved
            # for the function's own configuration and are typically
            # not relevant for path derivation (the typical case is
            # ``cache_dir = lambda session: session.paths.figures_dir``).
            if callable(cache_dir):
                cdir = Path(cache_dir(*args))
            else:
                cdir = Path(cache_dir)

            # Derive the hash payload
            if callable(hash_args):
                hash_payload = hash_args(*args, **kwargs)
            else:
                bound = sig.bind_partial(*args, **kwargs)
                bound.apply_defaults()
                hash_payload = {
                    name: bound.arguments.get(name)
                    for name in hash_args
                    if name in bound.arguments
                }

            tag       = data_hash(hash_payload, algorithm=hash_algorithm)
            cache_pth = cache_path_for(cdir, prefix_used, tag, extension)

            # Load if available
            if use_cache and not overwrite and cache_pth.exists():
                return loader(cache_pth)

            # Compute
            result = fn(*args, **kwargs)

            # Save
            if use_cache:
                tmp = cache_pth.with_suffix(cache_pth.suffix + ".tmp")
                try:
                    dumper(result, tmp)
                    tmp.replace(cache_pth)
                except Exception:
                    if tmp.exists():
                        try:
                            tmp.unlink()
                        except OSError:
                            pass
                    raise
            return result

        # Expose the wrapped resolver for test inspection
        wrapper._cached_compute_resolve_path = lambda *a, **kw: _resolve_path(  # type: ignore[attr-defined]
            sig, fn.__name__, prefix_used, cache_dir, hash_args,
            extension, hash_algorithm, *a, **kw,
        )
        return wrapper

    return decorator


# ─────────────────────────────────────────────────────────────────────── #
# Internals                                                                 #
# ─────────────────────────────────────────────────────────────────────── #

def _resolve_path(
    sig:            inspect.Signature,
    fn_name:        str,
    prefix_used:    str,
    cache_dir:      Callable | Path | str,
    hash_args:      Sequence[str] | Callable,
    extension:      str,
    hash_algorithm: str,
    *args, **kwargs,
) -> Path:
    """Compute the cache path *without* running the function."""
    if callable(cache_dir):
        cdir = Path(cache_dir(*args))
    else:
        cdir = Path(cache_dir)
    if callable(hash_args):
        hash_payload = hash_args(*args, **kwargs)
    else:
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        hash_payload = {
            name: bound.arguments.get(name)
            for name in hash_args
            if name in bound.arguments
        }
    tag = data_hash(hash_payload, algorithm=hash_algorithm)
    return cache_path_for(cdir, prefix_used, tag, extension)


def _pickle_loader(p: Path) -> Any:
    with open(p, "rb") as f:
        return pickle.load(f)


def _pickle_dumper(obj: Any, p: Path) -> None:
    with open(p, "wb") as f:
        pickle.dump(obj, f, protocol=4)
