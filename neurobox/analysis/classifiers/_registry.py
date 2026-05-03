"""
neurobox.analysis.classifiers._registry
========================================
Internal lookup table mapping backend identifier strings to
:class:`Classifier` subclasses.

A backend is "registered" simply by importing its defining module —
the module-level code uses :func:`register` to install itself.  This
lets the package skip importing torch / sklearn on bare ``import
neurobox`` and only pull in the heavy dependency when the user
actually constructs that backend.
"""

from __future__ import annotations

from typing import Callable

from .base import Classifier


# ──────────────────────────────────────────────────────────────────── #
# Registration                                                          #
# ──────────────────────────────────────────────────────────────────── #

_REGISTRY: dict[str, type[Classifier]] = {}


def register(name: str, cls: type[Classifier]) -> None:
    """Register *cls* under the backend identifier *name*."""
    _REGISTRY[name] = cls


# ──────────────────────────────────────────────────────────────────── #
# Lookup                                                                #
# ──────────────────────────────────────────────────────────────────── #

# Backend → import-path-of-the-module-that-defines-it.  Used by
# get_classifier_class to lazy-load.
_LAZY_BACKENDS: dict[str, tuple[str, Callable[[], None]]] = {}


def register_lazy(
    name:        str,
    module_path: str,
    importer:    Callable[[], None],
) -> None:
    """Register a backend whose definition lives in a separate module.

    The *importer* is called the first time a user constructs the
    backend by name, giving the module a chance to import its heavy
    dependencies (torch, sklearn) and register the concrete class.
    """
    _LAZY_BACKENDS[name] = (module_path, importer)


def get_classifier_class(name: str) -> type[Classifier]:
    """Look up a backend by name, lazy-importing its defining module if needed."""
    if name in _REGISTRY:
        return _REGISTRY[name]
    if name in _LAZY_BACKENDS:
        module_path, importer = _LAZY_BACKENDS[name]
        importer()
        if name in _REGISTRY:
            return _REGISTRY[name]
        raise RuntimeError(
            f"Backend {name!r} module {module_path!r} did not register itself"
        )
    raise KeyError(
        f"Unknown classifier backend: {name!r}.  "
        f"Available: {sorted(set(_REGISTRY) | set(_LAZY_BACKENDS))}"
    )


def list_backends() -> list[str]:
    """Return the names of all registered backends (eager + lazy)."""
    return sorted(set(_REGISTRY) | set(_LAZY_BACKENDS))
