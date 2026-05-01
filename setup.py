"""
Build script for the Cython extension(s) in neurobox.

Most package metadata lives in :file:`pyproject.toml` — this file
exists only to declare compiled extensions, which setuptools doesn't
yet support natively in pyproject.toml.

The extension build is **best-effort**: if Cython or numpy aren't
available at build time, the package still installs as pure-Python and
falls back to the slower numpy implementation in
:mod:`neurobox.analysis.spikes._ccg_python_fallback`.
"""

from __future__ import annotations

import os
from setuptools import setup

# Try to build the compiled extension(s).  Failure is non-fatal: the
# package still works in pure-Python mode.
_extensions: list = []
_build_failed = False
try:
    import numpy as np
    from Cython.Build import cythonize
    from setuptools import Extension

    _extensions = cythonize(
        [
            Extension(
                name="neurobox.analysis.spikes._ccg_engine",
                sources=["neurobox/analysis/spikes/_ccg_engine.pyx"],
                include_dirs=[np.get_include()],
                define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            ),
            Extension(
                name="neurobox.analysis.lfp._within_ranges_engine",
                sources=["neurobox/analysis/lfp/_within_ranges_engine.pyx"],
                include_dirs=[np.get_include()],
                define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            ),
        ],
        compiler_directives={
            "language_level":  "3",
            "boundscheck":     False,
            "wraparound":      False,
            "cdivision":       True,
        },
        # Allow the build to succeed even on environments where Cython/numpy
        # aren't fully cooperating — the user can still use the package.
        annotate=False,
    )
except Exception as e:                                                # pragma: no cover
    _build_failed = True
    if os.environ.get("NEUROBOX_REQUIRE_CYTHON"):
        raise
    import warnings
    warnings.warn(
        f"neurobox: Cython extension build failed ({e!r}).  "
        "The package will install in pure-Python mode.  "
        "Set NEUROBOX_REQUIRE_CYTHON=1 to make this an error.",
        stacklevel=2,
    )

setup(ext_modules=_extensions)
