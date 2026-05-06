"""
Build script for the Cython extensions in neurobox.

Most package metadata lives in :file:`pyproject.toml` — this file
exists only to declare compiled extensions, which setuptools doesn't
yet support natively in pyproject.toml.

Cython is a hard build-time requirement (declared in
``pyproject.toml`` under ``[build-system].requires``).  ``pip install
-e .`` and ``pip install .`` use isolated builds by default and will
install Cython into the build environment automatically; both
extensions are compiled and shipped with every installation.
"""

from __future__ import annotations

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

_extensions = cythonize(
    [
        Extension(
            name=    "neurobox.analysis.spikes._ccg_engine",
            sources= ["neurobox/analysis/spikes/_ccg_engine.pyx"],
            include_dirs=[np.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        ),
        Extension(
            name=    "neurobox.analysis.lfp._within_ranges_engine",
            sources= ["neurobox/analysis/lfp/_within_ranges_engine.pyx"],
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
)

setup(ext_modules=_extensions)
