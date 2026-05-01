"""
neurobox.analysis.decoding
===========================
Bayesian population-coding decoding from boxcar-smoothed unit firing
rates.

Ports of the MTA decoding stack:

* :mod:`bayesian` — :func:`decode_ufr_boxcar`, :class:`DecodingResult`,
  :func:`prepare_ratemap`, :func:`prepare_bin_coords`.  Port of
  :file:`MTA/analysis/decode_ufr_boxcar.m`.
* :mod:`accumulate` — :func:`accumulate_decoding_vars`,
  :class:`AccumulatedDecoding`.  Port of
  :file:`MTA/analysis/accumulate_decoding_vars.m`.
* :mod:`tensor_mask` — :func:`create_tensor_mask` plus boundary
  dataclasses (:class:`CircularBoundary`, :class:`SquareBoundary`,
  :class:`LineBoundary`).  Port of
  :file:`MTA/analysis/create_tensor_mask.m`.
* :mod:`theta_phase` — :func:`theta_phase`.  Port of
  :file:`MTA/transformations/load_theta_phase.m`.
* :mod:`state_matrix` — :func:`stc2mat`.  Port of
  :file:`MTA/utilities/stc/stc2mat.m`.
"""

from .bayesian      import (
    decode_ufr_boxcar,
    DecodingResult,
    prepare_ratemap,
    prepare_bin_coords,
)
from .accumulate    import accumulate_decoding_vars, AccumulatedDecoding
from .tensor_mask   import (
    create_tensor_mask,
    CircularBoundary,
    SquareBoundary,
    LineBoundary,
)
from .theta_phase   import theta_phase
from .state_matrix  import stc2mat

__all__ = [
    "decode_ufr_boxcar",
    "DecodingResult",
    "prepare_ratemap",
    "prepare_bin_coords",
    "accumulate_decoding_vars",
    "AccumulatedDecoding",
    "create_tensor_mask",
    "CircularBoundary",
    "SquareBoundary",
    "LineBoundary",
    "theta_phase",
    "stc2mat",
]
