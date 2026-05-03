"""
neurobox.analysis.transformations
=================================
Ports of :file:`MTA/transformations/*.m`.

What's here
-----------
* :mod:`bin_statistics` — :class:`BinAxis`,
  :func:`bin_statistic_2d`, :func:`bin_statistic_2d_circ`,
  :func:`bin_statistic_3d`, :func:`bin_statistic` (generic).  Ports of
  ``compute_2d_discrete_func.m``, ``compute_2d_discrete_stats.m``,
  ``compute_2d_discrete_circ_stats.m``, ``compute_3d_discrete_stats.m``.
* :mod:`axis_alignment` — :func:`rot_z_axis`, :func:`rot_y_axis`,
  :func:`detect_roll`.  Per-frame rotation matrices that align marker
  vectors with cardinal axes.
* :mod:`thetarc_phase` — :func:`thetarc_phase`.  Bipolar-reference
  variant of :func:`neurobox.analysis.decoding.theta_phase`.
* :mod:`body_motion_svd` — :func:`decompose_xy_motion_wrt_body`,
  :class:`BodyMotionSVDModel`.  SVD-based decomposition of horizontal
  body motion onto a body-fixed orthonormal basis.

Already covered in earlier rounds (don't re-port from
:file:`MTA/transformations/`):

* ``load_theta_phase.m`` → :func:`neurobox.analysis.decoding.theta_phase`
  (round 10)
"""

from .bin_statistics    import (
    BinAxis,
    BinStats2D, BinStats2DCirc, BinStats3D, BinStatsArbitrary,
    bin_statistic_2d,
    bin_statistic_2d_circ,
    bin_statistic_3d,
    bin_statistic,
)
from .axis_alignment    import rot_z_axis, rot_y_axis, detect_roll
from .thetarc_phase     import thetarc_phase
from .body_motion_svd   import (
    decompose_xy_motion_wrt_body,
    BodyMotionSVDModel,
)
from .quaternions       import quat2rotm, quaternion2rad
from .misc              import make_uniform_distr, shilbert, my_theta_phase

__all__ = [
    "BinAxis",
    "BinStats2D", "BinStats2DCirc", "BinStats3D", "BinStatsArbitrary",
    "bin_statistic_2d",
    "bin_statistic_2d_circ",
    "bin_statistic_3d",
    "bin_statistic",
    "rot_z_axis", "rot_y_axis", "detect_roll",
    "thetarc_phase",
    "decompose_xy_motion_wrt_body",
    "BodyMotionSVDModel",
    "quat2rotm", "quaternion2rad",
    "make_uniform_distr", "shilbert", "my_theta_phase",
]
