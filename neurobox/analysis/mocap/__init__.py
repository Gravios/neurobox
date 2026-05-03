"""
neurobox.analysis.mocap
========================
Motion-capture utility ports of :file:`MTA/utilities/mocap/*.m`.

What's here
-----------
* :mod:`rotations` — :func:`rotate_points_around_vectors` (Rodrigues
  primitive), :func:`rotate_point_around_vector` and
  :func:`rotate_marker_around_vector` (marker-based wrappers).
* :mod:`basis` — :func:`rigid_body_basis` (with both MATLAB names
  as aliases), :func:`intermarker_distances`, :func:`marker_triads`.
* :mod:`gap_filling` — :func:`fill_gaps` (PCHIP gap-filling, port of
  :file:`mocap_fill_gaps.m` with the ``gapPer`` typo bug fixed).
* :mod:`virtual_joint` — :func:`infer_virtual_joint` (port of
  :file:`infer_virtual_joint_from_rigidbody_kinematics.m`).
* :mod:`error_periods` — :func:`find_error_periods` (port of
  :file:`FindErrorPeriods.m`).
* :mod:`correct_errors` — :func:`correct_point_errors` (port of
  :file:`CorrectPointErrors.m`, re-derived using Hungarian
  assignment to fix the original's known assignment-uniqueness bug).
* :mod:`motive_csv` — :func:`parse_rbo_from_csv` (Motive CSV
  rigid-body-orientation parser).

What's not ported
-----------------
* ``add_virtual_marker_to_rigidbody.m`` — 437-line interactive GUI
  workflow; depends on ``mixGaussEm`` Bayesian GMM, ``select_ranges``
  GUI widget, ``ClusterPP`` GUI widget — none ported.
* ``transform_rigidBody.m`` — 553-line optimisation + figure display
  + tightly tied to ``MTASession`` infrastructure.
* ``parse_mkr_from_csv.m``, ``convert_motiveCSV_to_pos.m``,
  ``convert_MotiveCSV_to_pos.m``, ``concatenate_take_files.m`` —
  broken or stub MATLAB (incomplete statements, parse errors).
* ``preview_xyz.m``, ``PlotSessionErrors.m``,
  ``display_session_marker_error_summary.m`` — figure-plotting
  utilities; neurobox has no plotting layer.
* ``concatenate_vicon_files.m`` / ``concatViconFiles.m`` — directory
  walking + .mat loading; tightly tied to ``MTASession`` and a
  filesystem layout that's unlikely to apply outside MTA.
* ``load_take_files.m`` — 13-line ``.mat`` loader; trivial when
  needed.
* ``compute_rigidbody_basis_timeseries.m`` — exact byte-for-byte
  duplicate of ``compute_rb_basis_timeseries.m`` save for variable
  rename; both names are exposed as aliases of
  :func:`rigid_body_basis`.
"""

from .rotations    import (
    rotate_points_around_vectors,
    rotate_point_around_vector,
    rotate_marker_around_vector,
)
from .basis        import (
    rigid_body_basis,
    compute_rb_basis_timeseries,
    compute_rigidbody_basis_timeseries,
    intermarker_distances,
    marker_triads,
    MarkerTriadResult,
    marker_diff_matrix,
    inter_marker_distance,
    inter_marker_angles,
    inter_marker_orientation,
)
from .gap_filling  import fill_gaps
from .virtual_joint import infer_virtual_joint
from .error_periods import find_error_periods
from .correct_errors import correct_point_errors
from .motive_csv   import parse_rbo_from_csv, MotiveTakeResult

__all__ = [
    # rotations
    "rotate_points_around_vectors",
    "rotate_point_around_vector",
    "rotate_marker_around_vector",
    # basis
    "rigid_body_basis",
    "compute_rb_basis_timeseries",
    "compute_rigidbody_basis_timeseries",
    "intermarker_distances",
    "marker_triads",
    "MarkerTriadResult",
    "marker_diff_matrix",
    "inter_marker_distance",
    "inter_marker_angles",
    "inter_marker_orientation",
    # gap filling
    "fill_gaps",
    # virtual joint
    "infer_virtual_joint",
    # error detection / correction
    "find_error_periods",
    "correct_point_errors",
    # motive CSV
    "parse_rbo_from_csv",
    "MotiveTakeResult",
]
