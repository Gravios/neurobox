"""
neurobox.analysis.kinematics
=============================
Position pre-processing, derived markers, body-frame transforms, and
related kinematic feature helpers.
"""

from .augment import augment_xyz
from .helpers import finite_nonzero_mask, zscore_with_mask
from . import features

# Round 17 — spline-spine preprocessing + body-referenced features
from .spline_spine import (
    SplineSpineResult,
    spline_spine,
    preproc_xyz_spline_spine_head_eqi,
    preproc_xyz_spline_spine_head_eqd,
)
from .body_referenced import (
    BodyReferencedFeatures,
    body_referenced_features,
    body_referenced_xy_features,
)

# Round 19 — kitchen-sink 59-column feature set
from .fet_all import (
    FetAllResult,
    fet_all_features,
    lower_spine_yaw_ppc,
)

__all__ = [
    "augment_xyz",
    "finite_nonzero_mask",
    "zscore_with_mask",
    "features",
    # Spline-spine preprocessing
    "SplineSpineResult",
    "spline_spine",
    "preproc_xyz_spline_spine_head_eqi",
    "preproc_xyz_spline_spine_head_eqd",
    # Body-referenced features
    "BodyReferencedFeatures",
    "body_referenced_features",
    "body_referenced_xy_features",
    # Round 19
    "FetAllResult",
    "fet_all_features",
    "lower_spine_yaw_ppc",
]
