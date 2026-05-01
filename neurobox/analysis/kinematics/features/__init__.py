"""
neurobox.analysis.kinematics.features
======================================
Kinematic feature definitions — derived signals computed from
:class:`~neurobox.dtype.NBDxyz` position data.

Each function takes an ``NBDxyz`` (typically already augmented with
``bcom``/``hcom``/``nose`` via :func:`augment_xyz`) and returns an
:class:`~neurobox.dtype.NBDfet` with named columns.

Ports of the ``MTA/features/fet_*.m`` files.  Each Python function
documents which MATLAB file it derives from.

Currently ported
----------------
* :func:`fet_xy`           — head COM xy
* :func:`fet_dxy`          — head yaw + xy
* :func:`fet_xyhb`         — xy + head/body pitch
* :func:`fet_head_pitch`   — head_back→head_front pitch
* :func:`fet_HB_pitch`     — three sequential body→head pitches
* :func:`fet_HB_pitchB`    — Butter-filtered head/body pitch (uses fet_HB_pitch)
* :func:`fet_hzp`          — head height + body pitch
* :func:`fet_hba`          — head-body azimuth angle
* :func:`fet_hbav`         — head-body angular velocity (filtered fet_hba)

Not yet ported (deferred)
-------------------------
The other ~120 fet_*.m files; ``fet_lfp``, ``fet_spec``,
``fet_respiration_*``, ``fet_bref_*``, ``fet_tsne_*``, ``fet_all*``,
and the spline-spine ``procOpts`` modes of preproc_xyz which require
porting ``fet_spline_spine`` + ``arclength`` + ``interparc`` first.

If you need one of the deferred features, the cleanest approach is
to copy the small ones (~50-100 lines) into your project as a starting
point — most are self-contained once you have NBDxyz.add_marker,
augment_xyz, and NBDfet.
"""

from .basic       import fet_xy, fet_dxy
from .pitch       import (
    fet_head_pitch, fet_HB_pitch, fet_HB_pitchB,
    fet_xyhb, fet_hzp,
)
from .head_body   import fet_hba, fet_hbav
from .body_frame  import fet_href_HXY, fet_bref_BXY, fet_hvfl

__all__ = [
    "fet_xy", "fet_dxy",
    "fet_head_pitch", "fet_HB_pitch", "fet_HB_pitchB",
    "fet_xyhb", "fet_hzp",
    "fet_hba", "fet_hbav",
    "fet_href_HXY", "fet_bref_BXY", "fet_hvfl",
]
