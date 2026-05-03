"""
neurobox.analysis.transformations.body_motion_svd
=================================================
SVD-based decomposition of horizontal body motion onto a
body-fixed orthonormal basis.

Port of :file:`MTA/transformations/decompose_xy_motion_wrt_body.m`.

What it does
------------
For each frame the body's longitudinal axis is the unit vector from
``spine_lower`` to ``spine_upper`` (the "body unit vector").  Per-frame
rotation matrices ``rotMat`` rotate this body unit vector by user-
specified angles (default 0° and 90°), producing a small set of
orthonormal body-fixed directions.  The horizontal velocity
(centred-difference of xy positions) of each spine + head marker is
then projected onto these directions, giving a feature matrix
``walk_fet_rot`` of shape ``(T, n_markers * n_orientations, 2)`` where
the last axis is (x, y) world components.

The flat per-frame feature vector is **embedded** with a sliding
window of length ``window_length × samplerate`` (default ~64 samples
at 120 Hz).  In the *COMPUTE* mode this embedded matrix is decomposed
via SVD over walk and turn periods to obtain reusable eigenvectors;
in *RUN* mode those eigenvectors are loaded from a saved model and
used to project a new session's embedded feature matrix down to
*max_num_components* dimensions.

Differences from the MATLAB original
------------------------------------
The MATLAB function is tightly coupled to MTA's session list /
``Trial.load`` infrastructure and to the deferred spline-spine
:func:`preproc_xyz` modes.  This port decouples the maths from the
data-loading by accepting:

* an already-augmented :class:`NBDxyz` (use
  :func:`neurobox.analysis.kinematics.augment_xyz` to add the
  ``bcom`` / ``hcom`` virtual markers),
* an optional :class:`NBStateCollection` for COMPUTE-mode SVD
  (the MATLAB original used the ``walk + turn`` query
  ``stc{'w+n'}`` to mask training samples).

Saving/loading the model is the caller's responsibility — pass
``mode='COMPUTE'`` with no ``model`` to fit, ``mode='RUN'`` with a
fitted :class:`BodyMotionSVDModel` to project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from neurobox.analysis.lfp.filtering import butter_filter
from neurobox.analysis.kinematics.helpers import (
    finite_nonzero_mask, zscore_with_mask,
)
from neurobox.dtype.xyz import NBDxyz


__all__ = [
    "BodyMotionSVDModel",
    "decompose_xy_motion_wrt_body",
]


@dataclass
class BodyMotionSVDModel:
    """SVD model for body-motion decomposition.

    Attributes
    ----------
    mean, std:
        Per-feature normalisation parameters (matched to the
        flattened ``walk_fet_rot`` feature matrix used at fit time).
    eigen_values, eigen_vectors:
        Singular values and right singular vectors from SVD on the
        embedded feature matrix.
    metadata:
        Free-form dict capturing fit-time parameters (samplerate,
        window length, orientations, marker list).
    """
    mean:           np.ndarray
    std:            np.ndarray
    eigen_values:   np.ndarray
    eigen_vectors:  np.ndarray
    metadata:       dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────── #
# Feature builders                                                      #
# ─────────────────────────────────────────────────────────────────── #

def _shifted_diff_xy(
    xyz_data:    np.ndarray,           # (T, n_markers, n_dims)
    marker_idx:  Sequence[int],
    half_shift:  int,
) -> np.ndarray:
    """Centred-difference horizontal velocity of selected markers.

    Returns ``(T, n_markers_sub, 2)`` — the xy-only velocity at each
    frame, computed as ``circshift(-half) - circshift(+half)``.
    """
    sub = xyz_data[:, marker_idx, :2]   # (T, M, 2)
    fwd = np.roll(sub, -half_shift, axis=0)
    bwd = np.roll(sub, +half_shift, axis=0)
    out = fwd - bwd
    out[:half_shift, :, :]  = 0.0
    out[-half_shift:, :, :] = 0.0
    return out


def _build_body_basis(
    xyz_data:           np.ndarray,
    idx_lower:          int,
    idx_upper:          int,
    rotation_angles:    np.ndarray,
) -> np.ndarray:
    """Per-frame orthonormal body-fixed basis at each rotation angle.

    Parameters
    ----------
    xyz_data:
        ``(T, n_markers, n_dims)``.
    idx_lower, idx_upper:
        Column indices into ``xyz_data`` defining the body
        longitudinal axis ``(idx_upper) - (idx_lower)``.
    rotation_angles:
        ``(R,)`` angles in radians.  The unit vector is rotated by
        each angle to give one direction per angle.

    Returns
    -------
    unit : ``(T, R, 2)``
        Per-frame, per-angle unit vectors in xy.
    """
    body = xyz_data[:, idx_upper, :2] - xyz_data[:, idx_lower, :2]   # (T, 2)
    norm = np.linalg.norm(body, axis=1, keepdims=True)
    body_unit = np.divide(body, norm, out=np.zeros_like(body),
                           where=norm > 0)                       # (T, 2)
    # Rotate body_unit by each angle in rotation_angles
    out = np.empty((body_unit.shape[0], rotation_angles.size, 2),
                   dtype=np.float64)
    for r, theta in enumerate(rotation_angles):
        c, s = np.cos(theta), np.sin(theta)
        out[:, r, 0] =  c * body_unit[:, 0] - s * body_unit[:, 1]
        out[:, r, 1] =  s * body_unit[:, 0] + c * body_unit[:, 1]
    return out


def _embed_window(
    flat_features:     np.ndarray,    # (T, n_features)
    embedding_window:  int,
) -> np.ndarray:
    """Embed each row in a sliding window — equivalent of MATLAB ``GetSegs``.

    Returns ``(T, embedding_window * n_features)``.  Rows near the
    edges are zero-padded.  The ``circshift(w, e/2, 2)`` step in the
    MATLAB code centres the window on the current sample, which we
    reproduce here.
    """
    T, n_feat = flat_features.shape
    half = embedding_window // 2
    out = np.zeros((T, embedding_window, n_feat), dtype=np.float64)
    for k in range(embedding_window):
        shift = k - half
        if shift > 0:
            out[shift:, k, :] = flat_features[:T - shift, :]
        elif shift < 0:
            out[:T + shift, k, :] = flat_features[-shift:, :]
        else:
            out[:, k, :] = flat_features
    return out.reshape(T, embedding_window * n_feat)


# ─────────────────────────────────────────────────────────────────── #
# Public API                                                            #
# ─────────────────────────────────────────────────────────────────── #

def decompose_xy_motion_wrt_body(
    xyz:               NBDxyz,
    *,
    mode:              str = "RUN",
    model:             BodyMotionSVDModel | None = None,
    new_samplerate:    float = 119.881035,
    window_length_s:   float = 0.53386259,
    orientations_deg:  Sequence[float] = (0.0, 90.0),
    max_num_components: int = 5,
    spine_markers:     Sequence[str] = (
        "spine_lower", "pelvis_root", "spine_middle",
        "spine_upper", "hcom",
    ),
    body_axis_markers: tuple[str, str] = ("spine_lower", "spine_upper"),
    train_mask:        np.ndarray | None = None,
) -> tuple[np.ndarray, BodyMotionSVDModel]:
    """SVD-based decomposition of horizontal body motion.

    Port of :file:`MTA/transformations/decompose_xy_motion_wrt_body.m`.

    Parameters
    ----------
    xyz:
        Augmented :class:`NBDxyz` containing all of *spine_markers* and
        the two *body_axis_markers*.  The ``hcom`` virtual marker is
        typically required (use
        :func:`neurobox.analysis.kinematics.augment_xyz`).
    mode:
        ``'COMPUTE'`` (fit a fresh SVD model, requires *train_mask*) or
        ``'RUN'`` (project using a precomputed *model*).  Default
        ``'RUN'``.
    model:
        Required for ``mode='RUN'``.
    new_samplerate:
        Working samplerate in Hz.  Default 119.881035 matches the
        MATLAB default.  Pass ``None`` to keep the input rate.
    window_length_s:
        Embedding window length in seconds.  Default 0.534 matches
        MATLAB.
    orientations_deg:
        Rotation angles applied to the body unit vector before
        projecting velocity.  Default ``(0, 90)``.
    max_num_components:
        Number of SVD components to project onto in RUN mode.
        Default 5.
    spine_markers:
        Markers whose horizontal velocity is projected onto the body
        basis.  Default is the 4 spine markers + hcom.
    body_axis_markers:
        ``(lower, upper)`` markers defining the body longitudinal
        axis.  Default ``('spine_lower', 'spine_upper')``.
    train_mask:
        ``(T,)`` boolean mask marking samples to use for SVD fitting
        (typically walk + turn periods).  Required for COMPUTE.

    Returns
    -------
    projection_score : ``(T, max_num_components)``
        Per-sample score on the SVD principal components.
    model : :class:`BodyMotionSVDModel`
        Fitted model (in COMPUTE mode) or the input model (in RUN
        mode), so the caller can save / chain it.
    """
    if mode not in ("COMPUTE", "RUN"):
        raise ValueError(f"mode must be 'COMPUTE' or 'RUN'; got {mode!r}")

    if mode == "RUN" and model is None:
        raise ValueError("mode='RUN' requires a fitted model")

    # Verify markers exist
    needed = set(spine_markers) | set(body_axis_markers)
    missing = [m for m in needed if m not in xyz.model.markers]
    if missing:
        raise ValueError(
            f"decompose_xy_motion: xyz missing markers {missing}"
        )

    # Resample if requested
    work = xyz if (new_samplerate is None or new_samplerate == xyz.samplerate) \
           else xyz.resample(float(new_samplerate))
    fs = float(work.samplerate)

    # Half-shift in samples — MATLAB's round(sampleRate * 0.05) / 2
    half_shift = max(1, int(round(fs * 0.05) // 2))

    # 1. Centred-difference horizontal velocity of spine markers
    spine_idx = [work.model.index(m) for m in spine_markers]
    body_low_idx = work.model.index(body_axis_markers[0])
    body_up_idx  = work.model.index(body_axis_markers[1])
    rot_angles = np.deg2rad(np.asarray(orientations_deg, dtype=np.float64))

    tvec = _shifted_diff_xy(work.data, spine_idx, half_shift)   # (T, M, 2)

    # 2. Body-fixed basis directions (one per rotation angle)
    body_dirs = _build_body_basis(work.data, body_low_idx, body_up_idx,
                                   rot_angles)                  # (T, R, 2)

    # 3. Project velocity onto each (rotated) body direction
    # walk_fet_rot[t, m, r] = tvec[t, m, :] · body_dirs[t, r, :]
    walk_fet_rot = np.einsum("tmd,trd->tmr", tvec, body_dirs)   # (T, M, R)

    T_samples = walk_fet_rot.shape[0]
    flat = walk_fet_rot.reshape(T_samples, -1)                   # (T, M*R)
    n_features = flat.shape[1]

    # 4. Normalise (mean / std).  In COMPUTE mode we fit; in RUN mode
    # we use the stored model parameters.
    if mode == "COMPUTE":
        z, mean, std = zscore_with_mask(flat)
    else:
        z, mean, std = zscore_with_mask(flat, mean=model.mean, std=model.std)

    # 5. Window-embed
    embedding_window = int(round(window_length_s * fs))
    if embedding_window < 1:
        raise ValueError(
            f"window_length_s={window_length_s} too small for "
            f"samplerate={fs}"
        )
    embedded = _embed_window(z, embedding_window)               # (T, W*n_feat)
    embedded = np.nan_to_num(embedded, nan=0.0)

    # 6. SVD decomposition (COMPUTE) or projection (RUN)
    if mode == "COMPUTE":
        if train_mask is None:
            raise ValueError(
                "mode='COMPUTE' requires a train_mask to select walk/turn "
                "samples for SVD"
            )
        train_mask = np.asarray(train_mask, dtype=bool).ravel()
        if train_mask.size != T_samples:
            raise ValueError(
                f"train_mask length {train_mask.size} ≠ T={T_samples}"
            )
        wfw = embedded[train_mask, :]
        # MATLAB svd(...,0) returns thin SVD: [U, S, V]
        # We need the right singular vectors (V).
        _, sing, vh = np.linalg.svd(wfw, full_matrices=False)
        eigen_vectors = vh.T                                # (W*n_feat, k)
        eigen_values  = sing
        model = BodyMotionSVDModel(
            mean          = mean,
            std           = std,
            eigen_values  = eigen_values,
            eigen_vectors = eigen_vectors,
            metadata      = dict(
                new_samplerate    = fs,
                window_length_s   = window_length_s,
                orientations_deg  = list(orientations_deg),
                spine_markers     = list(spine_markers),
                body_axis_markers = list(body_axis_markers),
                embedding_window  = embedding_window,
                n_features        = n_features,
            ),
        )

    # Project onto the first max_num_components components
    n_keep = min(max_num_components, model.eigen_vectors.shape[1])
    projection = embedded @ model.eigen_vectors[:, :n_keep]      # (T, k)
    return projection, model
