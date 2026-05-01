"""
neurobox.analysis.decoding.accumulate
=====================================
Wrap a decoding result with posterior-derived ego- and
trajectory-centric features.

Port of :file:`MTA/analysis/accumulate_decoding_vars.m` (Anton
Sirota et al.).

What it does
------------
Given a :class:`DecodingResult` from :func:`decode_ufr_boxcar` and
the matching xyz tracking, compute the **error vector** between
each decoded position estimate and the rat's actual head COM, then
project that error into two body-fixed reference frames:

* the **head frame** ``hvec`` — longitudinal axis ``hcom → nose``,
  lateral axis 90° CCW, with optional yaw calibration
* the **trajectory frame** ``tvec`` — longitudinal axis
  ``hcom[t+Δ] - hcom[t-Δ]`` (a 200 ms centred trajectory direction
  at 250 Hz), lateral axis 90° CCW

For each of the five posterior-derived position estimates (``com``,
``sax``, ``max``, ``lom``, ``lax``) you get:

* ``e<x>`` = (estimate − hcom) projected onto head frame
* ``t<x>`` = (estimate − hcom) projected onto trajectory frame

These are the natural quantities for analysing how the population
decode is biased relative to the rat's current heading or motion
direction (the "ahead/behind/left/right" decomposition of decode
errors used throughout the lab's analyses).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from neurobox.dtype.xyz import NBDxyz

from .bayesian import DecodingResult


@dataclass
class AccumulatedDecoding:
    """Posterior-derived features in head- and trajectory-frame coordinates.

    All ``e*`` arrays are decoded position estimates expressed in the
    head frame (anteroposterior + lateral); all ``t*`` arrays are in
    the trajectory frame.  Last column when present holds the third
    dimension (typically pitch / height) passed unchanged.

    Attributes
    ----------
    decoding:
        The underlying :class:`DecodingResult` (same row count as
        all the per-step arrays below).
    xyz:
        ``(N, n_markers, 3)`` xyz subset at decoded indices.
    hvec:
        ``(N, 2, 2)`` head-frame rotation matrix per decoded sample.
    tvec:
        ``(N, 2, 2)`` trajectory-frame rotation matrix per decoded
        sample.
    ecom, esax, emax, elom, elax:
        ``(N, 2)`` head-frame projections of the corresponding
        posterior estimates relative to head COM.
    tcom, tsax, tmax, tlom, tlax:
        ``(N, 2)`` trajectory-frame projections.
    """

    decoding: DecodingResult
    xyz:      np.ndarray
    hvec:     np.ndarray
    tvec:     np.ndarray
    ecom:     np.ndarray
    esax:     np.ndarray
    emax:     np.ndarray
    elom:     np.ndarray
    elax:     np.ndarray
    tcom:     np.ndarray
    tsax:     np.ndarray
    tmax:     np.ndarray
    tlom:     np.ndarray
    tlax:     np.ndarray


# ─────────────────────────────────────────────────────────────────────────── #
# Frame builders                                                               #
# ─────────────────────────────────────────────────────────────────────────── #

def _build_head_frame(
    xyz_data:    np.ndarray,
    idx_hcom:    int,
    idx_nose:    int,
    head_yaw:    float = 0.0,
) -> np.ndarray:
    """Per-frame head rotation: ``hcom→nose`` longitudinal, 90° CCW lateral.

    Returns ``(T, 2, 2)`` where ``[:, :, 0]`` is the longitudinal
    unit vector and ``[:, :, 1]`` the lateral.  Optional
    ``head_yaw`` rotates the frame by that angle (radians).
    """
    vec = xyz_data[:, idx_nose, :2] - xyz_data[:, idx_hcom, :2]
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    u_long = np.divide(vec, norm, out=np.zeros_like(vec), where=norm > 0)
    u_lat = np.column_stack([-u_long[:, 1], u_long[:, 0]])
    R = np.stack([u_long, u_lat], axis=2)             # (T, 2, 2)
    if head_yaw != 0.0:
        c, s = np.cos(head_yaw), np.sin(head_yaw)
        rot = np.array([[c, -s], [s, c]])
        R = np.einsum("tij,jk->tik", R, rot)
    return R


def _build_trajectory_frame(
    xyz_data:    np.ndarray,
    idx_hcom:    int,
    delta:       int,
) -> np.ndarray:
    """Per-frame trajectory rotation: ``hcom[t+Δ] - hcom[t-Δ]`` longitudinal.

    Returns ``(T, 2, 2)``.  Frames at the boundaries (where the
    centred shift goes out of bounds) get zero direction vectors.
    """
    pos = xyz_data[:, idx_hcom, :2]                    # (T, 2)
    fwd = np.roll(pos, -delta, axis=0)
    bwd = np.roll(pos, +delta, axis=0)
    vec = fwd - bwd
    # Zero out wrap-around at the edges
    vec[:delta, :]  = 0.0
    vec[-delta:, :] = 0.0

    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    u_long = np.divide(vec, norm, out=np.zeros_like(vec), where=norm > 0)
    u_lat  = np.column_stack([-u_long[:, 1], u_long[:, 0]])
    return np.stack([u_long, u_lat], axis=2)


def _project_estimate(
    estimate:  np.ndarray,           # (N, n_dims) — typically n_dims==2 or 4
    hcom:      np.ndarray,           # (N, 2)
    R:         np.ndarray,           # (N, 2, 2)
) -> np.ndarray:
    """Project ``estimate - hcom`` onto the rotation frame ``R``.

    Returns ``(N, 2)`` egocentric coordinates.  If the estimate has
    extra columns beyond the first two spatial dims, they're dropped
    (use the ``extra_dims`` parameter of accumulate to forward them).
    """
    if estimate.ndim != 2:
        raise ValueError(f"estimate must be (N, n_dims); got {estimate.shape}")
    delta = estimate[:, :2] - hcom                  # (N, 2)
    return np.einsum("ti,tij->tj", delta, R)


# ─────────────────────────────────────────────────────────────────────────── #
# Public API                                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

def accumulate_decoding_vars(
    decoding:     DecodingResult,
    xyz:          NBDxyz,
    *,
    head_yaw:     float = 0.0,
    trajectory_window_s: float = 0.1,
    edge_pad:     int = 26,
) -> AccumulatedDecoding:
    """Build ego- and trajectory-frame posterior features.

    Port of :file:`MTA/analysis/accumulate_decoding_vars.m` (the
    section after the call to :func:`decode_ufr_boxcar`, lines 50-217).

    Parameters
    ----------
    decoding:
        Output of :func:`decode_ufr_boxcar`.
    xyz:
        Augmented xyz at the same samplerate as the decoding's UFR.
        Must contain ``hcom`` and ``nose``.
    head_yaw:
        Per-subject head-frame yaw correction (radians) applied to
        the head frame.  Pass ``Trial.meta.correction.headYaw`` from
        the relevant :class:`SubjectInfo`.  MATLAB stores this in
        ``Trial.meta.correction.headYaw``.
    trajectory_window_s:
        Half-window for the trajectory-frame computation: the
        forward / backward shift in seconds applied to ``hcom``
        before differencing.  Default 0.1 s (matches MATLAB
        ``round(250 * 0.1)`` at 250 Hz).
    edge_pad:
        Number of samples to drop at the start and end of the
        decoding output before further processing (matches MATLAB's
        ``[1:26, end-26:end]`` removal at 250 Hz).

    Returns
    -------
    AccumulatedDecoding
    """
    if "hcom" not in xyz.model.markers or "nose" not in xyz.model.markers:
        raise ValueError(
            "accumulate_decoding_vars: xyz must contain 'hcom' and 'nose' "
            "markers — pass through augment_xyz first."
        )

    if xyz.samplerate != decoding.samplerate:
        raise ValueError(
            f"samplerate mismatch: xyz={xyz.samplerate} Hz, "
            f"decoding={decoding.samplerate} Hz.  Resample xyz first."
        )

    # Drop edge samples — matches MATLAB's [1:26, end-26:end] removal
    n_dec = decoding.n
    if n_dec <= 2 * edge_pad:
        # Not enough samples — return empty
        return AccumulatedDecoding(
            decoding = decoding,
            xyz      = np.zeros((0, len(xyz.model.markers), 3)),
            hvec     = np.zeros((0, 2, 2)),
            tvec     = np.zeros((0, 2, 2)),
            **{k: np.zeros((0, 2)) for k in
               ("ecom", "esax", "emax", "elom", "elax",
                "tcom", "tsax", "tmax", "tlom", "tlax")},
        )

    keep = slice(edge_pad, n_dec - edge_pad)
    ind  = decoding.ind[keep]
    com  = decoding.com[keep]
    sax  = decoding.sax[keep]
    mx   = decoding.max[keep]
    lom  = decoding.lom[keep]
    lax  = decoding.lax[keep]

    # Subset xyz at decoded indices
    xyz_data = xyz.data
    if xyz_data is None:
        raise RuntimeError("xyz data is not loaded")
    xyz_sub = xyz_data[ind, :, :]                     # (N, n_markers, 3)
    idx_hcom = xyz.model.index("hcom")
    idx_nose = xyz.model.index("nose")
    hcom_sub = xyz_sub[:, idx_hcom, :2]               # (N, 2)

    # Head frame at decoded samples
    hvec = _build_head_frame(xyz_sub, idx_hcom, idx_nose, head_yaw=head_yaw)

    # Trajectory frame: built from the *full* xyz, then subsampled at
    # the decoded indices.  This matches MATLAB's
    # ``circshift(xyz(:,'hcom',[1,2]), ±round(250*0.1))``.
    delta = int(round(trajectory_window_s * xyz.samplerate))
    tvec_full = _build_trajectory_frame(xyz_data, idx_hcom, delta)
    tvec = tvec_full[ind]

    # Project each of the five estimates onto each frame
    ecom = _project_estimate(com, hcom_sub, hvec)
    esax = _project_estimate(sax, hcom_sub, hvec)
    emax = _project_estimate(mx,  hcom_sub, hvec)
    elom = _project_estimate(lom, hcom_sub, hvec)
    elax = _project_estimate(lax, hcom_sub, hvec)
    tcom = _project_estimate(com, hcom_sub, tvec)
    tsax = _project_estimate(sax, hcom_sub, tvec)
    tmax = _project_estimate(mx,  hcom_sub, tvec)
    tlom = _project_estimate(lom, hcom_sub, tvec)
    tlax = _project_estimate(lax, hcom_sub, tvec)

    # Trim the underlying decoding to match
    trimmed = DecodingResult(
        ind  = ind,
        max  = mx,
        com  = com,
        sax  = sax,
        lom  = lom,
        lax  = lax,
        post = decoding.post[keep],
        ucnt = decoding.ucnt[keep],
        uinc = decoding.uinc[keep],
        smoothing_weights = decoding.smoothing_weights,
        window            = decoding.window,
        samplerate        = decoding.samplerate,
    )

    return AccumulatedDecoding(
        decoding = trimmed,
        xyz      = xyz_sub,
        hvec     = hvec,
        tvec     = tvec,
        ecom = ecom, esax = esax, emax = emax, elom = elom, elax = elax,
        tcom = tcom, tsax = tsax, tmax = tmax, tlom = tlom, tlax = tlax,
    )
