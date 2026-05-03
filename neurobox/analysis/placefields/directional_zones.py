"""
neurobox.analysis.placefields.directional_zones
=================================================
Directional / distance-to-place-field-centre per-sample scores.

Ports of:

* ``MTA/analysis/placefields/compute_drz.m`` → :func:`compute_drz`
* ``MTA/analysis/placefields/compute_ddz.m`` → :func:`compute_ddz`
* ``MTA/analysis/placefields/compute_ghz.m`` → :func:`compute_ghz`
* ``MTA/analysis/placefields/compute_gdz.m`` → :func:`compute_gdz`
* ``MTA/analysis/placefields/compute_hdz.m`` → :func:`compute_hdz`
* ``MTA/analysis/placefields/compute_hrz.m`` → :func:`compute_hrz`
* ``MTA/analysis/placefields/compute_hpv.m`` → :func:`compute_hpv`
* ``MTA/analysis/placefields/compute_tpv.m`` → :func:`compute_tpv`

What these scores measure
-------------------------
For each unit (with a known 2-D place-field centre) and each frame
of a session, these functions return a real-valued **per-sample
score** that summarises one aspect of the rat's spatial relationship
to the unit's place field.

* **DRZ** (Directional Rate Zone) — heading-signed
  ``±(1 − rate_at_position / peak_rate)``.  Negative when the rat is
  moving toward the field centre, positive when moving away.  Useful
  for separating "approach" from "departure" trajectories around
  each place field.
* **DDZ** (Directional Distance Zone) — heading-signed Euclidean
  distance to centre.  Same sign convention as DRZ.
* **GDZ / GHZ** (Gaussian-distance / Gaussian-heading zone) — like
  DRZ but the rate-map lookup is replaced by a Gaussian function of
  distance with width *sigma*.  Useful when you don't have a clean
  rate-map peak.
* **HDZ / HRZ** (Head-anchored DDZ / DRZ) — same algorithm but the
  trajectory anchor is the rat's head-of-mass position (``hcom``)
  and the heading is taken from a separate marker (typically
  ``nose``).
* **HPV / TPV** — same as HRZ / HDZ but the trajectory tangent is
  computed using a 1-sample shift (rather than the standard
  0.2-second shift used by DRZ/DDZ).  In practice these capture
  finer-grained head / tangential motion features.

Sign convention
---------------
Across all 8 functions:

* The **bearing** ``b_t = ∠(centre − position_t)`` is the angle from
  the rat to the place-field centre.
* The **heading** ``h_t = ∠(position_{t+Δ} − position_{t-Δ})`` is the
  rat's instantaneous direction of motion.
* ``pfds = circ_dist(h_t, b_t)`` is the angular separation in radians.
* The sign factor ``pfd`` is ``+1`` when ``|pfds| < π/2`` (heading
  toward the centre) and ``-1`` otherwise — except for ``compute_ghz``
  / ``compute_gdz`` where the convention is **flipped** (the
  MATLAB original convention is preserved here).

What's deferred or not ported
-----------------------------
* ``compute_pfstats_bs.m`` — depends on the MTA ``MTAAknnpfs_bs``
  KNN-bootstrap placefield class, which is not ported and would
  require a substantial new module.  Round 9's
  :func:`place_field_stats` provides the per-patch statistics math
  that the MATLAB original used internally; the bootstrap-iteration
  wrapping is the MTA-specific part.

Inputs
------
All functions take:

* :class:`PlaceFieldResult` — the spatial placefield (2-D) used to
  locate each unit's field centre.
* :class:`NBDxyz` — augmented xyz with ``hcom``, ``nose``, etc.  Use
  :func:`neurobox.analysis.kinematics.augment_xyz` first.
* ``unit_ids`` — sequence of unit IDs to compute for.

Outputs
-------
``(score, field_centres)`` — ``score`` is ``(T, n_units)`` and
``field_centres`` is ``(n_units, 2)`` (xy of the place-field peak).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from neurobox.analysis.lfp.filtering import butter_filter
from neurobox.analysis.spatial.place_fields import PlaceFieldResult
from neurobox.analysis.stats.circular import circ_dist
from neurobox.dtype.xyz import NBDxyz


__all__ = [
    "compute_drz",
    "compute_ddz",
    "compute_ghz",
    "compute_gdz",
    "compute_hdz",
    "compute_hrz",
    "compute_hpv",
    "compute_tpv",
    "field_centres_from_result",
]


# ─────────────────────────────────────────────────────────────────────── #
# Helpers                                                                   #
# ─────────────────────────────────────────────────────────────────────── #

def field_centres_from_result(
    pf:        PlaceFieldResult,
    unit_ids:  Sequence[int],
) -> np.ndarray:
    """Locate each unit's place-field peak from a :class:`PlaceFieldResult`.

    Parameters
    ----------
    pf:
        2-D placefield result.  ``pf.rate_map`` shape is
        ``(n_x, n_y, n_units, n_iter)``.  This helper averages over
        iterations and selects the bin with maximum rate.
    unit_ids:
        Sequence of unit IDs to look up.  Must be present in
        ``pf.unit_ids``.

    Returns
    -------
    np.ndarray, shape ``(len(unit_ids), 2)``
        ``[x_peak, y_peak]`` per unit, in the same units as
        ``pf.bin_centres``.  NaN if a unit's averaged map is all-NaN.
    """
    if len(pf.bin_centres) != 2:
        raise ValueError(
            f"field_centres_from_result expects a 2-D placefield; got "
            f"{len(pf.bin_centres)} dimensions"
        )
    unit_ids = list(unit_ids)
    out = np.full((len(unit_ids), 2), np.nan, dtype=np.float64)
    for k, uid in enumerate(unit_ids):
        try:
            j = list(pf.unit_ids).index(uid)
        except ValueError as e:
            raise KeyError(
                f"unit {uid} not in PlaceFieldResult.unit_ids"
            ) from e
        rate_map = pf.rate_map[..., j, :]            # (n_x, n_y, n_iter)
        with np.errstate(invalid="ignore"):
            mean_map = np.nanmean(rate_map, axis=-1)     # (n_x, n_y)
        if np.all(np.isnan(mean_map)):
            continue
        flat_idx = int(np.nanargmax(mean_map))
        ix, iy = np.unravel_index(flat_idx, mean_map.shape)
        out[k, 0] = pf.bin_centres[0][ix]
        out[k, 1] = pf.bin_centres[1][iy]
    return out


def _shifted_position(
    xy:           np.ndarray,
    shift:        int,
) -> np.ndarray:
    """Return ``xy`` shifted by *shift* samples along axis 0 (with edge zeros)."""
    out = np.zeros_like(xy)
    if shift > 0:
        out[shift:] = xy[:-shift]
    elif shift < 0:
        out[:shift] = xy[-shift:]
    else:
        out[:] = xy
    return out


def _bandpass_filter_xy(
    xy:           np.ndarray,
    samplerate:   float,
    cutoff_hz:    float,
    order:        int = 3,
) -> np.ndarray:
    """Low-pass filter trajectory at *cutoff_hz* (zero-phase)."""
    return butter_filter(xy, cutoff=cutoff_hz, samplerate=samplerate,
                         order=order, btype="lowpass")


def _bearing_and_heading(
    pos_xy:       np.ndarray,           # (T, 2) — rat position
    head_xy:      np.ndarray | None,    # (T, 2) — heading source (None → use pos_xy itself)
    centre_xy:    np.ndarray,           # (2,)   — place-field centre
    samplerate:   float,
    tangent_mode: str,                  # 'fast' (1-sample) or 'slow' (0.2-sec)
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-frame ``(pfds, pfdd)``.

    * ``pfds`` = signed angular distance heading vs. bearing-to-centre, radians.
    * ``pfdd`` = Euclidean distance from *pos_xy* to *centre_xy*.

    *head_xy* selects the trajectory used for **heading**:
    * ``None`` → heading from *pos_xy* itself (DRZ/DDZ/GHZ/GDZ).
    * ``(T, 2)`` array → heading from this trajectory (HDZ/HRZ/HPV/TPV
      use the head-com, with the marker direction encoded elsewhere).

    *tangent_mode*:
    * ``'fast'`` → 1-sample circshift (HPV/TPV).
    * ``'slow'`` → ``round(samplerate / 5)``-sample circshift (DRZ/DDZ).
    """
    if tangent_mode == "fast":
        shift = 1
    elif tangent_mode == "slow":
        shift = max(1, int(round(samplerate / 5)))
    else:
        raise ValueError(f"tangent_mode must be 'fast' or 'slow'; got {tangent_mode!r}")

    # Choose anchor for the bearing (rat → centre)
    anchor = pos_xy if head_xy is None else head_xy

    # Heading: angle of (anchor[t+shift] - anchor[t]) — we use
    # circshift(anchor, shift) - anchor[t] for the SLOW mode (a forward
    # difference) to match MATLAB exactly:
    #     pfhxy(:, 1, :) = anchor(t)
    #     pfhxy(:, 2, :) = circshift(anchor, shift)
    #     cor = anchor(t+shift) - anchor(t)
    shifted = _shifted_position(anchor, shift)
    dx_h = shifted[:, 0] - anchor[:, 0]
    dy_h = shifted[:, 1] - anchor[:, 1]
    heading = np.arctan2(dy_h, dx_h)

    # Bearing: angle of (centre - anchor)
    dx_b = centre_xy[0] - anchor[:, 0]
    dy_b = centre_xy[1] - anchor[:, 1]
    bearing = np.arctan2(dy_b, dx_b)
    distance = np.hypot(dx_b, dy_b)

    pfds = circ_dist(heading, bearing)
    return pfds, distance


def _heading_sign(pfds: np.ndarray, flipped: bool = False) -> np.ndarray:
    """Convert angular separation into ±1 heading sign.

    Default MATLAB convention (DRZ/DDZ/HDZ/HRZ/HPV/TPV):
        ``+1`` when ``|pfds| < π/2``  (heading toward centre)
        ``-1`` otherwise

    Flipped convention (GDZ/GHZ — MATLAB original):
        ``-1`` when ``|pfds| < π/2``
        ``+1`` otherwise
    """
    pfd = np.zeros_like(pfds)
    inner = np.abs(pfds) < np.pi / 2
    if flipped:
        pfd[inner]  = -1
        pfd[~inner] =  1
    else:
        pfd[inner]  =  1
        pfd[~inner] = -1
    return pfd


def _peak_rates(
    pf:        PlaceFieldResult,
    unit_ids:  Sequence[int],
) -> np.ndarray:
    """Per-unit peak (mean over iterations) firing rate."""
    out = np.full(len(unit_ids), np.nan, dtype=np.float64)
    for k, uid in enumerate(unit_ids):
        j = list(pf.unit_ids).index(uid)
        with np.errstate(invalid="ignore"):
            mean_map = np.nanmean(pf.rate_map[..., j, :], axis=-1)
        if not np.all(np.isnan(mean_map)):
            out[k] = float(np.nanmax(mean_map))
    return out


def _rate_at_position(
    pf:        PlaceFieldResult,
    unit_idx:  int,
    pos_xy:    np.ndarray,             # (T, 2)
) -> np.ndarray:
    """Bilinear-interpolated rate at each (x, y) sample.

    Returns ``(T,)`` floats.  Samples outside the rate-map domain
    or in NaN bins return 0 (matching the MATLAB ``isnan→0`` patch).
    """
    from scipy.interpolate import RegularGridInterpolator
    rate_map = pf.rate_map[..., unit_idx, :]
    with np.errstate(invalid="ignore"):
        rate_map = np.nanmean(rate_map, axis=-1)
    rate_map = np.where(np.isnan(rate_map), 0.0, rate_map)
    interp = RegularGridInterpolator(
        pf.bin_centres, rate_map,
        method="linear", bounds_error=False, fill_value=0.0,
    )
    out = interp(pos_xy)
    out[~np.isfinite(out)] = 0.0
    out[out < 0] = 0.0
    return out


# ─────────────────────────────────────────────────────────────────────── #
# Trajectory helper — extract a 2-D trajectory from an NBDxyz              #
# ─────────────────────────────────────────────────────────────────────── #

def _xy_from_xyz(
    xyz:           NBDxyz,
    marker:        str,
    samplerate:    float | None,
    filt_cutoff:   float | None,
    filter_order:  int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(xy, finite_mask)`` for *marker* in xy plane.

    Resamples to *samplerate* if given, applies a low-pass filter at
    *filt_cutoff* Hz if given, and returns the validity mask
    (frames with finite, non-zero xy).
    """
    work = xyz if samplerate is None or samplerate == xyz.samplerate \
           else xyz.resample(float(samplerate))
    fs = float(work.samplerate)
    midx = work.model.index(marker)
    xy = work.data[:, midx, :2].astype(np.float64).copy()
    # Validity mask BEFORE filtering (to preserve "BUG: patch" comment from MATLAB)
    finite = np.isfinite(xy).all(axis=1) & (xy != 0).any(axis=1)
    if filt_cutoff is not None:
        xy = _bandpass_filter_xy(xy, fs, filt_cutoff, order=filter_order)
        xy[~finite, :] = 0.0
    return xy, finite


# ─────────────────────────────────────────────────────────────────────── #
# Public API                                                                #
# ─────────────────────────────────────────────────────────────────────── #

def compute_drz(
    pf:                PlaceFieldResult,
    xyz:               NBDxyz,
    unit_ids:          Sequence[int],
    *,
    marker:            str = "nose",
    samplerate:        float | None = None,
    filt_cutoff:       float = 2.5,
    max_rate_scale:    float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Directional Rate Zone score per unit, per sample.

    Port of :file:`MTA/analysis/placefields/compute_drz.m`.

    Returns ``±(1 − rate_at_position / peak_rate)``, signed by the
    rat's heading relative to the place-field centre.  Negative when
    moving toward the centre, positive when moving away.

    Parameters
    ----------
    pf:
        2-D place-field result.
    xyz:
        Augmented :class:`NBDxyz`.  Must contain *marker*.
    unit_ids:
        Units to compute scores for.
    marker:
        Trajectory marker.  Default ``'nose'`` matches MATLAB.
    samplerate:
        Optional resample target.  ``None`` keeps *xyz* rate.
    filt_cutoff:
        Low-pass cutoff (Hz) applied to trajectory before computing
        heading.  Default 2.5 matches MATLAB.
    max_rate_scale:
        Multiplier applied to the per-unit peak rate when normalising.
        Default 1.0.

    Returns
    -------
    score : np.ndarray, shape ``(T, n_units)``
    field_centres : np.ndarray, shape ``(n_units, 2)``
    """
    xy, finite = _xy_from_xyz(xyz, marker, samplerate, filt_cutoff)
    fs = float(samplerate) if samplerate is not None else xyz.samplerate
    centres = field_centres_from_result(pf, unit_ids)
    peak    = _peak_rates(pf, unit_ids) * max_rate_scale

    T = xy.shape[0]
    n_units = len(unit_ids)
    score = np.zeros((T, n_units), dtype=np.float64)

    for k, uid in enumerate(unit_ids):
        if not np.isfinite(centres[k]).all():
            continue
        if not np.isfinite(peak[k]) or peak[k] == 0:
            continue
        pfds, _ = _bearing_and_heading(xy, None, centres[k],
                                        fs, tangent_mode="slow")
        pfd = _heading_sign(pfds, flipped=False)
        j = list(pf.unit_ids).index(uid)
        rate = _rate_at_position(pf, j, xy)
        score[:, k] = pfd * (1.0 - rate / peak[k])
        score[~finite, k] = 0.0
    return score, centres


def compute_ddz(
    pf:                PlaceFieldResult,
    xyz:               NBDxyz,
    unit_ids:          Sequence[int],
    *,
    marker:            str = "hcom",
    samplerate:        float | None = None,
    filt_cutoff:       float = 2.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Directional Distance Zone score per unit, per sample.

    Port of :file:`MTA/analysis/placefields/compute_ddz.m`.

    Returns ``±|position − centre|`` signed by the rat's heading.
    Positive when moving toward the centre, negative when moving
    away.  Same heading-sign convention as DRZ.

    Parameters
    ----------
    pf:
        2-D place-field result.  Used only to locate field centres.
    xyz, unit_ids, samplerate, filt_cutoff:
        See :func:`compute_drz`.
    marker:
        Trajectory marker.  Default ``'hcom'`` matches MATLAB.

    Returns
    -------
    score : np.ndarray, shape ``(T, n_units)``
    field_centres : np.ndarray, shape ``(n_units, 2)``
    """
    xy, finite = _xy_from_xyz(xyz, marker, samplerate, filt_cutoff)
    fs = float(samplerate) if samplerate is not None else xyz.samplerate
    centres = field_centres_from_result(pf, unit_ids)

    T = xy.shape[0]
    n_units = len(unit_ids)
    score = np.zeros((T, n_units), dtype=np.float64)

    for k in range(n_units):
        if not np.isfinite(centres[k]).all():
            continue
        pfds, pfdd = _bearing_and_heading(xy, None, centres[k],
                                           fs, tangent_mode="slow")
        pfd = _heading_sign(pfds, flipped=False)
        score[:, k] = pfd * pfdd
        score[~finite, k] = 0.0
    return score, centres


def compute_ghz(
    pf:                PlaceFieldResult,
    xyz:               NBDxyz,
    unit_ids:          Sequence[int],
    *,
    marker:            str = "hcom",
    samplerate:        float | None = None,
    filt_cutoff:       float = 2.5,
    sigma:             float = 150.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian-distance Heading Zone score.

    Port of :file:`MTA/analysis/placefields/compute_ghz.m`.

    Replaces DRZ's rate-map lookup with a Gaussian function of
    distance.  Note the **flipped** sign convention vs. DRZ (matches
    the MATLAB original): negative outside, positive when heading
    toward the field.

    Parameters
    ----------
    sigma:
        Gaussian width in same units as ``pf.bin_centres``.  Default
        150 (mm) matches MATLAB.
    """
    xy, finite = _xy_from_xyz(xyz, marker, samplerate, filt_cutoff,
                                filter_order=4)
    fs = float(samplerate) if samplerate is not None else xyz.samplerate
    centres = field_centres_from_result(pf, unit_ids)

    T = xy.shape[0]
    n_units = len(unit_ids)
    score = np.zeros((T, n_units), dtype=np.float64)

    for k in range(n_units):
        if not np.isfinite(centres[k]).all():
            continue
        pfds, pfdd = _bearing_and_heading(xy, None, centres[k],
                                           fs, tangent_mode="slow")
        pfd = _heading_sign(pfds, flipped=True)
        gauss = (1.0 / (sigma * np.sqrt(2 * np.pi))) * \
                np.exp(-0.5 * (pfdd / sigma) ** 2)
        gmax = gauss.max()
        if gmax > 0:
            score[:, k] = pfd * (1.0 - gauss / gmax)
        score[~finite, k] = 0.0
    return score, centres


def compute_gdz(
    pf:                PlaceFieldResult,
    xyz:               NBDxyz,
    unit_ids:          Sequence[int],
    *,
    marker:            str = "hcom",
    samplerate:        float | None = None,
    filt_cutoff:       float = 2.5,
    sigma:             float = 150.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian Distance Zone score.

    Port of :file:`MTA/analysis/placefields/compute_gdz.m`.

    Identical to :func:`compute_ghz` in this port — the MATLAB
    originals differ only in some metadata strings.  Both names are
    exposed for API parity.
    """
    return compute_ghz(pf, xyz, unit_ids, marker=marker,
                        samplerate=samplerate, filt_cutoff=filt_cutoff,
                        sigma=sigma)


def _head_anchored_score(
    pf:                PlaceFieldResult,
    xyz:               NBDxyz,
    unit_ids:          Sequence[int],
    *,
    marker:            str,
    samplerate:        float | None,
    filt_cutoff:       float,
    rate_normalised:   bool,
    tangent_mode:      str,
    max_rate_scale:    float,
) -> tuple[np.ndarray, np.ndarray]:
    """Shared head-anchored zone-score machinery for HDZ/HRZ/HPV/TPV.

    Marker provides the heading source; ``hcom`` provides the position
    anchor (matches MATLAB's HDZ/HRZ/HPV/TPV layout).
    """
    if "hcom" not in xyz.model.markers:
        raise ValueError(
            "head-anchored zone scores require an 'hcom' virtual marker; "
            "use neurobox.analysis.kinematics.augment_xyz first"
        )

    feat_xy, finite_f = _xy_from_xyz(xyz, marker, samplerate, filt_cutoff)
    hcom_xy, finite_h = _xy_from_xyz(xyz, "hcom", samplerate, None)
    finite = finite_f & finite_h
    fs = float(samplerate) if samplerate is not None else xyz.samplerate

    centres = field_centres_from_result(pf, unit_ids)

    T = feat_xy.shape[0]
    n_units = len(unit_ids)
    score = np.zeros((T, n_units), dtype=np.float64)

    if rate_normalised:
        peak = _peak_rates(pf, unit_ids) * max_rate_scale

    for k, uid in enumerate(unit_ids):
        if not np.isfinite(centres[k]).all():
            continue
        # Heading from `feat_xy` (with shift), bearing from `hcom_xy → centre`
        pfds, _ = _bearing_and_heading(hcom_xy, feat_xy, centres[k],
                                        fs, tangent_mode=tangent_mode)
        pfd = _heading_sign(pfds, flipped=False)
        if rate_normalised:
            if not np.isfinite(peak[k]) or peak[k] == 0:
                continue
            j = list(pf.unit_ids).index(uid)
            rate = _rate_at_position(pf, j, feat_xy)
            score[:, k] = pfd * (1.0 - rate / peak[k])
        else:
            _, pfdd = _bearing_and_heading(hcom_xy, feat_xy, centres[k],
                                            fs, tangent_mode=tangent_mode)
            score[:, k] = pfd * pfdd
        score[~finite, k] = 0.0
    return score, centres


def compute_hdz(
    pf:                PlaceFieldResult,
    xyz:               NBDxyz,
    unit_ids:          Sequence[int],
    *,
    marker:            str = "nose",
    samplerate:        float | None = None,
    filt_cutoff:       float = 2.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Head-anchored Distance Zone score.

    Port of :file:`MTA/analysis/placefields/compute_hdz.m`.

    Like DDZ, but the position anchor is ``hcom`` (head-of-mass) and
    the heading source is *marker* (typically ``nose``).
    """
    return _head_anchored_score(
        pf, xyz, unit_ids, marker=marker, samplerate=samplerate,
        filt_cutoff=filt_cutoff, rate_normalised=False,
        tangent_mode="slow", max_rate_scale=1.0,
    )


def compute_hrz(
    pf:                PlaceFieldResult,
    xyz:               NBDxyz,
    unit_ids:          Sequence[int],
    *,
    marker:            str = "nose",
    samplerate:        float | None = None,
    filt_cutoff:       float = 2.5,
    max_rate_scale:    float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Head-anchored Rate Zone score.

    Port of :file:`MTA/analysis/placefields/compute_hrz.m`.

    Like DRZ, but the position anchor is ``hcom`` and the heading
    source is *marker*.
    """
    return _head_anchored_score(
        pf, xyz, unit_ids, marker=marker, samplerate=samplerate,
        filt_cutoff=filt_cutoff, rate_normalised=True,
        tangent_mode="slow", max_rate_scale=max_rate_scale,
    )


def compute_hpv(
    pf:                PlaceFieldResult,
    xyz:               NBDxyz,
    unit_ids:          Sequence[int],
    *,
    marker:            str = "nose",
    samplerate:        float | None = None,
    filt_cutoff:       float = 2.5,
    max_rate_scale:    float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Head-anchored Rate Zone score with **fast** tangent (1-sample shift).

    Port of :file:`MTA/analysis/placefields/compute_hpv.m`.

    Identical algorithm to :func:`compute_hrz` but the heading is
    computed from a 1-sample shift instead of the standard 0.2-second
    shift.  Captures finer-grained heading dynamics; useful when the
    rat is moving fast or when 1-sample tangents are needed for
    instantaneous-velocity decoding.
    """
    return _head_anchored_score(
        pf, xyz, unit_ids, marker=marker, samplerate=samplerate,
        filt_cutoff=filt_cutoff, rate_normalised=True,
        tangent_mode="fast", max_rate_scale=max_rate_scale,
    )


def compute_tpv(
    pf:                PlaceFieldResult,
    xyz:               NBDxyz,
    unit_ids:          Sequence[int],
    *,
    marker:            str = "nose",
    samplerate:        float | None = None,
    filt_cutoff:       float = 2.5,
    max_rate_scale:    float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Tangential per-velocity Zone score.

    Port of :file:`MTA/analysis/placefields/compute_tpv.m`.

    Identical to :func:`compute_hpv` in this port — the MATLAB
    originals are byte-for-byte equivalent except for variable names
    in the docstring.  Both names exposed for API parity.
    """
    return compute_hpv(pf, xyz, unit_ids, marker=marker,
                        samplerate=samplerate, filt_cutoff=filt_cutoff,
                        max_rate_scale=max_rate_scale)
