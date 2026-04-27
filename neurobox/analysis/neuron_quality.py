"""
neuron_quality.py
=================
Per-unit isolation and waveform quality metrics.

Port of MTA's ``NeuronQuality.m``.

Metrics
-------
isi_contamination : float
    Fraction of inter-spike intervals shorter than *refractory_ms* ms
    (the "refractory period violation" rate).  Values near 0 indicate
    well-isolated single units; > 0.05 suggests multi-unit activity.

snr : float
    Waveform signal-to-noise ratio, defined as::

        SNR = peak_amplitude / (2 × noise_std)

    where ``peak_amplitude = max(mean_wf) - min(mean_wf)`` across all
    channels and ``noise_std`` is estimated from the MAD of the raw
    spike waveform distribution.

spike_width_ms : float
    Trough-to-peak width of the mean waveform in milliseconds, measured
    on the channel with the largest amplitude.  Narrow-spiking
    (< 0.35 ms) units are putative interneurons; wide-spiking
    (> 0.45 ms) are putative pyramidal cells.

spike_width_left_ms : float
    Half-width to the left of the trough (trough to baseline crossing).

spike_width_right_ms : float
    Half-width to the right of the trough (trough to peak).

time_symmetry : float
    ``spike_width_right_ms / spike_width_left_ms``.  Values > 1 indicate
    a rightward-skewed waveform (typical of pyramidal cells).

n_spikes : int
    Total spike count for this unit.

mean_firing_rate : float
    Mean firing rate in Hz computed over the supplied recording duration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class NeuronQualityResult:
    """Quality metrics for one cluster.

    Attributes
    ----------
    unit_id : int
        Global cluster ID.
    shank : int | None
        Shank index (1-based), or None if unknown.
    n_spikes : int
    isi_contamination : float
        Fraction of ISIs < *refractory_ms*.
    snr : float | None
        Waveform SNR (None when waveforms not provided).
    spike_width_ms : float | None
        Trough-to-peak width in ms.
    spike_width_left_ms : float | None
    spike_width_right_ms : float | None
    time_symmetry : float | None
    mean_firing_rate : float | None
        Hz, requires *duration_sec* to be provided.
    """
    unit_id:               int
    shank:                 int | None  = None
    n_spikes:              int         = 0
    isi_contamination:     float       = float("nan")
    snr:                   float | None = None
    spike_width_ms:        float | None = None
    spike_width_left_ms:   float | None = None
    spike_width_right_ms:  float | None = None
    time_symmetry:         float | None = None
    mean_firing_rate:      float | None = None
    yaml_quality:          str | None   = None
    yaml_cell_type:        str | None   = None
    yaml_structure:        str | None   = None
    yaml_isolation_distance: float | None = None

    def is_single_unit(
        self,
        max_isi_contamination: float = 0.02,
        min_snr:               float = 2.0,
        min_spikes:            int   = 100,
        good_quality_tags: frozenset[str] | None = None,
    ) -> bool:
        """Return True if this unit passes basic single-unit criteria.

        If ``yaml_quality`` is set and *good_quality_tags* is provided (or
        a default set is used), the YAML tag is required to be in the
        accepted set — it serves as a hard veto.
        """
        if self.n_spikes < min_spikes:
            return False
        if self.isi_contamination > max_isi_contamination:
            return False
        if self.snr is not None and self.snr < min_snr:
            return False
        # If manual quality is available, require it to be a single-unit tag
        if self.yaml_quality is not None:
            tags = good_quality_tags or frozenset(
                {"good", "great", "excellent", "su", "single_unit",
                 "single-unit", "accepted"}
            )
            if self.yaml_quality.strip().lower() not in tags:
                return False
        return True

    def __repr__(self) -> str:
        parts = [f"unit={self.unit_id}"]
        if self.shank is not None:
            parts.append(f"sh={self.shank}")
        parts.append(f"n={self.n_spikes}")
        parts.append(f"ISI={self.isi_contamination:.3f}")
        if self.snr is not None:
            parts.append(f"SNR={self.snr:.1f}")
        if self.spike_width_ms is not None:
            parts.append(f"w={self.spike_width_ms:.3f}ms")
        if self.yaml_quality is not None:
            parts.append(f"q={self.yaml_quality!r}")
        return f"NQR({', '.join(parts)})"


# ---------------------------------------------------------------------------
# ISI contamination
# ---------------------------------------------------------------------------

def _isi_contamination(
    spike_times_sec: np.ndarray,
    refractory_ms:   float = 2.0,
) -> float:
    """Fraction of ISIs shorter than *refractory_ms* ms.

    Only considers ISIs between consecutive spikes — cross-unit
    refractory violations are not penalised here.
    """
    if len(spike_times_sec) < 2:
        return 0.0
    isis = np.diff(np.sort(spike_times_sec)) * 1000.0   # → ms
    return float((isis < refractory_ms).mean())


# ---------------------------------------------------------------------------
# Waveform metrics
# ---------------------------------------------------------------------------

def _waveform_metrics(
    waveforms:   np.ndarray,
    samplerate:  float,
) -> dict:
    """Compute SNR and spike width from per-spike waveform snippets.

    Parameters
    ----------
    waveforms : np.ndarray, shape ``(n_spikes, n_samples, n_channels)``
        Raw int16 or float waveform snippets.
    samplerate : float
        Wideband recording rate (Hz).

    Returns
    -------
    dict with keys:
        snr, spike_width_ms, spike_width_left_ms,
        spike_width_right_ms, time_symmetry
    """
    result = dict(
        snr                = None,
        spike_width_ms     = None,
        spike_width_left_ms  = None,
        spike_width_right_ms = None,
        time_symmetry      = None,
    )

    if waveforms is None or waveforms.size == 0:
        return result

    wf = waveforms.astype(np.float64)

    # Mean waveform per channel
    mean_wf = wf.mean(axis=0)   # (n_samples, n_channels)

    # ── Channel with largest amplitude ────────────────────────────────── #
    amp_per_ch = mean_wf.max(axis=0) - mean_wf.min(axis=0)
    best_ch    = int(amp_per_ch.argmax())
    mwf        = mean_wf[:, best_ch]   # (n_samples,)

    dt_ms      = 1000.0 / samplerate

    # ── SNR ───────────────────────────────────────────────────────────── #
    # Peak amplitude / (2 × robust noise estimate)
    peak_amp  = mwf.max() - mwf.min()
    noise_std = np.median(np.abs(wf - wf.mean())) / 0.6745   # MAD → σ
    if noise_std > 0:
        result["snr"] = float(peak_amp / (2.0 * noise_std))

    # ── Trough location ───────────────────────────────────────────────── #
    trough_idx = int(mwf.argmin())

    # ── Trough-to-peak width ──────────────────────────────────────────── #
    # Peak is the maximum AFTER the trough
    post_trough = mwf[trough_idx:]
    if len(post_trough) < 2:
        return result
    peak_after_trough = int(post_trough.argmax()) + trough_idx
    result["spike_width_ms"] = float((peak_after_trough - trough_idx) * dt_ms)

    # ── Half-width left of trough ─────────────────────────────────────── #
    # Find where the waveform crosses half the trough amplitude on the left
    half_amp    = mwf[trough_idx] / 2.0    # (trough value is negative)
    pre_trough  = mwf[:trough_idx]
    left_cross  = np.where(pre_trough >= half_amp)[0]
    if len(left_cross):
        result["spike_width_left_ms"] = float((trough_idx - left_cross[-1]) * dt_ms)

    # ── Half-width right of trough ────────────────────────────────────── #
    post_trough = mwf[trough_idx:]
    right_cross = np.where(post_trough >= half_amp)[0]
    if len(right_cross) > 1:
        result["spike_width_right_ms"] = float((right_cross[1] - 0) * dt_ms)

    # ── Time symmetry ─────────────────────────────────────────────────── #
    if (result["spike_width_left_ms"] is not None
            and result["spike_width_right_ms"] is not None
            and result["spike_width_left_ms"] > 0):
        result["time_symmetry"] = float(
            result["spike_width_right_ms"] / result["spike_width_left_ms"]
        )

    return result


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def neuron_quality(
    session_or_spk,
    refractory_ms:  float = 2.0,
    duration_sec:   float | None = None,
    include_noise:  bool  = False,
    save:           bool  = True,
    overwrite:      bool  = False,
) -> dict[int, NeuronQualityResult]:
    """Compute per-unit quality metrics for all clusters in a session.

    Port of MTA's ``NeuronQuality.m``.

    Parameters
    ----------
    session_or_spk:
        Either an ``NBSession`` object (quality is computed from
        ``session.spk`` and ``session.spk.spk`` waveforms) or a bare
        ``NBSpk`` object.
    refractory_ms:
        Refractory period threshold for ISI contamination (default 2 ms).
    duration_sec:
        Recording duration used to compute mean firing rate.  Read from
        ``session.sync`` when *session_or_spk* is an NBSession.
    include_noise:
        Include clusters 0 (noise) and 1 (MUA) in the output.
    save:
        Save the results to ``session.spath/<name>.NeuronQuality.npy``
        (only when *session_or_spk* is an NBSession).
    overwrite:
        Recompute and overwrite an existing .NeuronQuality.npy file.

    Returns
    -------
    dict[int, NeuronQualityResult]
        Maps each unit ID to its quality metrics.

    Examples
    --------
    >>> from neurobox.analysis import neuron_quality
    >>> nq = neuron_quality(session)
    >>> good_units = [uid for uid, r in nq.items() if r.is_single_unit()]
    >>> print(f"{len(good_units)} / {len(nq)} units pass quality criteria")
    """
    from neurobox.dtype.spikes import NBSpk

    # ── Resolve NBSpk and metadata from input ─────────────────────────── #
    is_session = not isinstance(session_or_spk, NBSpk)

    if is_session:
        session = session_or_spk
        spk = getattr(session, "spk", None)
        if spk is None:
            spk = session.load("spk")

        if duration_sec is None and hasattr(session, "sync") and session.sync is not None:
            try:
                duration_sec = float(
                    session.sync.data[-1, 1] - session.sync.data[0, 0]
                )
            except Exception:
                pass

        # ── Check for cached result ───────────────────────────────────── #
        if save and hasattr(session, "spath"):
            nq_file = Path(session.spath) / f"{session.name}.NeuronQuality.npy"
            if nq_file.exists() and not overwrite:
                cached = np.load(nq_file, allow_pickle=True).item()
                if isinstance(cached, dict):
                    return cached
    else:
        spk = session_or_spk
        session = None
        nq_file = None

    if spk is None or len(spk) == 0:
        return {}

    samplerate = spk.samplerate
    results: dict[int, NeuronQualityResult] = {}

    for uid in np.unique(spk.clu):
        if not include_noise and int(uid) <= 1:
            continue

        mask        = spk.clu == uid
        spike_times = spk.res[mask]
        n_spikes    = int(mask.sum())

        # Shank
        shank = spk.shank_for_unit(int(uid))

        # ISI contamination
        isi = _isi_contamination(spike_times, refractory_ms)

        # Waveform metrics
        wf_metrics = {}
        if spk.spk is not None:
            wf_metrics = _waveform_metrics(spk.spk[mask], samplerate)

        # Mean firing rate
        mfr = (float(n_spikes / duration_sec)
               if duration_sec and duration_sec > 0 else None)

        # Merge YAML curation annotation if available
        ann = spk.annotation_for(int(uid)) if hasattr(spk, 'annotation_for') else None

        results[int(uid)] = NeuronQualityResult(
            unit_id                  = int(uid),
            shank                    = shank,
            n_spikes                 = n_spikes,
            isi_contamination        = isi,
            mean_firing_rate         = mfr,
            yaml_quality             = ann.quality             if ann else None,
            yaml_cell_type           = ann.cell_type           if ann else None,
            yaml_structure           = ann.structure           if ann else None,
            yaml_isolation_distance  = ann.isolation_distance  if ann else None,
            **wf_metrics,
        )

    # ── Save ──────────────────────────────────────────────────────────── #
    if save and is_session and nq_file is not None:
        np.save(str(nq_file), results)
        print(f"  NeuronQuality saved → {nq_file.name}")

    return results


def print_neuron_quality_report(
    nq:          dict[int, NeuronQualityResult],
    max_isi:     float = 0.02,
    min_snr:     float = 2.0,
    min_spikes:  int   = 100,
) -> None:
    """Print a formatted quality report to stdout."""
    if not nq:
        print("  (no units)")
        return

    has_yaml = any(r.yaml_quality is not None for r in nq.values())
    q_col    = f"  {'Quality':>9}" if has_yaml else ""
    header   = (f"  {'Unit':>5}  {'Shank':>5}  {'N spk':>6}  "
                f"{'ISI%':>6}  {'SNR':>6}  {'W(ms)':>6}  {'FR(Hz)':>6}"
                + q_col + "  OK")
    print(header)
    print("  " + "-" * (len(header) - 2))

    n_ok = 0
    for uid, r in sorted(nq.items()):
        ok   = r.is_single_unit(max_isi, min_snr, min_spikes)
        n_ok += int(ok)
        snr_s = f"{r.snr:.1f}"  if r.snr  is not None else "  N/A"
        w_s   = f"{r.spike_width_ms:.3f}" if r.spike_width_ms is not None else "  N/A"
        fr_s  = f"{r.mean_firing_rate:.1f}" if r.mean_firing_rate is not None else "  N/A"
        q_s   = f"  {(r.yaml_quality or '-'):>9}" if has_yaml else ""
        print(f"  {uid:5d}  {str(r.shank or '-'):>5}  {r.n_spikes:6d}  "
              f"{r.isi_contamination*100:5.1f}%  {snr_s:>6}  {w_s:>6}  "
              f"{fr_s:>6}{q_s}  {'✓' if ok else '✗'}")

    print(f"\n  {n_ok}/{len(nq)} units pass quality criteria "
          f"(ISI<{max_isi*100:.0f}%"
          + (f", SNR>{min_snr}" if min_snr else "")
          + f", n>{min_spikes})")
