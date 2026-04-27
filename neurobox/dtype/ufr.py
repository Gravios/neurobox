"""
ufr.py  —  NBDufr
==================
Unit firing-rate time-series at LFP sample rate.

Port of ``MTADufr``.

Data layout: ``(T, N_units)`` — time × unit, in spikes/s.

The signal is computed by accumulating spike counts into a dense array at
``samplerate`` (typically 1250 Hz, matching the LFP) then convolving with
a smoothing kernel:

``gauss``   — Gaussian (default, σ = *window* × *samplerate* samples)
``boxcar``  — Rectangular, width *window* seconds
``count``   — Raw spike counts per sample bin (no smoothing)

Design differences from MTADufr
---------------------------------
* ``create`` / ``compute`` accept an ``NBSpk`` directly rather than using a
  session reference.
* The reference object (for setting sample rate and output length) is
  explicitly an ``NBDlfp`` or float samplerate rather than an opaque RefObj.
* The returned array is spikes/s (divided by the bin width) rather than raw
  convolved counts, so the units are always Hz.

Usage
-----
::

    from neurobox.dtype.ufr import NBDufr

    # From a loaded session
    ufr = NBDufr.compute(
        session.spk,
        samplerate  = session.lfp.samplerate,  # 1250 Hz
        duration_sec = total_time,
        units       = good_unit_ids,
        window      = 0.05,                    # 50 ms Gaussian
    )

    # Spike rate of unit 5 (Hz)
    rate_u5 = ufr[:, ufr.unit_ids.tolist().index(5)]

    # All units during walk state
    walk_rates = ufr[stc["walk"]]              # NBData.__getitem__ epoch select
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from neurobox.dtype.data  import NBData
from neurobox.dtype.epoch import NBEpoch


class NBDufr(NBData):
    """Unit firing-rate time-series at LFP (or any reference) sample rate.

    Parameters
    ----------
    data:
        Array of shape ``(T, N_units)``.  Pass *None* for lazy
        construction via :meth:`compute`.
    unit_ids:
        1-D array of unit global IDs corresponding to columns of *data*.
    samplerate:
        Sample rate in Hz (typically 1250, matching the LFP).
    sync, origin:
        Recording window.
    """

    def __init__(
        self,
        data: np.ndarray | None = None,
        unit_ids: np.ndarray | None = None,
        samplerate: float = 1250.0,
        sync: NBEpoch | None = None,
        origin: float = 0.0,
        path: Path | str | None = None,
        filename: str | None = None,
        name: str = "",
    ) -> None:
        super().__init__(
            path       = path,
            filename   = filename,
            data       = data,
            samplerate = samplerate,
            sync       = sync,
            origin     = origin,
            type_      = "TimeSeries",
            ext        = "ufr",
            name       = name,
            label      = "ufr",
            key        = "u",
        )
        self.unit_ids: np.ndarray = (
            np.asarray(unit_ids, dtype=np.int32)
            if unit_ids is not None
            else np.array([], dtype=np.int32)
        )

    # ------------------------------------------------------------------ #
    # Abstract interface                                                   #
    # ------------------------------------------------------------------ #

    def load(self, *args, **kwargs) -> "NBDufr":      # type: ignore[override]
        """Alias for :meth:`compute`."""
        return self.compute(*args, **kwargs)

    def create(self, *args, **kwargs) -> "NBDufr":    # type: ignore[override]
        """Alias for :meth:`compute`."""
        return self.compute(*args, **kwargs)

    # ------------------------------------------------------------------ #
    # Main computation                                                     #
    # ------------------------------------------------------------------ #

    @classmethod
    def compute(
        cls,
        spk,
        samplerate:   float             = 1250.0,
        duration_sec: float | None      = None,
        units:        "np.ndarray | list[int] | None" = None,
        window:       float             = 0.05,
        mode:         str               = "gauss",
        sync:         "NBEpoch | None"  = None,
    ) -> "NBDufr":
        """Compute firing-rate time-series for selected units.

        Parameters
        ----------
        spk:
            :class:`~neurobox.dtype.spikes.NBSpk` with spike times in
            seconds.
        samplerate:
            Output sample rate in Hz.  Typically the LFP rate (1250 Hz).
        duration_sec:
            Total recording duration in seconds.  Inferred from
            ``spk.res.max()`` when *None*.
        units:
            Unit IDs to include.  ``None`` → all units in ``spk.unit_ids``.
        window:
            Smoothing kernel half-width in seconds.
            For ``'gauss'``  : σ of the Gaussian.
            For ``'boxcar'`` : full width of the box.
            Ignored for ``'count'``.
        mode:
            ``'gauss'``   — Gaussian convolution (default).
            ``'boxcar'``  — rectangular (moving average).
            ``'count'``   — raw spike counts per bin (spikes per sample).
        sync:
            Recording-window epoch attached to the output object.

        Returns
        -------
        NBDufr
            ``data`` shape ``(T, N_units)``, values in spikes/s.
        """
        from scipy.ndimage import gaussian_filter1d, uniform_filter1d

        if len(spk.res) == 0:
            return cls(samplerate=samplerate, sync=sync)

        if duration_sec is None:
            duration_sec = float(spk.res.max())

        n_samples = int(np.ceil(duration_sec * samplerate))

        if units is None:
            units = spk.unit_ids
        units = np.asarray(units, dtype=np.int32)
        n_units = len(units)

        if n_units == 0:
            return cls(samplerate=samplerate, unit_ids=units, sync=sync)

        # ── Convert spike times to sample indices ──────────────────────── #
        # spk.res is in seconds; multiply by samplerate to get sample index
        res_samp = np.round(spk.res * samplerate).astype(np.int64)
        res_samp = np.clip(res_samp, 0, n_samples - 1)

        data = np.zeros((n_samples, n_units), dtype=np.float32)

        for col, uid in enumerate(units):
            mask = spk.clu == uid
            if not mask.any():
                continue
            idx = res_samp[mask]
            np.add.at(data[:, col], idx, 1.0)

        # ── Smooth ────────────────────────────────────────────────────────── #
        if mode == "gauss":
            sigma_samp = window * samplerate       # σ in samples
            for col in range(n_units):
                data[:, col] = gaussian_filter1d(
                    data[:, col].astype(np.float64), sigma=sigma_samp
                ).astype(np.float32)
            # Convert counts → spikes/s
            data = data * samplerate

        elif mode == "boxcar":
            width_samp = max(1, int(round(window * samplerate)))
            for col in range(n_units):
                data[:, col] = uniform_filter1d(
                    data[:, col].astype(np.float64), size=width_samp
                ).astype(np.float32)
            data = data * samplerate

        elif mode == "count":
            # Leave as raw counts per sample bin
            pass

        else:
            raise ValueError(f"Unknown mode {mode!r}. Use 'gauss', 'boxcar', or 'count'.")

        return cls(
            data       = data,
            unit_ids   = units,
            samplerate = samplerate,
            sync       = sync,
        )

    # ------------------------------------------------------------------ #
    # Convenience properties                                               #
    # ------------------------------------------------------------------ #

    @property
    def n_units(self) -> int:
        """Number of units in this firing-rate array."""
        return len(self.unit_ids)

    def unit_column(self, unit_id: int) -> int:
        """Return the column index for *unit_id*.

        Raises ``KeyError`` if the unit is not present.
        """
        matches = np.where(self.unit_ids == unit_id)[0]
        if len(matches) == 0:
            raise KeyError(f"Unit {unit_id} not found in this NBDufr.")
        return int(matches[0])

    def rates_for(self, unit_id: int) -> np.ndarray:
        """Return the firing-rate trace for a single unit (spikes/s).

        Parameters
        ----------
        unit_id:
            Global unit ID.

        Returns
        -------
        np.ndarray, shape ``(T,)``
        """
        if self._data is None:
            raise RuntimeError("Firing rates not computed yet.")
        return self._data[:, self.unit_column(unit_id)]

    # ------------------------------------------------------------------ #
    # Repr                                                                 #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        shape = self.shape if self._data is not None else "not computed"
        return (
            f"NBDufr(shape={shape}, n_units={self.n_units}, "
            f"sr={self.samplerate}Hz)"
        )
