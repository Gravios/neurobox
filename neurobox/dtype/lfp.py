"""
lfp.py  —  NBDlfp
==================
Port of MTADlfp.  Local field potential data.

Data layout: ``(T, N_channels)`` — time × channel.

Design differences from MTADlfp
---------------------------------
* Loading delegates to ``neurobox.io.load_binary`` via
  ``NBDlfp.load()``.  Lazy: data stays ``None`` until ``load()`` is
  called.
* ``channels`` stores the 0-based hardware channel indices that were
  actually loaded, so ``data[:, i]`` always refers to
  ``channels[i]``.
* CSD uses the standard finite-difference second derivative:
  ``csd[:, i] = (v[i] + v[i+2] - 2*v[i+1]) / (2 * d * dx)²``
  (Nicholson & Freeman 1975).
* ``filter()`` inherited from NBData (Butterworth via
  ``scipy.signal.sosfiltfilt``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from neurobox.dtype.data import NBData
from neurobox.dtype.epoch import NBEpoch


class NBDlfp(NBData):
    """Local field potential time-series.

    Parameters
    ----------
    data:
        Array of shape ``(T, N_channels)``.  Pass *None* for lazy
        loading.
    channels:
        0-based hardware channel indices corresponding to each column
        of *data*.
    samplerate:
        LFP sample rate in Hz (typically 1 250 Hz for Neurosuite
        ``.lfp`` files, or 20 000 Hz for ``.dat``).
    uv_per_bit:
        ADC conversion factor.  If not None, raw int16 data is
        multiplied by this to get µV.  Default None (keep raw int16).
    sync:
        NBEpoch defining the valid recording window.
    path / filename:
        Location of the binary source file for lazy loading.
    ext:
        ``'lfp'`` (1 250 Hz downsampled) or ``'dat'`` (wideband).
    """

    def __init__(
        self,
        data: np.ndarray | None = None,
        channels: list[int] | None = None,
        samplerate: float = 1250.0,
        uv_per_bit: float | None = None,
        sync: NBEpoch | None = None,
        origin: float = 0.0,
        path: Path | str | None = None,
        filename: str | None = None,
        name: str = "",
        ext: str = "lfp",
    ) -> None:
        super().__init__(
            path=path, filename=filename, data=data,
            samplerate=samplerate, sync=sync, origin=origin,
            type_="TimeSeries", ext=ext, name=name, label="lfp", key="l",
        )
        self.channels: list[int] = channels or []
        self.uv_per_bit: float | None = uv_per_bit

    # ------------------------------------------------------------------ #
    # Abstract interface                                                   #
    # ------------------------------------------------------------------ #

    def load(
        self,
        file_base: str | Path | None = None,
        channels: list[int] | None = None,
        periods: np.ndarray | None = None,
        par=None,
    ) -> "NBDlfp":
        """Load LFP data from the binary source file.

        Parameters
        ----------
        file_base:
            Session base path.  If None, ``self.fpath`` is used.
        channels:
            0-based channel list.  If None, use ``self.channels`` (or
            all channels if that is also empty).
        periods:
            ``(N, 2)`` array of ``[start_sample, end_sample)`` intervals.
            If None, load the entire file.
        par:
            Parsed parameter object.  Auto-loaded from the .yaml sidecar
            if not provided.
        """
        from neurobox.io import load_binary, load_par

        source = Path(file_base) if file_base is not None else self.fpath
        if source is None:
            raise ValueError("No source file specified.")

        # Determine suffix
        suffix = self.ext  # 'lfp' or 'dat'
        if not source.suffix:
            source = source.with_suffix(f".{suffix}")

        if par is None:
            par = load_par(str(source.with_suffix("")))

        if channels is None:
            channels = self.channels or list(range(int(par.acquisitionSystem.nChannels)))

        # Set sample rate from par
        if self.ext == "lfp":
            from neurobox.io.load_yaml import get_lfp_samplerate
            self.samplerate = get_lfp_samplerate(par, default=self.samplerate)
        else:
            try:
                self.samplerate = float(par.acquisitionSystem.samplingRate)
            except Exception:
                pass

        raw = load_binary(
            source,
            channels   = channels,
            par        = par,
            periods    = periods,
            uv_per_bit = self.uv_per_bit,
            channel_first = False,   # → (T, C)
        )
        self._data    = raw
        self.channels = channels
        return self

    def create(self, *args, **kwargs) -> "NBDlfp":
        """Alias for load()."""
        return self.load(*args, **kwargs)

    # ------------------------------------------------------------------ #
    # Current source density                                               #
    # ------------------------------------------------------------------ #

    def csd(
        self,
        channel_interval: int = 1,
        channel_pitch_um: float = 50.0,
    ) -> "NBDlfp":
        """Compute the current source density (CSD).

        Uses the standard central-difference approximation
        (Nicholson & Freeman, 1975):

            CSD[:, i] = (v[i] + v[i+2*h] - 2*v[i+h]) / (2*h*dx)^2

        where *h* = channel_interval and *dx* = channel_pitch_um (µm).

        Parameters
        ----------
        channel_interval:
            Step size between channels for the second spatial
            derivative (default 1 = adjacent channels).
        channel_pitch_um:
            Physical distance between electrode contacts in µm.

        Returns
        -------
        NBDlfp
            A new NBDlfp object whose data contains the CSD.  The
            number of channels is reduced by ``2 * channel_interval``.
        """
        if self._data is None:
            raise RuntimeError("LFP data not loaded.")
        h  = channel_interval
        dx = channel_pitch_um
        v  = self._data.astype(np.float64)   # (T, C)
        n_out = v.shape[1] - 2 * h
        if n_out <= 0:
            raise ValueError("Not enough channels for CSD with this interval.")
        csd_data = np.empty((v.shape[0], n_out), dtype=np.float64)
        for i in range(n_out):
            csd_data[:, i] = (
                (v[:, i] + v[:, i + 2 * h] - 2.0 * v[:, i + h])
                / (2.0 * h * dx) ** 2
            )
        new_channels = (self.channels[h:-h]
                        if len(self.channels) >= v.shape[1]
                        else list(range(n_out)))
        return NBDlfp(
            data       = csd_data,
            channels   = new_channels,
            samplerate = self.samplerate,
            sync       = self.sync,
            name       = self.name,
            ext        = self.ext,
        )

    # ------------------------------------------------------------------ #
    # Repr                                                                 #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        shape = self.shape if self._data is not None else "not loaded"
        return (f"NBDlfp(shape={shape}, channels={self.channels}, "
                f"sr={self.samplerate}Hz, ext={self.ext!r})")
