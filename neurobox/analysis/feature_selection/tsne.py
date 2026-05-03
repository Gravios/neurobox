"""
neurobox.analysis.feature_selection.tsne
==========================================
t-SNE wrapper with state-aware sub-sampling and colour mapping.

Port of :file:`MTA/analysis/mta_tsne.m` on top of
:class:`sklearn.manifold.TSNE`.

What it does
------------
1. Selects samples that are inside any of the listed states.
2. Down-samples by a stride for tractability (MATLAB default skip=2).
3. Runs t-SNE.
4. Returns the embedding with a per-sample state label, so callers
   can colour-code the scatter plot.

Differences from MATLAB
-----------------------
* Uses :class:`sklearn.manifold.TSNE` instead of MATLAB's :func:`tsne`
  (which was Laurens van der Maaten's reference C-mex
  implementation).  The two should agree closely on the
  large-perplexity regime used by the lab (default 100); for fine
  cluster structure they may differ.
* MATLAB's ``initDims=5`` (PCA preconditioning) is replaced by
  scikit-learn's default ``init='pca'``.
* The disk-cache is replaced by an in-memory return.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


__all__ = [
    "TSNEResult",
    "mta_tsne",
]


@dataclass
class TSNEResult:
    """Output of :func:`mta_tsne`.

    Attributes
    ----------
    embedding : np.ndarray, shape ``(n_samples_kept, n_dims)``
        t-SNE 2-D (or n-D) coordinates of each kept sample.
    sample_indices : np.ndarray, shape ``(n_samples_kept,)``
        Original time-axis indices of the kept samples (so the caller
        can map back to the recording).
    state_labels : np.ndarray, shape ``(n_samples_kept,)``
        Integer label per sample.  ``0`` = no state; ``1..n_states``
        = state index in the input *states* list.
    state_names : tuple[str, ...]
        Same as the input *states* argument; preserved for plotting.
    perplexity : float
    init_dims : int
    """
    embedding:      np.ndarray
    sample_indices: np.ndarray
    state_labels:   np.ndarray
    state_names:    tuple[str, ...]
    perplexity:     float
    init_dims:      int


def mta_tsne(
    feature_data:        np.ndarray,
    feature_samplerate:  float,
    *,
    state_labels:        Optional[np.ndarray] = None,
    state_names:         Sequence[str] = (
        "rear", "walk", "turn", "pause", "groom", "sit",
    ),
    n_dims:              int = 2,
    init_dims:           int = 5,
    perplexity:          float = 100.0,
    skip:                int = 2,
    subset:              Optional[tuple[int, int]] = None,
    rng:                 Optional[np.random.Generator | int] = None,
    progress:            bool = False,
) -> TSNEResult:
    """Run t-SNE on a feature matrix with state-aware sub-sampling.

    Port of :file:`MTA/analysis/mta_tsne.m`.

    Parameters
    ----------
    feature_data:
        ``(T, n_features)`` feature matrix.
    feature_samplerate:
        Hz of *feature_data*.  Stored on the result for downstream use.
    state_labels:
        ``(T,)`` integer state codes (``0`` = no state, ``1..K`` =
        state).  Used to drop samples that aren't in any of *state_names*.
        Pass ``None`` to skip filtering.
    state_names:
        Names corresponding to the integer codes 1..K.  Preserved on
        the result for plotting.
    n_dims:
        Output dimensionality.  Default 2.
    init_dims:
        PCA preconditioning components.  Default 5 matches MATLAB.
        Note: scikit-learn doesn't expose this directly; we apply PCA
        ourselves first when ``init_dims < n_features``.
    perplexity:
        t-SNE perplexity.  Default 100 matches MATLAB.
    skip:
        Sample stride.  Default 2 matches MATLAB (every 2nd sample).
    subset:
        ``(t_start, t_end)`` window in samples to restrict to.  Default
        ``None`` → full recording.
    rng:
        Random seed for reproducibility.
    progress:
        Print scikit-learn t-SNE progress.

    Returns
    -------
    TSNEResult
    """
    try:
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
    except ImportError as e:                 # pragma: no cover
        raise ImportError(
            "mta_tsne requires scikit-learn.  Install with "
            "`pip install 'neurobox[full]'` (or `pip install scikit-learn`)."
        ) from e

    feature_data = np.asarray(feature_data, dtype=np.float64)
    if feature_data.ndim != 2:
        raise ValueError(
            f"feature_data must be (T, n_features); got {feature_data.shape}"
        )
    T, n_feat = feature_data.shape

    # Build inclusion mask
    finite = np.isfinite(feature_data).all(axis=1)
    if state_labels is not None:
        sl = np.asarray(state_labels)
        if sl.shape != (T,):
            raise ValueError(
                f"state_labels shape {sl.shape} ≠ T={T}"
            )
        in_state = sl > 0
    else:
        sl = np.zeros(T, dtype=np.int32)
        in_state = np.ones(T, dtype=bool)

    if subset is not None:
        t0, t1 = subset
        in_window = np.zeros(T, dtype=bool)
        in_window[t0:t1] = True
    else:
        in_window = np.ones(T, dtype=bool)

    stride = np.zeros(T, dtype=bool)
    stride[::skip] = True

    keep = finite & in_state & in_window & stride
    if not keep.any():
        raise ValueError("no samples remain after filtering")

    keep_idx = np.flatnonzero(keep)
    fet_kept = feature_data[keep, :]
    labels_kept = sl[keep]

    seed = (rng if isinstance(rng, int)
            else (rng.integers(0, 2**31 - 1) if rng is not None
                   else None))

    # PCA preconditioning if requested
    if init_dims is not None and 0 < init_dims < n_feat:
        pca = PCA(n_components=init_dims, random_state=seed)
        fet_kept = pca.fit_transform(fet_kept)

    # Cap perplexity to a sane max relative to n samples
    eff_perp = float(min(perplexity, max(5.0, fet_kept.shape[0] / 4 - 1)))

    tsne = TSNE(
        n_components = n_dims,
        perplexity   = eff_perp,
        random_state = seed,
        init         = "pca",
        verbose      = 1 if progress else 0,
    )
    embedding = tsne.fit_transform(fet_kept)

    return TSNEResult(
        embedding      = embedding,
        sample_indices = keep_idx,
        state_labels   = labels_kept,
        state_names    = tuple(state_names),
        perplexity     = eff_perp,
        init_dims      = init_dims,
    )
