"""Tests for the training-cycle repairs (audit findings 1–3).

1. Reproducibility: a fixed ``rng`` on ``train_classifier_ensemble``
   now fixes the ENTIRE run — bootstrap draws, weight init, batch
   order — for torch and sklearn backends alike.
2. Epoch-unit contract: ``label_with_heuristics`` emits seconds (the
   NBEpoch contract) so its output composes with ``stc2mat`` and
   ``whole_state_bootstrap``; the bootstrap loudly rejects periods
   that cannot be seconds.
3. Missing-state protection: a state absent from every training
   session raises; a state absent from only some sessions warns and
   still yields full-width ``predict_proba``; ``fit(...,
   n_classes=S)`` pins output width even when trailing classes are
   absent from ``y``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from neurobox.dtype import NBEpoch, NBStateCollection, NBDxyz, NBModel
from neurobox.analysis.classifiers import (
    train_classifier_ensemble, whole_state_bootstrap, make_classifier,
    labels_from_stc, label_with_heuristics,
)
from neurobox.analysis.kinematics.augment import augment_xyz


FS = 10.0
STATES = ("walk", "rear")


def _session(seed=0, T=1200, gap=False):
    """Alternating 10 s bouts; state-separable Gaussian features.

    With ``gap=True`` the 'rear' state has no periods (session-level
    absence)."""
    rng = np.random.default_rng(seed)
    y = (np.arange(T) // 100) % 2
    if gap:
        y = np.zeros(T, dtype=int)
    X = np.where(y[:, None] == np.arange(2)[None, :], 2.0, 0.0)
    X = X + rng.normal(0, 1.0, size=(T, 2))
    stc = NBStateCollection()
    for i, s in enumerate(STATES):
        if gap and i == 1:
            segs = np.zeros((0, 2))
        else:
            segs = np.array([[a / FS, (a + 100) / FS]
                             for a in range(i * 100, T, 200)], float)
        stc.add_state(NBEpoch(segs, samplerate=1.0, label=s, key=s[0]))
    return stc, X, y


# ─────────────────────────────────────────────────────────────────────── #
# Finding 1 — reproducibility                                             #
# ─────────────────────────────────────────────────────────────────────── #

class TestReproducibility:
    def _train_probs(self, backend, rng_seed, **ckw):
        stc, X, _ = _session()
        ens = train_classifier_ensemble(
            [(stc, X, FS)], STATES, backend=backend, n_iter=2,
            classifier_kwargs=ckw,
            rng=np.random.default_rng(rng_seed))
        return ens.predict_proba(X[:200])

    def test_torch_same_rng_identical(self):
        a = self._train_probs("patternnet", 42, epochs=8)
        b = self._train_probs("patternnet", 42, epochs=8)
        np.testing.assert_array_equal(a, b)

    def test_torch_different_rng_differs(self):
        a = self._train_probs("patternnet", 42, epochs=8)
        b = self._train_probs("patternnet", 43, epochs=8)
        assert not np.allclose(a, b)

    def test_sklearn_same_rng_identical(self):
        a = self._train_probs("rf", 7, n_estimators=20)
        b = self._train_probs("rf", 7, n_estimators=20)
        np.testing.assert_array_equal(a, b)

    def test_explicit_seed_wins_over_rng(self):
        """A caller-pinned seed in classifier_kwargs suppresses
        injection, so all members share it regardless of rng."""
        stc, X, _ = _session()
        outs = []
        for rng_seed in (1, 2):
            ens = train_classifier_ensemble(
                [(stc, X, FS)], STATES, backend="patternnet", n_iter=1,
                classifier_kwargs=dict(epochs=8, seed=99),
                bootstrap_kwargs=dict(rng=np.random.default_rng(0)),
                rng=np.random.default_rng(rng_seed))
            outs.append(ens.predict_proba(X[:100]))
        np.testing.assert_array_equal(outs[0], outs[1])

    def test_backend_seed_kwarg_direct(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(300, 2)); y = (X[:, 0] > 0).astype(int)
        a = make_classifier("patternnet", epochs=8, seed=5).fit(X, y)
        b = make_classifier("patternnet", epochs=8, seed=5).fit(X, y)
        np.testing.assert_array_equal(
            a.predict_proba(X[:50]), b.predict_proba(X[:50]))


# ─────────────────────────────────────────────────────────────────────── #
# Finding 2 — epoch-unit contract                                         #
# ─────────────────────────────────────────────────────────────────────── #

MARKERS = ["spine_lower", "pelvis_root", "spine_middle", "spine_upper",
           "head_back", "head_left", "head_front", "head_right"]


def _walking_xyz(seed=1, T=120 * 60, fs=120.0):
    rng = np.random.default_rng(seed)
    t = np.arange(T) / fs
    speed = np.zeros(T)
    speed[(t > 10) & (t < 20)] = 6.0
    speed[(t > 35) & (t < 45)] = 6.0
    path = np.cumsum(np.stack([speed, speed * 0.3], 1), 0)
    base = np.concatenate([path, np.zeros((T, 1))], 1)[:, None, :]
    off = (np.linspace(0, 140, len(MARKERS))[None, :, None]
           * np.array([1., 0., 0.]))
    data = base + off + rng.normal(0, 1.2, size=(T, len(MARKERS), 3))
    data[:, :, 2] += 60.0
    return augment_xyz(NBDxyz(data, model=NBModel(MARKERS), samplerate=fs))


class TestEpochUnitContract:
    def test_stage1_emits_seconds(self):
        stc = label_with_heuristics(_walking_xyz())
        w = stc["walk"].data
        assert w.dtype == np.float64
        # bouts at 10–20 s and 35–45 s: values must be tens of
        # seconds, not thousands of samples
        assert w.max() < 60.0
        assert w[0][0] == pytest.approx(10.0, abs=0.6)

    def test_stage1_composes_with_stc2mat(self):
        """labels_from_stc (which trusts the seconds contract via
        to_mask) must rasterise stage-1 output correctly."""
        xyz = _walking_xyz()
        stc = label_with_heuristics(xyz)
        T = xyz.data.shape[0]
        lab = labels_from_stc(stc, ("walk",), T, 120.0)
        frac = (lab[int(12 * 120):int(18 * 120)] == 0).mean()
        assert frac > 0.95            # inside a detected bout
        assert (lab[: int(8 * 120)] == -1).all()

    def test_stage1_composes_with_bootstrap(self):
        """The audit probe, inverted into a regression test: rows
        drawn for 'walk' must come from walking frames."""
        xyz = _walking_xyz()
        stc = label_with_heuristics(xyz)
        T = xyz.data.shape[0]
        rng = np.random.default_rng(0)
        # 1-D surrogate feature: 2.0 while walking, 0.0 otherwise
        t = np.arange(T) / 120.0
        walkmask = ((t > 10) & (t < 20)) | ((t > 35) & (t < 45))
        X = np.where(walkmask[:, None], 2.0, 0.0) + rng.normal(
            0, 0.1, size=(T, 1))
        res = whole_state_bootstrap(
            stc, X, 120.0, states=("walk",), rng=rng,
            state_block_size=2000)
        assert res.features[res.labels == 0][:, 0].mean() > 1.8

    def test_bootstrap_rejects_sample_index_periods(self):
        """Periods that begin beyond the feature timeline cannot be
        seconds — the old behaviour drew 15 000 silent garbage rows."""
        stc = NBStateCollection()
        segs = np.array([[a * 12, (a + 100) * 12]
                         for a in range(0, 1200, 200)], float)  # samples
        stc.add_state(NBEpoch(segs, samplerate=120.0,
                              label="walk", key="w"))
        X = np.zeros((1200, 2))
        with pytest.raises(ValueError, match="must be in seconds"):
            whole_state_bootstrap(stc, X, FS, states=("walk",),
                                  rng=np.random.default_rng(0))

    def test_bootstrap_seconds_still_fine(self):
        stc, X, _ = _session()
        res = whole_state_bootstrap(stc, X, FS, states=STATES,
                                    rng=np.random.default_rng(0))
        walk = res.features[res.labels == 0]
        assert walk[:, 0].mean() > 1.5


# ─────────────────────────────────────────────────────────────────────── #
# Finding 3 — missing-state protection                                    #
# ─────────────────────────────────────────────────────────────────────── #

class TestMissingStateProtection:
    def test_globally_missing_state_raises(self):
        stc, X, _ = _session(gap=True)          # no 'rear' anywhere
        with pytest.raises(ValueError, match=r"\['rear'\].*no\s+labelled"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                train_classifier_ensemble(
                    [(stc, X, FS)], STATES, backend="patternnet",
                    n_iter=1, classifier_kwargs=dict(epochs=2),
                    rng=np.random.default_rng(0))

    def test_session_level_gap_warns_but_trains_full_width(self):
        full = _session(seed=0)
        gappy = _session(seed=1, gap=True)
        with pytest.warns(UserWarning, match="no periods"):
            ens = train_classifier_ensemble(
                [(full[0], full[1], FS), (gappy[0], gappy[1], FS)],
                STATES, backend="patternnet", n_iter=1,
                classifier_kwargs=dict(epochs=4),
                rng=np.random.default_rng(0))
        p = ens.predict_proba(full[1][:50])
        assert p.shape == (50, len(STATES))

    @pytest.mark.parametrize("backend,kw", [
        ("patternnet", dict(epochs=3)),
        ("rf",         dict(n_estimators=10)),
    ])
    def test_explicit_n_classes_pins_width(self, backend, kw):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 2))
        y = np.zeros(200, dtype=int)            # only class 0 present
        clf = make_classifier(backend, seed=1, **kw).fit(X, y, n_classes=3)
        assert clf.predict_proba(X[:10]).shape == (10, 3)

    def test_n_classes_smaller_than_labels_raises(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(100, 2))
        y = np.array([0, 1, 2, 3] * 25)
        with pytest.raises(ValueError, match="out of range"):
            make_classifier("patternnet", epochs=2).fit(X, y, n_classes=2)
