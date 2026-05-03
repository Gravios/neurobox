"""Tests for round-12 — behavioural-state classifiers."""

from __future__ import annotations

import os
import tempfile
from itertools import groupby

import numpy as np
import pytest

from neurobox.dtype import NBStateCollection, NBEpoch


# ─────────────────────────────────────────────────────────────────────── #
# Imports gated on optional deps                                            #
# ─────────────────────────────────────────────────────────────────────── #

# Skip the entire module if sklearn isn't available (it's the lighter dep)
sklearn = pytest.importorskip("sklearn")
torch = pytest.importorskip("torch")


from neurobox.analysis.classifiers import (
    Classifier, FitInfo,
    whole_state_bootstrap, BootstrapResult,
    make_classifier,
    train_classifier_ensemble,
    smooth_labels_to_state_collection,
    label_states,
    TrainedEnsemble,
    FeatureNormalisation, fit_normalisation,
)


# ─────────────────────────────────────────────────────────────────────── #
# Fixtures                                                                  #
# ─────────────────────────────────────────────────────────────────────── #

@pytest.fixture
def synth_session():
    """3-state, 2-feature synthetic session with clearly-separable centroids."""
    rng = np.random.default_rng(0)
    state_seq = np.repeat(np.tile([0, 1, 2], 22), 90)        # 5940 samples
    T = state_seq.size
    fs = 30.0
    centres = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0]])
    features = centres[state_seq] + rng.standard_normal((T, 2)) * 0.4

    stc = NBStateCollection()
    periods_by_state = {0: [], 1: [], 2: []}
    i = 0
    for s, group in groupby(state_seq):
        n = sum(1 for _ in group)
        periods_by_state[s].append([i / fs, (i + n) / fs])
        i += n
    for s, name, key in [(0, "walk", "w"), (1, "rear", "r"), (2, "pause", "p")]:
        stc.add_state(NBEpoch(
            data=np.array(periods_by_state[s], dtype=float),
            samplerate=1.0, label=name, key=key,
        ))
    return stc, features, fs, state_seq


# ─────────────────────────────────────────────────────────────────────── #
# Bootstrap                                                                 #
# ─────────────────────────────────────────────────────────────────────── #

class TestWholeStateBootstrap:
    def test_basic(self, synth_session):
        stc, features, fs, _ = synth_session
        res = whole_state_bootstrap(
            stc, features, fs, states=["walk", "rear", "pause"],
            state_block_size=300, prct_train=80,
            rng=np.random.default_rng(0),
        )
        assert res.features.shape == (900, 2)
        assert res.labels.shape   == (900,)
        # Equal samples per class
        for k in (0, 1, 2):
            assert (res.labels == k).sum() == 300

    def test_state_names_returned(self, synth_session):
        stc, features, fs, _ = synth_session
        res = whole_state_bootstrap(
            stc, features, fs, states=["pause", "walk", "rear"],
            state_block_size=100,
            rng=np.random.default_rng(0),
        )
        # Order is preserved: state 0 = pause, etc.
        assert res.state_names == ["pause", "walk", "rear"]

    def test_noise_zero_means_deterministic(self, synth_session):
        stc, features, fs, _ = synth_session
        a = whole_state_bootstrap(
            stc, features, fs, states=["walk", "rear", "pause"],
            state_block_size=200, noisy=False,
            rng=np.random.default_rng(123),
        )
        b = whole_state_bootstrap(
            stc, features, fs, states=["walk", "rear", "pause"],
            state_block_size=200, noisy=False,
            rng=np.random.default_rng(123),
        )
        np.testing.assert_array_equal(a.features, b.features)
        np.testing.assert_array_equal(a.labels,   b.labels)

    def test_noise_jitters_features(self, synth_session):
        stc, features, fs, _ = synth_session
        a = whole_state_bootstrap(
            stc, features, fs, states=["walk", "rear", "pause"],
            state_block_size=200, noisy=False,
            rng=np.random.default_rng(123),
        )
        b = whole_state_bootstrap(
            stc, features, fs, states=["walk", "rear", "pause"],
            state_block_size=200, noisy=True, noise_std=0.1,
            rng=np.random.default_rng(123),
        )
        # Same indices were selected (same rng) but features have jitter
        assert not np.array_equal(a.features, b.features)

    def test_unknown_state_raises(self, synth_session):
        stc, features, fs, _ = synth_session
        with pytest.raises(KeyError, match="not found"):
            whole_state_bootstrap(
                stc, features, fs, states=["walk", "ghost"],
                state_block_size=100,
            )


# ─────────────────────────────────────────────────────────────────────── #
# Normalisation                                                             #
# ─────────────────────────────────────────────────────────────────────── #

class TestFeatureNormalisation:
    def test_zero_mean_unit_std(self):
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
        norm = fit_normalisation(X)
        Xz = norm.transform(X)
        np.testing.assert_allclose(Xz.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(Xz.std(axis=0),  1.0, atol=1e-10)

    def test_clip(self):
        X = np.random.default_rng(0).standard_normal((200, 1)) * 10.0
        norm = fit_normalisation(X, clip=2.0)
        out = norm.transform(X)
        assert out.max() <= 2.0
        assert out.min() >= -2.0

    def test_serialisable(self):
        X = np.random.default_rng(0).standard_normal((100, 3))
        norm = fit_normalisation(X, clip=3.0)
        d = norm.to_dict()
        round_tripped = FeatureNormalisation.from_dict(d)
        np.testing.assert_array_equal(round_tripped.mean, norm.mean)
        np.testing.assert_array_equal(round_tripped.std,  norm.std)
        assert round_tripped.clip == norm.clip


# ─────────────────────────────────────────────────────────────────────── #
# Backends — every one runs end-to-end                                      #
# ─────────────────────────────────────────────────────────────────────── #

ALL_BACKENDS = ["patternnet", "mlp", "cnn", "lstm",
                "sklearn-mlp", "rf", "gbm"]


@pytest.mark.parametrize("backend", ALL_BACKENDS)
class TestBackendBasic:
    def test_label_states_runs(self, synth_session, backend):
        stc, features, fs, state_seq = synth_session
        out_stc, ensemble, probs = label_states(
            train_sessions=[(stc, features, fs)],
            target_features=features, target_samplerate=fs,
            states=["walk", "rear", "pause"],
            backend=backend, n_iter=2,
            bootstrap_kwargs=dict(state_block_size=300,
                                   prct_train=80, noise_std=0.05),
            rng=np.random.default_rng(42),
        )
        assert probs.shape == (state_seq.size, 3)
        assert isinstance(out_stc, NBStateCollection)
        # On clean synthetic data with separable classes we expect very
        # high accuracy.  CNN/LSTM are slightly noisier with n_iter=2.
        pred = np.argmax(probs, axis=1)
        accuracy = (pred == state_seq).mean()
        # Permissive bound: > 70% (synthetic with 3 well-separated classes)
        assert accuracy > 0.70, f"{backend}: accuracy {accuracy:.3f} < 0.70"


# ─────────────────────────────────────────────────────────────────────── #
# Save / load roundtrip                                                     #
# ─────────────────────────────────────────────────────────────────────── #

@pytest.mark.parametrize("backend", ALL_BACKENDS)
class TestEnsembleSaveLoad:
    def test_roundtrip(self, synth_session, backend, tmp_path):
        stc, features, fs, _ = synth_session
        _, ensemble, probs = label_states(
            train_sessions=[(stc, features, fs)],
            target_features=features, target_samplerate=fs,
            states=["walk", "rear", "pause"],
            backend=backend, n_iter=2,
            bootstrap_kwargs=dict(state_block_size=200, prct_train=80,
                                   noise_std=0.05),
            rng=np.random.default_rng(42),
        )
        save_dir = tmp_path / "ensemble"
        ensemble.save(save_dir)
        # ensemble.json + iter_0/ + iter_1/ exist
        assert (save_dir / "ensemble.json").exists()
        assert (save_dir / "iter_0").is_dir()
        assert (save_dir / "iter_1").is_dir()

        loaded = TrainedEnsemble.load(save_dir)
        assert loaded.backend     == ensemble.backend
        assert loaded.state_names == ensemble.state_names
        np.testing.assert_array_equal(loaded.normalisation.mean,
                                      ensemble.normalisation.mean)
        # Predictions must match bit-exactly (same model, same input, no RNG)
        re_probs = loaded.predict_proba(features)
        np.testing.assert_allclose(re_probs, probs, atol=1e-6)


# ─────────────────────────────────────────────────────────────────────── #
# Decision smoothing                                                        #
# ─────────────────────────────────────────────────────────────────────── #

class TestSmoothLabels:
    def test_periods_match_argmax(self):
        T = 1000
        fs = 100.0
        # Construct synthetic probs where state 0 is dominant for first half,
        # state 1 dominant for second half.
        probs = np.zeros((T, 2))
        probs[:T // 2, 0] = 0.9; probs[:T // 2, 1] = 0.1
        probs[T // 2:, 0] = 0.1; probs[T // 2:, 1] = 0.9
        out = smooth_labels_to_state_collection(
            probs, state_names=["a", "b"], feature_samplerate=fs,
            smoothing_window_s=0.05,
        )
        # Expect one period per state covering the right half
        a = out.get_state("a")
        b = out.get_state("b")
        assert a.data.shape[0] >= 1
        assert b.data.shape[0] >= 1
        # Period 0 of state a should start near 0
        assert a.data[0, 0] < 0.1
        # Period 0 of state b should start near 5 sec
        assert abs(b.data[0, 0] - 5.0) < 0.2

    def test_smoothing_eliminates_short_blips(self):
        T = 200
        fs = 100.0
        # Steady state-0 except a single-sample blip to state 1
        probs = np.zeros((T, 2)); probs[:, 0] = 0.9; probs[:, 1] = 0.1
        probs[100, 0] = 0.0; probs[100, 1] = 1.0
        out = smooth_labels_to_state_collection(
            probs, state_names=["a", "b"], feature_samplerate=fs,
            smoothing_window_s=0.1,
        )
        # The 1-sample blip should be smoothed away
        assert out.get_state("b").data.shape[0] == 0

    def test_valid_mask_excludes_samples(self):
        T = 300
        probs = np.full((T, 2), 0.5)
        probs[:, 0] = 0.7
        valid_mask = np.zeros(T, dtype=bool)
        valid_mask[100:200] = True
        out = smooth_labels_to_state_collection(
            probs, state_names=["a", "b"], feature_samplerate=100.0,
            valid_mask=valid_mask,
        )
        # Only the middle-third samples are eligible to be in any state
        a = out.get_state("a")
        if a.data.shape[0]:
            for s, e in a.data:
                assert s >= 1.0 - 0.1
                assert e <= 2.0 + 0.1


# ─────────────────────────────────────────────────────────────────────── #
# make_classifier                                                           #
# ─────────────────────────────────────────────────────────────────────── #

class TestMakeClassifier:
    def test_known_backends(self):
        for b in ALL_BACKENDS:
            clf = make_classifier(b)
            assert isinstance(clf, Classifier)
            assert clf.backend == b

    def test_unknown_backend_raises(self):
        with pytest.raises(KeyError, match="Unknown classifier backend"):
            make_classifier("not-a-real-backend")
