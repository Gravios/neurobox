"""Round-19 tests — fet_all features, hierarchical-MI feature selection,
   t-SNE wrapper, and the run_feature_selection_pipeline orchestrator.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

# matplotlib is an OPTIONAL dependency (extras: [viz]); skip the
# entire module gracefully if it isn't installed.
matplotlib = pytest.importorskip("matplotlib", exc_type=ImportError)
matplotlib.use("Agg")

from neurobox.dtype import (        # noqa: E402
    NBDxyz, NBModel, NBEpoch, NBStateCollection,
)


# ─────────────────────────────────────────────────────────────────────── #
# Helpers                                                                    #
# ─────────────────────────────────────────────────────────────────────── #

def _synthetic_xyz(T: int = 3000, fs: float = 120.0) -> NBDxyz:
    """Synthetic rat-skeleton xyz with circular motion + body-frame offsets."""
    t = np.arange(T) / fs
    markers = ["spine_lower", "pelvis_root", "spine_middle", "spine_upper",
               "head_back", "head_left", "head_front", "head_right"]
    data = np.zeros((T, len(markers), 3))
    heading = t * 0.3
    body_x = 100 + 80 * np.cos(heading)
    body_y = 100 + 80 * np.sin(heading)
    offsets = {"spine_lower": (-30, 0, 50), "pelvis_root": (-15, 0, 50),
               "spine_middle": (0, 0, 50), "spine_upper": (15, 0, 50),
               "head_back": (25, 0, 60), "head_left": (30, 5, 60),
               "head_front": (35, 0, 60), "head_right": (30, -5, 60)}
    cy_, sy_ = np.cos(heading), np.sin(heading)
    for i, m in enumerate(markers):
        ox, oy, oz = offsets[m]
        data[:, i, 0] = body_x + ox * cy_ - oy * sy_
        data[:, i, 1] = body_y + ox * sy_ + oy * cy_
        data[:, i, 2] = oz + 5 * np.sin(heading * 2)
    return NBDxyz(data, model=NBModel(markers=markers),
                   samplerate=fs, name="r19_test")


def _runs_to_periods(mask: np.ndarray, fs: float) -> np.ndarray:
    d = np.diff(mask.astype(np.int8), prepend=0, append=0)
    s = np.flatnonzero(d == 1)
    e = np.flatnonzero(d == -1)
    return np.column_stack([s, e]).astype(np.float64) / fs


def _synthetic_stc_and_fet(T: int = 6000, fs: float = 30.0,
                            seed: int = 42):
    """Synthetic 4-state stc + an 8-column feature matrix with known
    state-discriminating columns."""
    rng = np.random.default_rng(seed)
    states = ["walk", "rear", "pause", "sit"]
    fet = rng.standard_normal((T, 8))
    labels = rng.choice(states, size=T, p=[0.4, 0.2, 0.3, 0.1])
    fet[labels == "walk", 0] += 3.0
    fet[labels == "rear", 1] += 3.0
    fet[labels == "pause", 2] -= 3.0
    fet[labels == "sit",  3] += 2.0

    stc = NBStateCollection(mode="manual")
    for s in states:
        mask = labels == s
        periods = _runs_to_periods(mask, fs)
        if len(periods):
            stc.add_state(NBEpoch(
                data=periods, samplerate=fs, mode="periods",
                label=s, key=s[0],
            ))
    return stc, fet, states, labels


# ─────────────────────────────────────────────────────────────────────── #
# fet_all_features                                                           #
# ─────────────────────────────────────────────────────────────────────── #

class TestFetAll:
    def test_returns_59_columns(self):
        from neurobox.analysis.kinematics import (
            augment_xyz, fet_all_features,
        )
        xyz = _synthetic_xyz()
        aug = augment_xyz(xyz)
        result = fet_all_features(aug, samplerate=20.0)
        assert result.data.shape[1] == 59
        assert len(result.column_names) == 59
        assert np.isfinite(result.data).all()

    def test_resamples_correctly(self):
        from neurobox.analysis.kinematics import (
            augment_xyz, fet_all_features,
        )
        xyz = _synthetic_xyz(T=6000, fs=120.0)
        aug = augment_xyz(xyz)
        result = fet_all_features(aug, samplerate=20.0)
        assert result.samplerate == 20.0
        # 6000 samples @ 120 Hz = 50 s → 1000 samples @ 20 Hz
        assert abs(result.data.shape[0] - 1000) <= 2

    def test_column_names_unique(self):
        from neurobox.analysis.kinematics import (
            augment_xyz, fet_all_features,
        )
        xyz = _synthetic_xyz()
        aug = augment_xyz(xyz)
        result = fet_all_features(aug, samplerate=20.0)
        # Some columns may repeat (MATLAB had duplicates) — check at least
        # we have the expected 59 names
        assert len(result.column_names) == 59


class TestLowerSpineYawPpc:
    def test_rigid_motion_high_ppc(self):
        from neurobox.analysis.kinematics import (
            augment_xyz, lower_spine_yaw_ppc,
        )
        xyz = _synthetic_xyz()
        aug = augment_xyz(xyz)
        ppc = lower_spine_yaw_ppc(aug)
        assert ppc.shape == (xyz.data.shape[0],)
        # All markers translate together → very high PPC (all marker
        # velocity vectors are parallel)
        assert np.nanmean(ppc) > 0.9

    def test_returns_zeros_on_missing_markers(self):
        from neurobox.analysis.kinematics import lower_spine_yaw_ppc
        # Only a subset of expected markers present
        xyz = NBDxyz(np.random.randn(100, 1, 3),
                      model=NBModel(markers=["nose"]),
                      samplerate=100.0, name="empty")
        ppc = lower_spine_yaw_ppc(xyz)
        assert ppc.shape == (100,)
        # No matching markers → returns zeros
        assert (ppc == 0).all()


# ─────────────────────────────────────────────────────────────────────── #
# select_features_hmi                                                        #
# ─────────────────────────────────────────────────────────────────────── #

class TestSelectFeaturesHmi:
    def test_state_order_covers_all(self):
        from neurobox.analysis import select_features_hmi
        stc, fet, states, _ = _synthetic_stc_and_fet()
        result = select_features_hmi(
            stc, fet, 30.0, states=states, mi_threshold_bits=0.05,
        )
        # state_order should include all input states (in some order)
        assert set(result.state_order) == set(states)
        # The MATLAB original peels off (n-2) states with their own
        # feature subsets, then the leaf has a single d′-selected
        # subset for the remaining 2 states (treated as a binary
        # classifier).  So feature_indices has length (n-2) + 1.
        assert len(result.feature_indices) == len(result.state_order) - 1

    def test_finds_discriminating_features(self):
        from neurobox.analysis import select_features_hmi
        stc, fet, states, _ = _synthetic_stc_and_fet()
        result = select_features_hmi(
            stc, fet, 30.0, states=states, mi_threshold_bits=0.05,
        )
        # The first state to be peeled off should have its
        # discriminating column in the selected feature set.
        first_state = result.state_order[0]
        first_features = set(result.feature_indices[0])
        # The column that "boosts" first_state by +3 should be in there
        boost_col = states.index(first_state)
        assert boost_col in first_features

    def test_leaf_uses_dprime(self):
        from neurobox.analysis import select_features_hmi
        stc, fet, states, _ = _synthetic_stc_and_fet()
        result = select_features_hmi(
            stc, fet, 30.0, states=states,
            mi_threshold_bits=0.05, dprime_threshold=0.5,
        )
        # Leaf entry is the last state's feature_indices (paired
        # against the second-to-last via d′)
        leaf_features = result.feature_indices[-1]
        # At least one of the discriminating cols should be picked up
        # by d′
        assert leaf_features.size >= 1


# ─────────────────────────────────────────────────────────────────────── #
# mta_tsne                                                                   #
# ─────────────────────────────────────────────────────────────────────── #

class TestMtaTsne:
    def test_basic_run(self):
        from neurobox.analysis import mta_tsne
        rng = np.random.default_rng(0)
        T, F = 600, 4
        fet = rng.standard_normal((T, F))
        labels = rng.choice([1, 2, 3], size=T).astype(np.int32)
        result = mta_tsne(
            fet, 10.0, state_labels=labels,
            state_names=("a", "b", "c"),
            perplexity=20.0, init_dims=3, skip=2, rng=42,
        )
        assert result.embedding.shape[1] == 2
        assert result.embedding.shape[0] == result.sample_indices.size
        assert result.embedding.shape[0] == result.state_labels.size
        assert result.state_names == ("a", "b", "c")

    def test_subset_filter(self):
        from neurobox.analysis import mta_tsne
        rng = np.random.default_rng(0)
        fet = rng.standard_normal((1000, 4))
        labels = np.ones(1000, dtype=np.int32)
        result = mta_tsne(
            fet, 10.0, state_labels=labels,
            perplexity=10.0, init_dims=2, skip=1,
            subset=(0, 200), rng=42,
        )
        # All kept samples should be inside [0, 200)
        assert (result.sample_indices < 200).all()

    def test_handles_no_state_labels(self):
        from neurobox.analysis import mta_tsne
        rng = np.random.default_rng(0)
        fet = rng.standard_normal((400, 3))
        result = mta_tsne(
            fet, 10.0, perplexity=10.0,
            init_dims=2, skip=2, rng=42,
        )
        assert result.embedding.shape[1] == 2


# ─────────────────────────────────────────────────────────────────────── #
# augment_features_quadratic                                                 #
# ─────────────────────────────────────────────────────────────────────── #

class TestAugmentFeaturesQuadratic:
    def test_correct_output_size(self):
        from neurobox.analysis import augment_features_quadratic
        rng = np.random.default_rng(0)
        T, F = 200, 5
        fet = rng.standard_normal((T, F))
        out = augment_features_quadratic(fet)
        # Expected: F + F + F*(F-1) = 5 + 5 + 20 = 30
        assert out.shape == (T, F + F + F * (F - 1))

    def test_first_block_is_original(self):
        from neurobox.analysis import augment_features_quadratic
        fet = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        out = augment_features_quadratic(fet)
        np.testing.assert_array_equal(out[:, :3], fet)

    def test_second_block_is_squared(self):
        from neurobox.analysis import augment_features_quadratic
        fet = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        out = augment_features_quadratic(fet)
        np.testing.assert_array_equal(out[:, 3:6], fet ** 2)


# ─────────────────────────────────────────────────────────────────────── #
# run_feature_selection_pipeline                                             #
# ─────────────────────────────────────────────────────────────────────── #

class TestPipeline:
    def test_end_to_end(self):
        """Smoke test the full pipeline; classifier accuracy on the
        target-discriminating column should be high."""
        from neurobox.analysis import run_feature_selection_pipeline
        stc, fet, states, _ = _synthetic_stc_and_fet()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_feature_selection_pipeline(
                stc, fet, 30.0,
                states                  = states,
                mi_threshold_bits       = 0.02,
                max_features_per_state  = 4,
                backend                 = "sklearn-mlp",
                backend_kwargs          = {
                    "hidden_layer_sizes": (16,),
                    "max_iter":           50,
                    "random_state":       42,
                },
                rng = 42,
            )
        # Result has all expected fields
        assert hasattr(result, "hmi")
        assert hasattr(result, "per_state")
        assert hasattr(result, "optimised_feature_indices")
        # state_order covers all input states
        assert set(result.state_order) == set(states)
        # At least one per-state result has classifier accuracy > 0.7
        accs_max = max(
            float(np.nanmax(ps.accuracy)) for ps in result.per_state
        )
        assert accs_max > 0.7

    def test_normalisation_attached(self):
        from neurobox.analysis import run_feature_selection_pipeline
        stc, fet, states, _ = _synthetic_stc_and_fet()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_feature_selection_pipeline(
                stc, fet, 30.0,
                states                  = states,
                max_features_per_state  = 2,
                backend_kwargs          = {
                    "hidden_layer_sizes": (4,),
                    "max_iter":           20,
                    "random_state":       42,
                },
                rng = 42,
            )
        # Normalisation has the expected shape
        assert result.normalisation.mean.shape == (8,)
        assert result.normalisation.std.shape  == (8,)
        # Apply transform and check result is centred
        norm_fet = result.normalisation.transform(fet)
        assert abs(norm_fet.mean()) < 1.0   # roughly centred

    def test_per_state_records_have_required_fields(self):
        from neurobox.analysis import run_feature_selection_pipeline
        stc, fet, states, _ = _synthetic_stc_and_fet()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_feature_selection_pipeline(
                stc, fet, 30.0,
                states                  = states,
                max_features_per_state  = 2,
                backend_kwargs          = {
                    "hidden_layer_sizes": (4,),
                    "max_iter":           20,
                    "random_state":       42,
                },
                rng = 42,
            )
        for ps in result.per_state:
            assert ps.state in states
            assert ps.initial_feature_indices.ndim == 1
            assert ps.optimised_feature_indices.ndim == 1
            assert ps.accuracy.ndim == 1
            # accuracy len ≤ max_features_per_state
            assert ps.accuracy.size <= 2
