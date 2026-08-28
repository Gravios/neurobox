"""Tests for structured state decoding and the labelling evaluation
harness.

The decisive test for the Viterbi implementation is brute force: on
small problems every legal label sequence is enumerated and scored,
and the decoder must return (one of) the maximum-scoring paths — both
without and with minimum-duration constraints.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from neurobox.analysis.classifiers.state_decoding import (
    TransitionModel, fit_transition_matrix, labels_from_stc,
    viterbi_decode, decode_labels,
)
from neurobox.analysis.classifiers.evaluation import (
    segments_from_labels, frame_scores, segmental_f1, boundary_mae,
    bout_statistics, loso_splits, evaluate_labeling,
)
from neurobox.analysis.classifiers.label import (
    smooth_labels_to_state_collection,
)
from neurobox.dtype import NBEpoch, NBStateCollection


# ─────────────────────────────────────────────────────────────────────── #
# Brute-force reference                                                   #
# ─────────────────────────────────────────────────────────────────────── #

def _path_score(path, log_emit, log_T):
    s = log_emit[0, path[0]]
    for t in range(1, len(path)):
        s += log_T[path[t - 1], path[t]] + log_emit[t, path[t]]
    return s


def _interior_durations_ok(path, min_d):
    """Check duration floors on interior bouts only (edge bouts are
    exempt — the sequence may start or end mid-bout)."""
    segs = segments_from_labels(np.asarray(path))
    for i, (s, a, b) in enumerate(segs):
        touches_edge = (a == 0) or (b == len(path))
        if not touches_edge and (b - a) < min_d[s]:
            return False
    return True


def _brute_force(log_emit, model):
    T, S = log_emit.shape
    best, best_score = None, -np.inf
    for path in itertools.product(range(S), repeat=T):
        if not _interior_durations_ok(path, model.min_durations):
            continue
        sc = _path_score(path, log_emit, model.log_transitions)
        if sc > best_score:
            best, best_score = path, sc
    return np.asarray(best), best_score


def _uniform_model(S, min_d=None):
    lt = np.log(np.full((S, S), 1.0 / S))
    md = np.ones(S, dtype=int) if min_d is None else np.asarray(min_d)
    return TransitionModel(lt, md, tuple(f"s{i}" for i in range(S)), 10.0)


class TestViterbiAgainstBruteForce:
    @pytest.mark.parametrize("seed", range(5))
    def test_unconstrained_matches_brute_force(self, seed):
        rng = np.random.default_rng(seed)
        T, S = 8, 3
        log_emit = rng.normal(size=(T, S))
        # random (row-stochastic) transitions
        A = rng.dirichlet(np.ones(S), size=S)
        model = TransitionModel(np.log(A), np.ones(S, int),
                                ("a", "b", "c"), 10.0)
        got = viterbi_decode(log_emit, model)
        ref, ref_score = _brute_force(log_emit, model)
        assert _path_score(got, log_emit, model.log_transitions) \
            == pytest.approx(ref_score)

    @pytest.mark.parametrize("seed", range(5))
    def test_min_duration_matches_brute_force(self, seed):
        rng = np.random.default_rng(100 + seed)
        T, S = 9, 2
        log_emit = rng.normal(size=(T, S))
        A = rng.dirichlet(np.ones(S), size=S)
        model = TransitionModel(np.log(A), np.array([3, 2]),
                                ("a", "b"), 10.0)
        got = viterbi_decode(log_emit, model)
        assert _interior_durations_ok(got, model.min_durations)
        ref, ref_score = _brute_force(log_emit, model)
        assert _path_score(got, log_emit, model.log_transitions) \
            == pytest.approx(ref_score)

    def test_min_duration_suppresses_blips(self):
        """A one-frame emission spike for state 1 must not survive a
        3-frame floor when transitions penalise switching."""
        T, S = 20, 2
        log_emit = np.full((T, S), 0.0)
        log_emit[:, 1] = -2.0
        log_emit[10, 1] = 3.0        # single-frame temptation
        log_emit[10, 0] = -3.0
        A = np.log(np.array([[0.95, 0.05], [0.05, 0.95]]))
        blippy = viterbi_decode(
            log_emit, TransitionModel(A, np.array([1, 1]), ("a", "b"), 10.))
        floored = viterbi_decode(
            log_emit, TransitionModel(A, np.array([1, 3]), ("a", "b"), 10.))
        segs = segments_from_labels(floored)
        for s, a, b in segs:
            if s == 1 and a != 0 and b != T:
                assert b - a >= 3
        # And the floor genuinely changed the answer relative to no floor
        assert (blippy != floored).any() or (blippy == 0).all()

    def test_edge_bouts_exempt_from_floor(self):
        """A truncated bout at the sequence edge may be shorter than
        the floor — the recording started mid-bout."""
        log_emit = np.array([[3., -3.]] * 2 + [[-3., 3.]] * 10)
        model = TransitionModel(
            np.log(np.full((2, 2), 0.5)), np.array([5, 1]),
            ("a", "b"), 10.0)
        path = viterbi_decode(log_emit, model)
        assert (path[:2] == 0).all() and (path[2:] == 1).all()


class TestViterbiEdgeCases:
    def test_empty_input(self):
        assert viterbi_decode(
            np.zeros((0, 2)), _uniform_model(2)).size == 0

    def test_single_frame(self):
        p = viterbi_decode(np.array([[0.1, 2.0, 0.3]]), _uniform_model(3))
        assert p.tolist() == [1]

    def test_all_neg_inf_row_does_not_poison(self):
        log_emit = np.zeros((5, 2))
        log_emit[2] = -np.inf
        path = viterbi_decode(log_emit, _uniform_model(2))
        assert path.shape == (5,)

    def test_state_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="states"):
            viterbi_decode(np.zeros((4, 3)), _uniform_model(2))


class TestDecodeLabels:
    def test_argmax_matches_numpy(self):
        rng = np.random.default_rng(0)
        p = rng.dirichlet(np.ones(4), size=50)
        np.testing.assert_array_equal(
            decode_labels(p, method="argmax"), p.argmax(axis=1))

    def test_viterbi_requires_model(self):
        with pytest.raises(ValueError, match="TransitionModel"):
            decode_labels(np.ones((5, 2)) * 0.5, method="viterbi")

    def test_unknown_method(self):
        with pytest.raises(ValueError, match="unknown decode"):
            decode_labels(np.ones((5, 2)) * 0.5, method="magic")

    def test_sticky_transitions_smooth_noisy_probs(self):
        """The motivating case: noisy per-frame probabilities around a
        clean two-bout structure.  Argmax flickers; Viterbi with
        sticky transitions doesn't."""
        rng = np.random.default_rng(3)
        T = 200
        truth = np.zeros(T, int); truth[80:150] = 1
        logits = np.where(truth[:, None] == np.arange(2)[None, :], 1.2, -1.2)
        logits = logits + rng.normal(0, 1.4, size=logits.shape)
        p = np.exp(logits); p /= p.sum(1, keepdims=True)

        arg = decode_labels(p, method="argmax")
        model = TransitionModel(
            np.log(np.array([[0.98, 0.02], [0.02, 0.98]])),
            np.array([5, 5]), ("a", "b"), 10.0)
        vit = decode_labels(p, method="viterbi", model=model)

        n_seg_arg = len(segments_from_labels(arg))
        n_seg_vit = len(segments_from_labels(vit))
        assert n_seg_vit < n_seg_arg          # fewer spurious bouts
        assert n_seg_vit <= 5
        assert (vit == truth).mean() > (arg == truth).mean()


class TestFitTransitionMatrix:
    def test_counts_and_smoothing(self):
        seq = np.array([0, 0, 0, 1, 1, 0])
        m = fit_transition_matrix([seq], 2, pseudocount=0.0,
                                  samplerate=10.0)
        A = np.exp(m.log_transitions)
        # from 0: 0->0 twice, 0->1 once ; from 1: 1->1 once, 1->0 once
        np.testing.assert_allclose(A[0], [2/3, 1/3])
        np.testing.assert_allclose(A[1], [1/2, 1/2])

    def test_unlabelled_frames_break_transitions(self):
        seq = np.array([0, -1, 1, 1])
        m = fit_transition_matrix([seq], 2, pseudocount=0.0,
                                  samplerate=10.0)
        A = np.exp(m.log_transitions)
        # only the 1->1 transition is counted; row 0 has no counts and
        # collapses to NaN-free floor via the log clamp
        assert A[1, 1] == pytest.approx(1.0)

    def test_min_duration_seconds_to_frames(self):
        m = fit_transition_matrix(
            [np.array([0, 1, 0, 1])], 2,
            samplerate=10.0, min_duration_s=(0.3, 0.0))
        np.testing.assert_array_equal(m.min_durations, [3, 1])

    def test_multiple_sessions_pool(self):
        m1 = fit_transition_matrix(
            [np.array([0, 1]), np.array([0, 1])], 2,
            pseudocount=0.0, samplerate=10.0)
        m2 = fit_transition_matrix(
            [np.array([0, 1, 0, 1])], 2, pseudocount=0.0, samplerate=10.0)
        # both have 0->1 twice; second also has 1->0 once
        A1, A2 = np.exp(m1.log_transitions), np.exp(m2.log_transitions)
        assert A1[0, 1] == pytest.approx(1.0)
        assert A2[0, 1] == pytest.approx(1.0)

    def test_validation(self):
        with pytest.raises(ValueError, match="min_durations must be >= 1"):
            TransitionModel(np.zeros((2, 2)), np.array([0, 1]),
                            ("a", "b"), 10.0)
        with pytest.raises(ValueError, match="log_transitions"):
            TransitionModel(np.zeros((2, 3)), np.array([1, 1]),
                            ("a", "b"), 10.0)


class TestLabelsFromStc:
    def _stc(self, fs=10.0):
        stc = NBStateCollection()
        stc.add_state(NBEpoch(np.array([[0.0, 1.0]]), samplerate=1.0,
                              label="walk", key="w"))
        stc.add_state(NBEpoch(np.array([[1.0, 2.0]]), samplerate=1.0,
                              label="rear", key="r"))
        return stc

    def test_rasterise(self):
        lab = labels_from_stc(self._stc(), ("walk", "rear"), 25, 10.0)
        assert (lab[:10] == 0).all()
        assert (lab[10:20] == 1).all()
        assert (lab[20:] == -1).all()

    def test_earlier_state_wins_on_overlap(self):
        stc = NBStateCollection()
        stc.add_state(NBEpoch(np.array([[0.0, 2.0]]), samplerate=1.0,
                              label="walk", key="w"))
        stc.add_state(NBEpoch(np.array([[1.0, 3.0]]), samplerate=1.0,
                              label="rear", key="r"))
        lab = labels_from_stc(stc, ("walk", "rear"), 30, 10.0)
        assert (lab[10:20] == 0).all()      # overlap → walk (earlier)


# ─────────────────────────────────────────────────────────────────────── #
# Evaluation metrics                                                      #
# ─────────────────────────────────────────────────────────────────────── #

class TestSegments:
    def test_basic_rle(self):
        segs = segments_from_labels(np.array([0, 0, 1, 1, 1, 0]))
        assert segs == [(0, 0, 2), (1, 2, 5), (0, 5, 6)]

    def test_unlabelled_omitted(self):
        segs = segments_from_labels(np.array([-1, -1, 2, 2, -1]))
        assert segs == [(2, 2, 4)]

    def test_empty(self):
        assert segments_from_labels(np.array([])) == []


class TestFrameScores:
    def test_perfect(self):
        y = np.array([0, 1, 1, 2])
        acc, macro, per = frame_scores(y, y, 3)
        assert acc == 1.0 and macro == 1.0
        np.testing.assert_allclose(per, [1., 1., 1.])

    def test_unlabelled_excluded(self):
        yt = np.array([-1, 0, 1])
        yp = np.array([1, 0, 1])          # wrong on the -1 frame: ignored
        acc, _, _ = frame_scores(yt, yp, 2)
        assert acc == 1.0

    def test_absent_state_is_nan_not_zero(self):
        yt = np.array([0, 0, 0])
        yp = np.array([0, 0, 0])
        _, macro, per = frame_scores(yt, yp, 3)
        assert np.isnan(per[1]) and np.isnan(per[2])
        assert macro == 1.0               # NaNs skipped, not averaged in

    def test_hand_computed_f1(self):
        yt = np.array([0, 0, 1, 1])
        yp = np.array([0, 1, 1, 1])
        # state 0: tp=1 fp=0 fn=1 → 2/3 ; state 1: tp=2 fp=1 fn=0 → 4/5
        _, macro, per = frame_scores(yt, yp, 2)
        np.testing.assert_allclose(per, [2/3, 4/5])
        assert macro == pytest.approx((2/3 + 4/5) / 2)


class TestSegmentalF1:
    def test_perfect(self):
        y = np.array([0, 0, 1, 1, 0])
        assert segmental_f1(y, y, iou=0.5) == 1.0

    def test_fragmented_prediction_penalised(self):
        """Same frames covered, but the bout is split in two: frame
        accuracy is high, segmental F1 is not — exactly the failure
        mode the metric exists for."""
        yt = np.array([1] * 10)
        yp = np.array([1] * 4 + [0] + [1] * 5)
        assert frame_scores(yt, yp, 2)[0] == 0.9
        assert segmental_f1(yt, yp, iou=0.5) < 0.7

    def test_iou_threshold_matters(self):
        yt = np.zeros(20, int); yt[:10] = 1
        yp = np.zeros(20, int); yp[:4] = 1
        # state-1 pair has IoU 0.4; state-0 pair has IoU 10/16 = 0.625.
        # At τ=0.25 both match → F1 = 1.  At τ=0.5 only state-0 does:
        # tp=1, fp=1, fn=1 → F1 = 0.5.
        assert segmental_f1(yt, yp, iou=0.25) == pytest.approx(1.0)
        assert segmental_f1(yt, yp, iou=0.50) == pytest.approx(0.5)

    def test_class_must_match(self):
        yt = np.array([1, 1, 1, 0, 0])
        yp = np.array([2, 2, 2, 0, 0])
        assert segmental_f1(yt, yp, iou=0.1) == pytest.approx(2 / 4 * 2 / 2)

    def test_both_empty_is_nan(self):
        e = np.full(5, -1)
        assert np.isnan(segmental_f1(e, e, iou=0.5))


class TestBoundaryMae:
    def test_exact(self):
        y = np.array([0, 0, 1, 1])
        assert boundary_mae(y, y, samplerate=10.0) == 0.0

    def test_known_offset(self):
        yt = np.array([0, 0, 0, 1, 1, 1])
        yp = np.array([0, 0, 1, 1, 1, 1])       # boundary 1 frame early
        assert boundary_mae(yt, yp, 10.0) == pytest.approx(0.1)

    def test_no_pred_boundaries_is_inf(self):
        yt = np.array([0, 1])
        yp = np.array([0, 0])
        assert boundary_mae(yt, yp, 10.0) == np.inf

    def test_no_true_boundaries_is_nan(self):
        assert np.isnan(boundary_mae(np.zeros(5, int), np.zeros(5, int), 10.))


class TestBoutStatistics:
    def test_counts_and_durations(self):
        lab = np.array([0]*10 + [1]*5 + [0]*10)
        st = bout_statistics(lab, 2, samplerate=10.0)
        assert st[0]["n_bouts"] == 2 and st[1]["n_bouts"] == 1
        assert st[1]["total_s"] == pytest.approx(0.5)
        assert st[0]["mean_s"] == pytest.approx(1.0)

    def test_absent_state(self):
        st = bout_statistics(np.zeros(10, int), 2, 10.0)
        assert st[1]["n_bouts"] == 0 and np.isnan(st[1]["mean_s"])


class TestLoso:
    def test_yields_each_once(self):
        splits = list(loso_splits(4))
        assert [t for _, t in splits] == [0, 1, 2, 3]
        for train, test in splits:
            assert test not in train and len(train) == 3

    def test_too_few_sessions(self):
        with pytest.raises(ValueError, match=">= 2"):
            list(loso_splits(1))


class TestEvaluateLabeling:
    def test_bundle_and_summary(self):
        rng = np.random.default_rng(0)
        yt = np.zeros(100, int); yt[40:70] = 1
        yp = yt.copy(); yp[40:42] = 0            # boundary 2 frames late
        sc = evaluate_labeling(yt, yp, 2, samplerate=10.0)
        assert sc.accuracy == pytest.approx(0.98)
        assert sc.segmental[0.50] == 1.0
        assert sc.boundary_mae_s == pytest.approx(0.1)
        txt = sc.summary(states=("walk", "rear"))
        assert "segF1@50" in txt and "boundary MAE" in txt


# ─────────────────────────────────────────────────────────────────────── #
# Integration: decode= in smooth_labels_to_state_collection               #
# ─────────────────────────────────────────────────────────────────────── #

class TestSmoothLabelsDecodeParam:
    def _noisy_probs(self, seed=5, T=300):
        rng = np.random.default_rng(seed)
        truth = np.zeros(T, int); truth[100:220] = 1
        logits = np.where(truth[:, None] == np.arange(2)[None, :], 1.0, -1.0)
        logits += rng.normal(0, 1.5, size=logits.shape)
        p = np.exp(logits); p /= p.sum(1, keepdims=True)
        return truth, p

    def test_default_is_argmax_parity(self):
        truth, p = self._noisy_probs()
        a = smooth_labels_to_state_collection(
            p, ("walk", "rear"), 10.0)
        b = smooth_labels_to_state_collection(
            p, ("walk", "rear"), 10.0, decode="argmax")
        for s in ("walk", "rear"):
            np.testing.assert_array_equal(
                a.get_state(s).data, b.get_state(s).data)

    def test_viterbi_reduces_fragmentation(self):
        truth, p = self._noisy_probs()
        model = fit_transition_matrix(
            [truth], 2, states=("walk", "rear"),
            samplerate=10.0, min_duration_s=0.5)
        arg = smooth_labels_to_state_collection(
            p, ("walk", "rear"), 10.0, decode="argmax")
        vit = smooth_labels_to_state_collection(
            p, ("walk", "rear"), 10.0, decode="viterbi",
            transition_model=model)
        n_arg = sum(arg.get_state(s).data.shape[0] for s in ("walk", "rear"))
        n_vit = sum(vit.get_state(s).data.shape[0] for s in ("walk", "rear"))
        assert n_vit < n_arg
        assert n_vit <= 4

    def test_viterbi_without_model_raises(self):
        _, p = self._noisy_probs()
        with pytest.raises(ValueError, match="TransitionModel"):
            smooth_labels_to_state_collection(
                p, ("walk", "rear"), 10.0, decode="viterbi")

    def test_valid_mask_still_respected(self):
        truth, p = self._noisy_probs()
        model = fit_transition_matrix(
            [truth], 2, states=("walk", "rear"), samplerate=10.0)
        mask = np.ones(p.shape[0], bool); mask[:50] = False
        stc = smooth_labels_to_state_collection(
            p, ("walk", "rear"), 10.0, decode="viterbi",
            transition_model=model, valid_mask=mask)
        # First valid frame is index 50; thresh_cross places the period
        # edge on the last sample below threshold, so the earliest
        # permissible start is (50 - 1) / fs = 4.9 s.
        earliest = (50 - 1) / 10.0
        for s in ("walk", "rear"):
            d = stc.get_state(s).data
            if d.size:
                assert (d[:, 0] >= earliest - 1e-9).all()
