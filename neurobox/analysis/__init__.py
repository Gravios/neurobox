"""
neurobox.analysis
=================
Post-processing and quality-assessment modules.
"""
from .neuron_quality    import neuron_quality, NeuronQualityResult
from .lfp               import (
    SpectralParams, SpectrumResult,
    multitaper_spectrogram, multitaper_coherogram,
    multitaper_cross_spectrogram, multitaper_psd,
    whiten_ar, fet_spec,
    butter_filter, filter0, fir_filter,
    OscillationResult, detect_oscillations, detect_ripples,
    local_minima, thresh_cross, within_ranges,
    CSDResult, current_source_density,
    join_ranges, intersect_ranges, subtract_ranges, complement_ranges,
)
from .stats             import (
    RayleighResult,
    circ_mean, circ_r,
    rayleigh_test, ppc,
    von_mises_fit, von_mises_pdf, von_mises_rvs,
    bessel_ratio_inverse,
    # CircStat2012a additions
    circ_dist, circ_dist2,
    circ_var, circ_std,
    circ_median,
    circ_kappa,
    circ_moment,
    circ_skewness, circ_kurtosis,
    circ_axial,
    circ_ang2rad, circ_rad2ang,
    FDRResult, fdr_bh,
    BinSmoothResult, bin_smooth,
)
from .spikes            import (
    ccg, trains_to_ccg, CCGResult,
)
from .spatial           import (
    occupancy_map, OccupancyResult,
    place_field, PlaceFieldResult,
    place_field_stats, Patch, UnitStats,
    # Round 18
    knn_place_field, compute_pfstats_bs, PfsBsResult,
)
from .feature_dynamics  import (
    time_lagged_mutual_information,
    time_lagged_cross_correlation,
    TimeLaggedResult,
)
from .feature_selection import (
    pairwise_mutual_information_ranking,
    MIFeatureRanking,
    # Round 19
    select_features_hmi,
    HierarchicalMIResult,
    mta_tsne,
    TSNEResult,
    run_feature_selection_pipeline,
    augment_features_quadratic,
    FeatureSelectionPipelineResult,
    PerStateAccumulation,
)
from .placefields       import (
    compute_drz, compute_ddz,
    compute_ghz, compute_gdz,
    compute_hdz, compute_hrz, compute_hpv, compute_tpv,
    field_centres_from_result,
    egocentric_position,
    compute_ego_ratemap,
    compute_ego_ratemap_conditioned,
)
from .kinematics        import (
    augment_xyz,
    # Round 17
    SplineSpineResult,
    spline_spine,
    preproc_xyz_spline_spine_head_eqi,
    preproc_xyz_spline_spine_head_eqd,
    BodyReferencedFeatures,
    body_referenced_features,
    body_referenced_xy_features,
    # Round 19
    FetAllResult,
    fet_all_features,
    lower_spine_yaw_ppc,
)
from . import heuristics
from .decoding          import (
    decode_ufr_boxcar, DecodingResult,
    prepare_ratemap, prepare_bin_coords,
    accumulate_decoding_vars, AccumulatedDecoding,
    create_tensor_mask,
    CircularBoundary, SquareBoundary, LineBoundary,
    theta_phase, stc2mat,
)
from .mocap             import (
    rotate_points_around_vectors,
    rotate_point_around_vector,
    rotate_marker_around_vector,
    rigid_body_basis,
    intermarker_distances,
    marker_triads, MarkerTriadResult,
    marker_diff_matrix,
    inter_marker_distance,
    inter_marker_angles,
    inter_marker_orientation,
    fill_gaps,
    infer_virtual_joint,
    find_error_periods,
    correct_point_errors,
    parse_rbo_from_csv, MotiveTakeResult,
)
from .transformations   import (
    BinAxis,
    BinStats2D, BinStats2DCirc, BinStats3D, BinStatsArbitrary,
    bin_statistic_2d,
    bin_statistic_2d_circ,
    bin_statistic_3d,
    bin_statistic,
    rot_z_axis, rot_y_axis, detect_roll,
    thetarc_phase,
    decompose_xy_motion_wrt_body,
    BodyMotionSVDModel,
    quat2rotm, quaternion2rad,
    make_uniform_distr, shilbert, my_theta_phase,
)
from .transform_origin  import transform_origin, TransformResult


# ─────────────────────────────────────────────────────────────────────────── #
# Lazy HMM + classifiers import.                                              #
# torch / scikit-learn are only pulled in if a backend is actually requested. #
# ─────────────────────────────────────────────────────────────────────────── #

_LAZY_CLASSIFIER_NAMES = {
    "Classifier", "FitInfo",
    "whole_state_bootstrap", "BootstrapResult",
    "make_classifier",
    "train_classifier_ensemble", "predict_with_ensemble",
    "smooth_labels_to_state_collection",
    "label_states",
    "TrainedEnsemble", "FeatureNormalisation", "fit_normalisation",
    # stc-editing utilities
    "mat_to_stc",
    "confusion_matrix",
    "compare_stcs", "LabelComparisonStats",
    "swap_state_vector_ids",
    "reassign_short_periods",
    "reassign_state_by_duration",
    "reduce_stc_to_loc",
    "mutual_information_states_features",
    # session alignment
    "BehaviouralManifoldStats",
    "behavioural_manifold_stats",
    "map_to_reference_session",
}


def __getattr__(name: str):
    if name in ("gauss_hmm", "HMMResult"):
        from .stats import hmm as _hmm
        return getattr(_hmm, name)
    if name in _LAZY_CLASSIFIER_NAMES:
        from . import classifiers as _classifiers
        return getattr(_classifiers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "neuron_quality", "NeuronQualityResult",
    "transform_origin", "TransformResult",
    "SpectralParams", "SpectrumResult",
    "multitaper_spectrogram", "multitaper_coherogram",
    "multitaper_cross_spectrogram", "multitaper_psd",
    "whiten_ar", "fet_spec",
    "butter_filter", "filter0", "fir_filter", "rect_filter", "gauss_window",
    "OscillationResult", "detect_oscillations", "detect_ripples",
    "local_minima", "thresh_cross", "within_ranges",
    "CSDResult", "current_source_density",
    "join_ranges", "intersect_ranges", "subtract_ranges", "complement_ranges",
    "RayleighResult",
    "circ_mean", "circ_r",
    "rayleigh_test", "ppc",
    "von_mises_fit", "von_mises_pdf", "von_mises_rvs",
    "bessel_ratio_inverse",
    "circ_dist", "circ_dist2",
    "circ_var", "circ_std",
    "circ_median",
    "circ_kappa",
    "circ_moment",
    "circ_skewness", "circ_kurtosis",
    "circ_axial",
    "circ_ang2rad", "circ_rad2ang",
    "FDRResult", "fdr_bh",
    "BinSmoothResult", "bin_smooth",
    "ccg", "trains_to_ccg", "CCGResult",
    "occupancy_map", "OccupancyResult",
    "place_field", "PlaceFieldResult",
    "place_field_stats", "Patch", "UnitStats",
    # round 18 spatial — KNN ratemaps + multi-state pfstats aggregation
    "knn_place_field", "compute_pfstats_bs", "PfsBsResult",
    # round 18 feature dynamics — MI / cross-corr at variable lags
    "time_lagged_mutual_information",
    "time_lagged_cross_correlation",
    "TimeLaggedResult",
    # round 18 feature selection — MI ranking vs binary state labels
    "pairwise_mutual_information_ranking",
    "MIFeatureRanking",
    # round 19 feature selection — HMI + pipeline + t-SNE wrapper
    "select_features_hmi",
    "HierarchicalMIResult",
    "mta_tsne",
    "TSNEResult",
    "run_feature_selection_pipeline",
    "augment_features_quadratic",
    "FeatureSelectionPipelineResult",
    "PerStateAccumulation",
    # round 19 kinematics — fet_all + lower_spine_yaw_ppc
    "FetAllResult",
    "fet_all_features",
    "lower_spine_yaw_ppc",
    # directional zone scores (placefields module)
    "compute_drz", "compute_ddz",
    "compute_ghz", "compute_gdz",
    "compute_hdz", "compute_hrz", "compute_hpv", "compute_tpv",
    "field_centres_from_result",
    "egocentric_position",
    "compute_ego_ratemap",
    "compute_ego_ratemap_conditioned",
    "augment_xyz",
    # round 17 kinematics + heuristics
    "SplineSpineResult",
    "spline_spine",
    "preproc_xyz_spline_spine_head_eqi",
    "preproc_xyz_spline_spine_head_eqd",
    "BodyReferencedFeatures",
    "body_referenced_features",
    "body_referenced_xy_features",
    "heuristics",
    # decoding
    "decode_ufr_boxcar", "DecodingResult",
    "prepare_ratemap", "prepare_bin_coords",
    "accumulate_decoding_vars", "AccumulatedDecoding",
    "create_tensor_mask",
    "CircularBoundary", "SquareBoundary", "LineBoundary",
    "theta_phase", "stc2mat",
    # mocap
    "rotate_points_around_vectors",
    "rotate_point_around_vector",
    "rotate_marker_around_vector",
    "rigid_body_basis",
    "intermarker_distances",
    "marker_triads", "MarkerTriadResult",
    "marker_diff_matrix",
    "inter_marker_distance",
    "inter_marker_angles",
    "inter_marker_orientation",
    "fill_gaps",
    "infer_virtual_joint",
    "find_error_periods",
    "correct_point_errors",
    "parse_rbo_from_csv", "MotiveTakeResult",
    # transformations
    "BinAxis",
    "BinStats2D", "BinStats2DCirc", "BinStats3D", "BinStatsArbitrary",
    "bin_statistic_2d", "bin_statistic_2d_circ", "bin_statistic_3d",
    "bin_statistic",
    "rot_z_axis", "rot_y_axis", "detect_roll",
    "thetarc_phase",
    "decompose_xy_motion_wrt_body", "BodyMotionSVDModel",
    "quat2rotm", "quaternion2rad",
    "make_uniform_distr", "shilbert", "my_theta_phase",
    # hmm — requires `pip install 'neurobox[hmm]'`
    "gauss_hmm", "HMMResult",
    # classifiers — requires `pip install 'neurobox[classify]'`
    "Classifier", "FitInfo",
    "whole_state_bootstrap", "BootstrapResult",
    "make_classifier",
    "train_classifier_ensemble", "predict_with_ensemble",
    "smooth_labels_to_state_collection",
    "label_states",
    "TrainedEnsemble", "FeatureNormalisation", "fit_normalisation",
    "mat_to_stc",
    "confusion_matrix",
    "compare_stcs", "LabelComparisonStats",
    "swap_state_vector_ids",
    "reassign_short_periods",
    "reassign_state_by_duration",
    "reduce_stc_to_loc",
    "mutual_information_states_features",
    "BehaviouralManifoldStats",
    "behavioural_manifold_stats",
    "map_to_reference_session",
]
