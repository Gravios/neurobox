# labbox × MTA — AST Cross-Reference and Analysis Function Catalogue

**labbox:** 1,118 functions across 15 directories  
**MTA:** 1,623 functions across 55 top-level namespaces  
**Common (shared verbatim):** 78 functions  
**labbox-unique:** 1,040 functions  
**labbox functions actively called by MTA analysis code:** 88  

---

## 1. Common Functions (78 total)

These exist in both toolboxes, mostly as shared third-party libraries vendored by both.

### Shared third-party toolboxes (can be de-duplicated — use one copy)

| Library | Count | Functions |
|---------|------:|-----------|
| **CircStat** (circ_*) | 30 | `circ_mean`, `circ_std`, `circ_r`, `circ_kappa`, `circ_rtest`, `circ_vmpar/pdf/rnd`, `circ_corrcc/corrcl`, `circ_wwtest`, `circ_hktest`, `circ_dist/dist2`, `circ_axialmean`, `circ_stats`, `circ_plot` + 16 others |
| **FastICA** | 10 | `fastica`, `fasticag`, `fpica`, `pcamat`, `remmean`, `whitenv`, `demosig`, `dispsig`, `icaplot`, `gui_*` |
| **boundedline** | 1 | `boundedline` |
| **struct2xml** | 1 | `struct2xml` |

### Substantive functions shared by both (direct duplicates — identical implementations)

| Function | labbox path | MTA path | Notes |
|----------|-------------|----------|-------|
| `DefaultArgs` | `Helper/DefaultArgs.m` | `utilities/DefaultArgs.m` | Core argument-default utility used everywhere |
| `SelectPeriods` | `Helper/SelectPeriods.m` | `utilities/SelectPeriods.m` | Period-based indexing — already ported to `neurobox.dtype.epoch.select_periods` |
| `IntersectRanges` | `TF/IntersectRanges.m` | `utilities/IntersectRanges.m` | Range intersection — ported to `NBEpoch.__and__` |
| `NeuronQuality` | `Helper/NeuronQuality.m` | `utilities/NeuronQuality.m` | Already ported to `neurobox.analysis.neuron_quality` |
| `LoadBinary` | `IO/LoadBinary.m` | (via `@MTADlfp/load.m`) | Ported to `neurobox.io.load_binary` |
| `LoadClu` | `IO/LoadClu.m` | `utilities/` | Ported to `neurobox.io.load_clu_res` |
| `LoadCluRes` | `IO/LoadCluRes.m` | `utilities/` | Ported |
| `LoadRes` | `IO/LoadRes.m` | `utilities/` | Ported |
| `LoadPar` | `IO/LoadPar.m` | `utilities/` | Ported to `neurobox.io.load_par` |
| `LoadXml` | `IO/LoadXml.m` | `xml/` | Ported to `neurobox.io.load_xml` |
| `LoadEvents` | `IO/LoadEvents.m` | `utilities/` | Ported to `neurobox.io.load_evt` |
| `LoadSpk` | `IO/LoadSpk.m` | `utilities/` | Ported to `neurobox.io.load_spk` |
| `parseXML` | `IO/parseXML.m` | `xml/` | Ported |
| `DetectOscillations` | `TF/DetectOscillations.m` | `utilities/` | Not yet in neurobox |
| `DetectRipples` | `TF/DetectRipples.m` | `utilities/` | Not yet in neurobox |
| `CSDTrigAverTheta` | `basic_lfp_analysis/csd/` | `analysis/` | Not yet in neurobox |
| `csd` | `CSD/iCSD/csd.m` | `@MTADlfp/csd.m` | Ported to `NBDlfp.csd()` |
| `Comodugram` | `TF/Comodugram.m` | `utilities/` | Not yet in neurobox |
| `FileQuality` | `Helper/FileQuality.m` | `utilities/` | Not yet in neurobox |
| `MakeUniformDistr` | `Helper/MakeUniformDistr.m` | `utilities/transforms/` | Not yet in neurobox |
| `ResolvePath` | `IO/ResolvePath.m` | `utilities/` | Not yet in neurobox |
| `reportfig` | `Helper/reportfig.m` | `utilities/` | Not yet in neurobox |
| `unity` | `Helper/unity.m` | `@MTAData/unity.m` | Ported to `NBData` |
| `imagescnan` | `Graphics/imagescnan.m` | `utilities/graphics/` | Not yet in neurobox |
| `linkax` | `Graphics/linkax.m` | `utilities/graphics/` | Not yet in neurobox |
| `subplot2` | `Graphics/subplot2.m` | `utilities/graphics/` | Not yet in neurobox |

---

## 2. labbox Functions Called by MTA Analysis — High-Priority Ports (88)

These are labbox functions that MTA's analysis scripts call directly. They represent **immediate porting candidates** for `neurobox`.

### 2A. Signal processing & filtering

| Function | Description | MTA callers (sample) |
|----------|-------------|---------------------|
| `ButFilter` | Butterworth band-pass/low-pass/high-pass filter | `CSDTrigAverTheta`, `DetectOscillations`, `DetectStructuredOscilations` |
| `Filter0` | Zero-phase forward-backward filter | `CorrectedUFRCCGAnalysis`, `MTAAccg`, `MTAccg` |
| `FirFilter` | FIR filter design and application | `DetectRipples` |
| `MTFilter` | Multi-taper filter | — |
| `WhitenSignal` | AR whitening of LFP signal | `remove_emg_long`, `bhv_body_head`, `bhv_lfp_psd` |
| `AdaptiveFilter` | Adaptive filter for line noise removal | — |
| `EMG_rm` | Remove EMG noise from LFP via ICA | `remove_emg_from_dat`, `req20200118` |
| `EMG_rm_main` | Pipeline wrapper for EMG removal | `remove_emg_from_dat`, `remove_emg_main` |
| `EMG_rm_long` | EMG removal for long recordings | `remove_emg_long` |
| `EMG_rm_linenoise` | Line-noise removal | `remove_emg_linenoise` |
| `EMG_Cluster` | Cluster EMG activity above threshold | `remove_emg_cluster` |
| `EMG_Cluster_s` | Session-level EMG clustering | `remove_emg_cluster_s` |
| `StartEnd1d` | Start/end indices of 1D blocks | `remove_emg_cluster`, `remove_emg_long` |

### 2B. Spectral analysis (multi-taper)

| Function | Description | MTA callers |
|----------|-------------|-------------|
| `mtcsdlong` | Cross-spectral density (long recordings) | `bhv_ncp_distrb`, `bhv_ncp_distrib` |
| `mtcsdglong` | Cross-spectrogram (long) | `bhv_body_head`, `bhv_ncp_distrb` |
| `mtchglong` | Cross-coherogram (long) | `MyFirstPaperFigs`, `bhv_body_head`, `bhv_drhm` |
| `mtcsd` | Multi-taper CSD | `ed_HeadMotionAnalisysAllSessions_Spec` |
| `mtcsdfast` | Fast CSD (single-segment) | `remove_emg_linenoise` |
| `mtchd` | Coherence + phase | `ed_HeadMotionAnalisysAllSessions_Spec` |
| `mtcsglong` | Power spectrogram (long) | `label_lfp_states`, `ed_HeadMotionAnalisysAllSessions_Spec` |
| `mtfft` | Multi-taper FFT | `req20220103` |
| `mtptchd` | Point-process coherence | `unit_lfp_spkcohere`, `unit_rhm_spkcohere` |
| `Comodugram` | Comodulogram (cross-frequency coupling) | — |
| `Harmonix` | Harmonic power analysis | — |
| `CCG` | Cross/auto-correlogram of spike trains | `EgoProCode2D_f1_data`, `MjgER2016F3V1` |
| `CCGSignif` | CCG significance via convolution | — |
| `Trains2CCG` | Spike-train list → CCG | `MTAAccg`, `MTAccg` |
| `FiringRate` | Firing rate from `.res`/`.clu` | `rear_ccg_stats`, `rearing`, `rf_ccg_stats` |
| `SmoothFiring` | Gaussian-smoothed firing rates | — |
| `HistFiring` | Binned firing rate histogram | — |

### 2C. Oscillation detection & brain-state labelling

| Function | Description | MTA callers |
|----------|-------------|-------------|
| `DetectOscillations` | Detect theta/gamma oscillation bouts | `label_lfp_states` |
| `DetectRipples` | Detect SWR events | — |
| `HmmStateSegment` | HMM-based state segmentation | `label_lfp_states_keymap`, `req20211210` |
| `CheckEegStates` | Spectral state verification UI | `groom`, `labelTheta`, `MjgER2016_select_sessions` |
| `ThreshCross` | Threshold crossing detection | `DetectOscillations`, `DetectStructuredOscilations` |
| `LocalMinima` | Local minima in 1D signal | `DetectOscillations`, `EgoProCode2D_f3_generate_v2` |
| `LocalMinimaN` | Local minima with constraints | `bhv_lfp_psd`, `center_head` |
| `SchmittTrigger` | Hysteresis-based threshold crossing | `theta` |

### 2D. CSD analysis

| Function | Description | MTA callers |
|----------|-------------|-------------|
| `CurSrcDns` | Traditional (Vaknin) CSD | `CSDTrigAverTheta`, `req20210926` |
| `CSDTrigAverTheta` | Theta-triggered CSD average | — |
| `PlotCSD` | CSD visualization | `rscpt` |
| `PlotManyCh` | Multi-channel trace plot | `CSDTrigAverTheta` |

### 2E. I/O

| Function | Description | MTA callers |
|----------|-------------|-------------|
| `LoadBinary` | Low-level binary data loader | `CSDTrigAverTheta`, `MTABrowser`, `expl_theta` |
| `LoadClu` | Load `.clu` cluster file | `FileQuality`, `NeuronQuality` |
| `LoadCluRes` | Combined `.clu` + `.res` loader | `MjgER2016_supfig_multiKlusterValidation` |
| `LoadSpk` | Load waveform snippets | `NeuronQuality`, `compute_neuron_quality` |
| `LoadPar` | Load XML parameter file | `MTABrowser`, `get_dat_source_lengths` |
| `LoadXml` | Parse XML session file | `CSDTrigAverTheta`, `NeuronQuality` |
| `LoadEvents` | Load `.evt` event file | `check_xyz_nlx_consistency`, `convert_nlx_events_to_mta_epochs` |
| `LoadSegs` | Load segmented waveform data | `label_lfp_states_keymap` |
| `LoadFet` | Load `.fet` feature file | `FileQuality`, `link_clusters` |
| `LoadRes` | Load `.res` spike timestamps | `load_ndm_res`, `load_res` |
| `bload` | Load generic binary file | `label_lfp_states`, `load_spk` |
| `msave` | Save integer matrix as ASCII | `DetectRipples`, `DetectStructuredOscilations` |
| `Save2sts` | Save state periods as `.sts` | `convert_stc2sts` |
| `MakeEvtFile` | Write `.evt` event file | `DetectRipples` |
| `load_open_ephys_data_blocks` | Load Open Ephys continuous blocks | `convert_oephys_to_dat_subses_block` |
| `load_open_ephys_header` | Read Open Ephys file header | `convert_oephys_to_dat_subses_block` |
| `oephys2dat` | Convert Open Ephys → `.dat` | `ecube_map_continuous_to_dat` |
| `oephys2dat_subses_blocks` | Convert by sub-session blocks | `convert_oephys_to_dat_subses_block` |
| `mergedat` | Merge multiple `.dat` files | `ecube_map_continuous_to_dat` |
| `CleanProcessedSession` | Post-processing cleanup utility | `ecube_process_session` |
| `ProcessEcube` | eCube acquisition preprocessing | `ecube_process_session` |

### 2F. Statistics

| Function | Description | MTA callers |
|----------|-------------|-------------|
| `RankCorrelation` | Spearman rank correlation | `ex20130517`, `rear_ccg_stats` |
| `RayleighTest` | Rayleigh test for phase locking | `ed_HeadMotionAnalisysAllSessions_Spec` |
| `PPC` | Pairwise phase consistency | `MjgER2016_decode`, `MjgER2016_decode_xy` |
| `fdr_bh` | Benjamini-Hochberg FDR correction | `rear_ccg_stats`, `rearing` |
| `gausshmm` | Gaussian HMM fitting | `bhv_hmm`, `bhv_hmm_grp` |
| `erpPCA` | ERP PCA (Varimax) | `MjgER2016_figure_BhvThetaDecomposition` |
| `VonMisesRnd` | Von Mises random samples | `MjgER2016_figure5` |

### 2G. Utility / data manipulation

| Function | Description | MTA callers |
|----------|-------------|-------------|
| `Accumulate` | Fast histogram accumulation (2D) | `PlotPF_new` |
| `CatStruct` | Concatenate struct arrays | `EgoProCode2D_f1_data`, `MjgER2016_figure_BhvPlacefields` |
| `GetSegs` | Extract data segments by start-points | `CorrectPointErrors`, `CorrectedUFRCCG` |
| `Ind2Sub` | Linear index → subscripts | `MTAAPfknncorm`, `MTAAknnpfs` |
| `IntersectRanges` | Intersect epoch ranges | `CorrectedUFRCCG`, `JoinRanges` |
| `NearestNeighbour` | Nearest-neighbor lookup | `DetectOscillations`, `EgoProCode2D_decoding` |
| `WithinRanges` | Test if values fall within ranges | `DetectOscillations`, `EgoProCode2D_decoding` |
| `MedianFilter` | Running median filter | `req20220128`, `req20220223` |
| `sq` | Squeeze matrix dimensions | `CorrectPointErrors`, `CorrectedUFRCCG` |
| `mud` | Column mean utility | `MjgER2016_decode`, `MjgER2016_decode_xy` |
| `cell2array` | Flatten cell array to regular array | `MjgER2016_figure_BhvPlacefields` |
| `clip` | Clamp values to range | `Comodugram`, `MjgER2016_figure5` |

### 2H. Graphics

| Function | Description | MTA callers |
|----------|-------------|-------------|
| `ForAllSubplots` | Apply command to all subplots | `EgoProCode2D_decoding` |
| `Lines` | Draw lines on axes | `ConvertBhv2Stc`, `CorrectedUFRCCG` |
| `hist2` | 2D histogram with smoothing | `ClusterH2`, `EgoProCode2D_compute_egoratemaps` |
| `histcI` | Histogram with interval indices | `rear_ccg_stats`, `rearing` |
| `histcirc` | Circular histogram | `EgoProCode2D_f2_data`, `MjgER2016_figure_BhvClassification` |
| `PlotTraces` | Multi-trace LFP display | `label_lfp_states_keymap`, `req20200118` |
| `TrigRasters` | Peri-event raster plot | `bhv_gamma_burst`, `rear_ccg_stats` |
| `suptitle` | Figure super-title | `MjgER2016_figure5` |
| `tight_subplot` | Compact subplot layout | `EgoProCode2D_f4_generate`, `MjgER2016_figure5` |
| `subplotfit` | Subplot with aspect fitting | `MjgER2016_decode`, `MjgER2016_decode_xy` |
| `export_fig` | High-quality figure export | `MethodsPaper` |
| `append_pdfs` | Append PDF figures | `MjgER2016_figure_thetaPhasePrecession_alt2` |
| `copyax` | Copy axes between figures | `req20171205` |
| `linkx` | Link x-axes across subplots | `EgoProCode2D_f2_generate` |
| `ellipse` | Draw ellipse on axis | `error_ellipse` |

---

## 3. Full labbox Categorization (1,040 unique functions)

### Category summary

| Category | Functions | MTA-called | neurobox status | Priority |
|----------|----------:|----------:|-----------------|----------|
| **Signal filtering** (TF) | 80 | 26 | Partial — `NBDlfp.filter` placeholder | **HIGH** |
| **Spectral analysis** (TF/mtXxx) | 18 | 14 | Not ported | **HIGH** |
| **Oscillation detection** (TF) | 12 | 8 | Not ported | **HIGH** |
| **CSD analysis** (CSD + basic_lfp) | 382 + 81 = 463 | 7 | `NBDlfp.csd()` only | MEDIUM |
| **Ripple/SWR detection** | 12 | 0 | Not ported | MEDIUM |
| **EMG removal** | 42 | 18 | Not ported | MEDIUM |
| **Brain state detection** | 9 | 0 | Not ported — covered by `NBStateCollection` | LOW |
| **Statistics** (Stats) | 88 | 7 | Not ported | HIGH |
| **I/O** (IO) | 95 | 16 | Mostly ported | LOW |
| **Graphics** (Graphics) | 135 | 19 | Not ported — display only | LOW |
| **Helper utilities** (Helper) | 101 | 23 | Partial | MEDIUM |
| **Open Ephys processing** (oephys) | 57 | 7 | `sync_openephys_*` pipelines | LOW |
| **Behavior** | 12 | 0 | Not ported | LOW |

---

## 4. Recommended neurobox.analysis Module Structure

Based on what MTA analysis scripts actually call from labbox, here is the proposed module layout:

```
neurobox/
  analysis/
    neuron_quality.py      ✓ done
    transform_origin.py    ✓ done

    # --- HIGH PRIORITY (MTA analysis calls these directly) ---
    lfp/
      __init__.py
      filtering.py         ButFilter, Filter0, FirFilter  →  butter_filter(), fir_filter(), filter0()
      spectral.py          mtcsd, mtcsdglong, mtchglong, mtcsglong  →  multitaper_psd(), cross_spectrogram(), coherogram()
      oscillations.py      DetectOscillations, DetectRipples, ThreshCross  →  detect_oscillations(), detect_ripples()
      phase.py             phase_locking(), spike_phase(), instantaneous_phase()
      csd.py               CurSrcDns, CSDTrigAverTheta  →  current_source_density(), triggered_csd()

    spikes/
      __init__.py
      ccg.py               CCG, Trains2CCG, CCGSignif  →  cross_correlogram(), auto_correlogram()
      firing_rate.py       FiringRate, SmoothFiring, HistFiring  →  (already in NBDufr; add epoch-based wrappers)
      phase_locking.py     RayleighTest, PPC, spike_phase_histogram()

    spatial/
      __init__.py
      place_fields.py      PlaceField, PlotPF_new  →  occupancy_map(), rate_map(), place_field_detect()
      head_direction.py    (uses NBDang)  →  hd_tuning_curve(), hd_information()
      spatial_info.py      SpatialInformation, SkaggsBitsPerSpike

    stats/
      __init__.py
      circular.py          circ_mean, circ_r, circ_rtest, VonMisesFit, RayleighTest, PPC
      correction.py        fdr_bh, bonf_holm
      smoothing.py         BinSmooth, MeanSmooth, MedianSmooth
      hmm.py               gausshmm  →  fit_gaussian_hmm()

    emg/
      __init__.py
      removal.py           EMG_rm, EMG_rm_long, EMG_rm_main  →  remove_emg()
      detection.py         EMG_Cluster  →  detect_emg_epochs()
```

---

## 5. Immediate Port Candidates (ordered by scientific impact on B01)

The following 12 functions from labbox should be ported to `neurobox.analysis` in roughly this order:

1. **`ButFilter` / `Filter0`** → `neurobox.analysis.lfp.filtering`  
   Called by 20+ MTA analysis scripts; needed for theta detection, EMG removal, ripple detection.

2. **`mtcsdglong` / `mtchglong` / `mtcsglong`** → `neurobox.analysis.lfp.spectral`  
   Multi-taper spectrograms and coherograms. Core to most LFP characterization analyses.

3. **`DetectOscillations` / `ThreshCross`** → `neurobox.analysis.lfp.oscillations`  
   Theta bout detection feeds directly into `NBStateCollection`; `ThreshCross` is a primitive used by 8+ scripts.

4. **`CCG` / `Trains2CCG`** → `neurobox.analysis.spikes.ccg`  
   Auto/cross-correlograms. Required for inhibitory/excitatory interaction analysis.

5. **`RayleighTest` / `PPC`** → `neurobox.analysis.spikes.phase_locking`  
   Phase-locking significance tests. Required for theta-phase coding analyses.

6. **`CurSrcDns`** → `neurobox.analysis.lfp.csd`  
   Traditional Vaknin CSD. Needed for laminar profile analysis alongside the existing `NBDlfp.csd()`.

7. **`PlaceField` / occupancy maps** → `neurobox.analysis.spatial.place_fields`  
   Rate maps and occupancy normalization. Central to B01 project goals.

8. **`fdr_bh`** → `neurobox.analysis.stats.correction`  
   BH FDR correction; needed by ripple, CCG, and tuning-curve analyses.

9. **`gausshmm`** → `neurobox.analysis.stats.hmm`  
   Gaussian HMM. Used by `bhv_hmm` for continuous behavioral state decomposition.

10. **`EMG_rm` / `EMG_Cluster`** → `neurobox.analysis.emg`  
    EMG removal pipeline. Pre-processing step required before LFP spectral analysis.

11. **`histcirc` / `VonMisesFit`** → `neurobox.analysis.stats.circular`  
    Circular statistics for head-direction and theta-phase analyses.

12. **`DetectRipples`** → `neurobox.analysis.lfp.oscillations`  
    SWR event detection. Required for ripple-triggered CSD and replay analyses.
