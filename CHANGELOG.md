## [Unreleased]

### Added

**Structured state decoding** —
`neurobox.analysis.classifiers.state_decoding`
- `fit_transition_matrix()` fits log transition probabilities and
  per-state minimum-duration floors (given in seconds) from
  hand-labelled sequences; `-1` (unlabelled) frames break transition
  counting.  Zero-observation rows fall back to uniform rather than
  NaN.  `labels_from_stc()` rasterises an `NBStateCollection` into
  the per-frame integer labels the fitter consumes.
- `viterbi_decode()` finds the maximum-a-posteriori label path.
  Minimum durations are enforced exactly via the expanded-state
  construction; forced advances carry the self-transition cost, so
  the decoder maximises the standard HMM path score subject to the
  constraint (the floor changes the feasible set, not the
  objective).  Bouts truncated by the recording edges are exempt.
- `decode_labels(probs, method="argmax"|"viterbi")` is the uniform
  entry point, and `smooth_labels_to_state_collection()` gains
  matching `decode=` / `transition_model=` parameters (default
  `"argmax"` — exact MATLAB parity; the median filter applies only on
  that path, since Viterbi output is already duration-constrained).
- Not a port: MATLAB had no equivalent.  This is the principled
  replacement for `argmax` + `ThreshCross` recommended by the
  `label_behavior` audit, and it works with every registered backend
  including the MATLAB-parity `patternnet`.  It is expected to
  subsume most of what `optimize_stc_transition.m` (stage 3) did by
  hand; that will be confirmed empirically before stage 3 is ported
  or dropped.
- Verified against brute force: on randomised small problems the
  decoded path score equals exhaustive enumeration over all legal
  label sequences, both unconstrained and under duration floors.

**Labelling evaluation harness** —
`neurobox.analysis.classifiers.evaluation`
- Frame metrics (`frame_scores`: accuracy, macro-F1 with
  absent-state-as-NaN semantics), segmental F1 at configurable IoU
  thresholds (Lea et al. 2017 greedy matching), boundary MAE in
  seconds, per-state bout statistics, and `loso_splits()` for
  leave-one-session-out cross-validation.
- `evaluate_labeling()` bundles the lot into a `LabelingScore` with a
  printable `summary()`.
- Rationale: the MATLAB pipeline scored per-frame confusion only,
  which is nearly blind to fragmentation and boundary error — the
  exact failure modes stage 3 existed to repair.  Two labelings with
  identical frame accuracy can differ wildly in bout structure; the
  segmental metrics make that visible and give the upcoming
  TCN-vs-patternnet comparison an honest yardstick.

**`fet_mis` feature set** — `neurobox.analysis.kinematics.fet_mis`
- Port of `MTA/features/fet_mis.m`; the 18-column feature basis the
  behaviour-labelling pipeline (`label_behavior` → `label_bhv_msnn`)
  uses by default.  Five spine/head pitch angles, trajectory-yaw PPC,
  four marker heights, four log10 XY speeds, two log10 Z speeds, spine
  sinuosity, and summed inter-segment yaw deviation.
- Parameters live in a frozen, hashable `FetMisConfig` so the result
  can key a `cached_compute` cache — replacing the concatenated
  `MTAC_BATCH+...` model-identity string the MATLAB pipeline built by
  hand.
- The PPC column is computed at the source sample rate and only then
  low-passed and resampled, matching MATLAB (where it came from a
  cached `.lsppc` file).  Computing it post-downsample would silently
  change what the `shift` parameter means.

**Stage-1 heuristic labeller** —
`neurobox.analysis.classifiers.heuristic_labeling`
- Port of `MTA/classifiers/label_behavior_with_heuristics.m`.  Detects
  `gper` / `walk` / `rear` through a ~15-threshold cascade over
  windowed trajectory statistics.
- `windowed_trajectory_stats()` replaces five near-identical 20-line
  loop blocks in the MATLAB with one vectorised helper.
- Every magic number is a documented field of the frozen
  `HeuristicThresholds` dataclass instead of being hardcoded inline.
- Returns an `NBStateCollection` rather than writing `.stc` files and
  mutating a session object, so it is testable and composable.

**neurosuite-3 variant (chain-of-custody) file naming** — `NBSessionPaths`
- Per-shank artifacts are classified into three resolution classes:
  SessionWide (`dat`, `lfp`, `yaml`, …), MethodSpecific (`clu`, `clc`,
  `clp`, `fet`, `pca`, `col`), and Shared (`res`, `spk`).
- `ns3_class(type)`, `ns3_file(type, shank, method)` (name-only), and
  `resolve_ns3(type, shank, method)` (disk-aware, returns
  `(path, resolved_method)`).
- Type-specific helpers: `spk_file`, `fet_file`, `pca_file`, `clc_file`,
  `clp_file`, `col_file`, plus variant-tagged `clu_ns3_file` /
  `res_ns3_file`.  Legacy singular `res_file` / `clu_file` unchanged for
  backward compatibility.
- Retired `.spkD` / `.fetD` / `.pcaD` names are subsumed by
  `method="stderiv"`.

**neurosuite-3 binary readers** — `neurobox.io`
- `load_fet` → `FetData(features, timestamps, n_dimensions)`
  (`.fet[.method].N`).
- `load_pca` → `PcaBasis(means, eigenvectors, …)` (`.pca[.method].N`).
- `load_clc` (atom child-layer) and `load_clp` → `ClpMap(parent_of, header)`
  (hierarchical `.clc` / `.clp`), with `build_atom_to_fiber` /
  `build_fiber_to_atoms` helpers.
- `load_col` / `load_drift` (YAML), `load_loc` / `load_chunks`
  (small binary + text), plus a `LOC_COLUMNS` constant.
- All readers are naming-agnostic — they accept variant-tagged, standard,
  or untagged-legacy paths.

**neurosuite-3 writers** — `neurobox.io.ns3_writers`
- `save_res`, `save_clu`, `save_clc`, `save_clp`, `save_spk`, `save_fet`,
  `save_pca` (binary); `save_col`, `save_drift` (YAML); `save_loc`,
  `save_chunks` (small binary + text).
- Every writer's output is byte-exact-reversible by the matching `load_*`.
- Guarantees: little-endian regardless of host, atomic replace via
  `<path>.tmp` + `os.replace`, `overwrite=False` default, shape/dtype
  validation, parent-dir auto-creation.

**`NBSpk.save(base, method, overwrite)`** — object-level spike I/O
- Complement of `NBSpk.load()`: writes the in-memory spike train back to
  per-shank `.res.<method>.N` / `.clu.<method>.N` file pairs via the ns3
  writers.  Returns the list of paths written.
- Reverses the cross-shank global-ID remap (see the shank_map change below)
  so a load → save round-trip reproduces the source `.clu` byte pattern.

**`link_session` — padded subject-ID resolution**
- Sirota-lab processed data uses 6-digit zero-padded subject IDs on disk
  (`sirotaA-jg-000005`) while session names use the compact form
  (`sirotaA-jg-05`).  New `NBSessionPaths` resolvers
  (`subject_id_padded`, `resolve_processed_ephys`, `resolve_processed_mocap`,
  `resolve_processed_mocap_session`) try the canonical path first, then the
  padded variant.  `link_session`, `discover_mazes`, and
  `sync_pipelines._find_file` use them; `link_session` prints the resolved
  source directory.

**`sync_ephys_vicon`** — generic-ephys entry point
- Thin wrapper over `sync_nlx_vicon` (unified neurobox layout), registered
  in the dispatch table for `data_loggers=['ephys', 'vicon']` and
  re-exported from `neurobox.utils.sync`.

**Documentation**
- `docs/session-workflow.md` — full directory-layout reference, neurosuite-3
  format tables, and create/load/trial-restrict walkthrough.
- README "Directory structure" section expanded with the processed subtree,
  padded subject IDs, and the variant-naming convention.

### Changed

**Python 3.13 and 3.14 support**
- Full test suite verified on CPython 3.10, 3.12, 3.13 and 3.14
  (1083 passed / 6 skipped on each; 1214 passed / 1 skipped on 3.14
  with every optional dependency group installed).  Cython extensions
  build cleanly on all four.  No source changes were required —
  `requires-python = ">=3.10"` already permitted them; the classifiers
  now advertise 3.13 and 3.14 as well.
- Verified against numpy 2.5, scipy 1.18, pandas 3.0, matplotlib 3.11,
  scikit-learn 1.9, PySide6 6.11, hmmlearn 0.3.3 and torch 2.13.

**Packaging metadata modernised (PEP 639)**
- `license = { text = "MIT" }` → `license = "MIT"` (SPDX expression), and
  the now-redundant `License :: OSI Approved :: MIT License` classifier is
  dropped.  Together these clear the setuptools deprecation warnings that
  will otherwise become hard errors.
- Build requirement raised `setuptools>=68` → `setuptools>=77`, which is
  the minimum that accepts the SPDX form.  Standard PEP 517 build
  isolation installs this automatically; only `--no-build-isolation`
  builds need setuptools upgraded by hand.

**`imagescnan` uses `Colormap.with_extremes`**
- Replaces the `copy()` + `set_bad()` pair, which matplotlib 3.11
  deprecates.  `with_extremes` returns a new colormap, so it preserves
  the isolation the explicit copy provided (a caller-supplied or
  registered colormap is still never mutated) while removing the
  deprecation warning.

**`TestKernelEquivalence` generalised**
- Now compares every pair from `KERNELS` instead of a hardcoded
  python/cython pair, and reports an accurate skip reason.  The
  pure-Python kernel was removed when Cython became a hard dependency,
  so the test currently skips — but it will start running automatically
  if a second kernel variant is registered.

**`load_clu_res` shank_map is now 3-column**
- `[global_cluster_id, shank_index, local_cluster_id]` (was 2-column).  The
  third column preserves the on-disk cluster ID so `NBSpk.save()` can undo
  the cross-shank global remap.  Callers using only columns 0–1 are
  unaffected.

**`load_spk_from_par`** — Shared-artifact resolution
- Searches `.spk.<method>.N` → `.spk.standard.N` → `.spk.N` (untagged legacy)
  instead of preferring the retired `.spkD`.  The incorrect
  `n_channels - 1` adjustment for `.spkD` is removed (the stderiv transform
  is applied downstream at PCA time; there is no separate `.spk` variant).
  Also fixes channel-count extraction from Struct-typed YAML groups.

### Fixed

**`bfet` defect in the heuristic labeller — documented and switchable**
- MATLAB line 138 of `label_behavior_with_heuristics.m` builds the
  `btraj` window stack from `hfet` (2-column, head only) instead of
  `bfet` (5-column, full spine chain).  `bfet` is computed and then
  discarded, so the derived `bf` trace duplicates `hf` — yet `bf` is
  load-bearing, driving the `body_turn` gate that subtracts turn
  intervals out of the walk periods.
- Reproduced by default (`reproduce_bfet_defect=True`) because the
  surrounding thresholds were tuned against this behaviour; silently
  correcting it would invalidate every historical label.  Set the flag
  to `False` to use `bfet` as intended, but re-validate against
  hand-labelled sessions before trusting the result.

**Spots `.pos` reader used native byte order**
- `sync_pipelines` read the spots tracker's `.pos` file with a native
  `np.int16`, the only binary reader in the package that didn't pin
  byte order — the other twelve all use explicit `<i2` / `<i4` / `<i8`
  / `<f4` / `<f8`.  On a big-endian host this silently produced
  byte-swapped coordinates rather than failing.  Now `"<i2"`.
- Added `tests/test_binary_byte_order.py`, a source-level guard that
  walks the AST of every module under `neurobox/{io,dtype,analysis,
  viz,config,utils}` and asserts that each `np.fromfile` /
  `np.frombuffer` / `np.memmap` call names an explicit byte order.
  This is deliberately a source check, not a behavioural one: on
  little-endian hardware `np.int16` and `"<i2"` are byte-for-byte
  identical, so no runtime assertion running on x86 or Apple Silicon
  could distinguish them.  The guard is self-tested (it must reject
  `np.int16` and accept `<i2`, `>f8`, `|i1`, `u1`) and was confirmed
  to flag the original defect at its exact line.

**Cython kernels used bare C `long` for int64 buffers**
- `_ccg_engine.pyx` and `_within_ranges_engine.pyx` declared their
  integer buffers as `cnp.ndarray[long, ...]` while allocating the
  backing arrays as `np.int64` and receiving `np.int64` from every
  Python caller.  The two coincide under LP64 (Linux, macOS) but C
  `long` is 32-bit under Windows' LLP64 model, so the declared buffer
  type and the actual dtype disagree there.
- Retyped to fixed-width `cnp.int64_t` throughout — buffer
  declarations, scalars, pointers, memoryviews and casts — so the
  declared type and the allocated dtype stay in lockstep regardless of
  platform data model.  A note in each module docstring records why,
  to stop the change being "simplified" back.
- Verified as a pure no-op on LP64: a golden-reference harness hashing
  the exact output bytes of both kernels over 80 randomised trials
  produces an identical digest before and after, on CPython 3.10,
  3.12, 3.13 and 3.14.


## [0.1.2] — 2026-04-24

### Added

**`neurobox.dtype.NBDang`** (new class)
- Port of `MTADang` / `MTADang.create`.
- Computes pairwise inter-marker spherical coordinates ``(T, N, N, 3)``
  using vectorised numpy broadcasting (azimuth θ, elevation φ, distance r).
- `from_xyz(xyz)` classmethod, `between(i, j, component)` named-pair access,
  `head_direction(from, to)` shorthand.
- `session.load("ang")` auto-loads xyz if needed.

**`neurobox.dtype.NBDufr`** (new class)
- Port of `MTADufr` / `MTADufr.create`.
- Unit firing-rate time-series at any reference sample rate (default 1250 Hz).
- Three modes: `'gauss'` (Gaussian, default 50 ms σ), `'boxcar'`,
  `'count'` (raw bin counts).
- Returns spikes/s; inherits `NBData.__getitem__` for epoch selection
  (`ufr[stc["walk"]]`).
- `session.load("ufr", samplerate, units, window, mode)` session loader.

**`NBSession.load("ang")` / `load("ufr")`**
- New field branches alongside the existing `spk`, `lfp`, `xyz`, `stc`.

**`NBSession.load("spk", restrict, periods)` — two new keyword arguments**
- `restrict=True` (default): automatically restrict spikes to `self.sync`
  when called on an `NBTrial`.  `restrict=False` returns the full recording.
- `periods`: explicit `NBEpoch` or `(N, 2)` float64 array override.

**MTA method parity** — new methods on existing types
- `NBData`: `copy()`, `clear()`, `update_path()`, `update_filename()`,
  `phase(freq_range)` — band-pass + Hilbert analytic phase.
- `NBEpoch`: `cast(target_mode)`, `save(path)`, `load_file(path)`.
- `NBSpk`: `clear()`, `copy()`, `save_unit_set()`, `load_unit_set()`.
- `NBDxyz`: `subset(marker_names)`, `get_pose_index(marker, threshold)`.
- `NBSession`: `update_paths()`, `list_trial_names()`.

### Fixed

- **`NBSession` — par auto-loaded on every open** (`_load_ses_file` now calls
  `_init_par()` when par is None after restoring from `.ses.pkl`).
- **`NBSession.save()` excludes par** — par is always re-read from the YAML
  so edits to the `units:` block are picked up without re-running create.
- **`session.load("lfp")` and `load("xyz")` now respect `trial.sync`** —
  LFP reads only the trial sample range from disk; xyz is masked to trial
  frames.  Equivalent to `MTAData.resync` but implemented with three lines
  using `NBEpoch.to_mask` / `load_binary(periods=...)`.
- **`get_lfp_samplerate` — added legacy XML fallback** — checks
  `par.lfpSampleRate` (old ndManager XML / old YAML) when
  `par.fieldPotentials.lfpSamplingRate` is absent.
- **`autolabel` `fillgaps` wrong kwarg** — `min_gap_sec=` → `gap_sec=`;
  added 0.5 s minimum-duration filter to drop artefact bursts.
- **LFP sync sample-rate conversion robustness** — explicit `int64` dtype;
  safe fallback chain `self.par → load_par(disk) → sr_default`.

## [0.1.1] — 2026-04-22

### Added

**`neurobox.io`**
- `load_units` / `UnitAnnotation` / `map_annotations_to_global_ids` —
  parse the `units:` block from ndManager-yaml parameter files into typed
  `UnitAnnotation` dataclass objects; resolve per-cluster global IDs
  against `NBSpk.map` (port of the ndManager-yaml curation workflow)
- `get_lfp_samplerate(par, default)` — single helper for both YAML format
  variants; reads `par.fieldPotentials.lfpSamplingRate` (new ndManager-yaml
  format); exported from `neurobox.io` and `neurobox`

**`neurobox.dtype.NBSpk`**
- `annotations` field — automatically populated from the `units:` YAML
  block on `NBSpk.load()`
- `annotation_for(unit_id)` — look up a unit's `UnitAnnotation` by global
  or `(shank, local_cluster)` fallback
- `annotated_unit_ids(quality, cell_type, structure)` — filter units by
  YAML curation tags; compatible with `neural_scattering` workflows

**`neurobox.analysis.NeuronQualityResult`**
- `yaml_quality`, `yaml_cell_type`, `yaml_structure`,
  `yaml_isolation_distance` fields — merged from `UnitAnnotation` when
  annotation data is present
- `is_single_unit()` now uses `yaml_quality` as a hard veto when set
  (e.g. units tagged `mua` fail regardless of ISI / SNR metrics)
- `print_neuron_quality_report` shows a Quality column when annotations
  are present

**`scripts/`**
- `convert_session_list.py` — promoted to installable console script
  `nb-convert-sessions`; converts `get_session_list_v3.m` to
  `sessions.json` (argparse CLI, `scripts/__init__.py` added)

### Fixed

- `NBDlfp.load` and `sync_pipelines._load_record_sync` — replaced direct
  `par.lfpSampleRate` / `getattr(par, "lfpSampleRate", …)` accesses with
  `get_lfp_samplerate(par)`, fixing compatibility with the ndManager-yaml
  `fieldPotentials.lfpSamplingRate` layout used in neurosuite-3 ≥ 0.3
- `README.md` — `spk.by_unit` documented as property (was `by_unit()`)
- `pyproject.toml` — permanently exclude stale `test_pyqtgraph*.py` files
  from collection so plain `pytest tests/` works without extra flags

# Changelog


**`neurobox.io`** (additions)
- `load_processed_mat` / `concatenate_processed_mat` — load MTA-format
  processed ``.mat`` files from ``processed/mocap/.../session/maze/``
  (port of MTA ``concatenate_vicon_files``)
- `get_event_times` / `get_ttl_periods` — convenience TTL query wrappers
  (port of MTA ``sync_nlx_events`` and inline event-parsing idioms)
- `fill_gaps` / `fill_xyz_gaps` / `detect_gaps` — pchip/linear
  interpolation of dropout gaps in assembled xyz arrays
  (port of MTA ``mocap_fill_gaps``)

**`neurobox.dtype.sync_pipelines`** (updated)
- `_load_mocap_files` now tries processed ``.mat`` files first, then
  falls back to raw Motive CSV exports — matching the actual data flow
- All pipeline functions gained `save_xyz`, `tolerance_sec` parameters
  and per-phase progress output
- `_find_file` / `_has_paths` / `_ephys_base` helpers use
  ``NBSessionPaths`` for two-stage path resolution
- `NBSession.create` ensures par is loaded, spath exists, and injects
  ``save_xyz`` / ``tolerance_sec`` before dispatch

## [0.1.0] — unreleased

### Added

**`neurobox.io`**
- `load_par` — load neurosuite-3 YAML parameter files
- `load_yaml` — YAML → Struct parser; `get_channel_groups` helper
- `load_binary` — binary `.dat` / `.lfp` loader with mmap, period selection,
  µV conversion, `channel_first` flag; fixes time-axis indexing bug in
  original `load_binary`
- `load_clu_res` — binary `.res.N` (int64 LE) / `.clu.N` (int32 LE) loader
  with multi-shank global ID remapping, `as_seconds`, `sampling_rate` args
- `spikes_by_unit` — split concatenated arrays into `dict[unit → times]`
- `load_spk` / `load_spk_from_par` — `.spk.N` / `.spkD.N` waveform loader
- `load_evt` / `evt_to_periods` — `.evt` event file loader

**`neurobox.dtype`**
- `Struct` — MATLAB-style attribute dict
- `NBEpoch` — time period container (seconds); set operators `& | - +`;
  `to_mask` / `to_periods` conversion; `fillgaps`; `resample`
- `NBData` — abstract base for time-series data; `__getitem__` period
  selection; `filter` (Butterworth / Gaussian / rect); `resample`; `segs`
- `NBModel` — marker-name registry; `index` / `indices` / `resolve` /
  `subset`; `default_rat()`; `from_csv_headers()`
- `NBSpk` — spike container; `__getitem__` with epoch restriction;
  `restrict`; `by_unit`; `load` classmethod
- `NBDxyz` — 3-D position time-series; `sel`; `vel` / `acc` / `dist` /
  `com`; `from_motive_csv`; `save_npy` / `load`
- `NBDlfp` — LFP time-series; lazy `load`; `csd`; inherits `filter` /
  `resample`
- `NBStateCollection` — named epoch collection; DSL query language
  (`stc['walk&theta']`); `get_transitions`; `filter`; pickle persistence
- `NBSessionPaths` — path resolver for the 5-level source/processed/project
  hierarchy; `parse_session_name` / `build_session_name`
- `NBSession` — top-level session container; `create` sync dispatch;
  `load` field loader; `validate` quick-load (string / filebase / list /
  dict / existing object); `stc_mode` auto-load
- `NBTrial` — session subset by sync epoch; `validate`

**`neurobox.dtype.sync_pipelines`**
- `sync_nlx_vicon` — NLX primary + Vicon/Optitrack (event-based TTL)
- `sync_nlx_spots` — NLX primary + 2-LED tracker `.pos`
- `sync_nlx_whl`   — NLX primary + `.whl` (already on-clock)
- `sync_openephys_optitrack` — OpenEphys primary + Optitrack (pulse channel)
- `sync_openephys_vicon`     — OpenEphys primary + Vicon (pulse channel)
- `dispatch` — keyword-pattern lookup table

**`neurobox.config`**
- `configure_project` — create project skeleton + write `.env`
- `link_session` — create session directory with individual file symlinks
- `load_config` — read `.env` with three-way search
- CLI entry points `nb-configure` and `nb-link`
