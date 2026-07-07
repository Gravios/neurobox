## [Unreleased]

### Added

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
