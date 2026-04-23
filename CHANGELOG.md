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
