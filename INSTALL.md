# neurobox 0.1.1 — drop-in patch

## What this is

A minimal patch archive containing only the files that changed between
`a0c34b3` ("update") and `ea3ec54` ("0.1.1 release").  It is safe to
extract directly over an existing neurobox installation.

## Apply to an editable install (source checkout)

```bash
# From your neurobox source root:
tar -xzf neurobox-0.1.1-dropin.tar.gz --strip-components=1
pip install -e ".[dev]"   # reinstall to pick up new entry points
```

## Apply to a site-packages install

```bash
# Find where neurobox is installed:
python -c "import neurobox; print(neurobox.__file__)"
# e.g. /opt/anaconda3/envs/myenv/lib/python3.12/site-packages/neurobox/__init__.py

# Extract into that directory (adjust path):
SITE=$(python -c "import neurobox; import os; print(os.path.dirname(os.path.dirname(neurobox.__file__)))")
tar -xzf neurobox-0.1.1-dropin.tar.gz --strip-components=1 -C "$SITE"
```

## What changed

See `CHANGELOG.md` for full details.  Summary:

- **`neurobox/io/load_units.py`** (new) — parse `units:` block from
  ndManager-yaml par files into typed `UnitAnnotation` objects
- **`neurobox/io/load_yaml.py`** — `get_lfp_samplerate()` reads the new
  `par.fieldPotentials.lfpSamplingRate` key (ndManager-yaml ≥ 0.3)
- **`neurobox/dtype/spikes.py`** — `NBSpk.annotations`, `annotation_for()`,
  `annotated_unit_ids()` auto-populated from YAML `units:` on load
- **`neurobox/analysis/neuron_quality.py`** — `NeuronQualityResult` gains
  `yaml_quality / yaml_cell_type / yaml_structure / yaml_isolation_distance`;
  `is_single_unit()` uses YAML quality as a hard veto
- **`neurobox/dtype/lfp.py`** and **`sync_pipelines.py`** — replaced stale
  `par.lfpSampleRate` accesses with `get_lfp_samplerate(par)`
- **`scripts/convert_session_list.py`** — promoted to console script
  `nb-convert-sessions` via `pyproject.toml`
