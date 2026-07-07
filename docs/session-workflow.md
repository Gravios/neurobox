# Session workflow & directory structure

A reference for the neurobox data layout, the neurosuite-3 file formats,
and the commands to create and load a session with ephys + motion capture.
For a condensed version see the [README](../README.md); this document is
the full walkthrough.

---

## 1. Data directory layout

neurobox uses a single `<data_root>` (default `/data`, override with
`NB_DATA_PATH`) with three top-level trees. All per-session paths follow a
5-level hierarchy:

```
<type> / <sourceId> / <sourceId>-<userId> / <sourceId>-<userId>-<subjectId> / <session>
```

For session `sirotaA-jg-05-20120316` (sourceId `sirotaA`, userId `jg`,
subjectId `05`, date `20120316`) in project `B01`:

```
/data/
├── source/                                     raw acquisition data (read-only)
│   ├── nlx/    sirotaA/sirotaA-jg/sirotaA-jg-05/sirotaA-jg-05-20120316/
│   │              └── Events.nev, *.ncs, ...    Neuralynx raw
│   ├── mocap/  sirotaA/sirotaA-jg/sirotaA-jg-05/sirotaA-jg-05-20120316/
│   │              └── *.csv                     Vicon / Motive export
│   └── video/  sirotaA/sirotaA-jg/sirotaA-jg-05/sirotaA-jg-05-20120316/
│
├── processed/                                  format-converted, analysis-ready
│   ├── ephys/  sirotaA/sirotaA-jg/sirotaA-jg-05/sirotaA-jg-05-20120316/
│   │              ├── sirotaA-jg-05-20120316.yaml     session parameters
│   │              ├── sirotaA-jg-05-20120316.dat      wideband binary
│   │              ├── sirotaA-jg-05-20120316.lfp      downsampled LFP
│   │              ├── sirotaA-jg-05-20120316.all.evt  TTL events
│   │              └── (spike-sort artifacts — see §2)
│   └── mocap/  sirotaA/sirotaA-jg/sirotaA-jg-05/sirotaA-jg-05-20120316/
│                  └── <maze>/                   one subdir per maze (cof, nor, ...)
│                        └── sirotaA-jg-05-20120316.Trial001.mat
│
└── project/                                    analysis workspace (per project_id)
    └── B01/
        ├── .env                                NB_DATA_PATH + NB_PROJECT_ID
        ├── config/  figures/  scripts/  notebooks/  models/
        └── sirotaA-jg-05-20120316/             ← session workspace (spath)
              ├── sirotaA-jg-05-20120316.<file>          symlinks → processed/ephys
              ├── <maze>/sirotaA-jg-05-20120316.<file>   symlinks → processed/mocap
              └── (analysis outputs — see §3, real files not symlinks)
```

### Padded subject-ID variant

The Sirota lab's on-disk tree uses 6-digit zero-padded subject IDs even
though session names use the compact form. Both are handled transparently —
if the canonical path is absent, the padded variant is tried:

```
compact (session name):  .../sirotaA-jg-05/sirotaA-jg-05-20120316/
padded  (on disk):       .../sirotaA-jg-000005/sirotaA-jg-000005-20120316/
```

`link_session` prints which one it resolved to.

---

## 2. Spike-sort artifacts (neurosuite-3 variant naming)

Under `processed/ephys/.../<session>/`, per-shank files follow the
neurosuite-3 **variant (chain-of-custody) naming convention**. `<method>` is
`standard` (raw domain) or `stderiv` (spatial-derivative domain); `N` is the
1-based electrode-group index.

```
<base>.<type>.<method>.<N>          canonical (method-tagged)
<base>.<type>.<N>                    legacy (untagged; still readable)
<base>.<type>                        session-wide (no method, no group)
```

Three artifact classes with distinct resolution rules:

| Class            | Types                                | Naming                        | Resolution                                          |
| ---------------- | ------------------------------------ | ----------------------------- | --------------------------------------------------- |
| SessionWide      | dat, lfp, xml, yaml, fil, nrs, eeg   | `<base>.<type>`               | single path                                         |
| MethodSpecific   | clu, clc, clp, fet, pca, col         | `<base>.<type>.<method>.<N>`  | strict — no cross-variant fallback                  |
| Shared           | res, spk                             | `<base>.<type>.<method>.<N>`  | try `<method>` → `standard` → untagged legacy       |

Example files for group 1, standard variant:

```
sirotaA-jg-05-20120316.res.standard.1     spike times    (int64, no header)
sirotaA-jg-05-20120316.clu.standard.1     cluster IDs    (int32 header + int32 IDs)
sirotaA-jg-05-20120316.spk.standard.1     waveforms      (int16, sample-major)
sirotaA-jg-05-20120316.fet.standard.1     PCA features   (int32 header + int64 rows)
sirotaA-jg-05-20120316.pca.standard.1     PCA basis      (5×int32 header + doubles)
```

Hierarchical clustering adds a triple sharing one anchor:

```
sirotaA-jg-05-20120316.clu.standard.1     fiber (parent) layer
sirotaA-jg-05-20120316.clc.standard.1     atom  (child)  layer
sirotaA-jg-05-20120316.clp.standard.1     atom→fiber linkage map
```

Auxiliary formats:

```
sirotaA-jg-05-20120316.col.standard.1     collision decomposition (YAML)
sirotaA-jg-05-20120316.loc.1              per-spike source locations (5×float32)
sirotaA-jg-05-20120316.chunks.1           KiloKlustaKwik chunk boundaries (text)
sirotaA-jg-05-20120316.drift              session-level probe drift (YAML)
```

Retired: `.spkD` / `.fetD` / `.pcaD` → subsumed by `method="stderiv"`.

---

## 3. Analysis outputs (written to `spath`)

These are real files (not symlinks), created by `NBSession.create()` /
`.save()`. `cof` is the maze, `all` is the trial name:

```
sirotaA-jg-05-20120316.cof.all.ses.pkl          session checkpoint
sirotaA-jg-05-20120316.cof.all.pos.npz          assembled position array
sirotaA-jg-05-20120316.cof.all.stc.default.pkl  state collection
```

---

## 4. Package layout (the modules you'll import from)

```
neurobox/
├── config/
│   ├── config.py          configure_project, link_session, discover_mazes
│   ├── sessions_json.py    sessions.json catalog + EffectiveSession loader
│   └── session_lists.py    MATLAB session-list translation
├── dtype/
│   ├── session.py          NBSession, NBTrial, TrialWindow
│   ├── paths.py            NBSessionPaths (all path construction + ns3 naming)
│   ├── spikes.py           NBSpk (.load / .save)
│   ├── xyz.py  lfp.py  epoch.py  stc.py  ...   data containers
│   └── sync_pipelines.py   sync_nlx_vicon, sync_ephys_vicon, dispatch table
└── io/
    ├── load_clu_res.py     .res / .clu readers        + spikes_by_unit
    ├── load_spk.py         .spk reader (Shared fallback)
    ├── load_fet.py         .fet reader                 → FetData
    ├── load_pca.py         .pca reader                 → PcaBasis
    ├── load_clc_clp.py     .clc / .clp readers         + build_atom_to_fiber
    ├── load_yaml_ns3.py    .col / .drift readers
    ├── load_loc_chunks.py  .loc / .chunks readers      + LOC_COLUMNS
    └── ns3_writers.py      save_res/clu/clc/clp/spk/fet/pca/col/drift/loc/chunks
```

---

## 5. Commands — create & load a session with ephys + mocap

### One-time project setup

```python
from neurobox.config import configure_project

# Creates /data/project/B01/{config,figures,scripts,notebooks,models}
# and writes the .env file (NB_DATA_PATH + NB_PROJECT_ID).
configure_project("B01", data_root="/data")
```

### Step 1 — Link the session into the project tree

Creates `spath` and symlinks the processed ephys + mocap files into it.
Mazes are auto-discovered from the processed mocap tree when `mazes=None`.

```python
from neurobox.config import link_session

paths = link_session(
    session_name = "sirotaA-jg-05-20120316",
    project_id   = "B01",
    data_root    = "/data",          # or omit to read NB_DATA_PATH
    # mazes      = None,             # None → auto-discover; [] → ephys only
    # overwrite  = False,            # replace existing symlinks
)
# paths.spath           → /data/project/B01/sirotaA-jg-05-20120316
# paths.processed_ephys → .../processed/ephys/.../sirotaA-jg-05-20120316
```

### Step 2 — Create the session (synchronise ephys ↔ mocap)

`create()` reads the raw ephys + tracking data, aligns them to a common
clock, saves the assembled position array, and writes a `.ses.pkl`
checkpoint. This is a **one-time** operation per session.

```python
from neurobox.dtype import NBSession

session = NBSession(
    session_name = "sirotaA-jg-05-20120316",
    maze         = "cof",
    project_id   = "B01",
    data_root    = "/data",
)

session.create(
    data_loggers = ["nlx", "vicon"],   # primary (ephys), secondary (tracking)
    ttl_value    = "0x0040",           # TTL label marking mocap start
    stop_ttl     = "0x0000",           # TTL label marking mocap stop
    save_xyz     = True,               # write .pos.npz
)
```

`data_loggers` selects the sync pipeline:

```
['nlx', 'vicon']            Neuralynx + Vicon         (TTL events)
['nlx', 'optitrack']        Neuralynx + OptiTrack     (TTL events)
['nlx', 'motive']           Neuralynx + Motive        (TTL events)
['ephys', 'vicon']          generic ephys + Vicon     (TTL events)
['openephys', 'optitrack']  Open Ephys + OptiTrack    (ADC pulse channel)
['openephys', 'vicon']      Open Ephys + Vicon        (ADC pulse channel)
```

For OpenEphys pipelines, pass `sync_channel=<int>` and `threshold=<0-1>`
instead of `ttl_value` / `stop_ttl`.

After `create()`, the sync state is available on the object:

```python
session.window                    # TrialWindow — master-clock span of matched windows
session.xyz.data                  # (n_samples, n_markers, 3) position array
session.xyz.samplerate            # mocap sample rate (Hz)
session.xyz.stream_sync           # when mocap was actively recording
session.lfp.stream_sync           # LFP recording sync (single continuous segment)
```

### Step 3 — Load an existing session (subsequent runs)

Once `create()` has written the `.ses.pkl` checkpoint, later sessions just
load it — no re-sync.

```python
from neurobox.dtype import NBSession

session = NBSession(
    session_name = "sirotaA-jg-05-20120316",
    maze         = "cof",
    project_id   = "B01",
    data_root    = "/data",
)

session.load()             # loads .ses.pkl checkpoint
session.load("xyz")        # restore position array from .pos.npz
session.load("lfp")        # memory-map the .lfp binary
session.load("spk")        # load spikes (.res / .clu) → NBSpk
```

`load(field)` accepts: `'par'`, `'spk'`, `'lfp'`, `'dat'`, `'xyz'`,
`'stc'`, `'nq'`. With no argument it loads the `.ses.pkl`.

### Step 4 — Restrict to a trial window (optional)

```python
import numpy as np
from neurobox.dtype import NBTrial, TrialWindow

# A window covering 100–160 s on the master clock
window = TrialWindow(np.array([[100.0, 160.0]]), label="epoch1")

trial = NBTrial(
    session_name = "sirotaA-jg-05-20120316",
    maze         = "cof",
    trial_name   = "epoch1",
    project_id   = "B01",
    data_root    = "/data",
    window       = window,
)

xyz = trial.load("xyz")    # position restricted to the trial window
spk = trial.load("spk")    # spikes restricted to the trial window
```

---

## 6. Reading & writing spike-sort files directly

Path construction (naming lives here — pass resolved paths to the readers):

```python
from neurobox.dtype.paths import NBSessionPaths

p = NBSessionPaths("sirotaA-jg-05-20120316", "/data", "B01", maze="cof")

p.spk_file(1)                          # .spk.standard.1   (Shared)
p.fet_file(1, method="stderiv")        # .fet.stderiv.1    (MethodSpecific)
p.clc_file(1)                          # .clc.standard.1   (hierarchical child)

# Disk-aware resolution (Shared fallback: <method> → standard → untagged)
path, resolved_method = p.resolve_ns3("spk", 1, method="stderiv")
```

Readers (all naming-agnostic — take a path, return data):

```python
from neurobox.io import (
    load_clu_res, load_spk, load_fet, load_pca,
    load_clc, load_clp, build_atom_to_fiber,
    load_col, load_drift, load_loc, load_chunks,
)

res, clu, shank_map = load_clu_res("/data/.../sirotaA-jg-05-20120316")  # all shanks
fet = load_fet(p.fet_file(1))                        # FetData(features, timestamps, n_dimensions)
basis = load_pca(p.pca_file(1), n_samples=32)        # PcaBasis(means, eigenvectors, ...)
```

Writers (byte-exact round-trip with the readers; atomic, `overwrite=False`):

```python
from neurobox.io import save_res, save_clu, save_fet, save_pca

save_res(p.res_ns3_file(1), timestamps)
save_clu(p.clu_ns3_file(1), cluster_ids)
save_fet(p.fet_file(1, method="stderiv"), features, timestamps)
save_pca(p.pca_file(1, method="stderiv"), means, eigenvectors)
```

Object-level spike I/O:

```python
from neurobox.dtype.spikes import NBSpk

spk = NBSpk.load("/data/.../sirotaA-jg-05-20120316")   # reads .res + .clu, all shanks
# ... edit clusters ...
paths = spk.save("/data/.../sirotaA-jg-05-20120316", method="standard")
# writes .res.standard.N + .clu.standard.N per shank; returns the list of paths
```
