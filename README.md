# neurobox

Analysis toolbox for silicon-probe electrophysiology, integrated with
[neurosuite-3](https://github.com/Gravios/neurosuite-3).

## Features

- Load neurosuite-3 binary spike/LFP/waveform data (`.res`, `.clu`, `.spk`, `.lfp`, `.dat`)
- Parse YAML and XML session parameter files; par auto-loaded on every session open
- Synchronise ephys with motion-capture (Vicon, Optitrack) and 2-D tracking
- `NBSession` / `NBTrial` containers with lazy, sync-aware data loading
- `NBStateCollection` behavioural state management with set-algebra DSL (`stc["walk&theta"]`)
- `NBDang` — pairwise inter-marker spherical coordinates (head direction, pitch, distance)
- `NBDufr` — unit firing-rate time-series (Gaussian/boxcar smoothing, spikes/s)
- `neuron_quality` — ISI contamination, SNR, spike width metrics with YAML curation merge
- Structured project / data directory management with CLI entry points

## Installation

```bash
pip install neurobox
```

For optional dependencies (pandas, scikit-learn):

```bash
pip install "neurobox[full]"
```

Editable install from source:

```bash
git clone https://github.com/Gravios/neurobox.git
cd neurobox
pip install -e ".[dev]"
```

## Quick start

### Configure a project

```python
from neurobox.config import configure_project, link_session

# One-time project setup — writes /data/project/B01/.env
configure_project("B01", data_root="/data")

# Link a session into the project directory
link_session("sirotaA-jg-05-20120316", "B01", mazes=["cof"])
```

Or from the command line:

```bash
nb-configure --project-id B01 --data-root /data
nb-link --session sirotaA-jg-05-20120316 --project-id B01 --maze cof
```

### Load a session

```python
from neurobox.dtype import NBSession, NBTrial

# Load by session name (reads from .env in cwd or NB_DOTENV_PATH)
session = NBSession("sirotaA-jg-05-20120316", maze="cof")

# Or use validate() for flexible input (string / dict / list)
session = NBSession.validate("sirotaA-jg-05-20120316.cof.all")
trial   = NBTrial.validate({
    "sessionName": "sirotaA-jg-05-20120316",
    "mazeName":    "cof",
    "stcMode":     "default",
})

# Load data fields
par = session.load("par")     # YAML parameter file → Struct
spk = session.load("spk")     # spikes → NBSpk
lfp = session.load("lfp", channels=list(range(8)))  # LFP → NBDlfp
stc = session.load("stc")     # state collection → NBStateCollection
```

### Spike analysis

```python
spk = session.load("spk")              # full recording
spk = trial.load("spk")               # auto-restricted to trial.sync
spk = trial.load("spk", restrict=False)        # override — full recording
spk = trial.load("spk", periods=stc["walk"])   # explicit epoch override

# Spike times in seconds for unit 5
times = spk[5]

# Restrict to a behavioural state
times_walk = spk[5, stc["walk"]]

# Per-unit dict (compatible with neural_scattering)
spikes_dict = spk.by_unit  # property: dict[unit_id → spike_times_sec]

# YAML-curated unit filtering (requires klusters/phy quality tags in par YAML)
good_ids = spk.annotated_unit_ids(quality="good")
pyr_ids  = spk.annotated_unit_ids(quality="good", cell_type="pyr", structure="CA1")
```

### Unit annotations

After spike sorting with klusters or phy, quality tags written to the
session YAML (`quality: good`, `type: pyr`, `structure: CA1`, etc.) are
loaded automatically when you call `session.load("spk")`:

```python
spk = session.load("spk")

# Get IDs of manually curated single units
good_ids = spk.annotated_unit_ids(quality="good")

# Filter by cell type or anatomy
pyr_ids   = spk.annotated_unit_ids(quality="good", cell_type="pyr")
ca1_ids   = spk.annotated_unit_ids(structure="CA1")

# Inspect one unit's annotation
ann = spk.annotation_for(good_ids[0])
print(ann.quality, ann.cell_type, ann.isolation_distance)

# Compute quality metrics (ISI contamination, SNR, spike width)
from neurobox.analysis import neuron_quality, print_neuron_quality_report
nq = neuron_quality(session)
print_neuron_quality_report(nq)

# is_single_unit() uses YAML quality as a hard veto when present
good_units = [uid for uid, r in nq.items() if r.is_single_unit()]
```

### LFP

```python
lfp = session.load("lfp", channels=list(range(8)))

# Band-pass filter 6–12 Hz (theta)
lfp.filter("butter", cutoff=[6, 12], btype="band")

# Current source density
csd = lfp.csd(channel_pitch_um=50)

# Index by epoch periods
theta_lfp = lfp[stc["theta"].data]
```

### Angular kinematics (NBDang)

```python
ang = session.load("ang")        # computes from session.xyz
ang = NBDang.from_xyz(xyz)       # or from a loaded NBDxyz directly

# Head direction (azimuth, radians) — head_back → head_front vector
hd = ang.head_direction("head_back", "head_front")

# Any named pair and component
theta = ang.between("spine_lower", "head_back", "theta")  # yaw
phi   = ang.between("spine_lower", "head_back", "phi")    # pitch
r     = ang.between("spine_lower", "head_back", "r")      # distance mm

# Full (T, N, N, 3) array — [theta, phi, r] per pair
ang.data   # shape (T, n_markers, n_markers, 3)
```

### Unit firing rates (NBDufr)

```python
ufr = session.load("ufr")                              # all units, 50 ms Gaussian, 1250 Hz
ufr = trial.load("ufr", units=good_ids, window=0.05)  # curated units, trial-restricted

# Firing-rate trace for one unit (spikes/s)
rate = ufr.rates_for(unit_id)

# Epoch selection via inherited NBData.__getitem__
walk_rates = ufr[stc["walk"]]   # (T_walk, N_units)
```

### State collection

```python
stc = session.load("stc")

walk  = stc["walk"]               # NBEpoch
theta = stc["theta"]
walk_theta = stc["walk&theta"]    # intersection
active = stc["walk|rear"]         # union
```

### Synchronise a new session

```python
# Neuralynx primary + Vicon/Optitrack secondary (event-based)
session.create(["nlx", "vicon"], ttl_value="0x0040")

# Open Ephys primary + Optitrack secondary (pulse channel)
session.create(["openephys", "optitrack"], sync_channel=17)

# Neuralynx primary + whl file (already on-clock)
session.create(["nlx", "whl"])
```

## Directory structure

```
/data/
  source/<typeId>/<sourceId>/<sourceId>-<userId>/<..>-<subjectId>/<session>/
  processed/<typeId>/.../<session>/        ← standardised pipeline outputs
      ephys/  — .dat .lfp .yaml .res.N .clu.N .all.evt
      mocap/  — <maze>/*.mat
  project/<projectId>/<session>/           ← session.spath
      <symlinks to processed files>
      <maze>/                              ← real dir, mocap symlinks
      *.ses.pkl  *.stc.*.pkl  *.pos.npz   ← analysis outputs
```

Session name format: `<sourceId>-<userId>-<subjectId>-<date>`
e.g. `sirotaA-jg-05-20120316`

## Requirements

- Python ≥ 3.10
- numpy ≥ 1.26
- scipy ≥ 1.12
- pyyaml ≥ 6.0
- python-dotenv ≥ 1.0
