# neurobox

Analysis toolbox for silicon-probe electrophysiology, integrated with
[neurosuite-3](https://github.com/Gravios/neurosuite-3).

## Features

- Load neurosuite-3 binary spike/LFP/waveform data (`.res`, `.clu`, `.spk`, `.lfp`, `.dat`)
- Parse YAML session parameter files
- Synchronise ephys with motion-capture (Vicon, Optitrack) and 2-D tracking
- NBSession / NBTrial session containers with lazy data loading
- NBStateCollection behavioural state management with set-algebra query language
- Structured project / data directory management

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
spk = session.load("spk")

# Spike times in seconds for unit 5
times = spk[5]

# Restrict to a behavioural state
walk_epoch = stc["walk"]
times_walk = spk[5, walk_epoch]

# Per-unit dict (compatible with neural_scattering)
spikes_dict = spk.by_unit()
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
