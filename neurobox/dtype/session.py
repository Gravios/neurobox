"""
session.py  —  NBSession / NBTrial
====================================
Top-level session container.  Paths are resolved using the
``NB_DATA_PATH`` / ``NB_PROJECT_ID`` values from the project ``.env``
file (via :func:`~neurobox.config.load_config`).

Directory structure
-------------------
::

    /data/
      processed/ephys/<srcId>/<srcId>-<user>/<srcId>-<user>-<sub>/<session>/
          <session>.yaml   .dat   .lfp   .res.N   .clu.N   .all.evt

      processed/mocap/<srcId>/.../<session>/<maze>/
          <session>.TrialNNN.mat   (or .c3d etc.)

      project/<projectId>/<session>/          ← session.spath
          <symlinks to processed files>
          <maze>/                             ← real dir, links to mocap files
          <session>.cof.all.ses.pkl           ← analysis outputs (real files)
          <session>.cof.all.stc.*.pkl
          <session>.cof.all.pos.npz

Usage
-----
::

    from neurobox.dtype import NBSession, NBTrial

    # Load an existing session
    session = NBSession('sirotaA-jg-05-20120316', maze='cof', project_id='B01')

    # Load spikes
    spk = session.load('spk')

    # Create a trial restricted to a sync epoch
    trial = NBTrial(session, trial_name='run1', sync=np.array([[10.0, 300.0]]))
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from neurobox.dtype.struct import Struct
from neurobox.dtype.epoch  import NBEpoch
from neurobox.dtype.spikes import NBSpk
from neurobox.dtype.xyz    import NBDxyz
from neurobox.dtype.lfp    import NBDlfp
from neurobox.dtype.stc    import NBStateCollection
from neurobox.dtype.paths  import NBSessionPaths


# ---------------------------------------------------------------------------
# NBSession
# ---------------------------------------------------------------------------

class NBSession:
    """Top-level session container.

    Parameters
    ----------
    session_name:
        Full session name, e.g. ``'sirotaA-jg-05-20120316'``.
    maze:
        Arena code used to locate processed mocap data, e.g. ``'cof'``.
    trial:
        Trial label.  ``'all'`` = the full session (default).
    project_id:
        Project identifier, e.g. ``'B01'``.  Resolved from the
        ``.env`` file if *None*.
    data_root:
        Root of the data tree (``NB_DATA_PATH``).  Resolved from the
        ``.env`` file if *None*.
    overwrite:
        Re-create the session from scratch rather than loading a
        saved ``.ses.pkl``.
    """

    def __init__(
        self,
        session_name: str | None = None,
        maze:         str  = "cof",
        trial:        str  = "all",
        project_id:   str | None = None,
        data_root:    str | Path | None = None,
        overwrite:    bool = False,
    ) -> None:

        # ── Resolve project configuration ─────────────────────────────── #
        resolved = self._resolve_config(project_id, data_root)
        self._data_root:  Path = resolved["data_root"]
        self._project_id: str  = resolved["project_id"]

        if session_name is None:
            return

        # ── Build path object ──────────────────────────────────────────── #
        self.paths: NBSessionPaths = NBSessionPaths(
            session_name = session_name,
            data_root    = self._data_root,
            project_id   = self._project_id,
            maze         = maze,
        )

        # ── Core identity fields ───────────────────────────────────────── #
        self.name:      str  = session_name
        self.maze:      str  = maze
        self.trial:     str  = trial
        self.spath:     Path = self.paths.spath
        self.filebase:  str  = f"{session_name}.{maze}.{trial}"

        # ── Data containers (None until loaded) ───────────────────────── #
        self.par:        Struct | None            = None
        self.samplerate: float | None             = None
        self.sync:       NBEpoch | None           = None
        self.xyz:        NBDxyz  | None           = None
        self.lfp:        NBDlfp  | None           = None
        self.spk:        NBSpk   | None           = None
        self.stc:        NBStateCollection | None = None
        self.nq:         dict                     = {}
        self.meta:       dict                     = {}

        ses_file = self.paths.ses_file
        if ses_file.exists() and not overwrite:
            self._load_ses_file(ses_file)
        else:
            self._init_par()

    # ------------------------------------------------------------------ #
    # Configuration resolution                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_config(
        project_id: str | None,
        data_root:  str | Path | None,
    ) -> dict:
        """Read .env and merge with explicit arguments."""
        conf: dict[str, str] = {}
        try:
            from neurobox.config.config import load_config
            conf = load_config()
        except (FileNotFoundError, ImportError):
            pass

        resolved_root = Path(
            data_root if data_root is not None
            else conf.get("NB_DATA_PATH", "/data")
        )
        resolved_pid = (
            project_id if project_id is not None
            else conf.get("NB_PROJECT_ID", "")
        )
        return {"data_root": resolved_root, "project_id": resolved_pid}

    # ------------------------------------------------------------------ #
    # Parameter loading                                                   #
    # ------------------------------------------------------------------ #

    def _init_par(self) -> None:
        """Load the YAML parameter file and set core attributes."""
        try:
            from neurobox.io import load_par
            # Try spath first (symlink), then the processed ephys directory
            for base in (self.spath / self.name, self.paths.processed_ephys / self.name):
                try:
                    self.par = load_par(str(base))
                    self.samplerate = float(self.par.acquisitionSystem.samplingRate)
                    return
                except FileNotFoundError:
                    continue
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # create()                                                            #
    # ------------------------------------------------------------------ #

    def create(
        self,
        data_loggers: list[str] | None = None,
        save_xyz:      bool  = True,
        tolerance_sec: float = 0.2,
        **kwargs,
    ) -> "NBSession":
        """Synchronise raw data and populate session fields.

        This is a one-time operation per session.  It reads the raw
        ephys and tracking files, aligns them to a common clock, saves
        the processed position data, and writes a ``.ses.pkl``
        checkpoint so subsequent calls just load the checkpoint.

        Dispatches to a sync pipeline based on *data_loggers*:

        .. code-block:: text

            ['nlx', 'vicon']            sync_nlx_vicon      (TTL events)
            ['nlx', 'optitrack']        sync_nlx_vicon      (TTL events)
            ['nlx', 'motive']           sync_nlx_vicon      (TTL events)
            ['nlx', 'spots']            sync_nlx_spots      (binary .pos)
            ['nlx', 'whl']              sync_nlx_whl        (on-clock .whl)
            ['openephys', 'optitrack']  sync_openephys_optitrack  (pulse ch)
            ['openephys', 'vicon']      sync_openephys_vicon      (pulse ch)

        Parameters
        ----------
        data_loggers:
            List of system names identifying the primary (ephys) and
            secondary (tracking) acquisition systems.
        save_xyz:
            Save the assembled position array to
            ``<spath>/<filebase>.pos.npz`` (default True).
        tolerance_sec:
            Maximum duration mismatch tolerated when matching tracking
            data chunks to ephys windows (default 0.2 s).
        **kwargs:
            Forwarded to the chosen pipeline function.

            NLX pipelines
              ``ttl_value`` (str)   TTL label marking mocap start
              ``stop_ttl``  (str)   TTL label marking mocap stop

            OpenEphys pipelines
              ``sync_channel`` (int)   ADC channel carrying the sync pulse
              ``threshold``    (float) normalised detection threshold (0-1)

        Returns
        -------
        self

        Examples
        --------
        >>> session = NBSession('sirotaA-jg-05-20120316', maze='cof')
        >>> session.create(['nlx', 'vicon'], ttl_value='0x0040')

        >>> session.create(['openephys', 'optitrack'], sync_channel=17)
        """
        from neurobox.dtype.sync_pipelines import dispatch

        if data_loggers is None:
            raise ValueError(
                "data_loggers must be specified, e.g. ['nlx', 'vicon'].\n"
                "Supported primaries  : nlx, neuralynx, openephys, oephys\n"
                "Supported secondaries: vicon, optitrack, motive, spots, whl"
            )

        # Ensure par is loaded before dispatching
        if self.par is None:
            self._init_par()

        # Ensure the project session directory exists
        self.spath.mkdir(parents=True, exist_ok=True)

        # Inject shared kwargs that every pipeline accepts
        kwargs.setdefault("save_xyz",      save_xyz)
        kwargs.setdefault("tolerance_sec", tolerance_sec)

        dispatch(self, data_loggers, **kwargs)
        self.save()
        return self

    # ------------------------------------------------------------------ #
    # load()                                                              #
    # ------------------------------------------------------------------ #

    def load(self, field: str | None = None, *args, **kwargs):
        """Load a specific data field from disk.

        Parameters
        ----------
        field:
            ``'par'``, ``'spk'``, ``'lfp'``, ``'dat'``, ``'xyz'``,
            ``'stc'``, ``'nq'``.  If *None*, load the session ``.ses.pkl``.

        Returns
        -------
        The loaded data object (or *self* when *field* is None).
        """
        if field is None:
            self._load_ses_file(self.paths.ses_file)
            return self

        if field == "par":
            self._init_par()
            return self.par

        elif field == "spk":
            # Prefer the symlinked spath, fall back to processed_ephys
            for base in (self.spath / self.name,
                         self.paths.processed_ephys / self.name):
                try:
                    spk = NBSpk.load(
                        str(base),
                        samplerate = self.samplerate,
                        **kwargs,
                    )
                    self.spk = spk
                    return spk
                except FileNotFoundError:
                    continue
            raise FileNotFoundError(
                f"No .res/.clu files found for {self.name!r} "
                f"in {self.spath} or {self.paths.processed_ephys}"
            )

        elif field in ("lfp", "dat"):
            ext = field   # 'lfp' or 'dat'
            lfp_path = self.spath / f"{self.name}.{ext}"
            if not lfp_path.exists():
                lfp_path = self.paths.processed_ephys / f"{self.name}.{ext}"
            channels = kwargs.get("channels", None)
            lfp = NBDlfp(
                path       = lfp_path.parent,
                filename   = lfp_path.name,
                samplerate = (1250.0 if ext == "lfp"
                              else (self.samplerate or 20000.0)),
                name       = self.name,
                ext        = ext,
            )
            lfp.load(str(lfp_path.with_suffix("")),
                     channels=channels, par=self.par)
            self.lfp = lfp
            return lfp

        elif field == "xyz":
            pos_file = self.paths.pos_file
            if pos_file.exists():
                xyz = NBDxyz(path=self.spath,
                             filename=pos_file.name)
                xyz.load(pos_file)
                self.xyz = xyz
                return xyz
            raise FileNotFoundError(
                f"Position file not found: {pos_file}\n"
                "Run session.create() first to build it from raw mocap data."
            )

        elif field == "stc":
            mode     = kwargs.get("mode", "default")
            stc_file = self.paths.stc_file(mode)
            if stc_file.exists():
                self.stc = NBStateCollection.load_file(stc_file)
            else:
                self.stc = NBStateCollection(
                    path     = self.spath,
                    filename = stc_file.name,
                    mode     = mode,
                    sync     = self.sync,
                )
            return self.stc

        elif field == "nq":
            nq_file = self.spath / f"{self.name}.NeuronQuality.npy"
            if nq_file.exists():
                self.nq = np.load(nq_file, allow_pickle=True).item()
            return self.nq

        raise ValueError(
            f"Unknown field {field!r}.  "
            "Choose from: par, spk, lfp, dat, xyz, stc, nq."
        )

    # ------------------------------------------------------------------ #
    # Save / load session pickle                                          #
    # ------------------------------------------------------------------ #

    def save(self) -> None:
        """Persist the session skeleton to ``<spath>/<filebase>.ses.pkl``."""
        self.spath.mkdir(parents=True, exist_ok=True)
        state = {k: v for k, v in self.__dict__.items()
                 if not k.startswith("_")}
        with open(self.paths.ses_file, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_ses_file(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Session file not found: {path}")
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.__dict__.update(state)

    # ------------------------------------------------------------------ #
    # Convenience                                                         #
    # ------------------------------------------------------------------ #

    def peek(self) -> None:
        """Print a summary of loaded fields."""
        for key, val in self.__dict__.items():
            if key.startswith("_"):
                continue
            print(f"  {key:12s}: {repr(val)[:80]}")

    def __repr__(self) -> str:
        return (
            f"NBSession(name={getattr(self,'name','?')!r}, "
            f"maze={getattr(self,'maze','?')!r}, "
            f"trial={getattr(self,'trial','?')!r})"
        )

    @staticmethod
    def validate(ref, **kwargs) -> "NBSession":
        """Load or coerce *ref* into an NBSession.

        Accepted input forms
        --------------------
        ``NBSession`` / ``NBTrial``
            Returned as-is (already loaded).

        ``'sirotaA-jg-05-20120316'``
            Session name — loads the full session (trial ``'all'``).

        ``'sirotaA-jg-05-20120316.cof.all'``
            Filebase string — parsed into ``(name, maze, trial)`` and
            forwarded to ``NBTrial.validate``.

        ``['sirotaA-jg-05-20120316', 'cof', 'all']``
            3-element list/tuple — same as filebase string.

        ``{'sessionName': '...', 'mazeName': 'cof', 'trialName': 'all',
           'stcMode': 'hmm', 'subject': {...}}``
            Dict — extracts recognised keys; extra keys are stored in
            ``session.meta``.

        Keyword arguments
        -----------------
        stc_mode : str, optional
            If given, ``session.load('stc', mode=stc_mode)`` is called
            automatically after construction.
        project_id, data_root, overwrite
            Forwarded to the constructor.

        Examples
        --------
        >>> s = NBSession.validate('sirotaA-jg-05-20120316')
        >>> s = NBSession.validate('sirotaA-jg-05-20120316.cof.all')
        >>> s = NBSession.validate({'sessionName': 'sirotaA-jg-05-20120316',
        ...                         'mazeName': 'cof', 'stcMode': 'hmm'})
        """
        stc_mode = kwargs.pop("stc_mode", None)

        # ── Already an NBSession ────────────────────────────────────────── #
        if isinstance(ref, NBSession):
            session = ref

        # ── Filebase string  'name.maze.trial' ─────────────────────────── #
        elif isinstance(ref, str) and ref.count(".") == 2:
            session = NBTrial.validate(ref, **kwargs)

        # ── Session-name string  'name' ─────────────────────────────────── #
        elif isinstance(ref, str):
            session = NBSession(ref, **kwargs)

        # ── List / tuple  ['name', 'maze', 'trial'] ─────────────────────── #
        elif isinstance(ref, (list, tuple)):
            if len(ref) == 1:
                session = NBSession.validate(ref[0], stc_mode=stc_mode,
                                             **kwargs)
            elif len(ref) == 3:
                session = NBTrial(ref[0], maze=ref[1], trial_name=ref[2],
                                  **kwargs)
            else:
                raise ValueError(
                    f"List/tuple ref must have 1 or 3 elements, got {len(ref)}."
                )

        # ── Dict ─────────────────────────────────────────────────────────── #
        elif isinstance(ref, dict):
            name       = ref.get("sessionName", ref.get("name"))
            maze       = ref.get("mazeName",    kwargs.pop("maze",   "cof"))
            trial      = ref.get("trialName",   kwargs.pop("trial",  "all"))
            stc_mode   = stc_mode or ref.get("stcMode")
            meta       = {k: v for k, v in ref.items()
                          if k not in ("sessionName", "name", "mazeName",
                                       "trialName", "stcMode")}
            if trial == "all":
                session = NBSession(name, maze=maze, **kwargs)
            else:
                session = NBTrial(name, maze=maze, trial_name=trial, **kwargs)
            if meta:
                session.meta.update(meta)

        else:
            raise TypeError(
                f"Cannot coerce {type(ref).__name__!r} to NBSession.  "
                "Pass a session name string, filebase, list, dict, or "
                "an existing NBSession."
            )

        # ── Optional stc load ─────────────────────────────────────────────── #
        if stc_mode is not None:
            try:
                session.load("stc", mode=stc_mode)
            except Exception as e:
                print(f"  [warn] Could not load stc mode {stc_mode!r}: {e}")

        return session


# ---------------------------------------------------------------------------
# NBTrial
# ---------------------------------------------------------------------------

class NBTrial(NBSession):
    """A subset of an NBSession defined by synchronisation periods.

    Parameters
    ----------
    session_name:
        Session name or an existing NBSession to copy from.
    maze:
        Arena code.
    trial_name:
        Trial label (default ``'all'``).
    overwrite:
        Re-create from scratch.
    sync:
        NBEpoch or ``(N, 2)`` float64 array in seconds defining
        which periods to include.  If *None*, the full session
        sync is used.
    project_id, data_root:
        Forwarded to NBSession; read from .env if not provided.
    """

    def __init__(
        self,
        session_name: "str | NBSession | None" = None,
        maze:         str  = "cof",
        trial_name:   str  = "all",
        overwrite:    bool = False,
        sync: "NBEpoch | np.ndarray | None" = None,
        project_id:   str | None = None,
        data_root:    str | Path | None = None,
    ) -> None:

        if isinstance(session_name, NBSession):
            parent = session_name
            # Copy parent state
            super().__init__(None)
            for k, v in parent.__dict__.items():
                setattr(self, k, v)
        else:
            super().__init__(
                session_name = session_name,
                maze         = maze,
                trial        = trial_name,
                project_id   = project_id,
                data_root    = data_root,
                overwrite    = overwrite,
            )

        self.trial    = trial_name
        self.filebase = f"{self.name}.{self.maze}.{self.trial}"

        trl_file = self.spath / f"{self.filebase}.trl.pkl"

        if trl_file.exists() and not overwrite:
            with open(trl_file, "rb") as f:
                ds = pickle.load(f)
            self.sync = ds.get("sync", self.sync)
        elif sync is not None:
            if isinstance(sync, np.ndarray):
                self.sync = NBEpoch(sync, samplerate=1.0, mode="periods")
            else:
                self.sync = sync.copy()

    def save(self, overwrite: bool = False) -> None:  # type: ignore[override]
        """Persist trial sync to ``<filebase>.trl.pkl``."""
        trl_file = self.spath / f"{self.filebase}.trl.pkl"
        if trl_file.exists() and not overwrite:
            raise FileExistsError(f"{trl_file} exists.  Pass overwrite=True.")
        with open(trl_file, "wb") as f:
            pickle.dump({"sync": self.sync}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)

    def __repr__(self) -> str:
        return (
            f"NBTrial(name={getattr(self,'name','?')!r}, "
            f"maze={getattr(self,'maze','?')!r}, "
            f"trial={getattr(self,'trial','?')!r})"
        )

    @staticmethod
    def validate(ref, **kwargs) -> "NBTrial":
        """Load or coerce *ref* into an NBTrial.

        Accepted input forms
        --------------------
        ``NBTrial``
            Returned as-is.

        ``NBSession``
            Wrapped as ``NBTrial(session)`` using its existing sync.

        ``'sirotaA-jg-05-20120316'``
            Session name — loads the ``'all'`` trial.

        ``'sirotaA-jg-05-20120316.cof.run1'``
            Filebase string — parsed into ``(name, maze, trial)``.

        ``['name', 'maze', 'trial']``
            3-element list/tuple.

        ``{'sessionName': '...', 'mazeName': 'cof', 'trialName': 'run1',
           'stcMode': 'hmm'}``
            Dict — same keys as ``NBSession.validate``.

        Keyword arguments
        -----------------
        stc_mode : str, optional
            If given, ``trial.load('stc', mode=stc_mode)`` is called
            automatically.
        project_id, data_root, overwrite, sync
            Forwarded to the constructor.

        Examples
        --------
        >>> t = NBTrial.validate('sirotaA-jg-05-20120316.cof.all')
        >>> t = NBTrial.validate(['sirotaA-jg-05-20120316', 'cof', 'run1'])
        >>> t = NBTrial.validate({'sessionName': 'sirotaA-jg-05-20120316',
        ...                       'mazeName': 'cof', 'trialName': 'run1',
        ...                       'stcMode': 'default'})
        """
        stc_mode = kwargs.pop("stc_mode", None)

        # ── Already an NBTrial ───────────────────────────────────────────── #
        if isinstance(ref, NBTrial):
            trial = ref

        # ── NBSession → wrap ─────────────────────────────────────────────── #
        elif isinstance(ref, NBSession):
            trial = NBTrial(ref, **kwargs)

        # ── Filebase string  'name.maze.trial' ─────────────────────────── #
        elif isinstance(ref, str) and ref.count(".") == 2:
            name, maze, trial_name = ref.split(".")
            trial = NBTrial(name, maze=maze, trial_name=trial_name, **kwargs)

        # ── Session-name string  'name' ─────────────────────────────────── #
        elif isinstance(ref, str):
            trial = NBTrial(ref, **kwargs)

        # ── List / tuple ─────────────────────────────────────────────────── #
        elif isinstance(ref, (list, tuple)):
            if len(ref) == 1:
                trial = NBTrial.validate(ref[0], stc_mode=stc_mode, **kwargs)
            elif len(ref) == 3:
                trial = NBTrial(ref[0], maze=ref[1], trial_name=ref[2],
                                **kwargs)
            else:
                raise ValueError(
                    f"List/tuple ref must have 1 or 3 elements, got {len(ref)}."
                )

        # ── Dict ─────────────────────────────────────────────────────────── #
        elif isinstance(ref, dict):
            name       = ref.get("sessionName", ref.get("name"))
            maze       = ref.get("mazeName",    kwargs.pop("maze",       "cof"))
            trial_name = ref.get("trialName",   kwargs.pop("trial_name", "all"))
            stc_mode   = stc_mode or ref.get("stcMode")
            meta       = {k: v for k, v in ref.items()
                          if k not in ("sessionName", "name", "mazeName",
                                       "trialName", "stcMode")}
            trial = NBTrial(name, maze=maze, trial_name=trial_name, **kwargs)
            if meta:
                trial.meta.update(meta)

        else:
            raise TypeError(
                f"Cannot coerce {type(ref).__name__!r} to NBTrial.  "
                "Pass a session name, filebase, list, dict, NBSession, "
                "or an existing NBTrial."
            )

        # ── Optional stc load ─────────────────────────────────────────────── #
        if stc_mode is not None:
            try:
                trial.load("stc", mode=stc_mode)
            except Exception as e:
                print(f"  [warn] Could not load stc mode {stc_mode!r}: {e}")

        return trial
