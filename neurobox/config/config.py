"""
config.py
=========
Project configuration and session linking utilities.

Port of MTA's ``link_session.m`` / ``link_sessions_to_project.m`` family,
adapted for the neurobox directory layout.

Quick start
-----------
One-time project setup::

    from neurobox.config import configure_project
    configure_project(project_id="B01", data_root="/data")

Link a single session (mazes auto-discovered if omitted)::

    from neurobox.config import link_session
    link_session("sirotaA-jg-05-20120316", "B01")

Batch-link from a session list::

    from neurobox.config import link_sessions
    link_sessions("B01", sessions=[
        {"sessionName": "sirotaA-jg-05-20120316"},
        {"sessionName": "sirotaA-jg-06-20120320", "mazes": ["cof", "nor"]},
    ])

Show what is and isn't linked for a session::

    from neurobox.config import link_session_status
    link_session_status("sirotaA-jg-05-20120316", "B01")

Command-line::

    nb-configure --project-id B01 --data-root /data
    nb-link       --session sirotaA-jg-05-20120316 --project-id B01
    nb-link-batch --project-id B01 --session-list sessions.txt

Directory layout
----------------
::

    /data/
      processed/
        ephys/<srcId>/.../<session>/        ← .yaml .dat .lfp .res.N .clu.N .evt
        mocap/<srcId>/.../<session>/<maze>/ ← .mat trial files per maze

      project/<projectId>/
        .env
        config/ figures/ scripts/ notebooks/ models/
        <session>/                          ← spath (real directory)
            <symlinks → processed/ephys/.../<session>/*>
            <maze>/                         ← real subdirectory
                <symlinks → processed/mocap/.../<session>/<maze>/*>
            *.ses.pkl *.stc.*.pkl *.pos.npz ← analysis outputs (real files)

Rules
-----
* The session directory (``spath``) is always a **real** directory.
* Maze subdirectories inside ``spath`` are **real** directories.
* Every data file is a **symlink** pointing to the canonical processed path.
* Analysis outputs written by ``NBSession.create()`` are real files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from neurobox.dtype.paths import NBSessionPaths


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _log(msg: str, dry_run: bool = False, prefix: str = "  ") -> None:
    tag = "[dry-run] " if dry_run else ""
    print(f"{prefix}{tag}{msg}")


def _link_file(src: Path, dst: Path,
               overwrite: bool, dry_run: bool) -> str:
    """Create one symlink ``dst → src``.

    Returns ``'linked'``, ``'skipped'``, or ``'replaced'``.
    """
    exists = dst.exists() or dst.is_symlink()
    if exists and not overwrite:
        return "skipped"
    if not dry_run:
        if exists:
            dst.unlink()
        dst.symlink_to(src)
    return "replaced" if exists else "linked"


def _link_dir(src_dir: Path, dst_dir: Path,
               overwrite: bool, dry_run: bool,
               label: str) -> tuple[int, int, int]:
    """Symlink all files from *src_dir* into *dst_dir*.

    Returns ``(linked, replaced, skipped)`` counts.
    """
    if not src_dir.exists():
        _log(f"[warn] not found: {src_dir}")
        return 0, 0, 0

    if not dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)

    linked = replaced = skipped = 0

    for src in sorted(src_dir.iterdir()):
        if src.is_dir():
            continue                       # never link directories
        dst    = dst_dir / src.name
        result = _link_file(src, dst, overwrite, dry_run)
        if result == "linked":
            linked   += 1
        elif result == "replaced":
            replaced += 1
        else:
            skipped  += 1
        if dry_run and result != "skipped":
            _log(f"  {result}: {label}/{dst.name} → {src}", dry_run=dry_run)

    return linked, replaced, skipped


# ---------------------------------------------------------------------------
# discover_mazes  (port of the maze-scanning step in linkSession.m)
# ---------------------------------------------------------------------------

def discover_mazes(
    session_name: str,
    data_root:    str | Path = "/data",
    project_id:   str = "",
) -> list[str]:
    """Return all maze codes available for *session_name* in the processed tree.

    Scans ``/data/processed/mocap/.../<session>/`` for subdirectories.
    Each subdirectory name is treated as a maze code (e.g. ``'cof'``,
    ``'nor'``).

    Parameters
    ----------
    session_name:
        Full session name, e.g. ``'sirotaA-jg-05-20120316'``.
    data_root:
        Root of the data tree.
    project_id:
        Project identifier (used only to construct the NBSessionPaths
        object; not needed for discovery itself).

    Returns
    -------
    list[str]
        Sorted list of maze codes.  Empty if no processed mocap data exists.
    """
    paths = NBSessionPaths(
        session_name = session_name,
        data_root    = Path(data_root),
        project_id   = project_id or "_discover",
        maze         = "_discover",
    )
    # processed_mocap includes the maze subdir; the parent is the session dir
    mocap_session_dir = paths.processed_mocap.parent

    if not mocap_session_dir.exists():
        return []

    return sorted(
        d.name
        for d in mocap_session_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )


# ---------------------------------------------------------------------------
# configure_project
# ---------------------------------------------------------------------------

def configure_project(
    project_id:  str,
    data_root:   str | Path = "/data",
    dotenv_path: str | Path | None = None,
    overwrite:   bool = False,
) -> None:
    """Create the project directory skeleton and write a ``.env`` file.

    Creates::

        /data/project/<projectId>/
            config/  figures/  scripts/  notebooks/  models/

    Parameters
    ----------
    project_id:
        Short project identifier, e.g. ``'B01'``.
    data_root:
        Root of the data tree.  Default ``'/data'``.
    dotenv_path:
        Where to write the ``.env`` file.  Defaults to
        ``<data_root>/project/<projectId>/.env``.
    overwrite:
        Overwrite an existing ``.env`` file if True.
    """
    data_root   = Path(data_root)
    project_dir = data_root / "project" / project_id

    for sub in ("config", "figures", "scripts", "notebooks", "models"):
        (project_dir / sub).mkdir(parents=True, exist_ok=True)

    if dotenv_path is None:
        dotenv_path = project_dir / ".env"
    dotenv_path = Path(dotenv_path)

    if dotenv_path.exists() and not overwrite:
        print(f"[info] .env already exists at {dotenv_path} — "
              "pass overwrite=True to replace.")
    else:
        dotenv_path.write_text(
            f"NB_DATA_PATH={data_root}\n"
            f"NB_PROJECT_ID={project_id}\n"
        )
        print(f"[info] Wrote {dotenv_path}")

    print(f"[info] Project directory: {project_dir}")


# ---------------------------------------------------------------------------
# link_session   (port of link_session.m / link_session_Dpath.m)
# ---------------------------------------------------------------------------

def link_session(
    session_name: str,
    project_id:   str,
    data_root:    str | Path = "/data",
    mazes:        list[str] | None = None,
    overwrite:    bool = False,
    dry_run:      bool = False,
    verbose:      bool = True,
) -> NBSessionPaths:
    """Create the project session directory and populate it with symlinks.

    Port of MTA's ``link_session_Dpath`` / ``linkSession``, adapted for
    the neurobox directory layout.

    **What this creates**::

        project/<projectId>/<session>/          ← real directory (spath)
            <session>.yaml → processed/ephys/.../<session>/<session>.yaml
            <session>.lfp  → processed/ephys/.../<session>/<session>.lfp
            ...                                 (all ephys files, individually)
            <maze>/                             ← real subdirectory per maze
                <session>.Trial001.mat → processed/mocap/.../<session>/<maze>/...

    Rules
    -----
    * ``spath`` itself is always a real directory (never a symlink).
    * ``spath/<maze>/`` subdirectories are real directories.
    * Every file inside them is a symlink to the processed data.
    * Analysis outputs (``.ses.pkl``, ``.stc.*.pkl``, ``.pos.npz``)
      are written as real files by ``NBSession.create()`` later.

    Parameters
    ----------
    session_name:
        Full session name, e.g. ``'sirotaA-jg-05-20120316'``.
    project_id:
        Project identifier, e.g. ``'B01'``.
    data_root:
        Root of the data tree.  Default ``'/data'``.
    mazes:
        Maze codes to link processed mocap data from.  Pass ``None``
        (default) to **auto-discover** all mazes present in the
        processed mocap tree.  Pass ``[]`` to skip mocap entirely.
    overwrite:
        Replace existing symlinks if True.
    dry_run:
        Print what would be done without creating anything.
    verbose:
        Print a summary of linked / skipped counts.

    Returns
    -------
    NBSessionPaths
        Resolved paths for this session.

    Examples
    --------
    >>> link_session("sirotaA-jg-05-20120316", "B01")           # auto-discover mazes
    >>> link_session("sirotaA-jg-05-20120316", "B01", mazes=[]) # ephys only
    >>> link_session("sirotaA-jg-05-20120316", "B01",
    ...              mazes=["cof", "nor"], overwrite=True)
    """
    data_root = Path(data_root)

    # ── Resolve mazes ──────────────────────────────────────────────────── #
    if mazes is None:
        mazes = discover_mazes(session_name, data_root, project_id)
        if verbose and mazes:
            print(f"  discovered mazes: {mazes}")
        elif verbose:
            print("  no processed mocap found — linking ephys only")

    paths = NBSessionPaths(
        session_name = session_name,
        data_root    = data_root,
        project_id   = project_id,
        maze         = mazes[0] if mazes else "cof",
    )

    prefix = "[dry-run] " if dry_run else ""
    if verbose:
        print(f"{prefix}link_session: {session_name} → {paths.spath}")

    # ── Create session directory ───────────────────────────────────────── #
    if not dry_run:
        paths.spath.mkdir(parents=True, exist_ok=True)
    elif verbose:
        _log(f"mkdir  {paths.spath}", dry_run=True)

    # ── Link ephys files (flat into spath) ─────────────────────────────── #
    ln, rp, sk = _link_dir(
        src_dir  = paths.processed_ephys,
        dst_dir  = paths.spath,
        overwrite = overwrite,
        dry_run  = dry_run,
        label    = "ephys",
    )
    if verbose:
        print(f"  {'[dry-run] ' if dry_run else ''}"
              f"ephys:     linked={ln}  replaced={rp}  skipped={sk}")

    # ── Link mocap files per maze ──────────────────────────────────────── #
    for maze in mazes:
        paths_m = NBSessionPaths(
            session_name = session_name,
            data_root    = data_root,
            project_id   = project_id,
            maze         = maze,
        )
        ln, rp, sk = _link_dir(
            src_dir  = paths_m.processed_mocap,
            dst_dir  = paths.spath / maze,
            overwrite = overwrite,
            dry_run  = dry_run,
            label    = f"mocap/{maze}",
        )
        if verbose:
            print(f"  {'[dry-run] ' if dry_run else ''}"
                  f"mocap/{maze}: linked={ln}  replaced={rp}  skipped={sk}")

    return paths


# ---------------------------------------------------------------------------
# link_sessions  (port of link_sessions_to_project.m)
# ---------------------------------------------------------------------------

def link_sessions(
    project_id:  str,
    sessions:    list[str | dict[str, Any]],
    data_root:   str | Path = "/data",
    mazes:       list[str] | None = None,
    overwrite:   bool = False,
    dry_run:     bool = False,
    stop_on_error: bool = False,
) -> dict[str, NBSessionPaths | Exception]:
    """Link multiple sessions into a project in one call.

    Port of MTA's ``link_sessions_to_project``.

    Parameters
    ----------
    project_id:
        Project identifier.
    sessions:
        Session list.  Each entry is either:

        * A **string** — the session name; mazes are auto-discovered.
        * A **dict** with keys:

          ``'sessionName'`` (required)
            Full session name.
          ``'mazes'`` (optional)
            List of maze codes.  Overrides the *mazes* argument for
            this session.

        The dict format is the same as the ``sessionList`` entries in
        MTA's ``get_session_list``.

    data_root:
        Root of the data tree.
    mazes:
        Default maze list applied to every session that doesn't
        specify its own.  ``None`` (default) → auto-discover per session.
    overwrite:
        Replace existing symlinks.
    dry_run:
        Print what would be done without creating anything.
    stop_on_error:
        Raise immediately on the first error.  Default False: log errors
        and continue to the next session.

    Returns
    -------
    dict[str, NBSessionPaths | Exception]
        Maps each session name to its ``NBSessionPaths`` on success, or
        to the caught exception on failure.

    Examples
    --------
    >>> results = link_sessions("B01", [
    ...     "sirotaA-jg-05-20120316",
    ...     {"sessionName": "sirotaA-jg-06-20120320", "mazes": ["cof"]},
    ... ])
    >>> # Filter failures
    >>> failed = {k: v for k, v in results.items() if isinstance(v, Exception)}
    """
    data_root = Path(data_root)
    results:  dict[str, NBSessionPaths | Exception] = {}
    n_total  = len(sessions)

    for i, entry in enumerate(sessions, 1):
        # Parse entry
        if isinstance(entry, str):
            name         = entry
            session_mazes = mazes          # use the default
        elif isinstance(entry, dict):
            name         = entry.get("sessionName", entry.get("name", ""))
            session_mazes = entry.get("mazes", mazes)
        else:
            raise TypeError(
                f"sessions entries must be str or dict, got {type(entry)}"
            )

        if not name:
            print(f"  [warn] entry {i}: no sessionName, skipping")
            continue

        print(f"\n[{i}/{n_total}] {name}")
        try:
            paths = link_session(
                session_name = name,
                project_id   = project_id,
                data_root    = data_root,
                mazes        = session_mazes,
                overwrite    = overwrite,
                dry_run      = dry_run,
                verbose      = True,
            )
            results[name] = paths
        except Exception as exc:
            print(f"  [error] {exc}")
            results[name] = exc
            if stop_on_error:
                raise

    n_ok  = sum(1 for v in results.values() if isinstance(v, NBSessionPaths))
    n_err = sum(1 for v in results.values() if isinstance(v, Exception))
    print(f"\nlink_sessions: {n_ok}/{n_total} OK"
          + (f"  {n_err} errors" if n_err else ""))

    return results


# ---------------------------------------------------------------------------
# link_session_status  (diagnostic: show what's linked / missing / stale)
# ---------------------------------------------------------------------------

def link_session_status(
    session_name: str,
    project_id:   str,
    data_root:    str | Path = "/data",
    mazes:        list[str] | None = None,
) -> dict[str, Any]:
    """Print and return a status report for a linked session.

    Checks each symlink in ``spath`` and each maze subdirectory to
    determine whether it:

    * **linked** — symlink exists and resolves to an existing file
    * **stale**  — symlink exists but the target is missing
    * **missing** — no symlink exists (file is in processed but not linked)

    Parameters
    ----------
    session_name:
        Full session name.
    project_id:
        Project identifier.
    data_root:
        Root of the data tree.
    mazes:
        Maze codes to check.  Auto-discovered if *None*.

    Returns
    -------
    dict with keys:

    ``'spath'``
        Path to the project session directory.
    ``'ephys'``
        ``{'linked': [...], 'stale': [...], 'missing': [...]}``
    ``'mocap'``
        ``{maze: {'linked': [...], 'stale': [...], 'missing': [...]}}``
    ``'analysis'``
        List of real (non-symlink) files present in spath.
    ``'ok'``
        True if there are no stale or missing ephys files.
    """
    data_root = Path(data_root)

    if mazes is None:
        mazes = discover_mazes(session_name, data_root, project_id)

    paths = NBSessionPaths(
        session_name = session_name,
        data_root    = data_root,
        project_id   = project_id,
        maze         = mazes[0] if mazes else "cof",
    )

    def _check_dir(src_dir: Path, dst_dir: Path) -> dict:
        result = {"linked": [], "stale": [], "missing": []}

        # Files that *should* be linked (from the source directory)
        src_names: set[str] = set()
        if src_dir.exists():
            for src in sorted(src_dir.iterdir()):
                if src.is_dir():
                    continue
                src_names.add(src.name)
                dst = dst_dir / src.name
                if dst.is_symlink():
                    if dst.exists():          # follows the link — target ok
                        result["linked"].append(src.name)
                    else:                     # dangling symlink
                        result["stale"].append(src.name)
                else:
                    result["missing"].append(src.name)

        # Also scan the destination for dangling symlinks not in the source
        # (e.g. source was deleted after the link was created)
        if dst_dir.exists():
            for dst in sorted(dst_dir.iterdir()):
                if (dst.is_symlink()
                        and not dst.exists()
                        and dst.name not in src_names
                        and dst.name not in result["stale"]):
                    result["stale"].append(dst.name)

        return result

    # ── Ephys ──────────────────────────────────────────────────────────── #
    ephys_status = _check_dir(paths.processed_ephys, paths.spath)

    # ── Mocap per maze ────────────────────────────────────────────────── #
    mocap_status: dict[str, dict] = {}
    for maze in mazes:
        pm = NBSessionPaths(
            session_name = session_name,
            data_root    = data_root,
            project_id   = project_id,
            maze         = maze,
        )
        mocap_status[maze] = _check_dir(
            pm.processed_mocap,
            paths.spath / maze,
        )

    # ── Analysis outputs (real files in spath) ────────────────────────── #
    analysis_files: list[str] = []
    if paths.spath.exists():
        for f in sorted(paths.spath.iterdir()):
            if f.is_file() and not f.is_symlink():
                analysis_files.append(f.name)

    ok = (not ephys_status["stale"] and not ephys_status["missing"]
          and all(not v["stale"] and not v["missing"]
                  for v in mocap_status.values()))

    # ── Print report ──────────────────────────────────────────────────── #
    status_char = "✓" if ok else "✗"
    print(f"\n{status_char} {session_name}  [{project_id}]")
    print(f"  spath: {paths.spath}"
          + (" (exists)" if paths.spath.exists() else " (MISSING)"))

    def _print_section(label: str, st: dict) -> None:
        n_ok  = len(st["linked"])
        parts = [f"{n_ok} linked"]
        if st["stale"]:
            parts.append(f"{len(st['stale'])} STALE: {st['stale']}")
        if st["missing"]:
            parts.append(f"{len(st['missing'])} missing: {st['missing'][:3]}"
                         + ("…" if len(st["missing"]) > 3 else ""))
        print(f"  {label:12s}: {',  '.join(parts)}")

    _print_section("ephys", ephys_status)
    for maze, st in mocap_status.items():
        _print_section(f"mocap/{maze}", st)
    if analysis_files:
        print(f"  analysis   : {len(analysis_files)} file(s): "
              f"{analysis_files[:4]}"
              + ("…" if len(analysis_files) > 4 else ""))

    return {
        "spath":    paths.spath,
        "ephys":    ephys_status,
        "mocap":    mocap_status,
        "analysis": analysis_files,
        "ok":       ok,
    }


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

def load_config(dotenv_path: str | Path | None = None) -> dict[str, str]:
    """Read the project ``.env`` file and return key→value pairs.

    Search order (first found wins):
    1. *dotenv_path* argument
    2. ``$NB_DOTENV_PATH`` environment variable
    3. Current working directory ``.env``

    Raises
    ------
    FileNotFoundError if no ``.env`` can be located.
    """
    candidates: list[Path] = []
    if dotenv_path:
        candidates.append(Path(dotenv_path))
    env_var = os.environ.get("NB_DOTENV_PATH")
    if env_var:
        candidates.append(Path(env_var))
    candidates.append(Path.cwd() / ".env")

    for c in candidates:
        if c.exists():
            conf: dict[str, str] = {}
            for line in c.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                conf[k.strip()] = v.strip()
            return conf

    raise FileNotFoundError(
        f"No .env file found.  Searched: {[str(c) for c in candidates]}\n"
        "Run configure_project() to create one, or set NB_DOTENV_PATH."
    )


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------

def _resolve_project(args_project_id, args_data_root) -> tuple[str, str]:
    """Resolve project_id and data_root from CLI args or .env."""
    conf: dict = {}
    try:
        conf = load_config()
    except FileNotFoundError:
        pass
    project_id = args_project_id or conf.get("NB_PROJECT_ID", "")
    data_root  = args_data_root  or conf.get("NB_DATA_PATH", "/data")
    return project_id, data_root


def _cli_configure() -> None:
    """Entry point: nb-configure"""
    import argparse
    p = argparse.ArgumentParser(
        description="Create a neurobox project skeleton and write a .env file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--project-id",  required=True,  help="Project ID, e.g. B01")
    p.add_argument("--data-root",   default="/data", help="Root of the data tree")
    p.add_argument("--dotenv-path", default=None,    help="Path to write the .env file")
    p.add_argument("--overwrite",   action="store_true")
    args = p.parse_args()
    configure_project(
        project_id  = args.project_id,
        data_root   = args.data_root,
        dotenv_path = args.dotenv_path,
        overwrite   = args.overwrite,
    )


def _cli_link() -> None:
    """Entry point: nb-link"""
    import argparse
    p = argparse.ArgumentParser(
        description="Link a processed session into a project directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--session",    required=True, help="Session name")
    p.add_argument("--project-id", default=None,  help="Project ID (from .env if omitted)")
    p.add_argument("--data-root",  default=None,  help="Data root (from .env if omitted)")
    p.add_argument("--maze",       action="append", dest="mazes", default=None,
                   help="Maze to link (repeat for multiple; auto-discovered if omitted)")
    p.add_argument("--no-mocap",   action="store_true",
                   help="Skip mocap linking entirely")
    p.add_argument("--overwrite",  action="store_true")
    p.add_argument("--dry-run",    action="store_true")
    p.add_argument("--status",     action="store_true",
                   help="Show link status after linking")
    args = p.parse_args()

    project_id, data_root = _resolve_project(args.project_id, args.data_root)
    if not project_id:
        p.error("--project-id is required (or set NB_PROJECT_ID in .env)")

    mazes = [] if args.no_mocap else args.mazes  # None → auto-discover
    link_session(
        session_name = args.session,
        project_id   = project_id,
        data_root    = data_root,
        mazes        = mazes,
        overwrite    = args.overwrite,
        dry_run      = args.dry_run,
    )
    if args.status:
        link_session_status(args.session, project_id, data_root, mazes)


def _cli_link_batch() -> None:
    """Entry point: nb-link-batch"""
    import argparse
    p = argparse.ArgumentParser(
        description="Link multiple sessions into a project.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--project-id",    default=None, help="Project ID (from .env if omitted)")
    p.add_argument("--data-root",     default=None, help="Data root (from .env if omitted)")
    p.add_argument("--session-list",  required=True,
                   help="Text file with one session name per line")
    p.add_argument("--maze",          action="append", dest="mazes", default=None,
                   help="Maze to link for all sessions (auto-discovered if omitted)")
    p.add_argument("--overwrite",     action="store_true")
    p.add_argument("--dry-run",       action="store_true")
    p.add_argument("--stop-on-error", action="store_true")
    args = p.parse_args()

    project_id, data_root = _resolve_project(args.project_id, args.data_root)
    if not project_id:
        p.error("--project-id is required (or set NB_PROJECT_ID in .env)")

    session_file = Path(args.session_list)
    if not session_file.exists():
        p.error(f"Session list file not found: {session_file}")

    sessions = [
        line.strip()
        for line in session_file.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    print(f"Linking {len(sessions)} sessions into project {project_id!r}")

    link_sessions(
        project_id    = project_id,
        sessions      = sessions,
        data_root     = data_root,
        mazes         = args.mazes,
        overwrite     = args.overwrite,
        dry_run       = args.dry_run,
        stop_on_error = args.stop_on_error,
    )
