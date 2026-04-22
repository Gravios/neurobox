"""Tests for neurobox.config — link_session, link_sessions, discover_mazes, status."""

import pytest
from pathlib import Path


# ── Shared fixture ────────────────────────────────────────────────────────── #

@pytest.fixture
def project_tree(tmp_path):
    """Minimal source/processed/project tree for two sessions."""
    sessions = [
        ("sirotaA-jg-05-20120316", ["cof", "nor"]),
        ("sirotaA-jg-06-20120320", ["cof"]),
    ]
    from neurobox.dtype.paths import parse_session_name
    for name, mazes in sessions:
        parts  = parse_session_name(name)
        src_id = parts["sourceId"]
        u_id   = parts["userId"]
        sub_id = parts["subjectId"]
        u      = f"{src_id}-{u_id}"
        us     = f"{u}-{sub_id}"

        # processed ephys
        eph = tmp_path / "processed" / "ephys" / src_id / u / us / name
        eph.mkdir(parents=True)
        (eph / f"{name}.yaml").write_text(
            "acquisitionSystem:\n  nChannels: 32\n  samplingRate: 20000\n"
        )
        (eph / f"{name}.res.1").write_bytes(b"\x00" * 8)
        (eph / f"{name}.lfp").write_bytes(b"\x00" * 16)

        # processed mocap per maze
        for maze in mazes:
            moc = tmp_path / "processed" / "mocap" / src_id / u / us / name / maze
            moc.mkdir(parents=True)
            (moc / f"{name}.Trial001.mat").write_bytes(b"\x00" * 4)

    return tmp_path, sessions


# ── configure_project ─────────────────────────────────────────────────────── #

def test_configure_creates_subdirs(project_tree):
    from neurobox.config import configure_project
    root, _ = project_tree
    configure_project("B01", data_root=root, dotenv_path=root / ".env")
    for sub in ("config", "figures", "scripts", "notebooks", "models"):
        assert (root / "project" / "B01" / sub).is_dir()


def test_configure_writes_env(project_tree):
    from neurobox.config import configure_project, load_config
    root, _ = project_tree
    configure_project("B01", data_root=root, dotenv_path=root / ".env")
    conf = load_config(root / ".env")
    assert conf["NB_DATA_PATH"] == str(root)
    assert conf["NB_PROJECT_ID"] == "B01"


def test_configure_no_overwrite_keeps_existing(project_tree):
    from neurobox.config import configure_project
    root, _ = project_tree
    env = root / ".env"
    env.write_text("NB_DATA_PATH=/old\nNB_PROJECT_ID=OLD\n")
    configure_project("B01", data_root=root, dotenv_path=env, overwrite=False)
    assert "OLD" in env.read_text()


def test_configure_overwrite_replaces_env(project_tree):
    from neurobox.config import configure_project, load_config
    root, _ = project_tree
    env = root / ".env"
    env.write_text("NB_DATA_PATH=/old\nNB_PROJECT_ID=OLD\n")
    configure_project("B01", data_root=root, dotenv_path=env, overwrite=True)
    assert load_config(env)["NB_PROJECT_ID"] == "B01"


# ── discover_mazes ────────────────────────────────────────────────────────── #

def test_discover_mazes_finds_all(project_tree):
    from neurobox.config import discover_mazes
    root, sessions = project_tree
    name, mazes = sessions[0]
    found = discover_mazes(name, data_root=root)
    assert sorted(found) == sorted(mazes)


def test_discover_mazes_single_maze(project_tree):
    from neurobox.config import discover_mazes
    root, sessions = project_tree
    name, mazes = sessions[1]   # only "cof"
    found = discover_mazes(name, data_root=root)
    assert found == ["cof"]


def test_discover_mazes_empty_when_no_mocap(tmp_path):
    from neurobox.config import discover_mazes
    # No processed mocap directory at all
    found = discover_mazes("nonexistent-jg-01-20240101", data_root=tmp_path)
    assert found == []


# ── link_session ──────────────────────────────────────────────────────────── #

def _setup(root, tmp_path=None):
    """Configure project and return project_id."""
    from neurobox.config import configure_project
    configure_project("B01", data_root=root, dotenv_path=root / ".env")
    return "B01"


def test_link_creates_spath(project_tree):
    from neurobox.config import link_session
    root, sessions = project_tree
    name, _ = sessions[0]
    _setup(root)
    paths = link_session(name, "B01", data_root=root)
    assert paths.spath.is_dir() and not paths.spath.is_symlink()


def test_link_ephys_are_symlinks(project_tree):
    from neurobox.config import link_session
    root, sessions = project_tree
    name, _ = sessions[0]
    _setup(root)
    paths = link_session(name, "B01", data_root=root)
    yaml_link = paths.spath / f"{name}.yaml"
    assert yaml_link.is_symlink() and yaml_link.resolve().exists()


def test_link_auto_discovers_mazes(project_tree):
    from neurobox.config import link_session
    root, sessions = project_tree
    name, mazes = sessions[0]
    _setup(root)
    paths = link_session(name, "B01", data_root=root, mazes=None)
    # All maze dirs should have been created as real directories
    for maze in mazes:
        maze_dir = paths.spath / maze
        assert maze_dir.is_dir() and not maze_dir.is_symlink()


def test_link_maze_dirs_are_real(project_tree):
    from neurobox.config import link_session
    root, sessions = project_tree
    name, mazes = sessions[0]
    _setup(root)
    paths = link_session(name, "B01", data_root=root, mazes=mazes)
    for maze in mazes:
        d = paths.spath / maze
        assert d.is_dir() and not d.is_symlink(), f"{maze}/ must be a real dir"


def test_link_mocap_files_are_symlinks(project_tree):
    from neurobox.config import link_session
    root, sessions = project_tree
    name, mazes = sessions[0]
    _setup(root)
    paths = link_session(name, "B01", data_root=root, mazes=mazes)
    for maze in mazes:
        mat = paths.spath / maze / f"{name}.Trial001.mat"
        assert mat.is_symlink() and mat.resolve().exists()


def test_link_skip_mocap_when_empty_list(project_tree):
    from neurobox.config import link_session
    root, sessions = project_tree
    name, mazes = sessions[0]
    _setup(root)
    paths = link_session(name, "B01", data_root=root, mazes=[])
    # No maze subdirs should exist
    for maze in mazes:
        assert not (paths.spath / maze).exists()


def test_link_dry_run_no_files_created(project_tree):
    from neurobox.config import link_session
    root, sessions = project_tree
    name, _ = sessions[0]
    _setup(root)
    link_session(name, "B01", data_root=root, dry_run=True)
    spath = root / "project" / "B01" / name
    assert not spath.exists()


def test_link_overwrite_replaces_symlinks(project_tree):
    from neurobox.config import link_session
    root, sessions = project_tree
    name, _ = sessions[0]
    _setup(root)
    paths = link_session(name, "B01", data_root=root)
    # Second call with overwrite should not raise
    link_session(name, "B01", data_root=root, overwrite=True)
    assert (paths.spath / f"{name}.yaml").is_symlink()


def test_link_skips_existing_without_overwrite(project_tree):
    from neurobox.config import link_session
    root, sessions = project_tree
    name, _ = sessions[0]
    _setup(root)
    paths = link_session(name, "B01", data_root=root)
    # Replace one symlink with a real file
    yaml_link = paths.spath / f"{name}.yaml"
    yaml_link.unlink()
    yaml_link.write_text("real file")
    # Without overwrite, real file is preserved (skipped)
    link_session(name, "B01", data_root=root, overwrite=False)
    assert not yaml_link.is_symlink()
    assert yaml_link.read_text() == "real file"


# ── link_sessions (batch) ────────────────────────────────────────────────── #

def test_link_sessions_batch(project_tree):
    from neurobox.config import link_sessions
    root, sessions = project_tree
    _setup(root)
    names = [name for name, _ in sessions]
    results = link_sessions("B01", names, data_root=root)
    assert len(results) == 2
    for name in names:
        from neurobox.dtype.paths import NBSessionPaths
        assert isinstance(results[name], NBSessionPaths)


def test_link_sessions_dict_entries(project_tree):
    from neurobox.config import link_sessions
    root, sessions = project_tree
    _setup(root)
    name0, mazes0 = sessions[0]
    entries = [
        {"sessionName": name0, "mazes": mazes0[:1]},
    ]
    results = link_sessions("B01", entries, data_root=root)
    assert name0 in results


def test_link_sessions_continues_on_error(tmp_path):
    """Malformed session names raise ValueError — batch should catch and continue."""
    from neurobox.config import configure_project, link_sessions
    configure_project("B01", data_root=tmp_path, dotenv_path=tmp_path / ".env")
    # "invalid" and "also_invalid" don't match <srcId>-<user>-<subId(digits)>-<date(8dig)>
    results = link_sessions(
        "B01",
        ["invalid", "also_invalid"],
        data_root     = tmp_path,
        stop_on_error = False,
    )
    for v in results.values():
        assert isinstance(v, Exception)


def test_link_sessions_stop_on_error(tmp_path):
    """stop_on_error=True should re-raise on first malformed name."""
    from neurobox.config import configure_project, link_sessions
    configure_project("B01", data_root=tmp_path, dotenv_path=tmp_path / ".env")
    with pytest.raises(Exception):
        link_sessions(
            "B01",
            ["invalid"],
            data_root     = tmp_path,
            stop_on_error = True,
        )


# ── link_session_status ───────────────────────────────────────────────────── #

def test_status_all_linked(project_tree):
    from neurobox.config import configure_project, link_session, link_session_status
    root, sessions = project_tree
    name, mazes = sessions[0]
    configure_project("B01", data_root=root, dotenv_path=root / ".env")
    link_session(name, "B01", data_root=root, mazes=mazes)
    report = link_session_status(name, "B01", data_root=root, mazes=mazes)
    assert report["ok"]
    assert report["ephys"]["missing"] == []
    assert report["ephys"]["stale"]   == []


def test_status_detects_stale_link(project_tree):
    from neurobox.config import configure_project, link_session, link_session_status
    root, sessions = project_tree
    name, mazes = sessions[0]
    configure_project("B01", data_root=root, dotenv_path=root / ".env")
    paths = link_session(name, "B01", data_root=root, mazes=mazes)

    # Break the symlink by redirecting it to a nonexistent path
    # (leave the source file in place so _check_dir's source-scan still sees it)
    link = paths.spath / f"{name}.res.1"
    link.unlink()
    link.symlink_to(Path("/nonexistent/path/does_not_exist.res.1"))

    report = link_session_status(name, "B01", data_root=root, mazes=mazes)
    assert not report["ok"]
    assert f"{name}.res.1" in report["ephys"]["stale"]


def test_status_detects_missing_link(project_tree):
    from neurobox.config import configure_project, link_session, link_session_status
    root, sessions = project_tree
    name, mazes = sessions[0]
    configure_project("B01", data_root=root, dotenv_path=root / ".env")
    paths = link_session(name, "B01", data_root=root, mazes=mazes)

    # Remove the symlink but leave the target — simulates "not yet linked"
    link = paths.spath / f"{name}.lfp"
    link.unlink()

    report = link_session_status(name, "B01", data_root=root, mazes=mazes)
    assert not report["ok"]
    assert f"{name}.lfp" in report["ephys"]["missing"]


def test_status_reports_analysis_files(project_tree):
    from neurobox.config import configure_project, link_session, link_session_status
    root, sessions = project_tree
    name, mazes = sessions[0]
    configure_project("B01", data_root=root, dotenv_path=root / ".env")
    paths = link_session(name, "B01", data_root=root, mazes=mazes)

    # Write a fake analysis output
    (paths.spath / f"{name}.cof.all.ses.pkl").write_bytes(b"fake")

    report = link_session_status(name, "B01", data_root=root, mazes=mazes)
    assert f"{name}.cof.all.ses.pkl" in report["analysis"]
