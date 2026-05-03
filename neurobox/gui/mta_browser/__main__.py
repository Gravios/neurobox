"""
neurobox.gui.mta_browser.__main__
===================================
Command-line entry point for the MTA Browser.

Usage::

    python -m neurobox.gui.mta_browser              # empty browser, choose project via dialog
    python -m neurobox.gui.mta_browser --project /data/project/B01
    python -m neurobox.gui.mta_browser --session jg05-20120316 --maze cof
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from neurobox.dtype.session import NBSession
from neurobox.dtype.stc     import NBStateCollection

from .main_window import MTABrowserWindow


def main(argv: list[str] | None = None) -> int:                # noqa: D401
    parser = argparse.ArgumentParser(
        prog="python -m neurobox.gui.mta_browser",
        description="3-D motion-capture browser and behavioural-state "
                    "editor.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Project root directory (overrides .env NB_DATA_PATH)",
    )
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Session name (e.g. 'jg05-20120316') — auto-load on start",
    )
    parser.add_argument(
        "--maze",
        type=str,
        default="cof",
        help="Maze code for --session (default 'cof')",
    )
    parser.add_argument(
        "--trial",
        type=str,
        default="all",
        help="Trial label for --session (default 'all')",
    )
    args = parser.parse_args(argv)

    session = None
    if args.session is not None:
        session = NBSession(
            session_name = args.session,
            maze         = args.maze,
            trial        = args.trial,
        )
        session.load("xyz")
        try:
            session.load("stc")
        except Exception:
            session.stc = NBStateCollection(mode="manual")

    MTABrowserWindow.launch(
        session      = session,
        project_root = Path(args.project) if args.project else None,
    )
    return 0


if __name__ == "__main__":                          # pragma: no cover
    sys.exit(main())
