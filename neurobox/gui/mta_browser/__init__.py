"""
neurobox.gui.mta_browser
==========================
Interactive 3-D motion-capture browser and behavioural-state editor.

Port of the MATLAB ``MTABrowser.m`` / ``MTABrowser.fig`` GUI (Anton
Sirota / Justin Graboski lab) to Python using PySide6.

What it does
------------
Loads an :class:`NBSession`, displays the rat's 3-D skeleton over time
with a time-slider scrubber, shows configurable feature traces aligned
to the timeline, and lets you label behavioural states by holding a
keyboard key while playing forward or backward.  Saves labels to
disk via :class:`NBStateCollection`.

Usage
-----
From a script::

    from neurobox.gui.mta_browser import MTABrowserWindow
    from neurobox.dtype import NBSession

    session = NBSession("jg05-20120316", maze="cof")
    session.load("xyz")
    session.load("stc")            # optional — adds existing labels

    MTABrowserWindow.launch(session=session)

Or from the command line::

    python -m neurobox.gui.mta_browser SESSION_NAME [--maze cof] [--trial all]

What's ported
-------------
* Session data-management panel (subject / date / maze / trial picker)
* Motion-labelling 3-D viewer with playback controls
* State editor table with add/remove/rename and period statistics
* Multi-feature time-trace panel aligned with the playback head
* Keyboard shortcuts matching MATLAB:
    - Space          : play / pause
    - ←/→            : step backward / forward
    - Ctrl+←/→       : jump 50 frames
    - ↑/↓            : adjust play speed
    - 1 / 2 / 3      : view from X / Y / Z axis
    - <state-key>    : toggle labelling for that state
    - Delete         : erase mode (clears state at current frame)

What's NOT ported (and why)
---------------------------
* LFP-states panel (BSlfpStates) — see :mod:`neurobox.gui.check_eeg_states`
* Setup panel (BSSetup) — single-purpose admin
* Auxiliary video panel — niche use case
* MATLAB GUIDE "socket" / "view" layout management — replaced by
  :class:`QStackedWidget` and a clean signal/slot architecture

The MATLAB original was 2,568 LoC of GUIDE-generated code; this port
is ~2,000 LoC of clean PySide6 organised into focused modules.
"""

from __future__ import annotations

# Most of the API is locked behind a PySide6 import, which is an
# optional dependency.  Defer the import until a caller actually
# requests something from this module so that bare `import neurobox`
# doesn't pull in Qt.
def __getattr__(name):
    if name == "MTABrowserWindow":
        from .main_window import MTABrowserWindow
        return MTABrowserWindow
    raise AttributeError(
        f"module 'neurobox.gui.mta_browser' has no attribute {name!r}"
    )


__all__ = [
    "MTABrowserWindow",
]
