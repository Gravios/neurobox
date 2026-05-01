"""
neurobox.config
===============
Project configuration and session linking utilities.
"""
from .config import (
    configure_project,
    discover_mazes,
    link_session,
    link_sessions,
    link_session_status,
    load_config,
)
from .session_lists import (
    SessionList,
    TrialSpec,
    SubjectInfo,
    HeadCorrection,
    ChannelGroup,
    AnatLocation,
    available_session_lists,
    load_session_list,
)
from .standards import (
    MazeInfo,
    load_mazes,
    get_maze,
    load_markers,
    load_marker_connections,
    make_standard_model,
)
__all__ = [
    "configure_project",
    "discover_mazes",
    "link_session",
    "link_sessions",
    "link_session_status",
    "load_config",
    "SessionList",
    "TrialSpec",
    "SubjectInfo",
    "HeadCorrection",
    "ChannelGroup",
    "AnatLocation",
    "available_session_lists",
    "load_session_list",
    # Standard reference data: mazes, markers, skeleton connections
    "MazeInfo",
    "load_mazes",
    "get_maze",
    "load_markers",
    "load_marker_connections",
    "make_standard_model",
]
