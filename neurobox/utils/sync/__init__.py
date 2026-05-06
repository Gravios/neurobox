"""
neurobox.utils.sync
=====================
Re-export of synchronisation pipeline functions.

The real implementations live in :mod:`neurobox.dtype.sync_pipelines`
(they're co-located with the dtype types they populate).  This
namespace exists for callers that prefer the more obvious utility
import path.
"""

from neurobox.dtype.sync_pipelines import (
    dispatch,
    sync_nlx_vicon,
    sync_ephys_vicon,
    sync_nlx_spots,
    sync_nlx_whl,
    sync_openephys_optitrack,
    sync_openephys_vicon,
)

__all__ = [
    "dispatch",
    "sync_nlx_vicon",
    "sync_ephys_vicon",
    "sync_nlx_spots",
    "sync_nlx_whl",
    "sync_openephys_optitrack",
    "sync_openephys_vicon",
]
