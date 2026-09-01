#!/usr/bin/env python3
"""DEPRECATED.

The Blender local-pose copy path incorrectly mixed BVH/FBX bone axes and could
swap limb directions. Use ``visualization/direct_copy_mesh.py`` instead
(CopyQuat / fourway-compatible model-space quaternion copy + LBS).

This file is kept only so old commands fail with a clear message.
"""

from __future__ import annotations

import sys

raise SystemExit(
    "direct_copy_mesh_blender.py is deprecated due to bone-axis bugs.\n"
    "Use visualization/direct_copy_mesh.py (CopyQuat) via:\n"
    "  python visualization/batch_arp_sequence_smal33.py --retarget_mode direct ..."
)
