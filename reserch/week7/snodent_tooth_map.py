"""
SNODENT-to-FDI Tooth Mapping (week7 – identical to week6)
=========================================================
Re-exported here so week7 modules can import locally.
Uses importlib to avoid shadowing the week6 module.
"""

import importlib.util
from pathlib import Path

_WEEK6_FILE = Path(__file__).resolve().parent.parent / "week6" / "snodent_tooth_map.py"
_spec = importlib.util.spec_from_file_location("_week6_snodent_tooth_map", str(_WEEK6_FILE))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

SNODENT_TO_FDI          = _mod.SNODENT_TO_FDI            # noqa: F401
SNODENT_SURFACE_MAP     = _mod.SNODENT_SURFACE_MAP       # noqa: F401
DISPLAY_NAME_TO_SURFACE = _mod.DISPLAY_NAME_TO_SURFACE   # noqa: F401
FDI_TOOTH_NAMES         = _mod.FDI_TOOTH_NAMES            # noqa: F401
FDI_TO_SNODENT          = _mod.FDI_TO_SNODENT             # noqa: F401
snodent_display_to_fdi  = _mod.snodent_display_to_fdi     # noqa: F401
