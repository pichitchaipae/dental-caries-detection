"""
AIM XML Ground Truth Parser (week7 – re-export from week6)
==========================================================
Uses importlib to load the week6 module by absolute path so that
this file's name does not shadow the original.
"""

import importlib.util
from pathlib import Path

_WEEK6_FILE = Path(__file__).resolve().parent.parent / "week6" / "xml_ground_truth_parser.py"
_spec = importlib.util.spec_from_file_location("_week6_xml_ground_truth_parser", str(_WEEK6_FILE))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

parse_aim_xml = _mod.parse_aim_xml       # noqa: F401
parse_case_xmls = _mod.parse_case_xmls   # noqa: F401
