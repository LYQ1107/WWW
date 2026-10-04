#!/usr/bin/env python3
"""Run ``test_net.py`` with an isolated GMT package overlay.

This is for proxy/screening runs while canonical training owns the checkout.
The overlay is placed before the repository on ``sys.path``; all other GMT
code, configs, datasets, and checkpoints remain the repository versions.
"""

import os
from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parents[1]
overlay_value = os.environ.get("GMT_GTR_OVERLAY")
if not overlay_value:
    raise SystemExit("GMT_GTR_OVERLAY must point to an isolated source root")
overlay = Path(overlay_value).resolve()
if not (overlay / "gtr").is_dir():
    raise SystemExit(f"isolated GMT overlay is missing gtr/: {overlay}")

sys.path.insert(0, str(overlay))
sys.path.insert(1, str(ROOT))
runpy.run_path(str(ROOT / "test_net.py"), run_name="__main__")
