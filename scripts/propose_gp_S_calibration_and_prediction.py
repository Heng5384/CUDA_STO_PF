#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
script = ROOT / "tools" / "analysis" / "gp_assisted_beta_nucleation.py"
raise SystemExit(subprocess.call([sys.executable, str(script), "--out-dir", str(ROOT / "reports" / "gp_assisted_beta_nucleation_audit")]))
