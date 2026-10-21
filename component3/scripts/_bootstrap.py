"""Make `xaicomp` importable when a script is run as `python scripts/<name>.py` from component3/.

Also redirects bytecode caches to component3/.pycache before Component 1 is imported (and sets
PYTHONPYCACHEPREFIX so worker processes inherit the same rule).
"""
import os
import sys
from pathlib import Path

C3 = Path(__file__).resolve().parents[1]
os.environ.setdefault("PYTHONPYCACHEPREFIX", str(C3 / ".pycache"))
if sys.pycache_prefix is None:
    sys.pycache_prefix = str(C3 / ".pycache")
if str(C3 / "src") not in sys.path:
    sys.path.insert(0, str(C3 / "src"))

import xaicomp  # noqa: E402,F401  (adds ../component1/src to sys.path)
