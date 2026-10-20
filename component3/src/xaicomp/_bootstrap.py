"""Import bridge to Component 1 (read-only).

* Component 1 is never pip-installed (an editable install writes egg-info into component1/).
  Its `src/` folder is appended to sys.path instead.
* Python normally writes `__pycache__/` next to every imported module. For Component 1 that
  would be inside component1/src, so all bytecode caches are redirected to component3/.pycache
  (sys.pycache_prefix is read at every import, so setting it here is enough as long as this
  module is imported before `eegengine`).
"""

from __future__ import annotations

import sys
from pathlib import Path

C3_ROOT = Path(__file__).resolve().parents[2]          # component3/
REPO_ROOT = C3_ROOT.parent
C1_ROOT = REPO_ROOT / "component1"
C1_SRC = C1_ROOT / "src"

if sys.pycache_prefix is None:
    sys.pycache_prefix = str(C3_ROOT / ".pycache")
if str(C1_SRC) not in sys.path:
    sys.path.append(str(C1_SRC))
