"""pytest bootstrap: make `xaicomp` and Component 1's `eegengine` importable (read-only).

Bytecode caches are redirected into component3/.pycache BEFORE anything from component1 is
imported, so running the tests never writes into component1/.
"""

import sys
from pathlib import Path

C3 = Path(__file__).resolve().parent
if sys.pycache_prefix is None:
    sys.pycache_prefix = str(C3 / ".pycache")
for p in (C3 / "src", C3.parent / "component1" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
