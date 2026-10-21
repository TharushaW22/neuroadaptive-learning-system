"""xaicomp — comparative evaluation of XAI methods for cognitive-state models (Component 3).

Importing the package first runs `_bootstrap`, which makes Component 1 importable read-only
and keeps every bytecode cache inside component3/.
"""

from xaicomp import _bootstrap  # noqa: F401  (side effect: sys.path + pycache prefix)

__version__ = "0.1.0"
