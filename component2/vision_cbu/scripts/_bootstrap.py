"""Make `src` importable when scripts are run as `python scripts/<name>.py` from the project root."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXIT_OK = 0
EXIT_REFUSED = 2      # real data / model missing - the command refused to run
EXIT_ERROR = 1
