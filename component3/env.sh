# Source this from component3/:   source env.sh
# Makes Component 3 and Component 1's library importable WITHOUT installing Component 1
# (pip install -e would write egg-info into component1/), and sends every __pycache__
# into component3/.pycache so nothing is ever written inside component1/.
C3_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export PYTHONPATH="$C3_DIR/src:$C3_DIR/../component1/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPYCACHEPREFIX="$C3_DIR/.pycache"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
if [ -f "$C3_DIR/.venv/bin/activate" ]; then
  . "$C3_DIR/.venv/bin/activate"
fi
