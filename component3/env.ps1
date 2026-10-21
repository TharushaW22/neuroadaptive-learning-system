# Run from component3\ in PowerShell:   . .\env.ps1
# Same purpose as env.sh: import Component 1 read-only and keep all bytecode caches in component3\.pycache.
$C3 = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = "$C3\src;$C3\..\component1\src"
$env:PYTHONPYCACHEPREFIX = "$C3\.pycache"
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"
if (Test-Path "$C3\.venv\Scripts\Activate.ps1") { . "$C3\.venv\Scripts\Activate.ps1" }
