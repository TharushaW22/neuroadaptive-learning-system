# Runs the dataset-free verification sequence. Usage (project root, venv active):
#   powershell -ExecutionPolicy Bypass -File scripts\run_checks.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

function Step($name, [scriptblock]$cmd, [int[]]$okCodes = @(0)) {
    Write-Host "`n=== $name ===" -ForegroundColor Cyan
    & $cmd
    if ($okCodes -notcontains $LASTEXITCODE) { throw "$name failed (exit $LASTEXITCODE)" }
}

Step "Environment check"            { python scripts\check_environment.py }
Step "Automated tests"              { python -m pytest -q }
Step "Dataset audit (may be empty)" { python scripts\audit_dataset.py --no-probe }
Step "Synthetic demo pipeline"      { python scripts\generate_demo_data.py --train-demo-models --evaluate --epochs 5 }
Step "Mock webcam extractor"        { python scripts\run_webcam.py --mock --seconds 3 --no-display }
# Real-data commands: exit 2 (refused) is the expected result until DAiSEE is added.
Step "Real training guard"          { python scripts\train_baseline.py } @(0, 2)
Step "Real evaluation guard"        { python scripts\evaluate.py } @(0, 2)
Write-Host "`nAll checks completed." -ForegroundColor Green
