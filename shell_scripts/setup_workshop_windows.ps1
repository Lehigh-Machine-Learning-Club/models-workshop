param(
    [switch]$RecomputeToy,
    [switch]$RecomputeMnist
)

$ErrorActionPreference = "Stop"

# Workshop setup script for Windows PowerShell.
# Run from the shell_scripts folder:
#   powershell -ExecutionPolicy Bypass -File .\setup_workshop_windows.ps1
# Optional flags:
#   -RecomputeToy    Regenerate toy checkpoints
#   -RecomputeMnist  Retrain MNIST and regenerate artifacts

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $ProjectRoot

Write-Host "==> Project root: $ProjectRoot"

$PythonCmd = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }
try {
    & $PythonCmd --version | Out-Null
} catch {
    Write-Error "Python not found. Install Python 3.9+ and retry."
}

Write-Host "==> Creating virtual environment (.venv)"
& $PythonCmd -m venv .venv

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$VenvPip = Join-Path $ProjectRoot ".venv\Scripts\pip.exe"

Write-Host "==> Installing/refreshing dependencies"
& $VenvPython -m pip install --upgrade pip
& $VenvPip install -r requirements.txt

$ToyFile = Join-Path $ProjectRoot "models\toy_checkpoints_sigmoid.npz"
$MnistFile = Join-Path $ProjectRoot "models\mnist_mlp.pt"

if ($RecomputeToy) {
    Write-Host "==> Recomputing toy checkpoints"
    & $VenvPython scripts/precompute_toy_training.py
} elseif (-not (Test-Path $ToyFile)) {
    Write-Host "==> Toy checkpoints missing, generating now"
    & $VenvPython scripts/precompute_toy_training.py
} else {
    Write-Host "==> Toy checkpoints found, skipping recompute"
}

if ($RecomputeMnist) {
    Write-Host "==> Recomputing MNIST artifacts"
    & $VenvPython scripts/train_mnist.py
} elseif (-not (Test-Path $MnistFile)) {
    Write-Host "==> MNIST model artifact missing, training now"
    & $VenvPython scripts/train_mnist.py
} else {
    Write-Host "==> MNIST model artifact found, skipping retrain"
}

Write-Host "==> Launching Streamlit app"
& $VenvPython -m streamlit run app.py
