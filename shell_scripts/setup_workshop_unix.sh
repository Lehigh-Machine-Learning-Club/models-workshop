#!/usr/bin/env bash
set -euo pipefail

# Workshop setup script for macOS/Linux.
# Run this script from the shell_scripts folder:
#   bash setup_workshop_unix.sh
# Optional flags:
#   --recompute-toy    Regenerate toy checkpoints
#   --recompute-mnist  Retrain MNIST and regenerate artifacts

RECOMPUTE_TOY=false
RECOMPUTE_MNIST=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --recompute-toy)
      RECOMPUTE_TOY=true
      shift
      ;;
    --recompute-mnist)
      RECOMPUTE_MNIST=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: bash setup_workshop_unix.sh [--recompute-toy] [--recompute-mnist]"
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "==> Project root: ${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} not found. Install Python 3.9+ and retry."
  exit 1
fi

echo "==> Creating virtual environment (.venv)"
"${PYTHON_BIN}" -m venv .venv

echo "==> Activating virtual environment"
# shellcheck disable=SC1091
source ".venv/bin/activate"

echo "==> Installing/refreshing dependencies"
python -m pip install --upgrade pip
pip install -r requirements.txt

TOY_FILE="models/toy_checkpoints_sigmoid.npz"
MNIST_FILE="models/mnist_mlp.pt"

if [[ "${RECOMPUTE_TOY}" == "true" ]]; then
  echo "==> Recomputing toy checkpoints"
  python scripts/precompute_toy_training.py
elif [[ ! -f "${TOY_FILE}" ]]; then
  echo "==> Toy checkpoints missing, generating now"
  python scripts/precompute_toy_training.py
else
  echo "==> Toy checkpoints found, skipping recompute"
fi

if [[ "${RECOMPUTE_MNIST}" == "true" ]]; then
  echo "==> Recomputing MNIST artifacts"
  python scripts/train_mnist.py
elif [[ ! -f "${MNIST_FILE}" ]]; then
  echo "==> MNIST model missing, training now"
  python scripts/train_mnist.py
else
  echo "==> MNIST model artifact found, skipping retrain"
fi

echo "==> Launching Streamlit app"
streamlit run app.py
