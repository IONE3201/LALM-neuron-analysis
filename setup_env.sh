#!/usr/bin/env bash
# One-time environment setup on a Linux GPU machine (tested with an NVIDIA A100 80GB).
# The venv is created under $CAGA_BASE so that data, weights and environment live on the same disk.
set -e
export CAGA_BASE="${CAGA_BASE:-$HOME/caga}"
VENV="$CAGA_BASE/venv"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

apt-get update && apt-get install -y ffmpeg

python -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip

# torch: pick the index matching the CUDA version reported by `nvidia-smi` (cu118 / cu121 / cu124 ...)
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r "$REPO/requirements.txt" ipykernel

python -m ipykernel install --user --name caga --display-name "Python (caga)"
echo
echo "Done. Select the Jupyter kernel 'Python (caga)', keep CAGA_BASE=$CAGA_BASE, and run notebooks/01 -> 02 -> 03."
