#!/usr/bin/env bash
# Create/prepare a Python environment for Beyond Images.
#
# Usage:
#   bash scripts/setup_env.sh              # CUDA build (default tag below)
#   CUDA_TAG=cpu bash scripts/setup_env.sh # CPU-only install
#   CUDA_TAG=cu126 bash scripts/setup_env.sh
#
# Uses the currently active Python (conda env or venv). Python >= 3.10 required.
set -euo pipefail

CUDA_TAG="${CUDA_TAG:-cu132}"
PYTHON="${PYTHON:-python}"

echo "== Beyond Images environment setup =="
"$PYTHON" --version

if [ "$CUDA_TAG" = "cpu" ]; then
    echo "-> Installing CPU-only PyTorch"
    "$PYTHON" -m pip install torch torchvision
else
    echo "-> Installing PyTorch for $CUDA_TAG"
    "$PYTHON" -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
fi

echo "-> Installing pipeline dependencies"
"$PYTHON" -m pip install -r requirements.txt

echo "-> Verifying installation"
"$PYTHON" - <<'EOF'
import torch, transformers, sentence_transformers, h5py, bs4, yaml
print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()}")
print(f"transformers {transformers.__version__}")
print("Environment OK.")
EOF
