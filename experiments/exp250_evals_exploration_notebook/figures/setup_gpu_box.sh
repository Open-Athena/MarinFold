#!/usr/bin/env bash
# Set up a Linux GPU box to run the figure notebooks: checkout, Python environment, vLLM, and
# PyMOL for the cartoon renders. Written for a rented single-GPU machine, host-agnostic.
#
# Notebook 1 needs the GPU; 2-4 and the assembler run anywhere. PyMOL is only needed for the
# structure panel and is installed through micromamba because conda-forge is where it ships.
set -euo pipefail
cd "$HOME"

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

if [ ! -d MarinFold ]; then
  git clone --branch exp250/evals-exploration-notebook \
      https://github.com/Open-Athena/MarinFold.git MarinFold
fi
cd MarinFold
git fetch origin exp250/evals-exploration-notebook
git checkout -B exp250/evals-exploration-notebook origin/exp250/evals-exploration-notebook
git log --oneline -1

cd "$HOME"
uv venv --python 3.12 nbenv
export VIRTUAL_ENV="$HOME/nbenv"
uv pip install -q -e "$HOME/MarinFold/marinfold[transformers]" \
    jupyterlab nbconvert ipykernel papermill pandas pyarrow scikit-learn matplotlib \
    py3Dmol svgutils cairosvg

# vLLM: the cu129 wheel from vLLM's own index (the default PyPI wheel is CUDA-13-only in a way
# that has bitten Colab; cu129 runs on this box's CUDA-13 driver).
uv pip install -q "vllm==0.20.2+cu129" \
    --extra-index-url https://wheels.vllm.ai/0.20.2/cu129/ \
    --extra-index-url https://download.pytorch.org/whl/cu129 || echo "VLLM_INSTALL_FAILED"

"$HOME/nbenv/bin/python" - <<'PY'
import importlib.util
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
      "cc", torch.cuda.get_device_capability() if torch.cuda.is_available() else "")
import transformers, marinfold
print("transformers", transformers.__version__)
print("vllm", "yes" if importlib.util.find_spec("vllm") else "NO")
PY
echo "SETUP_DONE"

# PyMOL for the ray-traced cartoon panels. Not a Python dependency: conda-forge is where the
# open-source build lives, so micromamba rather than uv. The plot notebook finds it at this path.
command -v micromamba >/dev/null || \
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
"$HOME/bin/micromamba" create -y -p "$HOME/pymolenv" -c conda-forge pymol-open-source
"$HOME/pymolenv/bin/pymol" -cq -d "print(cmd.get_version())"

echo "SETUP_DONE (also: export FIGLIB_MACHINE_LABEL to keep a public hostname out of metadata)"
