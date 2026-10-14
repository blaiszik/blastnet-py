# Source on a Polaris or Sophia login or compute node: `source jobs/polaris/env.sh`.
# Layers zarr + an editable blastnet over the ALCF conda module's CUDA torch.
# Everything lives on the project filesystem, not $HOME (small quota). Edit BLASTNET_ROOT for your
# allocation; the repository is expected at $BLASTNET_ROOT/src/blastnet-py.
export BLASTNET_ROOT=${BLASTNET_ROOT:-/lus/eagle/projects/YOUR_PROJECT/$USER/blastnet}
export BLASTNET_DATA=$BLASTNET_ROOT/data/momentum128
export BLASTNET_RUNS=$BLASTNET_ROOT/runs
export UV_CACHE_DIR=$BLASTNET_ROOT/.cache/uv
export PIP_CACHE_DIR=$BLASTNET_ROOT/.cache/pip
export XDG_CACHE_HOME=$BLASTNET_ROOT/.cache
export TORCH_HOME=$BLASTNET_ROOT/.cache/torch
export MPLCONFIGDIR=$BLASTNET_ROOT/.cache/mpl

# Polaris and Sophia share /eagle but not a Python stack: one venv per machine.
module use /soft/modulefiles
if [[ $(hostname) == sophia* ]]; then
  module load conda/2026-06-08 >/dev/null 2>&1
  _venv=$BLASTNET_ROOT/venv-sophia
else
  module load conda/2026-10-01 >/dev/null 2>&1
  _venv=$BLASTNET_ROOT/venv
fi
eval "$(conda shell.bash hook)"
conda activate base

if [ ! -x "$_venv/bin/python" ]; then
  python -m venv --system-site-packages "$_venv"
  "$_venv/bin/python" -m pip install -q "zarr>=3"
  "$_venv/bin/python" -m pip install -q --no-deps -e "$BLASTNET_ROOT/src/blastnet-py"
fi
source "$_venv/bin/activate"
unset _venv

# Proxy for compute nodes (login nodes reach the internet directly).
if [[ $(hostname) != polaris-login-* && $(hostname) != sophia-login-* ]]; then
  export HTTP_PROXY=http://proxy.alcf.anl.gov:3128 HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
  export http_proxy=$HTTP_PROXY https_proxy=$HTTPS_PROXY
fi
