# Shared setup for the blastnet PBS jobs on ALCF Polaris; sourced by every jobs/polaris/*.pbs.
# Never submitted itself. Owns no environment: it sources env.sh.
#
# PBS header conventions. qsub reads #PBS lines only from the submitted script, so every .pbs
# repeats them (command-line -q / -l / -v override the defaults in the file):
#   #!/bin/bash -l                       login shell, so `module` exists for env.sh
#   #PBS -A YOUR_PROJECT                 your ALCF allocation (or pass qsub -A)
#   #PBS -l select=1:system=polaris      one node = 4x A100 40 GB, 32-core EPYC, 512 GB RAM
#   #PBS -l place=scatter
#   #PBS -l filesystems=home:eagle       data, venv, caches and runs all live on /lus/eagle
#   #PBS -l walltime=HH:MM:SS
#   #PBS -q <queue>    debug        <= 1 h, 1-2 nodes: smoke tests, analyses, short evals
#                      preemptable  1-10 nodes, up to 72 h, can be killed by demand jobs; every
#                                   job here resumes (train: last.ckpt; eval: skips done rows)
#                      prod         routing queue for >= 10 nodes (ALCF docs), so not for these
#                                   1-node jobs; check `qstat -Qf` if the limits have changed
#   #PBS -N <name>  and  #PBS -j oe     job name; stdout+stderr in one file in $PBS_O_WORKDIR
#
# Submit from the repository root, so that $PBS_O_WORKDIR/jobs/polaris/env.sh exists.
# qsub -v takes comma-separated NAME=value pairs: values may contain spaces but not commas.
# Outputs go to $BLASTNET_RUNS (set by env.sh).

BLASTNET_REPO=${PBS_O_WORKDIR:-$PWD}
source "$BLASTNET_REPO/jobs/polaris/env.sh"
cd "$BLASTNET_REPO" || exit 1
mkdir -p "$BLASTNET_RUNS/logs"

# Single-node DDP over NVLink. NCCL settings recommended by ALCF for Polaris; the rendezvous is
# local (Lightning's subprocess launcher picks a free MASTER_PORT on 127.0.0.1).
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_COLLNET_ENABLE=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_ADDR=127.0.0.1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

JOBID=${PBS_JOBID:-local}

blastnet_info() {
  echo "[$(date -Is)] $(hostname) job=$JOBID $*"
  nvidia-smi -L 2>/dev/null || true
}
