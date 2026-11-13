#!/bin/bash -l
#SBATCH --job-name=mol46_reweight
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=08:00:00
set -euo pipefail
: "${REWEIGHT_SOURCE_ROOT:?Set source postprocessing collection}"
: "${REWEIGHT_ENERGY_DIR:?Set snapshotted energy directory}"
: "${REWEIGHT_OUTPUT_BASE:?Set output base}"
: "${REWEIGHT_PYTHON:?Set compatible Python}"
: "${REWEIGHT_ENERGY_COMMIT:?Set energy commit}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
getconf GNU_LIBC_VERSION
"$REWEIGHT_PYTHON" -c 'import numpy,scipy,h5py; print(numpy.__version__,scipy.__version__,h5py.__version__)'
"$REWEIGHT_PYTHON" tools/run_reweight_collection.py \
  --source-root "$REWEIGHT_SOURCE_ROOT" --energy-dir "$REWEIGHT_ENERGY_DIR" \
  --output-root "$REWEIGHT_OUTPUT_BASE/run_$SLURM_JOB_ID" \
  --energy-commit "$REWEIGHT_ENERGY_COMMIT" --workers "$SLURM_CPUS_PER_TASK"
