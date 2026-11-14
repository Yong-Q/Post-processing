#!/bin/bash -l
#SBATCH --job-name=xi_kernel_pilot
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:nvidia_geforce_rtx_4090:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail
: "${XI_SOURCE:?Set water postprocessing source}"
: "${XI_OUTPUT_BASE:?Set new output base}"
: "${XI_PYTHON:?Set Python with torch}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
getconf GNU_LIBC_VERSION
for XI_RUNTIME_PYTHON in "$XI_PYTHON" /home/qiuyong/.conda/envs/Uni-Core/bin/python; do
  if "$XI_RUNTIME_PYTHON" -c 'import torch,numpy,scipy,h5py; print(torch.__version__,numpy.__version__,scipy.__version__,h5py.__version__); assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0)); print(torch.ones(1,device="cuda").item())'; then
    exec "$XI_RUNTIME_PYTHON" tools/run_xi_pilot.py --source "$XI_SOURCE" --output "$XI_OUTPUT_BASE/run_$SLURM_JOB_ID" --epochs 200 --device cuda
  fi
done
echo 'No CUDA-compatible Python found; CPU training fallback prohibited' >&2
exit 1
