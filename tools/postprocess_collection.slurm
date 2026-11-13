#!/bin/bash -l
#SBATCH --job-name=mol46_postprocess
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=12:00:00
set -euo pipefail

: "${POSTPROCESS_INPUT_DIR:?Set POSTPROCESS_INPUT_DIR}"
: "${POSTPROCESS_OUTPUT_BASE:?Set POSTPROCESS_OUTPUT_BASE}"
: "${POSTPROCESS_PYTHON:?Set POSTPROCESS_PYTHON to a compatible Python environment}"
POSTPROCESS_CODE_ROOT="${POSTPROCESS_CODE_ROOT:-$PWD}"
POSTPROCESS_RUN_DIR="$POSTPROCESS_OUTPUT_BASE/run_${SLURM_JOB_ID}"
mkdir -p "$POSTPROCESS_RUN_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export POSTPROCESS_RUN_DIR
export MPLCONFIGDIR="$POSTPROCESS_RUN_DIR/.matplotlib_cache"
cleanup_task_font_cache() {
  "$POSTPROCESS_PYTHON" - <<'PY'
import os,shutil
from pathlib import Path
p=Path(os.environ['MPLCONFIGDIR']);run=Path(os.environ['POSTPROCESS_RUN_DIR']).resolve()
base=Path(os.environ['POSTPROCESS_OUTPUT_BASE']).resolve()
if p.is_dir() and not p.is_symlink() and p.name=='.matplotlib_cache' and p.resolve().parent==run and run.parent==base and run.name=='run_'+os.environ['SLURM_JOB_ID']:
    shutil.rmtree(p)
PY
}
trap cleanup_task_font_cache EXIT
cd "$POSTPROCESS_CODE_ROOT"

getconf GNU_LIBC_VERSION
"$POSTPROCESS_PYTHON" - <<'PY'
import os,json,sys,socket,platform
from pathlib import Path
import numpy,scipy,ase,h5py,networkx,freud
info={'hostname':socket.gethostname(),'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
      'allocated_cpus':int(os.environ.get('SLURM_CPUS_PER_TASK','1')),'python':sys.version,
      'libc':platform.libc_ver(),'versions':{m.__name__:m.__version__ for m in [numpy,scipy,ase,h5py,networkx,freud]},
      'input_dir':os.environ['POSTPROCESS_INPUT_DIR'],'code_root':str(Path.cwd())}
name=f"preflight_{os.environ['SLURM_JOB_ID']}_{os.environ.get('SLURM_RESTART_COUNT','0')}.json"
with (Path(os.environ['POSTPROCESS_RUN_DIR'])/name).open('x') as f:json.dump(info,f,indent=2)
print(json.dumps(info,indent=2),flush=True)
PY

"$POSTPROCESS_PYTHON" tools/run_collection.py \
  --input-dir "$POSTPROCESS_INPUT_DIR" \
  --output-root "$POSTPROCESS_RUN_DIR" \
  --workers "$SLURM_CPUS_PER_TASK" --phase all \
  --env-cut 8.5 --nrad 3 --lmax 3 --resume --retry-failed
