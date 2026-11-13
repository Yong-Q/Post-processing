#!/bin/bash
#SBATCH --job-name=cdft_energy
#SBATCH --partition=amd_m9_768
#SBATCH --account=bscc-m9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/publicfs10/fs10-m9/home/m9s005258/qy/cdft/energy_out/slurm-%j.out
set -euo pipefail
source /etc/profile.d/modules.sh
module purge
module use /publicfs10/fs10-share/soft/share-soft/modulefiles
module load gcc/14.2.0
module load openmpi/4.1.8-gcc14.2.0
export LD_LIBRARY_PATH=/publicfs10/fs10-m9/home/m9s005258/qy/deps/blas/lib:/publicfs10/fs10-m9/home/m9s005258/qy/build-cpu-mpi/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1
export ENERGY_OUT=/publicfs10/fs10-m9/home/m9s005258/qy/cdft/energy_out_hp
unset OMP_PROC_BIND OMP_PLACES 2>/dev/null || true
echo "=== cdft_energy node=$(hostname) cpus=$SLURM_CPUS_PER_TASK start=$(date) ==="
cd /publicfs10/fs10-m9/home/m9s005258/qy/cdft
cat energy_list.txt | xargs -P 24 -I{} /publicfs10/fs10-m9/home/m9s005258/qy/cdft/energy_eval_one.sh {}
echo "=== done $(date) : $(wc -l < $ENERGY_OUT/progress.txt) states ==="
