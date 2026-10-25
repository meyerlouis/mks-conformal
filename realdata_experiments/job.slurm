#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Generic array job submitted by submit_all.sh. @SEED is replaced by the array index.

set -u

# Set up your Python environment here, for example
# module load <python module>
source /path/to/your/env/bin/activate
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CMD=()
for a in "$@"; do
    CMD+=("${a//@SEED/$SLURM_ARRAY_TASK_ID}")
done

echo "=============================="
echo "Command: ${CMD[*]}"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo "=============================="

"${CMD[@]}"

echo "Finished: $(date)"
