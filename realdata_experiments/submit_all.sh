#!/bin/bash
# Submits every cluster run behind the real-data results. Run from realdata_experiments/.
#
#   DRY=1 ./submit_all.sh               print the sbatch commands only
#   FAMILIES="grid" ./submit_all.sh     one family only (main, highdim, grid)
#
# Output goes to $OUT/<run>/seed_XXXX_<model>.json. Existing files are skipped.

set -u

P=${P:-hmem,compute,stats}
OUT=${OUT:-results_clean}
SEEDS=${SEEDS:-0-49}
GRID_SEEDS=${GRID_SEEDS:-0-29}
FAMILIES=${FAMILIES:-main highdim grid}
DRY=${DRY:-0}

SCORERS="Mahal Bonferroni Density_med2 Kernel_med2 OTCP"
ALPHAS="0.01 0.02 0.05 0.1"

[ -f run_common.py ] || { echo "Run this from realdata_experiments/"; exit 1; }
for f in bio.csv house.csv blog_data.csv; do
    [ -f "../data/feldman/$f" ] || { echo "Missing ../data/feldman/$f"; exit 1; }
done
mkdir -p logs

submit () {   # submit <job-name> <array> <mem> <command ...>     (@SEED -> array index)
    local name=$1 array=$2 mem=$3
    shift 3
    if [ "$DRY" = 1 ]; then
        echo "sbatch --job-name=$name --partition=$P --array=$array --mem=$mem job.slurm $*"
        return
    fi
    while true; do
        out=$(sbatch --parsable --job-name="$name" --partition="$P" \
                     --array="$array" --mem="$mem" job.slurm "$@" 2>&1)
        if [ $? -eq 0 ]; then
            echo "  $name -> $out"
            return
        fi
        case "$out" in
            *[Ll]imit*|*QOS*|*policy*)
                echo "  cap hit on $name, waiting 5 min"
                sleep 300 ;;
            *)
                echo "  FAILED $name: $out"
                return ;;
        esac
    done
}

for FAM in $FAMILIES; do
    case $FAM in

    main)
        for RUN in blog_4d blog_3d blog_2d bio_4d bio_3d bio_2d house_4d house_3d house_2d; do
            for MODEL in Ridge MLP; do
                submit "${RUN}_${MODEL}" "$SEEDS" 16G \
                    python run.py "$RUN" @SEED "$MODEL" \
                    --scorers $SCORERS --alphas $ALPHAS \
                    --kappa 3 --max-kappa 2000 --target-rel-se 0.02 --max-samples 400000 \
                    --out-dir "$OUT/$RUN"
            done
        done ;;

    highdim)
        for RUN in house_6d house_8d house_10d; do
            for MODEL in Ridge MLP; do
                submit "${RUN}_${MODEL}" "$SEEDS" 16G \
                    python run.py "$RUN" @SEED "$MODEL" \
                    --scorers $SCORERS --alphas $ALPHAS \
                    --kappa 3 --max-kappa 30000 --target-rel-se 0.03 --max-samples 10000000 \
                    --out-dir "$OUT/$RUN"
            done
        done ;;

    grid)
        for RUN in house_4d blog_2d bio_3d; do
            submit "grid_${RUN}" "$GRID_SEEDS" 32G \
                python hyperparam_grid/run_grid.py @SEED Ridge --run "$RUN" --alphas 0.1 \
                --ell-mults 16 8 4 2 1 0.5 0.25 0.125 0.0625 \
                --gamma-qs 5 10 30 50 70 90 95 \
                --max-kappa 20000 --target-rel-se 0.02 --max-samples 400000 \
                --out-dir "$OUT/grid_$RUN"
        done ;;

    *)
        echo "Unknown family: $FAM"
        exit 1 ;;
    esac
done

[ "$DRY" = 1 ] || squeue -u "${USER:-$(whoami)}" -h | wc -l | xargs echo "tasks queued:"
