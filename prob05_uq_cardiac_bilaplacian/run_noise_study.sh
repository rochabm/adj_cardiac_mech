#!/usr/bin/env bash
# =============================================================================
# run_noise_study.sh
# -----------------------------------------------------------------------------
# MAP comparison (L-BFGS-B vs Newton-CG), m = log C, calibrated BiLaplacian
# prior, for data noise levels 0%, 1% and 3% (relative to RMS(Fd - I)).
# Noise is added to the data only; for p > 0 the likelihood std of each case
# is matched to the actual noise std (--match-noise-std). For p = 0 the
# likelihood uses ex06's --noise-std (default 1e-3).
#
# Phase 1: Newton-CG for all noise levels and both cases (fast -> check first)
# Phase 2: L-BFGS-B for all noise levels and both cases
# Same data, seeds and initial guess in both phases. Each method writes to its
# own folder; make_latex_tables.py merges them:
#     $OUT/newtoncg/noise_XXX/<case>/...
#     $OUT/lbfgsb/noise_XXX/<case>/...
#
# Usage:
#     bash run_noise_study.sh                      # both phases (use bash, not sh)
#     PHASES="newtoncg" bash run_noise_study.sh    # Newton-CG only
#     PHASES="lbfgsb"   bash run_noise_study.sh    # L-BFGS-B only (e.g. later)
#     OUT=my_study bash run_noise_study.sh         # other output root
#     bash run_noise_study.sh --gtol 1e-6          # extra args -> ex06
#
# Runs are sequential on purpose: parallel runs compile the same FFCx forms
# at the same time and can hit the JIT-cache timeout.
# =============================================================================
set -euo pipefail
export PYTHONUNBUFFERED=1   # print line by line through the pipe to tee

OUT=${OUT:-out_noise_study}
LEVELS=(0 0.01 0.03)
PHASES=${PHASES:-"newtoncg lbfgsb"}
PYTHON=${PYTHON:-python}

mkdir -p "$OUT"
echo "Noise study -> $OUT   phases: $PHASES   (extra ex06 args: $*)"

for method in $PHASES; do
    echo
    echo "######################################################################"
    echo "# PHASE: $method"
    echo "######################################################################"
    for lvl in "${LEVELS[@]}"; do
        pct=$("$PYTHON" -c "print(f'{round($lvl*100):03d}')")
        dir="$OUT/$method/noise_${pct}"
        mkdir -p "$dir"
        echo
        echo "======================================================================"
        echo " $method | noise level ${lvl}  ->  $dir"
        echo "======================================================================"
        start=$(date +%s)
        "$PYTHON" -u ex06_map_comparison.py \
            --methods "$method" \
            --data-noise "$lvl" \
            --data-noise-seed 1234 \
            --match-noise-std \
            --gtol 1e-6 \
            --ftol 1e-12 \
            --output-dir "$dir" \
            "$@" 2>&1 | tee "$dir/run.log"
        echo " done in $(( $(date +%s) - start )) s"
    done
    # tables after each phase (after phase 1 they show Newton-CG only)
    echo
    echo "Building LaTeX tables from what is available ..."
    "$PYTHON" make_latex_tables.py --root "$OUT" --out "$OUT/tables.tex" \
        --standalone "$OUT/tables_standalone.tex"
done

echo
echo "All phases done. Tables: $OUT/tables.tex  (preview: $OUT/tables_standalone.tex)"
