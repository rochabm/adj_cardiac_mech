#!/usr/bin/env bash
# =============================================================================
# run_noise_study.sh
# -----------------------------------------------------------------------------
# MAP comparison (L-BFGS-B vs Newton-CG), m = log C, BiLaplacian prior
# calibrated PER CASE, for data noise levels 0%, 1% and 3% (relative to
# RMS(Fd - I)). Noise is added to the data only; for p > 0 the likelihood std
# of each case is matched to the actual noise std (--match-noise-std). For
# p = 0 the likelihood uses ex06's --noise-std (default 1e-3).
#
# Case-specific priors (override with the environment variables):
#   fibrosis : sigma = log(3)/2,   rho ~ 11.6 mm  (focal)
#   linear   : sigma = log(3/2)/2, rho ~ heart size (smooth)
#
# Phase 1: Newton-CG, all noise levels, both cases (fast -> check first)
# Phase 2: L-BFGS-B,  all noise levels, both cases
# Output:  $OUT/<method>/noise_XXX/<case>/...   (merged by make_latex_tables.py)
#
# Usage (use bash, not sh):
#     bash run_noise_study.sh --gtol 1e-6 --ftol 1e-12 --maxiter 1000
#     PHASES="newtoncg" bash run_noise_study.sh ...      # Newton-CG only
#     CASES="linear"    bash run_noise_study.sh ...      # one case only
#     PRIOR_LINEAR="--gamma 6.22 --delta 9.85e-3" bash run_noise_study.sh ...
#     OUT=my_study bash run_noise_study.sh ...
# Extra arguments are passed to every ex06 run.
#
# Runs are sequential on purpose (FFCx JIT cache).
# =============================================================================
set -euo pipefail
export PYTHONUNBUFFERED=1   # print line by line through the pipe to tee

OUT=${OUT:-out_noise_study}
LEVELS=(0 0.01 0.03)
PHASES=${PHASES:-"newtoncg lbfgsb"}
CASES=${CASES:-"linear fibrosis"}
PYTHON=${PYTHON:-python}

# ---- case-specific priors (gamma, delta of the BiLaplacian on m = log C) ----
PRIOR_LINEAR=${PRIOR_LINEAR:-"--gamma 6.22 --delta 9.85e-3"}
PRIOR_FIBROSIS=${PRIOR_FIBROSIS:-"--gamma 1.457 --delta 1.441e-2"}

prior_for() {
    case "$1" in
        linear)   echo "$PRIOR_LINEAR" ;;
        fibrosis) echo "$PRIOR_FIBROSIS" ;;
        *) echo "unknown case $1" >&2; exit 1 ;;
    esac
}

mkdir -p "$OUT"
echo "Noise study -> $OUT"
echo "  phases : $PHASES"
echo "  cases  : $CASES"
echo "  prior linear   : $PRIOR_LINEAR"
echo "  prior fibrosis : $PRIOR_FIBROSIS"
echo "  extra ex06 args: $*"

for method in $PHASES; do
    echo
    echo "######################################################################"
    echo "# PHASE: $method"
    echo "######################################################################"
    for lvl in "${LEVELS[@]}"; do
        pct=$("$PYTHON" -c "print(f'{round($lvl*100):03d}')")
        dir="$OUT/$method/noise_${pct}"
        mkdir -p "$dir"
        for case in $CASES; do
            prior=$(prior_for "$case")
            echo
            echo "======================================================================"
            echo " $method | noise ${lvl} | $case | prior: $prior"
            echo "======================================================================"
            start=$(date +%s)
            # shellcheck disable=SC2086   (prior is intentionally word-split)
            "$PYTHON" -u ex06_map_comparison.py \
                --methods "$method" \
                --cases "$case" \
                $prior \
                --data-noise "$lvl" \
                --data-noise-seed 1234 \
                --match-noise-std \
                --output-dir "$dir" \
                "$@" 2>&1 | tee "$dir/run_${case}.log"
            echo " done in $(( $(date +%s) - start )) s"
        done
    done
    # tables after each phase (after phase 1 they show Newton-CG only)
    echo
    echo "Building LaTeX tables from what is available ..."
    "$PYTHON" make_latex_tables.py --root "$OUT" --out "$OUT/tables.tex" \
        --standalone "$OUT/tables_standalone.tex"
    "$PYTHON" make_latex_tables.py --root "$OUT" --paper --out "$OUT/table_paper.tex" \
        --standalone "$OUT/table_paper_standalone.tex"
done

echo
echo "All phases done. Paper table: $OUT/table_paper.tex   (all tables: $OUT/tables.tex)"
