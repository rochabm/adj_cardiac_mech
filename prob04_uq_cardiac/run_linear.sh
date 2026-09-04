#!/bin/bash
# =============================================================================
# sweep_linear.sh
# Sweeps regularization parameter alpha AND number of nodes for LINEAR case.
#
# Usage:
#   chmod +x sweep_linear.sh
#   ./sweep_linear.sh
#
# Results saved to: results_linear/case_alpha_<alpha>_nodes_<nodes>/
# =============================================================================

CASE_TYPE="linear"
GTOL=1e-8
FTOL=1e-30

ALPHAS=(1e-3 1e-2 1e-1 1e+0 1e+1 1e+2)
NODES_LIST=(32 64 128)

for NODES in "${NODES_LIST[@]}"; do
for ALPHA in "${ALPHAS[@]}"; do

    OUTDIR="results_linear/case_alpha_${ALPHA}_nodes_${NODES}"
    mkdir -p "$OUTDIR"

    LOGFILE="${OUTDIR}/log_${CASE_TYPE}_alpha${ALPHA}_nodes${NODES}.txt"

    echo "============================================================"
    echo " Running: case_type=${CASE_TYPE}  alpha=${ALPHA}  nodes=${NODES}"
    echo " Output : ${OUTDIR}"
    echo "============================================================"

    python -u ex04_ventricle_inverse_discrete_var_novo.py \
        --case-type  "$CASE_TYPE"  \
        --num-nodes  "$NODES"      \
        --alpha      "$ALPHA"      \
        --gtol       "$GTOL"       \
        --ftol       "$FTOL"       \
        --output-dir "$OUTDIR"     \
        2>&1 | tee "$LOGFILE"

    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: run failed for alpha=${ALPHA} nodes=${NODES} (exit code ${EXIT_CODE})"
        echo "See ${LOGFILE} for details"
    else
        echo "Done: alpha=${ALPHA} nodes=${NODES} -> ${OUTDIR}"
    fi

    echo ""

done
done

echo "============================================================"
echo " All runs complete  (${#ALPHAS[@]} alphas x ${#NODES_LIST[@]} node counts)"
echo " Results in: results_linear/"
ls results_linear/
echo "============================================================"
