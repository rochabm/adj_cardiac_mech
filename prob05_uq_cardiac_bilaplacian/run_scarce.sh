#!/usr/bin/env bash
# Scarce-data scenario (Case 2, fibrosis): 64 vs 16 sensors, 0% noise, same prior.
# Run step by step:   bash run_scarce.sh 1     (then 2, 3, ...)
#                     bash run_scarce.sh all
# Assumes the 64-sensor results are already in out_ex07/fibrosis: ex07 (--k-eig 500),
# uq_regions.py and uq_qoi.py (their json files are used by the table in step 4).
set -euo pipefail
export PYTHONUNBUFFERED=1

FULL=out_ex07          # 64 sensors (existing)
SCARCE=out_ex07_s64    # 16 sensors (new)
NS=64
STEP=${1:-}

step1() {  # UQ with 16 sensors: new MAP + eigenproblem (~ minutes)
  python -u ex07_uq.py --case fibrosis --data-noise 0 --num-nodes $NS --k-eig 250 \
         --output-dir $SCARCE 2>&1 | tee $SCARCE.log
}
step2() {  # regional statistics, 16 sensors (64-sensor results already in $FULL)
  python uq_regions.py --root $SCARCE --case fibrosis --num-nodes $NS --vr-clim 0 1
}
step3() {  # regional stiffness intervals, 16 sensors (needs dolfinx)
  python uq_qoi.py --root $SCARCE --case fibrosis --n-samples 1000
}
step4() {  # comparison table + draft paragraph + overlaid spectra
  python compare_sensors.py --runs $FULL:64 $SCARCE:$NS
}
step5() {  # the combined figure
  python plot_scarce.py --root64 $FULL --root16 $SCARCE --n-scarce $NS
}

case "$STEP" in
  1|2|3|4|5) step$STEP ;;
  all) step1; step2; step3; step4; step5 ;;
  *) echo "usage: bash run_scarce.sh {1|2|3|4|5|all}"
     echo "  1 ex07 UQ with $NS sensors      2 uq_regions ($NS)"
     echo "  3 uq_qoi ($NS)                   4 compare_sensors (table, text)"
     echo "  5 plot_scarce (figure)"; exit 1 ;;
esac
