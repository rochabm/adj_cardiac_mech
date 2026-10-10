#!/usr/bin/env bash
# Prior calibration (BiLaplacian, Bui-Thanh et al. 2013) for the Results section.
# sigma and rho are SELECTED FROM THE GROUND TRUTH (--match-truth):
#   sigma = max|truth - m0| / 2          (truth inside the pointwise 95 % band)
#   rho   = correlation length of the truth field (same estimator as the prior)
# and (gamma, delta) are calibrated numerically so that the prior really has them.
# Needs paper_style.py in this folder.
set -e
LOG2=0.693147   # m0 = log(2): healthy C = 2 kPa

# --- Case 2 (fibrosis) selects the prior for the ellipsoid --------------------
python prior_calibration.py --mesh ellipsoid --order 2 --log --m0 $LOG2 \
    --truth fibrosis --match-truth --output-dir calib_case2
python plot_prior_calibration.py calib_case2 --plane-point -5 -2 -9 --plane-normal 0 1 0

# --- Case 1 (linear) uses the SAME prior (same targets -> same gamma, delta) --
SIG=$(awk '/^target_sigma/{print $2}' calib_case2/calib_summary.txt)
RHO=$(awk '/^target_rho/{print $2}'   calib_case2/calib_summary.txt)
python prior_calibration.py --mesh ellipsoid --order 2 --log --m0 $LOG2 \
    --sigma $SIG --rho $RHO --calibrate-rho --truth linear --output-dir calib_case1
python plot_prior_calibration.py calib_case1 --plane-normal 0 1 0

# --- Case 3 (patient LV), own truth-matched prior (~10 min: exact variance) ---
python prior_calibration.py --mesh Patient_7_lv_tissue.vtu --order 2 --log --m0 $LOG2 \
    --truth tissue --match-truth --n-ref 30 --output-dir calib_case3
python plot_prior_calibration.py calib_case3

# gamma_calibrated / delta_calibrated in calib_case*/calib_summary.txt -> ex05 --gamma/--delta
grep -H "gamma_calibrated\|delta_calibrated\|target_sigma\|target_rho" calib_case*/calib_summary.txt
