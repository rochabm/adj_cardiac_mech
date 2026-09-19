"""
analyze_fibrosis_region.py
---------------------------
Computes UQ and reconstruction metrics INSIDE vs OUTSIDE the fibrosis
region defined in ex03 (fibrosis case).

Fibrosis region (from ex03 c_expr):
    center: (xc, yc, zc) = (-5, -2, -9)
    r0 = 5   mm  -- core (CC = 4.0, fully fibrotic)
    r1 = 10  mm  -- outer boundary (CC = 2.0, healthy outside)

Metrics computed per region (core / transition / healthy):
    - CC_true         : mean, std, min, max
    - CC_map          : mean, std, min, max
    - rel_error       : mean, std, max  (|CC_map - CC_true| / |CC_true|)
    - prior_variance  : mean, std
    - posterior_variance: mean, std
    - variance_reduction: mean (= (prior-post)/prior)
    - n_sensors_in_region

Usage:
    python analyze_fibrosis_region.py --output-dir results_uq/fibrosis_alpha1e-3
"""

import argparse
import numpy as np
import h5py
from pathlib import Path

# =============================================================================
# arguments
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=str, default=".",
                    help="Directory containing ex05 output files")
parser.add_argument("--xc",  type=float, default=-5.0)
parser.add_argument("--yc",  type=float, default=-2.0)
parser.add_argument("--zc",  type=float, default=-9.0)
parser.add_argument("--r0",  type=float, default=5.0,
                    help="Core radius (fully fibrotic, default 5 mm)")
parser.add_argument("--r1",  type=float, default=10.0,
                    help="Outer transition radius (default 10 mm)")
args = parser.parse_args()

OUT = Path(args.output_dir)
xc, yc, zc = args.xc, args.yc, args.zc
r0, r1      = args.r0, args.r1

# =============================================================================
# load mesh geometry + scalar fields from HDF5
# =============================================================================

def load_scalar_fields(h5_file, field_names):
    """Load geometry node coordinates and named scalar fields."""
    with h5py.File(h5_file, "r") as f:
        mesh_key = list(f["Mesh"].keys())[0]
        points   = f[f"Mesh/{mesh_key}/geometry"][:]   # (n_nodes, 3)
        fields   = {}
        for name in field_names:
            if name in f["Function"]:
                steps = list(f[f"Function/{name}"].keys())
                data  = f[f"Function/{name}/{steps[-1]}"][:]
                fields[name] = data
            else:
                print(f"  [warn] field '{name}' not found in {h5_file}")
    return points, fields


h5_main = str(OUT / "out_uq_cardiac.h5")
h5_defo = str(OUT / "out_uq_deformation.h5")

print(f"Loading: {h5_main}")
points, fields = load_scalar_fields(h5_main, [
    "c_true", "c_map", "c_error_map",
    "prior_variance", "posterior_variance",
])

# load displacement magnitude (P1 scalar, one value per node)
if Path(h5_defo).exists():
    print(f"Loading: {h5_defo}")
    _, defo_fields = load_scalar_fields(h5_defo, [
        "displacement_magnitude",
        "F_frobenius_map",
        "J_det_map",
    ])
    # displacement_magnitude is P1 → n_pts values, add directly
    if "displacement_magnitude" in defo_fields:
        fields["displacement_magnitude"] = defo_fields["displacement_magnitude"]

    # F_frobenius and J are DG0 → n_cells values, need cell centres
    # for regional analysis we use cell-centre coordinates
    has_cell_fields = ("F_frobenius_map" in defo_fields or
                       "J_det_map"       in defo_fields)
else:
    print(f"  [warn] {h5_defo} not found -- re-run ex04_uq_cardiac.py")
    defo_fields    = {}
    has_cell_fields = False

n_pts = len(points)
print(f"Mesh nodes: {n_pts}")

# =============================================================================
# classify each node by distance from fibrosis center
# =============================================================================

dx = points[:, 0] - xc
dy = points[:, 1] - yc
dz = points[:, 2] - zc
r  = np.sqrt(dx**2 + dy**2 + dz**2)

mask_core    = r <= r0
mask_trans   = (r > r0) & (r < r1)
mask_healthy = r >= r1

regions = {
    "core       (r <= r0)   fibrotic ": mask_core,
    "transition (r0<r<r1)   blend    ": mask_trans,
    "healthy    (r >= r1)   normal   ": mask_healthy,
}

print(f"\nFibrosis center: ({xc}, {yc}, {zc}) mm")
print(f"r0 = {r0} mm  (core),  r1 = {r1} mm  (outer boundary)")
print(f"Nodes in core      : {mask_core.sum()}")
print(f"Nodes in transition: {mask_trans.sum()}")
print(f"Nodes in healthy   : {mask_healthy.sum()}")

# =============================================================================
# check if sensor indicator is available
# =============================================================================

_, ind_fields = load_scalar_fields(h5_main, ["measurement_indicator"])
has_indicator = "measurement_indicator" in ind_fields

# measurement_indicator may be on cells not points -- skip if shape mismatch
sensor_mask = None
if has_indicator:
    ind = ind_fields["measurement_indicator"]
    if ind.shape[0] == n_pts:
        sensor_mask = ind > 0.5
    else:
        print("  [info] measurement_indicator is cell-based -- "
              "sensor counts per region not computed")

# compute variance reduction
if "prior_variance" in fields and "posterior_variance" in fields:
    pv_prior = fields["prior_variance"]
    pv_post  = fields["posterior_variance"]
    var_red  = np.clip((pv_prior - pv_post) / (pv_prior + 1e-30), 0, 1)
    fields["variance_reduction"] = var_red

# =============================================================================
# compute and print metrics per region
# =============================================================================

def stats(arr, mask, label):
    """Print statistics of arr[mask]."""
    v = arr[mask]
    if len(v) == 0:
        print(f"    {label:30s}: NO DATA")
        return
    print(f"    {label:30s}: "
          f"mean={v.mean():.4f}  std={v.std():.4f}  "
          f"min={v.min():.4f}  max={v.max():.4f}  n={len(v)}")


separator = "="*70

print(f"\n{separator}")
print("  METRICS PER REGION")
print(separator)

for region_name, mask in regions.items():
    print(f"\n--- {region_name} ---")
    print(f"    Nodes in region: {mask.sum()}")

    if sensor_mask is not None:
        n_sensors = (mask & sensor_mask).sum()
        print(f"    Sensors in region: {n_sensors}")

    for fname, flabel in [
        ("c_true",                 "CC_true              "),
        ("c_map",                  "CC_map (MAP)         "),
        ("c_error_map",            "rel error |e|/|CT|   "),
        ("displacement_magnitude", "displacement |u| (mm)"),
        ("prior_variance",         "prior variance       "),
        ("posterior_variance",     "posterior variance   "),
        ("variance_reduction",     "variance reduction   "),
    ]:
        if fname in fields:
            stats(fields[fname], mask, flabel)

# --- DG0 fields: F_frobenius and J need cell-centre coordinates ---
if has_cell_fields:
    print(f"\n{'='*70}")
    print("  DEFORMATION GRADIENT METRICS (cell-averaged, DG0)")
    print(f"{'='*70}")
    print("  (cell centres classified by distance from fibrosis centre)\n")

    # load topology to compute cell centres
    with h5py.File(h5_defo, "r") as f:
        mesh_key  = list(f["Mesh"].keys())[0]
        pts_all   = f[f"Mesh/{mesh_key}/geometry"][:]
        cells_all = f[f"Mesh/{mesh_key}/topology"][:]

    cell_centres = pts_all[cells_all].mean(axis=1)   # (n_cells, 3)

    dx_c = cell_centres[:, 0] - xc
    dy_c = cell_centres[:, 1] - yc
    dz_c = cell_centres[:, 2] - zc
    r_c  = np.sqrt(dx_c**2 + dy_c**2 + dz_c**2)

    cmask_core    = r_c <= r0
    cmask_trans   = (r_c > r0) & (r_c < r1)
    cmask_healthy = r_c >= r1

    cell_regions = {
        "core       (r <= r0)": cmask_core,
        "transition (r0<r<r1)": cmask_trans,
        "healthy    (r >= r1)": cmask_healthy,
    }

    for fname, flabel in [
        ("F_frobenius_map", "||F||_F (Frobenius norm)"),
        ("J_det_map",       "J = det(F)              "),
    ]:
        if fname not in defo_fields:
            continue
        arr = defo_fields[fname]
        print(f"  {flabel}:")
        for rname, cmask in cell_regions.items():
            v = arr[cmask]
            if len(v) == 0:
                continue
            print(f"    {rname}: mean={v.mean():.5f}  "
                  f"std={v.std():.5f}  "
                  f"min={v.min():.5f}  max={v.max():.5f}")

# =============================================================================
# summary table: key metrics side by side
# =============================================================================

print(f"\n{separator}")
print("  SUMMARY TABLE")
print(separator)

header = f"{'Metric':<28} {'Core':>12} {'Transition':>12} {'Healthy':>12}"
print(header)
print("-"*70)

summary_fields = [
    ("c_true",                 "CC_true mean",           "mean"),
    ("c_map",                  "CC_map mean",            "mean"),
    ("c_error_map",            "Rel. error mean",        "mean"),
    ("c_error_map",            "Rel. error max",         "max"),
    ("displacement_magnitude", "Displacement mean (mm)", "mean"),
    ("displacement_magnitude", "Displacement max  (mm)", "max"),
    ("prior_variance",         "Prior var mean",         "mean"),
    ("posterior_variance",     "Post. var mean",         "mean"),
    ("variance_reduction",     "Var. reduction mean",    "mean"),
]

for fname, label, stat in summary_fields:
    if fname not in fields:
        continue
    arr = fields[fname]
    vals = []
    for mask in [mask_core, mask_trans, mask_healthy]:
        v = arr[mask]
        if len(v) == 0:
            vals.append("         N/A")
        elif stat == "max":
            vals.append(f"{v.max():>12.4f}")
        else:
            vals.append(f"{v.mean():>12.4f}")
    print(f"{label:<28} {vals[0]} {vals[1]} {vals[2]}")

# =============================================================================
# key question: is uncertainty larger or smaller in the fibrosis region?
# =============================================================================

print(f"\n{separator}")
print("  KEY QUESTION: Is uncertainty larger/smaller in the fibrosis region?")
print(separator)

if "posterior_variance" in fields and "variance_reduction" in fields:
    pv_core    = fields["posterior_variance"][mask_core].mean()
    pv_healthy = fields["posterior_variance"][mask_healthy].mean()
    vr_core    = fields["variance_reduction"][mask_core].mean()
    vr_healthy = fields["variance_reduction"][mask_healthy].mean()
    er_core    = fields["c_error_map"][mask_core].mean() \
                 if "c_error_map" in fields else float("nan")
    er_healthy = fields["c_error_map"][mask_healthy].mean() \
                 if "c_error_map" in fields else float("nan")

    print(f"\n  Posterior variance:")
    print(f"    Core    : {pv_core:.4e}")
    print(f"    Healthy : {pv_healthy:.4e}")
    if pv_core > pv_healthy:
        print(f"    → Uncertainty is LARGER in the fibrosis core "
              f"({pv_core/pv_healthy:.2f}× higher)")
    else:
        print(f"    → Uncertainty is SMALLER in the fibrosis core "
              f"({pv_healthy/pv_core:.2f}× lower than healthy)")

    print(f"\n  Variance reduction (data informativeness):")
    print(f"    Core    : {vr_core:.4f}  ({100*vr_core:.1f}% of prior reduced)")
    print(f"    Healthy : {vr_healthy:.4f}  ({100*vr_healthy:.1f}% of prior reduced)")

    print(f"\n  MAP reconstruction error:")
    print(f"    Core    : {er_core:.4f}  ({100*er_core:.1f}% relative error)")
    print(f"    Healthy : {er_healthy:.4f}  ({100*er_healthy:.1f}% relative error)")

# --- hypothesis test: does stiffer mean less displacement/deformation? ---
print(f"\n{separator}")
print("  HYPOTHESIS TEST: stiffer → less deformation → less information?")
print(separator)

if "displacement_magnitude" in fields:
    u_core    = fields["displacement_magnitude"][mask_core].mean()
    u_healthy = fields["displacement_magnitude"][mask_healthy].mean()
    print(f"\n  Mean displacement magnitude:")
    print(f"    Core    : {u_core:.6f} mm")
    print(f"    Healthy : {u_healthy:.6f} mm")
    ratio_u = u_healthy / u_core if u_core > 0 else float("nan")
    if u_core < u_healthy:
        print(f"    → Core displaces LESS ({ratio_u:.2f}× less than healthy)")
        print(f"      ✓ Confirms: stiffer region deforms less")
    else:
        print(f"    → Core displaces MORE ({u_core/u_healthy:.2f}× more than healthy)")
        print(f"      ✗ Does not confirm hypothesis")

if has_cell_fields and "F_frobenius_map" in defo_fields:
    F_core    = defo_fields["F_frobenius_map"][cmask_core].mean()
    F_healthy = defo_fields["F_frobenius_map"][cmask_healthy].mean()
    J_core    = defo_fields["J_det_map"][cmask_core].mean()   \
                if "J_det_map" in defo_fields else float("nan")
    J_healthy = defo_fields["J_det_map"][cmask_healthy].mean() \
                if "J_det_map" in defo_fields else float("nan")
    print(f"\n  Mean ||F||_F (deformation gradient Frobenius norm):")
    print(f"    Core    : {F_core:.6f}")
    print(f"    Healthy : {F_healthy:.6f}")
    print(f"    Reference (no deformation): ||I||_F = {np.sqrt(3):.6f}")
    ratio_F = (F_healthy - np.sqrt(3)) / (F_core - np.sqrt(3) + 1e-12)
    if F_core < F_healthy:
        print(f"    → Core deforms LESS (deviation from I is "
              f"{ratio_F:.2f}× smaller than healthy)")
        print(f"      ✓ Confirms: stiffer region has smaller F deviation")
    else:
        print(f"    → Core deforms MORE or equally")
        print(f"      ✗ Does not confirm hypothesis")

    print(f"\n  Mean J = det(F) (volume change):")
    print(f"    Core    : {J_core:.6f}  (1.0 = no volume change)")
    print(f"    Healthy : {J_healthy:.6f}")
    print(f"    → Core volume change: {abs(J_core-1)*100:.4f}%")
    print(f"    → Healthy volume change: {abs(J_healthy-1)*100:.4f}%")

print(f"\n{separator}")
print("Done.")
