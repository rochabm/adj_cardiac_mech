"""
plot_ex05_uq_cardiac.py
------------------------
PyVista visualization of all UQ results from ex04_uq_cardiac.py (ex05).

Files read (all from --output-dir):
    out_uq_cardiac.h5         -- main fields (c_true, c_map, variance, etc.)
    out_uq_samples.h5         -- posterior samples
    out_uq_eigenvectors.h5    -- first 6 generalized eigenvectors
    out_uq_eigvals.npy        -- all k eigenvalues
    out_uq_newton_J.npy       -- Newton-CG J history
    out_uq_newton_gnorm.npy   -- Newton-CG gradient norm history

Figures produced:
    fig4_pyvista_MAP.png          -- true CC, MAP CC, error, displacement
    fig5_pyvista_variance.png     -- prior var, posterior var, stddev, reduction
    fig6_pyvista_eigenvectors.png -- first 6 generalized eigenvectors
    fig7_pyvista_samples.png      -- 5 posterior samples
    fig8_convergence.png          -- Newton-CG J and grad norm history
    fig9_eigenvalue_decay.png     -- eigenvalue spectrum with lambda=1 line

Usage:
    python plot_ex05_uq_cardiac.py --output-dir results_uq/fibrosis_alpha1e-3
"""

import argparse
import numpy as np
import h5py
from pathlib import Path
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# =============================================================================
# arguments
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=str, default=".",
                    help="Directory containing ex05 output files")
parser.add_argument("--off-screen", action="store_true",
                    help="Render off-screen (no display required)")
args = parser.parse_args()

OUT = Path(args.output_dir)
pv.global_theme.background = "white"
pv.global_theme.font.color = "black"
OFF = args.off_screen

# =============================================================================
# helper: read HDF5 mesh + fields
# =============================================================================

def read_h5(h5_file):
    """Read dolfinx XDMF/HDF5 output into a PyVista UnstructuredGrid."""
    fields = {}
    with h5py.File(h5_file, "r") as f:
        mesh_key = list(f["Mesh"].keys())[0]
        points   = f[f"Mesh/{mesh_key}/geometry"][:]
        cells    = f[f"Mesh/{mesh_key}/topology"][:]
        for name in f["Function"].keys():
            steps = list(f[f"Function/{name}"].keys())
            data  = f[f"Function/{name}/{steps[-1]}"][:]
            fields[name] = data

    if points.shape[1] == 2:
        points = np.hstack([points, np.zeros((len(points), 1))])

    npc      = cells.shape[1]
    vtk_type = {3: 5, 4: 10}.get(npc, 10)
    cell_arr = np.hstack([
        np.full((len(cells), 1), npc, dtype=np.int64), cells
    ]).ravel()
    celltypes = np.full(len(cells), vtk_type, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cell_arr, celltypes, points)

    n_pts   = len(points)
    n_cells = len(cells)

    for name, data in fields.items():
        n = data.shape[0]
        if n == n_pts:
            # scalar or vector point data (CG1)
            grid.point_data[name] = data
            if data.ndim == 2 and data.shape[1] == 3:
                grid.point_data[f"{name}_mag"] = np.linalg.norm(data, axis=1)
        elif n == n_cells:
            # scalar cell data (DG0)
            grid.cell_data[name] = data
        else:
            # tensor or multi-component DG0: reshape to (n_cells, ncomp)
            # e.g. Fd_reference: 9*n_cells values -> (n_cells, 9)
            if n % n_cells == 0:
                ncomp = n // n_cells
                arr   = data.reshape(n_cells, ncomp)
                grid.cell_data[name] = arr
                # also store magnitude for tensors
                grid.cell_data[f"{name}_mag"] = np.linalg.norm(
                    arr.reshape(n_cells, -1), axis=1)
            elif n % n_pts == 0:
                ncomp = n // n_pts
                arr   = data.reshape(n_pts, ncomp)
                grid.point_data[name] = arr
                grid.point_data[f"{name}_mag"] = np.linalg.norm(
                    arr.reshape(n_pts, -1), axis=1)
            else:
                print(f"  [warn] skipping field '{name}': "
                      f"size {n} not divisible by n_pts={n_pts} or n_cells={n_cells}")

    return grid, fields


def add_sensor_overlay(pl, grid, indicator_key="measurement_indicator",
                       color="cyan", size=15):
    """Add sensor locations as spheres visible inside the semi-transparent domain."""
    if indicator_key in grid.point_data:
        ind  = grid.point_data[indicator_key]
        mask = ind > 0.5
        pts  = pv.PolyData(grid.points[mask])
    elif indicator_key in grid.cell_data:
        ind  = grid.cell_data[indicator_key]
        mask = ind > 0.5
        pts  = pv.PolyData(grid.cell_centers().points[mask])
    else:
        return
    if mask.sum() == 0:
        return
    pl.add_mesh(pts, color=color, point_size=size,
                render_points_as_spheres=True,
                label="sensors", lighting=True)


def scalar_plot(pl, grid, field, cmap="viridis", clim=None, title="",
                show_sensors=True, sensor_grid=None, opacity=1.0):
    """Add a scalar field to a plotter subplot with opacity so sensors are visible."""
    if field in grid.point_data:
        d = grid.point_data[field]
    elif field in grid.cell_data:
        d = grid.cell_data[field]
        if d.ndim > 1:
            d = np.linalg.norm(d.reshape(len(d), -1), axis=1)
    else:
        pl.add_text(f"{field}\nnot found", font_size=9)
        pl.view_isometric()
        return

    if clim is None:
        clim = (float(np.nanmin(d)), float(np.nanmax(d)))
    pl.add_mesh(grid, scalars=field, cmap=cmap,
                clim=clim, show_edges=False, opacity=opacity)
    pl.add_scalar_bar(title, n_labels=5, fmt="%.2e",
                      label_font_size=10, title_font_size=11)
    if show_sensors and sensor_grid is not None:
        add_sensor_overlay(pl, sensor_grid)
    pl.add_title(title, font_size=10)
    pl.view_isometric()

# =============================================================================
# load data
# =============================================================================

print("Loading data...")

h5_main = str(OUT / "out_uq_cardiac.h5")
h5_samp = str(OUT / "out_uq_samples.h5")
h5_eig  = str(OUT / "out_uq_eigenvectors.h5")

grid_main, _ = read_h5(h5_main)
print(f"  Main fields : {list(grid_main.point_data.keys())}")

grid_samp = None
if Path(h5_samp).exists():
    grid_samp, _ = read_h5(h5_samp)
    print(f"  Sample fields: {list(grid_samp.point_data.keys())}")
else:
    print(f"  [warn] {h5_samp} not found -- re-run ex04_uq_cardiac.py to generate")

grid_eig = None
if Path(h5_eig).exists():
    grid_eig, _ = read_h5(h5_eig)
    print(f"  Eigvec fields: {list(grid_eig.point_data.keys())}")
else:
    print(f"  [warn] {h5_eig} not found -- re-run ex04_uq_cardiac.py to generate")

eigvals_file = OUT / "out_uq_eigvals.npy"
eigvals = np.load(eigvals_file) if eigvals_file.exists() else None

newton_J_file = OUT / "out_uq_newton_J.npy"
newton_J = np.load(newton_J_file) if newton_J_file.exists() else None

newton_gnorm_file = OUT / "out_uq_newton_gnorm.npy"
newton_gnorm = np.load(newton_gnorm_file) if newton_gnorm_file.exists() else None

if eigvals is not None:
    print(f"  Eigenvalues: {len(eigvals)} values, top 5: {eigvals[:5]}")

# =============================================================================
# fig4: MAP results -- interactive viewer
# =============================================================================

print("\nPlotting fig4: MAP results...")

if OFF:
    pl4 = pv.Plotter(shape=(1, 4), off_screen=True,
                     window_size=(1800, 500))
    for i, (field, cmap, title) in enumerate([
            ("c_true",           "viridis", "True CC"),
            ("c_map",            "viridis", "MAP estimate CC*"),
            ("c_error_map",      "inferno", "Relative error"),
            ("displacement_map_mag", "plasma", "Displacement |u|"),
    ]):
        pl4.subplot(0, i)
        scalar_plot(pl4, grid_main, field, cmap=cmap, title=title,
                    show_sensors=(i > 0), sensor_grid=grid_main)
    pl4.link_views()
    pl4.screenshot(str(OUT / "fig4_pyvista_MAP.png"))
    pl4.close()
    print("  Saved: fig4_pyvista_MAP.png")
else:
    # interactive: one window per field, stays open until closed
    map_fields = [
        ("c_true",           "viridis", "True CC  (close to continue)"),
        ("c_map",            "viridis", "MAP estimate CC*  (close to continue)"),
        ("c_error_map",      "inferno", "Pointwise relative error  (close to continue)"),
    ]
    if "displacement_map_mag" in grid_main.point_data:
        map_fields.append(("displacement_map_mag", "plasma",
                           "Displacement magnitude  (close to continue)"))

    print("  Opening interactive MAP windows (close each to proceed)...")
    for field, cmap, title in map_fields:
        if field not in grid_main.point_data and field not in grid_main.cell_data:
            print(f"    [skip] {field} not found")
            continue
        pl = pv.Plotter(window_size=(900, 700))
        pl.add_mesh(grid_main, scalars=field, cmap=cmap,
                    show_edges=False, opacity=1.0)
        pl.add_scalar_bar(field, n_labels=5, fmt="%.2e")
        add_sensor_overlay(pl, grid_main)
        pl.add_title(title, font_size=11)
        pl.show()   # blocks until window is closed

    # also save screenshot of the 4-panel version
    pl4 = pv.Plotter(shape=(1, len(map_fields)), off_screen=True,
                     window_size=(400*len(map_fields), 500))
    for i, (field, cmap, title) in enumerate(map_fields):
        pl4.subplot(0, i)
        scalar_plot(pl4, grid_main, field, cmap=cmap, title=title,
                    show_sensors=(i > 0), sensor_grid=grid_main)
    pl4.link_views()
    pl4.screenshot(str(OUT / "fig4_pyvista_MAP.png"))
    pl4.close()
    print("  Saved: fig4_pyvista_MAP.png")

# =============================================================================
# fig5: variance fields -- interactive viewer
# =============================================================================

print("Plotting fig5: variance fields...")

# compute variance reduction
pv_prior = grid_main.point_data["prior_variance"]
pv_post  = grid_main.point_data["posterior_variance"]
var_red  = np.clip((pv_prior - pv_post) / (pv_prior + 1e-30), 0, 1)
grid_main.point_data["variance_reduction"] = var_red

var_fields = [
    ("prior_variance",             "inferno", "Prior variance diag(H_prior⁻¹)"),
    ("posterior_variance",         "inferno", "Posterior variance diag(Σ_post)"),
    ("posterior_stddev_calibrated","plasma",  "Posterior std dev (calibrated)"),
    ("variance_reduction",         "viridis", "Variance reduction fraction [0,1]"),
]

if OFF:
    pl5 = pv.Plotter(shape=(1, 4), off_screen=True,
                     window_size=(1800, 500))
    for i, (field, cmap, title) in enumerate(var_fields):
        pl5.subplot(0, i)
        clim = (0, 1) if field == "variance_reduction" else None
        scalar_plot(pl5, grid_main, field, cmap=cmap, clim=clim,
                    title=title, sensor_grid=grid_main)
    pl5.link_views()
    pl5.screenshot(str(OUT / "fig5_pyvista_variance.png"))
    pl5.close()
    print("  Saved: fig5_pyvista_variance.png")
else:
    print("  Opening interactive variance windows (close each to proceed)...")
    for field, cmap, title in var_fields:
        pl = pv.Plotter(window_size=(900, 700))
        clim = (0, 1) if field == "variance_reduction" else None
        if field not in grid_main.point_data and field not in grid_main.cell_data:
            continue
        pl = pv.Plotter(window_size=(900, 700))
        kw = {"clim": clim} if clim else {}
        pl.add_mesh(grid_main, scalars=field, cmap=cmap,
                    show_edges=False, opacity=1.0, **kw)
        pl.add_scalar_bar(field, n_labels=5, fmt="%.2e")
        add_sensor_overlay(pl, grid_main)
        pl.add_title(f"{title}  (close to continue)", font_size=11)
        pl.show()

    # save screenshot
    pl5 = pv.Plotter(shape=(1, 4), off_screen=True,
                     window_size=(1800, 500))
    for i, (field, cmap, title) in enumerate(var_fields):
        pl5.subplot(0, i)
        clim = (0, 1) if field == "variance_reduction" else None
        scalar_plot(pl5, grid_main, field, cmap=cmap, clim=clim,
                    title=title, sensor_grid=grid_main)
    pl5.link_views()
    pl5.screenshot(str(OUT / "fig5_pyvista_variance.png"))
    pl5.close()
    print("  Saved: fig5_pyvista_variance.png")

# =============================================================================
# fig6: first 6 generalized eigenvectors
# =============================================================================

print("Plotting fig6: eigenvectors...")

if grid_eig is not None:
    eig_keys = sorted([k for k in grid_eig.point_data.keys()
                       if k.startswith("eigenvector_")])[:6]
ncols = len(eig_keys)
if ncols > 0:
    pl6 = pv.Plotter(shape=(1, ncols), off_screen=OFF,
                      window_size=(300*ncols, 500))
    for i, key in enumerate(eig_keys):
        pl6.subplot(0, i)
        d    = grid_eig.point_data[key]
        amax = np.abs(d).max()
        scalar_plot(pl6, grid_eig, key, "coolwarm",
                    clim=(-amax, amax),
                    title=f"Eigvec {i}\n(λ={eigvals[i]:.2f})",
                    show_sensors=False)
    pl6.link_views()
    pl6.screenshot(str(OUT / "fig6_pyvista_eigenvectors.png"))
    if not OFF:
        pl6.show()
    print("  Saved: fig6_pyvista_eigenvectors.png")
else:
    print("  Skipped fig6 (out_uq_eigenvectors.h5 not found)")

# =============================================================================
# fig7: posterior samples
# =============================================================================

print("Plotting fig7: posterior samples...")

if grid_samp is not None:
    samp_keys = sorted([k for k in grid_samp.point_data.keys()
                        if k.startswith("posterior_sample_")])
if len(samp_keys) > 0:
    # shared color scale across all samples
    all_vals = np.concatenate([grid_samp.point_data[k] for k in samp_keys])
    s_clim   = (float(all_vals.min()), float(all_vals.max()))

    pl7 = pv.Plotter(shape=(1, len(samp_keys)), off_screen=OFF,
                      window_size=(300*len(samp_keys), 500))
    for i, key in enumerate(samp_keys):
        pl7.subplot(0, i)
        scalar_plot(pl7, grid_samp, key, "viridis",
                    clim=s_clim, title=f"Post. sample {i+1}",
                    show_sensors=False)
    pl7.link_views()
    pl7.screenshot(str(OUT / "fig7_pyvista_samples.png"))
    if not OFF:
        pl7.show()
    print("  Saved: fig7_pyvista_samples.png")
else:
    print("  Skipped fig7 (out_uq_samples.h5 not found)")

# =============================================================================
# fig8: Newton-CG convergence
# =============================================================================

print("Plotting fig8: Newton-CG convergence...")

if newton_J is not None:
    fig8, axes8 = plt.subplots(1, 2, figsize=(10, 4))

    axes8[0].semilogy(newton_J, "o-b", markersize=5)
    axes8[0].set_xlabel("Newton-CG iteration")
    axes8[0].set_ylabel("J")
    axes8[0].set_title("Cost functional J")
    axes8[0].grid(True, which="both", alpha=0.3)

    if newton_gnorm is not None:
        axes8[1].semilogy(newton_gnorm, "o-r", markersize=5)
        axes8[1].axhline(1e-8, color="k", linestyle="--", label="gtol=1e-8")
        axes8[1].set_xlabel("Newton-CG iteration")
        axes8[1].set_ylabel(r"$\|\nabla J\|_\infty$")
        axes8[1].set_title("Gradient norm")
        axes8[1].legend()
        axes8[1].grid(True, which="both", alpha=0.3)

    plt.suptitle("MAP convergence (inexact Newton-CG)", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(OUT / "fig8_convergence.png"), dpi=150)
    if not OFF:
        plt.show()
    print("  Saved: fig8_convergence.png")
else:
    print("  Skipped fig8 (out_uq_newton_J.npy not found)")

# =============================================================================
# fig9: eigenvalue decay
# =============================================================================

print("Plotting fig9: eigenvalue decay...")

fig9, ax9 = plt.subplots(figsize=(8, 5))
k = len(eigvals)
ax9.semilogy(range(k), eigvals, "b*", markersize=7)
ax9.axhline(1.0, color="r", linestyle="-", linewidth=1.5, label="λ=1")
ax9.fill_between(range(k), eigvals, 1.0,
                  where=eigvals > 1.0, alpha=0.15, color="green",
                  label="data-informed (λ>1)")
ax9.fill_between(range(k), eigvals, 1.0,
                  where=eigvals < 1.0, alpha=0.10, color="orange",
                  label="prior-dominated (λ<1)")
n_informed = int(np.sum(eigvals > 1))
ax9.set_xlabel("index", fontsize=12)
ax9.set_ylabel("eigenvalue λ", fontsize=12)
ax9.set_title(f"Hessian misfit spectrum — H_misfit v = λ H_prior v\n"
              f"{n_informed} data-informed modes (λ>1) out of {k}",
              fontsize=11)
ax9.legend(fontsize=10)
ax9.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig(str(OUT / "fig9_eigenvalue_decay.png"), dpi=150)
if not OFF:
    plt.show()
print("  Saved: fig9_eigenvalue_decay.png")

# =============================================================================
# summary
# =============================================================================

print("\n=== PyVista plots done ===")
print("Output figures:")
for f in ["fig4_pyvista_MAP.png",
          "fig5_pyvista_variance.png",
          "fig6_pyvista_eigenvectors.png",
          "fig7_pyvista_samples.png",
          "fig8_convergence.png",
          "fig9_eigenvalue_decay.png"]:
    fpath = OUT / f
    status = "✓" if fpath.exists() else "✗ NOT FOUND"
    print(f"  {status}  {fpath}")
