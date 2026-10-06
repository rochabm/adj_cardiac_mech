"""
Gaussian (Matern/Whittle) priors on a cardiac LV ellipsoid mesh — FEniCSx.

Reuses the PDE-based prior implemented in gaussian_priors_fenicsx.py
(MaternPrior) and applies it to the left-ventricle ellipsoid mesh built the
same way as ex03_ventricle_discrete_forward.py (via cardiac_geometries).

Renders samples, pointwise variance and a correlation structure with PyVista,
laid out in a single multi-panel window / screenshot per case:

    lv_01_laplacian.png     alpha = 1  (Laplacian-like)
    lv_02_bilaplacian.png   alpha = 2  (BiLaplacian-like, trace class)
    lv_03_anisotropic.png   alpha = 2 with fibre-aligned anisotropy

Notes
-----
* On a 3D unstructured mesh the *exact* column-by-column pointwise variance is
  prohibitively expensive (one linear solve per DOF). We use the randomized
  (sample-average) estimator instead.
* correlation_structure in the imported module hardcodes 2D coordinates, so we
  reimplement it here for 3D using the prior's public matrices/solvers.
* Anisotropy uses a fibre-aligned 3x3 tensor Theta = theta_par * f0 f0^T +
  theta_perp * (I - f0 f0^T), built from the geometry's fibre field.

Requires: dolfinx, petsc4py, ufl, numpy, pyvista, cardiac_geometries.
Run:  python3 lv_priors.py
"""

from pathlib import Path
import numpy as np

from mpi4py import MPI
import ufl
from dolfinx import fem
from dolfinx.plot import vtk_mesh

import pyvista as pv

# --- reuse the prior implementation from the 2D tutorial script ------------
from gaussian_priors_fenicsx import MaternPrior, V


# ---------------------------------------------------------------------------
# geometry (same recipe as ex03)
# ---------------------------------------------------------------------------
def build_lv_geometry(geodir="lv_ellipsoid", psize_ref=3):
    import cardiac_geometries
    geo = cardiac_geometries.mesh.lv_ellipsoid(
        outdir=Path(geodir),
        create_fibers=True,
        fiber_space="P_1",
        psize_ref=psize_ref,
        r_short_epi=10,
        aha=True,
        fiber_angle_endo=40.0,
        fiber_angle_epi=-50.0,
    )
    return geo


# ---------------------------------------------------------------------------
# fibre-aligned anisotropic diffusion tensor (3x3 UFL matrix)
# ---------------------------------------------------------------------------
def fiber_anis_tensor(f0, theta_par=1.0, theta_perp=6.0):
    """Theta = theta_par * f0 f0^T + theta_perp * (I - f0 f0^T).

    Larger diffusion (theta) along a direction -> longer correlation length in
    that direction. theta_perp > theta_par gives long correlation across the
    fibre and short along it (or swap to taste).
    """
    I = ufl.Identity(3)
    ff = ufl.outer(f0, f0)
    return theta_par * ff + theta_perp * (I - ff)


# ---------------------------------------------------------------------------
# 3D correlation structure (local reimplementation; module version is 2D-only)
# ---------------------------------------------------------------------------
def correlation_structure_3d(prior, point):
    coords = prior.Vh.tabulate_dof_coordinates()[:, :3]
    idx = int(np.argmin(np.sum((coords - np.asarray(point)) ** 2, axis=1)))

    rhs = prior.A.createVecRight()
    rhs.set(0.0)
    rhs.setValue(idx, 1.0)
    rhs.assemble()

    out = prior.A.createVecRight()
    tmp = prior.A.createVecRight()
    if prior.alpha == 2:
        prior.Asolver.solve(rhs, tmp)
        prior.M.mult(tmp, out)
        prior.Asolver.solve(out, rhs)
        return rhs.getArray().copy()
    else:
        prior.Asolver.solve(rhs, out)
        return out.getArray().copy()


# ---------------------------------------------------------------------------
# PyVista helpers
# ---------------------------------------------------------------------------
def make_grid(Vh):
    """dolfinx CG1 space -> PyVista UnstructuredGrid (cached topology)."""
    cells, cell_types, points = vtk_mesh(Vh)
    return pv.UnstructuredGrid(cells, cell_types, points)


def _orient_apex_up(plotter):
    """LV long axis is z in cardiac_geometries. Camera looks along -y with the
    up-direction set to -z, so the apex points down and the basal opening sits
    at the top of the frame."""
    plotter.view_vector((0.0, -1.0, 0.0), viewup=(1.0, 0.0, 0.0))
    plotter.camera.zoom(1.3)


def add_field(plotter, grid, values, title, cmap, clim=None, symmetric=False,
              show_bar=False, bar_title="", bar_args=None):
    """Attach a nodal field to a copy of the grid and add it to a subplot.

    A scalar bar is drawn only when show_bar=True, so a group of subplots can
    share a single bar (add it on the last member of the group).
    """
    g = grid.copy()
    vals = np.asarray(values)
    g.point_data["f"] = vals
    if symmetric and clim is None:
        a = np.abs(vals).max()
        clim = (-a, a)

    kwargs = dict(scalars="f", cmap=cmap, clim=clim, show_scalar_bar=show_bar)
    if show_bar:
        default_bar = {"title": bar_title, "n_labels": 3, "fmt": "%.2g",
                       "vertical": False, "position_x": 0.15,
                       "position_y": 0.05, "width": 0.7, "height": 0.08}
        if bar_args:
            default_bar.update(bar_args)
        kwargs["scalar_bar_args"] = default_bar

    plotter.add_mesh(g, **kwargs)
    plotter.add_text(title, font_size=9)
    _orient_apex_up(plotter)


def compute_case(Vh, gamma, delta, alpha, n_samples=3, r_var=100,
                 Theta=None, robin_bc=True, seed=1, corr_point=None):
    """Build one prior and evaluate all its fields (no plotting).

    Returns a dict with the prior, samples, variance and correlation, plus
    this case's own natural limits (used when computing shared limits).
    """
    prior = MaternPrior(Vh, gamma, delta, alpha=alpha,
                        Theta=Theta, robin_bc=robin_bc, seed=seed)
    samples = prior.samples(n_samples)
    var = prior.pointwise_variance_randomized(r=r_var)
    cs = (correlation_structure_3d(prior, corr_point)
          if corr_point is not None else None)

    return {
        "prior": prior, "alpha": alpha, "samples": samples,
        "var": var, "cs": cs,
        "sample_absmax": max(np.abs(s).max() for s in samples),
        "var_max": float(var.max()),
        "cs_absmax": (float(np.abs(cs).max()) if cs is not None else None),
    }


def render_case(case, grid, fname, sample_clim, var_clim, corr_clim,
                sample_cmap="RdBu_r", var_cmap="turbo",
                window_size=(1600, 560), off_screen=True):
    """Render a precomputed case with the given (shared) colour limits."""
    n_samples = len(case["samples"])
    has_cs = case["cs"] is not None
    ncols = n_samples + (2 if has_cs else 1)

    plotter = pv.Plotter(shape=(1, ncols), window_size=window_size,
                         off_screen=off_screen, border=True)
    kind = {1: "Laplacian (a=1)", 2: "BiLaplacian (a=2)"}[case["alpha"]]

    for i, s in enumerate(case["samples"]):
        plotter.subplot(0, i)
        add_field(plotter, grid, s, f"{kind}\nsample {i + 1}",
                  cmap=sample_cmap, clim=sample_clim,
                  show_bar=(i == n_samples - 1), bar_title="samples")

    plotter.subplot(0, n_samples)
    add_field(plotter, grid, case["var"], "pointwise variance",
              cmap=var_cmap, clim=var_clim, show_bar=True,
              bar_title="variance")

    if has_cs:
        plotter.subplot(0, n_samples + 1)
        add_field(plotter, grid, case["cs"], "correlation structure",
                  cmap=var_cmap, clim=corr_clim, show_bar=True,
                  bar_title="corr")

    plotter.link_views()
    plotter.screenshot(fname)
    print(f"  saved {fname}")
    if not off_screen:
        plotter.show()
    plotter.close()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(off_screen=True):
    # gamma/delta set the correlation length rho ~ sqrt(gamma/delta) and the
    # marginal variance. The LV ellipsoid is ~O(10) units across, so use a
    # larger rho than the unit-square tutorial's 0.2.
    gamma, delta = 1.0, 0.5          # rho ~ sqrt(2) ~ 1.4 units
    gamma, delta = 128.0, 1.0          # rho ~ sqrt(2) ~ 1.4 units

    print("Building LV ellipsoid geometry...")
    geo = build_lv_geometry()
    domain = geo.mesh
    Vh = V(domain)                   # CG1 scalar space for the prior
    grid = make_grid(Vh)
    centroid = domain.geometry.x.mean(axis=0)

    Theta = fiber_anis_tensor(geo.f0, theta_par=1.0, theta_perp=6.0)

    # --- 1) compute every case first (so limits can be shared) -------------
    print("Computing cases...")
    print("  lv_01 Laplacian (alpha=1)")
    c_lap = compute_case(Vh, gamma, delta, alpha=1, corr_point=centroid,
                         robin_bc=False, seed=1)
    print("  lv_02 BiLaplacian (alpha=2)")
    c_bi = compute_case(Vh, gamma, delta, alpha=2, corr_point=centroid,
                        robin_bc=True, seed=1)
    print("  lv_03 Anisotropic BiLaplacian")
    c_an = compute_case(Vh, gamma, delta, alpha=2, Theta=Theta,
                        corr_point=centroid, robin_bc=True, seed=3)
    cases = [c_lap, c_bi, c_an]

    # --- 2) shared colour limits across all three cases --------------------
    s_absmax = max(c["sample_absmax"] for c in cases)
    v_max = max(c["var_max"] for c in cases)
    corr_absmax = max(c["cs_absmax"] for c in cases)

    sample_clim = (-s_absmax, s_absmax)   # symmetric, shared by all samples
    var_clim = (0.0, v_max)               # shared variance scale
    corr_clim = (-corr_absmax, corr_absmax)
    print(f"Shared limits: samples +/-{s_absmax:.3g}, "
          f"variance [0,{v_max:.3g}], corr +/-{corr_absmax:.3g}")

    # --- 3) render each case with identical scales -------------------------
    print("Rendering...")
    render_case(c_lap, grid, "lv_01_laplacian.png",
                sample_clim, var_clim, corr_clim, off_screen=off_screen)
    render_case(c_bi, grid, "lv_02_bilaplacian.png",
                sample_clim, var_clim, corr_clim, off_screen=off_screen)
    render_case(c_an, grid, "lv_03_anisotropic.png",
                sample_clim, var_clim, corr_clim, off_screen=off_screen)

    print("\nDone. Wrote lv_01_laplacian.png, lv_02_bilaplacian.png, "
          "lv_03_anisotropic.png")


if __name__ == "__main__":
    # set off_screen=False to open interactive windows instead of screenshots
    main(off_screen=True)
