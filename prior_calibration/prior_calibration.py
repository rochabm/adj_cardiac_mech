"""
prior_calibration.py
--------------------
Calibration of the BiLaplacian-type Gaussian prior used in ex05
(Results section "Prior calibration"):

    C = (R^{-1} M)^{k-1} R^{-1},   R = gamma*K + delta*M + beta*M_boundary,
    beta = sqrt(gamma*delta)/1.42 (Robin, Daon & Stadler 2018),
    k = 2: BiLaplacian (Matern nu = 1/2 in 3D),  k = 4: nu = 5/2.

Given the TARGET marginal std sigma and correlation length rho, (gamma, delta)
follow from the Matern relations (Lindgren et al. 2011). Those relations hold
in an unbounded domain; in the thin LV wall (and on a coarse mesh) the
achieved std / correlation length can differ. This script MEASURES them:

    1. pointwise prior std  sqrt(diag C)   (exact, no sampling noise)
    2. correlation function  corr(x_ref, x)  from exact covariance columns
       C e_ref at n_ref reference nodes, binned by distance, compared with the
       Matern curve; empirical rho = distance where it drops to the value the
       Matern curve has at rho (~0.135)
    3. a CALIBRATED (gamma, delta): rescaling R by c leaves rho unchanged and
       multiplies C by c^{-k}, so c = (mean var / sigma^2)^{1/k} gives the
       target mean std exactly.
    4. prior samples, m0 +- 2 std fields, and (optionally) the true field
       of Case 1 / 2 / 3 for the figure.

Everything is numpy/scipy (P1 stiffness, mass, boundary mass assembled here;
identical to the dolfinx forms of ex05 for P1). dolfinx is only needed to get
the ellipsoid of Cases 1-2 from cardiac_geometries.

Usage:
    # Cases 1-2 (ellipsoid from cardiac_geometries, same as ex05)
    python prior_calibration.py --mesh ellipsoid --sigma 1.5 --rho 10 --m0 2.5 \\
        --truth fibrosis --output-dir calib_ellipsoid
    # Case 3 (patient LV)
    python prior_calibration.py --mesh Patient_7_lv_tissue.vtu --truth tissue \\
        --sigma 0.7 --rho 15 --order 4 --log --output-dir calib_lv
    # quick sweep of rho (prints a table, no files)
    python prior_calibration.py --mesh ellipsoid --sweep-rho 5 7.5 10 15 20

Outputs (--output-dir):
    calib_summary.txt   parameters, achieved vs target, calibrated gamma/delta
    calib_corr.txt      r, corr_mean, corr_p10, corr_p90, n_pairs, matern
    calib_fields.npz    points, cells, std, samples, truth, m0  (for plotting)
    calib_fields.xdmf/.h5  ParaView fields for the Fig. 6.3-style figure:
                        dev_plus2sigma, dev_minus2sigma, dev_sample_i, dev_truth
                        (deviation from m0), the same in C units (C_*),
                        prior_std; cell field tissue for Case 3
    calib_fields.vtu    same fields (single file)
Plot with plot_prior_calibration.py.
"""

import argparse
from math import gamma as gammafn, pi, sqrt
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.special import kv


# =============================================================================
# mesh
# =============================================================================

def load_mesh(spec, psize_ref=3.0):
    """Return points (n,3), cells (nc,4), tissue (nc,) or None."""
    if spec == "ellipsoid":
        import cardiac_geometries
        geodir = Path("lv_ellipsoid")
        try:
            geo = cardiac_geometries.geometry.Geometry.from_folder(geodir)
        except Exception:
            geo = cardiac_geometries.mesh.lv_ellipsoid(
                outdir=geodir, create_fibers=True, fiber_space="DG_0",
                psize_ref=psize_ref, r_short_epi=10,
                fiber_angle_endo=40.0, fiber_angle_epi=-50.0)
        dom = geo.mesh
        pts = np.asarray(dom.geometry.x[:, :3], dtype=float)
        nloc = dom.topology.index_map(dom.topology.dim).size_local
        cells = np.asarray(dom.geometry.dofmap[:nloc], dtype=np.int64)
        return pts, cells, None
    import meshio
    m = meshio.read(spec)
    pts = np.asarray(m.points, dtype=float)
    cells = np.asarray(m.cells_dict["tetra"], dtype=np.int64)
    tissue = None
    for key in ("tissue", "gmsh:physical"):
        if key in m.cell_data_dict and "tetra" in m.cell_data_dict[key]:
            tissue = np.rint(m.cell_data_dict[key]["tetra"]).astype(int)
            if key == "gmsh:physical":       # .msh groups 1/2/3 -> 0/1/2
                tissue = tissue - 1
            break
    return pts, cells, tissue


def boundary_facets(cells):
    f = np.sort(np.vstack([cells[:, [1, 2, 3]], cells[:, [0, 2, 3]],
                           cells[:, [0, 1, 3]], cells[:, [0, 1, 2]]]), axis=1)
    u, c = np.unique(f, axis=0, return_counts=True)
    return u[c == 1]


def assemble(points, cells):
    """P1 stiffness K, consistent mass M, boundary mass Mb (scipy CSC)."""
    n = len(points)
    x = points[cells]
    Mt = np.stack([x[:, 1] - x[:, 0], x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]], 1)
    Minv = np.linalg.inv(Mt)
    G = np.empty((len(cells), 4, 3))
    G[:, 1:, :] = np.transpose(Minv, (0, 2, 1))
    G[:, 0, :] = -G[:, 1:, :].sum(1)
    vol = np.abs(np.linalg.det(Mt)) / 6.0
    r, c = np.repeat(cells, 4, 1).ravel(), np.tile(cells, (1, 4)).ravel()
    K = sp.coo_matrix(((vol[:, None, None] * np.einsum("eik,ejk->eij", G, G)).ravel(),
                       (r, c)), (n, n)).tocsc()
    Me = (np.ones((4, 4)) + np.eye(4)) / 20.0
    M = sp.coo_matrix(((vol[:, None, None] * Me).ravel(), (r, c)), (n, n)).tocsc()
    F = boundary_facets(cells)
    xf = points[F]
    area = 0.5 * np.linalg.norm(np.cross(xf[:, 1] - xf[:, 0], xf[:, 2] - xf[:, 0]), axis=1)
    Mbe = (np.ones((3, 3)) + np.eye(3)) / 12.0
    Mb = sp.coo_matrix(((area[:, None, None] * Mbe).ravel(),
                        (np.repeat(F, 3, 1).ravel(), np.tile(F, (1, 3)).ravel())),
                       (n, n)).tocsc()
    return K, M, Mb, vol, F


# =============================================================================
# prior
# =============================================================================

def matern_coefficients(sigma, rho, order, d=3):
    """gamma, delta for C = A^{-order}, A = gamma(-Lap) + delta (Lindgren 2011)."""
    nu = order - d / 2.0
    if nu <= 0:
        raise ValueError("order must be > d/2")
    kappa = sqrt(8.0 * nu) / rho
    const = gammafn(nu) / (gammafn(order) * (4 * pi) ** (d / 2) * kappa ** (2 * nu))
    g = (const / sigma ** 2) ** (1.0 / order)
    return g, g * kappa ** 2, nu, kappa


def matern_corr(r, nu, kappa):
    r = np.asarray(r, float)
    out = np.ones_like(r)
    z = kappa * r[r > 0]
    out[r > 0] = 2 ** (1 - nu) / gammafn(nu) * z ** nu * kv(nu, z)
    return out


class Prior:
    def __init__(self, K, M, Mb, gamma, delta, order=2, mass="lumped"):
        self.order = order
        self.gamma, self.delta = gamma, delta
        self.beta = sqrt(gamma * delta) / 1.42
        R = (gamma * K + delta * M + self.beta * Mb).tocsc()
        self.lu = sla.splu(R)
        self.Ml = np.asarray(M.sum(axis=1)).ravel()
        self.M = M if mass == "consistent" else None    # covariance mass
        self.n = K.shape[0]

    def _Rs(self, v):
        return self.lu.solve(v)

    def _Mm(self, v):
        if self.M is not None:
            return self.M @ v
        return self.Ml[:, None] * v if v.ndim == 2 else self.Ml * v

    def cov_apply(self, v):            # C v = (R^{-1} M)^{k-1} R^{-1} v
        y = self._Rs(v)
        for _ in range(self.order - 1):
            y = self._Rs(self._Mm(y))
        return y

    def diag(self, block=256):
        """exact diag(C) = y_i^T M y_i, y_i = (R^{-1} M)^{k/2-1} R^{-1} e_i
        (blocks of `block` right-hand sides per LU solve)."""
        d = np.empty(self.n)
        for s0 in range(0, self.n, block):
            idx = np.arange(s0, min(s0 + block, self.n))
            E = np.zeros((self.n, len(idx)))
            E[idx, np.arange(len(idx))] = 1.0
            Y = self._Rs(E)
            for _ in range(self.order // 2 - 1):
                Y = self._Rs(self._Mm(Y))
            d[idx] = np.einsum("ij,ij->j", Y, self._Mm(Y))
        return d

    def sample(self, rng):             # z = (R^{-1}Ml)^{k/2-1} R^{-1} Ml^{1/2} w
        z = self._Rs(np.sqrt(self.Ml) * rng.standard_normal(self.n))
        for _ in range(self.order // 2 - 1):
            z = self._Rs(self.Ml * z)
        return z


# =============================================================================
# true fields (for the figure only)
# =============================================================================

def truth_field(kind, points, cells, tissue, c_healthy=2.0, factor=3.0):
    if kind == "none":
        return None
    x = points
    if kind == "linear":                      # ex03 'linear'
        return 2.0 + (x[:, 0] - x[:, 0].min()) / (np.ptp(x[:, 0]) + 1e-30)
    if kind == "fibrosis":                    # ex03 'fibrosis'
        r = np.linalg.norm(x - np.array([-5.0, -2.0, -9.0]), axis=1)
        t = np.clip((10.0 - r) / 5.0, 0.0, 1.0)
        return 2.0 + 4.0 * (3 * t**2 - 2 * t**3)
    if kind == "tissue":                      # Case 3: nodal average of cell C
        if tissue is None:
            raise SystemExit("--truth tissue needs a mesh with a 'tissue' field")
        Cc = np.where(np.isin(tissue, (1, 2)), c_healthy * factor, c_healthy)
        cnt = np.bincount(cells.ravel(), minlength=len(x))
        return np.bincount(cells.ravel(), weights=np.repeat(Cc, 4),
                           minlength=len(x)) / np.maximum(cnt, 1)
    raise ValueError(kind)


# =============================================================================
# calibration
# =============================================================================

def correlation_curve(prior, std, points, interior, n_ref, rng, nbins, rmax):
    refs = rng.choice(interior, size=min(n_ref, len(interior)), replace=False)
    edges = np.linspace(0, rmax, nbins + 1)
    vals = [[] for _ in range(nbins)]
    for i in refs:
        e = np.zeros(prior.n)
        e[i] = 1.0
        col = prior.cov_apply(e)
        corr = col / (std * std[i])
        dist = np.linalg.norm(points - points[i], axis=1)
        b = np.digitize(dist, edges) - 1
        ok = (b >= 0) & (b < nbins)
        for bi, cv in zip(b[ok], corr[ok]):
            vals[bi].append(cv)
    rc = 0.5 * (edges[1:] + edges[:-1])
    mean = np.array([np.mean(v) if v else np.nan for v in vals])
    p10 = np.array([np.percentile(v, 10) if v else np.nan for v in vals])
    p90 = np.array([np.percentile(v, 90) if v else np.nan for v in vals])
    cnt = np.array([len(v) for v in vals])
    return rc, mean, p10, p90, cnt


def truth_correlation(dev, points, rmax, nbins, n_ref, rng):
    """Correlation of a FIXED field 'dev' (deviation from the prior mean),
    with the same non-centred estimator one would apply to prior samples:
        cov(r)  = mean over node pairs at distance ~ r of dev_i * dev_j
        corr(r) = cov(r) / mean(dev^2)
    Reference nodes are drawn uniformly (all nodes if n <= n_ref)."""
    n = len(points)
    refs = np.arange(n) if n <= n_ref else rng.choice(n, n_ref, replace=False)
    edges = np.linspace(0, rmax, nbins + 1)
    num = np.zeros(nbins)
    cnt = np.zeros(nbins)
    for i in refs:
        dist = np.linalg.norm(points - points[i], axis=1)
        b = np.digitize(dist, edges) - 1
        ok = (b >= 0) & (b < nbins)
        np.add.at(num, b[ok], dev[i] * dev[ok])
        np.add.at(cnt, b[ok], 1.0)
    c0 = np.mean(dev ** 2)
    corr = np.where(cnt > 0, num / np.maximum(cnt, 1) / max(c0, 1e-300), np.nan)
    return 0.5 * (edges[1:] + edges[:-1]), corr


def first_crossing(r, c, level):
    ok = ~np.isnan(c)
    r, c = r[ok], c[ok]
    idx = np.where(c <= level)[0]
    if len(idx) == 0:
        return np.nan
    j = idx[0]
    if j == 0:
        return r[0]
    return r[j - 1] + (level - c[j - 1]) * (r[j] - r[j - 1]) / (c[j] - c[j - 1])


def calibrate(points, cells, sigma, rho, order, mass, n_ref, nbins, seed,
              K=None, M=None, Mb=None, F=None, verbose=True):
    if K is None:
        K, M, Mb, _, F = assemble(points, cells)
    g, d, nu, kappa = matern_coefficients(sigma, rho, order)
    pr = Prior(K, M, Mb, g, d, order, mass)
    var = pr.diag()
    std = np.sqrt(var)
    Ml = pr.Ml
    mean_var = float((Ml * var).sum() / Ml.sum())          # volume-weighted
    bnd = np.unique(F)
    interior = np.setdiff1d(np.arange(len(points)), bnd)
    thin = len(interior) < 10                               # wall ~1 element
    if thin:
        interior = np.arange(len(points))
    rng = np.random.default_rng(seed)
    rmax = min(2.5 * rho, float(np.linalg.norm(np.ptp(points, axis=0))))
    rc, cm, p10, p90, cnt = correlation_curve(pr, std, points, interior, n_ref,
                                              rng, nbins, rmax)
    level = float(matern_corr(np.array([rho]), nu, kappa)[0])
    rho_emp = first_crossing(rc, cm, level)
    c = (mean_var / sigma ** 2) ** (1.0 / order)            # R -> c R
    edge = np.linalg.norm(points[cells[:, 0]] - points[cells[:, 1]], axis=1).mean()
    res = dict(gamma=g, delta=d, beta=pr.beta, nu=nu, kappa=kappa,
               std=std, mean_std=sqrt(mean_var),
               std_interior=float(std[interior].mean()) if len(interior) else np.nan,
               std_boundary=float(std[bnd].mean()),
               std_min=float(std.min()), std_max=float(std.max()),
               rho_emp=rho_emp, corr_level=level,
               gamma_cal=g * c, delta_cal=d * c, scale=c,
               r=rc, corr_mean=cm, corr_p10=p10, corr_p90=p90, n_pairs=cnt,
               matern=matern_corr(rc, nu, kappa), h=edge, prior=pr,
               n_interior=0 if thin else len(interior), n_nodes=len(points))
    if verbose:
        print(f"  order {order} (nu = {nu:g}), sigma = {sigma}, input rho = {rho:.3f} mm  ->  "
              f"gamma = {g:.4g}, delta = {d:.4g}, beta = {pr.beta:.4g}")
        n_int = "no interior nodes (wall ~1 element thick)" if thin else f"{len(interior)} interior"
        print(f"  mesh: {len(points)} nodes ({n_int}), mean edge h = {edge:.2f} mm,"
              f" 1/kappa = {1/kappa:.2f} mm, rho/h = {rho/edge:.1f}")
        print(f"  achieved std: mean {res['mean_std']:.3f} (target {sigma}), interior "
              f"{res['std_interior']:.3f}, boundary {res['std_boundary']:.3f}, range "
              f"[{res['std_min']:.3f}, {res['std_max']:.3f}]")
        print(f"  correlation length: measured {rho_emp:.2f} mm (input {rho:.2f} mm) "
              f"(corr level {level:.3f})")
        print(f"  calibrated (mean std = sigma): gamma = {g*c:.4g}, delta = {d*c:.4g}"
              f"  (scale {c:.3f})")
    return res


def main():
    ap = argparse.ArgumentParser(description="BiLaplacian prior calibration")
    ap.add_argument("--mesh", default="ellipsoid",
                    help="'ellipsoid' (cardiac_geometries, Cases 1-2) or a .vtu/.msh file")
    ap.add_argument("--psize-ref", type=float, default=3.0)
    ap.add_argument("--sigma", type=float, default=1.5)
    ap.add_argument("--rho", type=float, default=10.0)
    ap.add_argument("--order", type=int, default=2, choices=[2, 4])
    ap.add_argument("--mass", default="lumped", choices=["lumped", "consistent"],
                    help="mass matrix in the covariance (ex05 LV: lumped)")
    ap.add_argument("--m0", type=float, default=2.5, help="prior mean (C units, or log C with --log)")
    ap.add_argument("--log", action="store_true",
                    help="prior is on m = log C (samples shown as exp(m))")
    ap.add_argument("--truth", default="none",
                    choices=["none", "linear", "fibrosis", "tissue"])
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--n-ref", type=int, default=60, help="reference nodes for corr(r)")
    ap.add_argument("--nbins", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--match-truth", action="store_true",
                    help="choose sigma and rho FROM the ground truth (needs "
                         "--truth): sigma = max|truth - m0| / 2 (truth inside "
                         "the pointwise 95%% band), rho = correlation length "
                         "of the truth field (same estimator as for the "
                         "prior); implies --calibrate-rho")
    ap.add_argument("--truth-ref", type=int, default=2000,
                    help="reference nodes for the truth correlation estimate")
    ap.add_argument("--calibrate-rho", action="store_true",
                    help="also adjust the INPUT rho until the measured "
                         "correlation length equals --rho (fixed-point, few "
                         "iterations); then rescale for sigma")
    ap.add_argument("--sweep-rho", type=float, nargs="+", default=None,
                    help="only print a table for these rho values (no files)")
    ap.add_argument("--output-dir", default="calib")
    a = ap.parse_args()

    pts, cells, tissue = load_mesh(a.mesh, a.psize_ref)
    K, M, Mb, vol, F = assemble(pts, cells)
    print(f"mesh '{a.mesh}': {len(pts)} nodes, {len(cells)} tets, "
          f"volume {vol.sum():.1f} mm^3")

    if a.sweep_rho:
        print(f"\n{'rho':>6} {'rho_emp':>8} {'mean std':>9} {'interior':>9} "
              f"{'boundary':>9} {'gamma_cal':>10} {'delta_cal':>10}")
        for rho in a.sweep_rho:
            r = calibrate(pts, cells, a.sigma, rho, a.order, a.mass, a.n_ref,
                          a.nbins, a.seed, K, M, Mb, F, verbose=False)
            print(f"{rho:>6.1f} {r['rho_emp']:>8.2f} {r['mean_std']:>9.3f} "
                  f"{r['std_interior']:>9.3f} {r['std_boundary']:>9.3f} "
                  f"{r['gamma_cal']:>10.4g} {r['delta_cal']:>10.4g}")
        return

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    truth = truth_field(a.truth, pts, cells, tissue)
    truth_corr = None
    if a.match_truth:
        if truth is None:
            raise SystemExit("--match-truth needs --truth linear|fibrosis|tissue")
        dev = (np.log(truth) if a.log else truth) - a.m0
        rmax_t = float(np.linalg.norm(np.ptp(pts, axis=0)))      # bbox diagonal
        rt, ct = truth_correlation(dev, pts, rmax_t, 40, a.truth_ref,
                                   np.random.default_rng(a.seed + 7))
        level = float(matern_corr(np.array([1.0]), a.order - 1.5,
                                  sqrt(8 * (a.order - 1.5)))[0])
        rho_t = first_crossing(rt, ct, level)
        sig_t = 0.5 * float(np.abs(dev).max())
        print(f"\nground truth ({a.truth}) in {'log C' if a.log else 'C'} units, "
              f"deviation from m0 = {a.m0:g}:")
        print(f"  max |dev| = {2*sig_t:.4f}  -> sigma = max|dev|/2 = {sig_t:.4f}"
              f"   (rms dev {np.sqrt(np.mean(dev**2)):.4f})")
        if np.isfinite(rho_t):
            print(f"  correlation length of the truth = {rho_t:.2f} mm "
                  f"(corr level {level:.3f})")
        else:
            rho_t = rmax_t
            print(f"  truth correlation never drops to {level:.3f} within "
                  f"{rmax_t:.1f} mm (smooth trend): using rho = {rho_t:.1f} mm")
        a.sigma, a.rho, a.calibrate_rho = sig_t, rho_t, True
        truth_corr = np.c_[rt, ct]
        np.savetxt(out / "calib_corr_truth.txt", truth_corr,
                   header="r_mm corr_truth   (non-centred, deviation from m0)")
    rho_in = prev = a.rho
    if a.calibrate_rho:
        diag = float(np.linalg.norm(np.ptp(pts, axis=0)))
        if a.rho > 0.5 * diag:
            print(f"\nWARNING: target rho = {a.rho:.1f} mm is comparable to the domain "
                  f"size ({diag:.1f} mm): it cannot be measured inside the mesh, so "
                  f"rho is NOT calibrated (input rho = target). A field this smooth "
                  f"is a trend; any prior with rho >~ {0.5*diag:.0f} mm represents it.")
        else:
            print(f"\ncalibrating input rho so that the measured rho = {a.rho:.2f} mm")
            for it in range(10):
                r = calibrate(pts, cells, a.sigma, rho_in, a.order, a.mass, a.n_ref,
                              a.nbins, a.seed, K, M, Mb, F, verbose=False)
                print(f"  it {it}: input rho {rho_in:7.3f} -> measured {r['rho_emp']:7.3f}")
                if not np.isfinite(r["rho_emp"]):
                    rho_in = prev
                    print("  measured rho undefined (too long for the mesh): "
                          "keeping the previous input")
                    break
                if abs(r["rho_emp"] - a.rho) < 0.01 * a.rho:
                    break
                prev = rho_in
                rho_in *= a.rho / r["rho_emp"]
            print(f"  -> use input rho = {rho_in:.3f} mm\n")
    res = calibrate(pts, cells, a.sigma, rho_in, a.order, a.mass, a.n_ref,
                    a.nbins, a.seed, K, M, Mb, F)

    # samples with the CALIBRATED prior (so the figure shows what is used)
    pr_cal = Prior(K, M, Mb, res["gamma_cal"], res["delta_cal"], a.order, a.mass)
    std_cal = res["std"] / res["scale"] ** (a.order / 2.0)
    rng = np.random.default_rng(a.seed + 1)
    samples = np.array([a.m0 + pr_cal.sample(rng) for _ in range(a.n_samples)])

    np.savez(out / "calib_fields.npz", points=pts, cells=cells, std=std_cal,
             std_uncal=res["std"], samples=samples, m0=a.m0, log=a.log,
             truth=truth if truth is not None else np.array([]),
             sigma=a.sigma, rho=a.rho, rho_in=rho_in, order=a.order)
    np.savetxt(out / "calib_corr.txt",
               np.c_[res["r"], res["corr_mean"], res["corr_p10"], res["corr_p90"],
                     res["n_pairs"], res["matern"]],
               header="r_mm corr_mean corr_p10 corr_p90 n_pairs matern")
    lines = [f"mesh {a.mesh}", f"n_nodes {len(pts)}", f"n_interior {res['n_interior']}",
             f"mean_edge_h {res['h']:.4f}", f"order {a.order}", f"nu {res['nu']}",
             f"mass {a.mass}", f"log_parametrisation {a.log}", f"m0 {a.m0}",
             f"match_truth {a.match_truth}",
             f"target_sigma {a.sigma:.6g}", f"target_rho {a.rho:.6g}",
             f"input_rho {rho_in:.6g}",
             f"gamma {res['gamma']:.8g}", f"delta {res['delta']:.8g}",
             f"beta {res['beta']:.8g}", f"kappa {res['kappa']:.6g}",
             f"achieved_mean_std {res['mean_std']:.6g}",
             f"achieved_std_interior {res['std_interior']:.6g}",
             f"achieved_std_boundary {res['std_boundary']:.6g}",
             f"achieved_std_range {res['std_min']:.6g} {res['std_max']:.6g}",
             f"empirical_rho {res['rho_emp']:.6g}", f"corr_level {res['corr_level']:.4f}",
             f"calibration_scale {res['scale']:.6g}",
             f"gamma_calibrated {res['gamma_cal']:.8g}",
             f"delta_calibrated {res['delta_cal']:.8g}",
             "# use gamma_calibrated / delta_calibrated in ex05 via --gamma/--delta"]
    (out / "calib_summary.txt").write_text("\n".join(lines) + "\n")
    write_paraview_fields(out, pts, cells, tissue, std_cal, samples, truth,
                          a.m0, a.log)


def write_paraview_fields(out, pts, cells, tissue, std, samples, truth, m0, log):
    """
    Fields for the Bui-Thanh-style figure in ParaView, written to
    calib_fields.xdmf (+ calib_fields.h5) and calib_fields.vtu.

    Point data (P1, one value per mesh node):
        dev_plus2sigma, dev_minus2sigma     +-2 std           (deviation from m0)
        dev_sample_1 ... dev_sample_N       prior samples - m0
        dev_truth                           truth - m0        (if --truth)
        prior_std                           pointwise prior std
        C_plus2sigma, C_minus2sigma, C_sample_i, C_truth   same fields in C units
                                            (exp(.) if --log, else m0 + dev)
    Cell data: tissue (Case 3 only).
    Deviations are in the PARAMETER units (log C with --log), as in
    Bui-Thanh et al. (2013, Fig. 6.3). Use ONE symmetric colour range for all
    dev_* fields: +-colour_range_dev written to calib_summary.txt.
    """
    import meshio
    to_C = np.exp if log else (lambda x: x)
    pd = {"dev_plus2sigma": 2.0 * std, "dev_minus2sigma": -2.0 * std,
          "prior_std": std,
          "C_plus2sigma": to_C(m0 + 2.0 * std), "C_minus2sigma": to_C(m0 - 2.0 * std)}
    for i, sm in enumerate(samples):
        pd[f"dev_sample_{i + 1}"] = sm - m0
        pd[f"C_sample_{i + 1}"] = to_C(sm)
    if truth is not None:
        pd["dev_truth"] = (np.log(truth) if log else truth) - m0
        pd["C_truth"] = truth
    pd = {k: np.ascontiguousarray(v, dtype=np.float64) for k, v in pd.items()}
    cd = {}
    if tissue is not None:
        cd = {"tissue": [np.asarray(tissue, dtype=np.float64)]}
    mesh = meshio.Mesh(pts, [("tetra", cells)], point_data=pd, cell_data=cd)

    vmax = max(float(np.abs(v).max()) for k, v in pd.items() if k.startswith("dev_"))
    with open(out / "calib_summary.txt", "a") as fh:
        fh.write(f"colour_range_dev {vmax:.4g}   # use [-{vmax:.3g}, {vmax:.3g}] "
                 f"for all dev_* fields in ParaView\n")

    written = []
    meshio.write(out / "calib_fields.vtu", mesh)
    written.append("calib_fields.vtu")
    try:
        meshio.write(out / "calib_fields.xdmf", mesh)       # + calib_fields.h5
        written.append("calib_fields.xdmf/.h5")
    except Exception as err:                                # h5py missing, ...
        print(f"  [warn] XDMF not written ({err}); the .vtu has the same fields")
    print(f"\nSaved calib_summary.txt, calib_corr.txt, calib_fields.npz, "
          f"{', '.join(written)} in {out}")
    print(f"  ParaView: colour all dev_* fields with the symmetric range "
          f"[-{vmax:.3g}, {vmax:.3g}] (diverging map, e.g. 'Cool to Warm')")


if __name__ == "__main__":
    main()
