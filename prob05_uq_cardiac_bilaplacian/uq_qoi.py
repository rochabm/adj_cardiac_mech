"""
uq_qoi.py
---------
Posterior of scalar quantities of interest from the saved Laplace approximation
of ex07 -- NO forward solves and NO eigenproblem, only the prior is rebuilt:

    q_core     = mean stiffness in the fibrotic core
    q_healthy  = mean stiffness in the healthy tissue
    (--mean geometric, default: exp( (1/|R|) int_R m dx );
     --mean arithmetic:              (1/|R|) int_R C dx )
    q_ratio    = q_core / q_healthy                    (stiffness contrast)

Posterior samples exactly as ex07 (same formula, same prior):
    m = m_MAP + z - U diag(1 - 1/sqrt(1+lambda)) U^T H_prior z,   z ~ N(0, Gamma_prior)
with (lambda, U) read from uq_<case>.npz. Prior samples m0 + z are drawn alongside,
so the prior interval of each QoI is reported as well (what the data taught us).

Regions as ex06/ex07: r = |x - (-5,-2,-9)|, core r <= 5 mm, healthy r >= 10 mm;
integrals with the lumped mass (P1), as in the prior.

Usage (same folder as ex07, needs dolfinx + the lv_ellipsoid geometry):
    python uq_qoi.py --case fibrosis --n-samples 1000

Reads  <root>/<case>/uq_<case>.npz, metrics_uq_<case>.json (prior settings)
Writes <root>/<case>/qoi_<case>.json, qoi_<case>.tex, qoi_<case>.pdf/.png
"""

from mpi4py import MPI
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import petsc4py.PETSc as PETSc

import ufl
import dolfinx.fem.petsc
from dolfinx import fem
import cardiac_geometries

from paper_style import apply_paper_style, save_figure, COLORS

CENTER, R0, R1 = np.array([-5.0, -2.0, -9.0]), 5.0, 10.0      # as ex03/ex06/ex07
QOI_LABEL = {"core": r"$\bar C_{\mathrm{core}}$ [kPa]",
             "healthy": r"$\bar C_{\mathrm{healthy}}$ [kPa]",
             "ratio": r"$\bar C_{\mathrm{core}} / \bar C_{\mathrm{healthy}}$"}
QOI_TEX = {"core": r"\bar C_{\mathrm{core}}", "healthy": r"\bar C_{\mathrm{healthy}}",
           "ratio": r"\bar C_{\mathrm{core}}/\bar C_{\mathrm{healthy}}"}


# =============================================================================
# BiLaplacian prior -- identical to ex07 (only what is needed here)
# =============================================================================

class BiLaplacianPrior:
    def __init__(self, Va, gamma, delta, dx, domain, theta=None):
        self.ndofs = Va.dofmap.index_map.size_local * Va.dofmap.index_map_bs
        self.gamma, self.delta = float(gamma), float(delta)
        self.beta = np.sqrt(self.gamma * self.delta) / 1.42
        q, dq = ufl.TrialFunction(Va), ufl.TestFunction(Va)
        ds_all = ufl.Measure("ds", domain=domain)
        grad_q = ufl.grad(q) if theta is None else theta * ufl.grad(q)
        R_form = (self.gamma * ufl.inner(grad_q, ufl.grad(dq)) * dx
                  + self.delta * ufl.inner(q, dq) * dx
                  + self.beta * ufl.inner(q, dq) * ds_all)
        self._R = fem.petsc.assemble_matrix(fem.form(R_form))
        self._R.assemble()
        M = fem.petsc.assemble_matrix(fem.form(ufl.inner(q, dq) * dx))
        M.assemble()
        self._x = self._R.createVecRight()
        self._b = self._R.createVecRight()
        self._b.array[:] = 1.0
        M.mult(self._b, self._x)
        self.M_lump = np.maximum(self._x.array.copy(), 1e-30)
        M.destroy()
        self._ksp = PETSc.KSP().create(domain.comm)
        self._ksp.setOperators(self._R)
        self._ksp.setType("preonly")
        self._ksp.getPC().setType("lu")
        self._ksp.getPC().setFactorSolverType("mumps")
        self._ksp.setUp()

    def _R_mult(self, v):
        self._b.array[:] = v
        self._R.mult(self._b, self._x)
        return self._x.array.copy()

    def _R_solve(self, v):
        self._b.array[:] = v
        self._ksp.solve(self._b, self._x)
        return self._x.array.copy()

    def mult(self, v):
        return self._R_mult(self._R_mult(v) / self.M_lump)

    def sample(self, rng):
        return self._R_solve(np.sqrt(self.M_lump) * rng.standard_normal(self.ndofs))


# =============================================================================
# sampling and QoIs (pure numpy given a prior object)
# =============================================================================

def draw(prior, U, lam, m_map, m0, n_samples, rng):
    """Prior and posterior samples of m (rows), same z for both (as ex07)."""
    dp = 1.0 - 1.0 / np.sqrt(1.0 + lam)
    prior_s, post_s = [], []
    for _ in range(n_samples):
        z = prior.sample(rng)
        prior_s.append(m0 + z)
        post_s.append(m_map + z - U @ (dp * (U.T @ prior.mult(z))))
    return np.array(prior_s), np.array(post_s)


def qois(m, w, core, healthy, kind="geometric"):
    """m: (n_samples, n) or (n,) -> dict of arrays.
    geometric : exp( mean of m over the region )  -- linear in m, so exactly
                log-normal under the Laplace approximation, median = MAP value
    arithmetic: mean of C = exp(m) over the region -- its posterior mean is shifted
                above the MAP value by ~ exp(sigma_post^2 / 2) (log-normal bias)"""
    m = np.atleast_2d(m)
    avg = lambda f, k: f[:, k] @ w[k] / w[k].sum()
    if kind == "geometric":
        q_c, q_h = np.exp(avg(m, core)), np.exp(avg(m, healthy))
    else:
        C = np.exp(m)
        q_c, q_h = avg(C, core), avg(C, healthy)
    return {"core": q_c, "healthy": q_h, "ratio": q_c / q_h}


def summary(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return dict(mean=float(x.mean()), std=float(x.std(ddof=1)), q025=float(lo), q975=float(hi))


def plot_hist(q_post, q_true, q_map, out):
    keys = ["core", "healthy", "ratio"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    for ax, k in zip(axes, keys):
        x = q_post[k]
        lo, hi = np.percentile(x, [2.5, 97.5])
        ax.axvspan(lo, hi, color=COLORS["blue"], alpha=0.12, lw=0, label="95% CI")
        a, b = np.percentile(x, [0.5, 99.5])
        a, b = min(a, q_true[k], q_map[k]), max(b, q_true[k], q_map[k])
        pad = 0.05 * (b - a)
        x = x[(x >= a) & (x <= b)]
        ax.hist(x, bins=40, range=(a, b), density=True, color=COLORS["blue"], alpha=0.75,
                edgecolor="white", linewidth=0.3, label="posterior")
        ax.axvline(q_true[k], color="black", ls="--", lw=1.0, label="truth")
        ax.axvline(q_map[k], color=COLORS["orange"], ls="-", lw=1.2, label="MAP")
        ax.set_xlim(a - pad, b + pad)
        ax.set_xlabel(QOI_LABEL[k])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    axes[0].set_ylabel("density")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(w_pad=0.8, rect=(0, 0, 1, 0.88))
    save_figure(fig, str(out))
    plt.close(fig)


# =============================================================================
# main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="out_ex07")
    ap.add_argument("--case", default="fibrosis", choices=["fibrosis", "linear"])
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--geometry", default="lv_ellipsoid")
    ap.add_argument("--mean", choices=["geometric", "arithmetic"], default="geometric",
                    help="Regional mean of C: exp(<m>) (default) or <exp(m)>")
    ap.add_argument("--use-tex", action="store_true")
    a = ap.parse_args()

    d = Path(a.root) / a.case
    dat = np.load(d / f"uq_{a.case}.npz")
    lam, U, m_map, m_true = dat["eigvals"], dat["eigvecs"], dat["m_map"], dat["m_true"]
    s = json.loads((d / f"metrics_uq_{a.case}.json").read_text())["settings"]
    print(f"{a.case}: {len(lam)} eigenpairs (lambda_k = {lam[-1]:.2e}), "
          f"gamma = {s['gamma']}, delta = {s['delta']}, fiber ratio = "
          f"{s.get('prior_fiber_ratio', 1.0)}, noise = {100 * s.get('data_noise', 0):g}%")
    if lam[-1] > 0.1:
        print(f"  WARNING: lambda_k = {lam[-1]:.2f} > 0.1 -- spectrum truncated early; "
              "posterior spread will be overestimated (conservative).")

    # ---- geometry and prior exactly as ex07 ---------------------------------
    try:                                   # newer cardiac_geometries: (comm, folder)
        geo = cardiac_geometries.geometry.Geometry.from_folder(MPI.COMM_WORLD, Path(a.geometry))
    except Exception:                      # same fallback as ex07 (regenerates the mesh)
        geo = cardiac_geometries.mesh.lv_ellipsoid(
            outdir=Path(a.geometry), create_fibers=True, fiber_space="DG_0",
            psize_ref=3, r_short_epi=10, fiber_angle_endo=40.0, fiber_angle_epi=-50.0)
    domain = geo.mesh
    Va = fem.functionspace(domain, ("Lagrange", 1))
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 4})
    theta = None
    fr = float(s.get("prior_fiber_ratio", 1.0))
    if fr != 1.0:
        f0n = geo.f0 / ufl.sqrt(ufl.inner(geo.f0, geo.f0))
        theta = ufl.Identity(domain.geometry.dim) + (fr ** 2 - 1.0) * ufl.outer(f0n, f0n)
    prior = BiLaplacianPrior(Va, s["gamma"], s["delta"], dx, domain, theta=theta)
    if prior.ndofs != len(m_map):
        raise RuntimeError(f"mesh has {prior.ndofs} dofs, npz has {len(m_map)}")

    # consistency: the saved eigenvectors must be H_prior-orthonormal for THIS prior
    kc = min(20, U.shape[1])
    G = U[:, :kc].T @ np.column_stack([prior.mult(U[:, j]) for j in range(kc)])
    err = np.abs(G - np.eye(kc)).max()
    print(f"  check U^T H_prior U = I (first {kc}): max error {err:.2e}")
    if err > 1e-6:
        raise RuntimeError("rebuilt prior does not match the one used in ex07 "
                           "(different mesh, gamma/delta or fiber ratio?)")

    # ---- samples and QoIs ---------------------------------------------------
    xyz = Va.tabulate_dof_coordinates()
    r = np.linalg.norm(xyz - CENTER, axis=1)
    core, healthy = r <= R0, r >= R1
    w = prior.M_lump
    m0 = float(s.get("C0", 2.0))
    rng = np.random.default_rng(a.seed)
    print(f"  drawing {a.n_samples} prior/posterior samples ...")
    m_pr, m_po = draw(prior, U, lam, m_map, np.log(m0), a.n_samples, rng)

    q_true = {k: float(v[0]) for k, v in qois(m_true, w, core, healthy, a.mean).items()}
    q_map = {k: float(v[0]) for k, v in qois(m_map, w, core, healthy, a.mean).items()}
    q_post, q_prior = qois(m_po, w, core, healthy, a.mean), qois(m_pr, w, core, healthy, a.mean)
    print(f"  regional mean of C: {a.mean}")

    # sanity: sample variance of m vs. the Laplace pointwise variance saved by ex07
    rel = np.median(np.abs(m_po.var(0, ddof=1) / dat["post_var"] - 1.0))
    print(f"  sample vs. Laplace pointwise variance: median rel. diff {rel:.3f} "
          f"(~{np.sqrt(2.0 / a.n_samples):.3f} expected from {a.n_samples} samples)")

    res = {}
    print(f"\n{'QoI':<9s} {'truth':>7s} {'MAP':>7s} {'post mean':>9s} "
          f"{'post 95% CI':>17s} {'prior 95% CI':>17s} {'in CI':>6s}")
    for k in ["core", "healthy", "ratio"]:
        po, pr = summary(q_post[k]), summary(q_prior[k])
        inside = po["q025"] <= q_true[k] <= po["q975"]
        res[k] = dict(truth=q_true[k], map=q_map[k], posterior=po, prior=pr,
                      truth_in_95CI=bool(inside))
        print(f"{k:<9s} {q_true[k]:7.3f} {q_map[k]:7.3f} {po['mean']:9.3f} "
              f"[{po['q025']:6.3f}, {po['q975']:6.3f}] [{pr['q025']:6.3f}, {pr['q975']:6.3f}]"
              f" {'yes' if inside else 'NO':>6s}")

    (d / f"qoi_{a.case}.json").write_text(json.dumps(
        dict(case=a.case, mean=a.mean, n_samples=a.n_samples, seed=a.seed, n_eig=int(len(lam)),
             lambda_k=float(lam[-1]), qoi=res), indent=2))
    lines = [f"% posterior mean [95% credible interval]; truth; {a.n_samples} samples"]
    for k in ["core", "healthy", "ratio"]:
        p = res[k]["posterior"]
        lines.append(f"${QOI_TEX[k]} = {p['mean']:.2f}\\;[{p['q025']:.2f},\\,{p['q975']:.2f}]$"
                     f" (truth ${res[k]['truth']:.2f}$)")
    (d / f"qoi_{a.case}.tex").write_text("\n".join(lines) + "\n")

    apply_paper_style(use_tex=a.use_tex)
    plot_hist(q_post, q_true, q_map, d / f"qoi_{a.case}")
    print(f"Wrote {d}/qoi_{a.case}.json, .tex, .pdf/.png")


if __name__ == "__main__":
    main()
