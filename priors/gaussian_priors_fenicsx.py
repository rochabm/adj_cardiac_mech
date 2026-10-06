"""
Gaussian priors in infinite dimensions — FEniCSx (dolfinx) reimplementation.

Port of the hIPPYlib G2S3-2018 tutorial:
    http://g2s3.com/labs/notebooks/Gaussian_priors.html

The original relies on hippylib.LaplacianPrior / BiLaplacianPrior, which do not
exist in FEniCSx. Here those PDE-based (Matern/Whittle) priors are rebuilt from
scratch with dolfinx + petsc4py.

Math recap
----------
Covariance C = A^{-alpha},  A = gamma (-div(Theta grad)) + delta I.

alpha = 2  (BiLaplacian, trace-class in 2D):
    precision  R = A M^{-1} A
    sample     A x = b,   b ~ N(0, M)      (so x ~ N(0, A^{-1} M A^{-1}))
    We draw b using a *lumped* mass square root:  b = sqrt(M_lumped) * w, w~N(0,I).
    (hIPPYlib uses a consistent-mass sqrt; lumped is the common, robust choice
     and gives the same qualitative behaviour.)

alpha = 1  (Laplacian, NOT trace-class in 2D -> mesh-dependent samples):
    R = gamma K + delta M   (K = stiffness, M = mass)
    sample by a Cholesky-like factor of R:  x = R^{-1} sqrt(R) w  is expensive;
    instead we use the standard trick   x solves  R x = sqrt(M_lumped) w ... NO.
    For alpha=1 the correct white-noise map is  x = L^{-T} w with L L^T = R.
    We form a sparse Cholesky of R via PETSc and apply L^{-T}. This deliberately
    reproduces the mesh-dependent (non-trace-class) pathology the tutorial shows.

Requires: dolfinx (>=0.8), ufl, petsc4py, numpy, matplotlib, mpi4py.
Install via conda-forge or the official dolfinx docker image (no pip wheel exists).

Run:  python3 gaussian_priors_fenicsx.py
"""

import math
import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI
from petsc4py import PETSc

import ufl
import dolfinx
import dolfinx.fem as fem
import dolfinx.fem.petsc as petsc
import dolfinx.mesh as dmesh
from dolfinx.mesh import (create_unit_square, CellType, locate_entities,
                          refine, compute_incident_entities)


# ---------------------------------------------------------------------------
# Meshes
# ---------------------------------------------------------------------------
def locally_refined_mesh(n0=16, n_refine=4):
    """Unit square, locally refined inside the box [0.2,0.5] x [0.3,0.7].

    dolfinx 0.8: refine(mesh, edges) returns ONLY the mesh (no parent maps).
    We mark edges whose midpoint lies in the box and refine on them.
    """
    mesh = create_unit_square(MPI.COMM_WORLD, n0, n0, CellType.triangle)

    for _ in range(n_refine):
        tdim = mesh.topology.dim          # 2
        edim = 1                          # edges
        mesh.topology.create_entities(edim)
        mesh.topology.create_connectivity(edim, 0)

        # midpoints of all local edges -> (N,3) array in 0.8
        n_edges = mesh.topology.index_map(edim).size_local
        emids = dmesh.compute_midpoints(
            mesh, edim, np.arange(n_edges, dtype=np.int32))

        inside = ((emids[:, 1] < 0.7) & (emids[:, 1] > 0.3) &
                  (emids[:, 0] > 0.2) & (emids[:, 0] < 0.5))
        edges = np.flatnonzero(inside).astype(np.int32)

        # Most 0.8 builds return a single Mesh; some return a tuple.
        res = refine(mesh, edges)
        mesh = res[0] if isinstance(res, tuple) else res

    return mesh


# ---------------------------------------------------------------------------
# Anisotropic diffusion tensor  Theta  (constant per element)
# ---------------------------------------------------------------------------
def anis_tensor(theta0, theta1, alpha_angle):
    """SPD tensor with eigenvalues theta0, theta1 and rotation alpha_angle.

    Returns a 2x2 UFL constant matrix.
        R = [[cos, -sin],[sin, cos]]
        Theta = R diag(theta0, theta1) R^T
    """
    c, s = math.cos(alpha_angle), math.sin(alpha_angle)
    t00 = theta0 * c * c + theta1 * s * s
    t01 = (theta0 - theta1) * c * s
    t11 = theta0 * s * s + theta1 * c * c
    return ufl.as_matrix(((t00, t01), (t01, t11)))


# ---------------------------------------------------------------------------
# PDE-based prior
# ---------------------------------------------------------------------------
class MaternPrior:
    """Gaussian prior with covariance A^{-alpha},
    A = gamma * (-div(Theta grad)) + delta * I,  alpha in {1, 2}.

    Boundary conditions:
        robin_bc = False -> natural (homogeneous Neumann)
        robin_bc = True  -> Robin term  beta * <u, v>_bdry  with
                            beta = sqrt(gamma*delta)/1.42  (hIPPYlib default)
    """

    def __init__(self, Vh, gamma, delta, alpha=2,
                 Theta=None, robin_bc=False, seed=1):
        assert alpha in (1, 2)
        self.Vh = Vh
        self.alpha = alpha
        self.gamma = gamma
        self.delta = delta
        self.rng = np.random.default_rng(seed)

        u = ufl.TrialFunction(Vh)
        v = ufl.TestFunction(Vh)

        # diffusion (stiffness) term
        if Theta is None:
            grad_form = ufl.inner(ufl.grad(u), ufl.grad(v))
        else:
            grad_form = ufl.inner(Theta * ufl.grad(u), ufl.grad(v))

        a_form = gamma * grad_form * ufl.dx + delta * ufl.inner(u, v) * ufl.dx

        if robin_bc:
            beta = math.sqrt(gamma * delta) / 1.42
            a_form += beta * ufl.inner(u, v) * ufl.ds

        m_form = ufl.inner(u, v) * ufl.dx          # mass matrix

        # assemble PETSc matrices
        self.A = petsc.assemble_matrix(fem.form(a_form))
        self.A.assemble()
        self.M = petsc.assemble_matrix(fem.form(m_form))
        self.M.assemble()

        # lumped mass (row sums) and its sqrt, stored as PETSc vecs
        self.ml = self.A.createVecRight()
        self.M.getRowSum(self.ml)
        self.ml_sqrt = self.ml.copy()
        self.ml_sqrt.sqrtabs()

        # --- solvers -------------------------------------------------------
        # A^{-1} via direct LU (mumps if available, else petsc lu)
        self.Asolver = self._lu_solver(self.A)

        if alpha == 1:
            # R = A ;  need a factor L with L L^T = A to map white noise.
            # Use a Cholesky factor through PETSc; apply L^{-T}.
            self.Rsolver = self.Asolver
            self._chol = self._cholesky_factor(self.A)
        else:
            # R = A M^{-1} A ; Rsolver x = A^{-1} M A^{-1} x
            pass

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _lu_solver(Amat):
        ksp = PETSc.KSP().create(Amat.comm)
        ksp.setOperators(Amat)
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
        try:
            pc.setFactorSolverType("mumps")
        except Exception:
            pass
        ksp.setFromOptions()
        return ksp

    @staticmethod
    def _cholesky_factor(Amat):
        """Return a PETSc PC holding an (I)Cholesky factorization of Amat,
        used to apply L^{-T} for alpha=1 sampling."""
        pc = PETSc.PC().create(Amat.comm)
        pc.setOperators(Amat)
        pc.setType("cholesky")
        try:
            pc.setFactorSolverType("mumps")
        except Exception:
            pass
        pc.setUp()
        return pc

    # -------------------------------------------------------------- sampling
    def _white_noise_scaled(self):
        """Return b = sqrt(M_lumped) * w  as a PETSc vec,  w ~ N(0, I)."""
        b = self.A.createVecRight()
        w = self.rng.standard_normal(b.getLocalSize())
        b.setArray(w)
        b.pointwiseMult(b, self.ml_sqrt)      # b <- b .* sqrt(ml)
        return b

    def sample(self):
        """Draw one sample as a numpy array of nodal values."""
        b = self._white_noise_scaled()
        x = self.A.createVecRight()

        if self.alpha == 2:
            # x = A^{-1} b   => x ~ N(0, A^{-1} M A^{-1})
            self.Asolver.solve(b, x)
        else:
            # alpha = 1: want x ~ N(0, A^{-1}).  Map:  A x = sqrt(M) w is
            # x ~ N(0, A^{-1} M A^{-1}) which is NOT N(0, A^{-1}).
            # The trace-class-violating Laplacian sample uses white noise w
            # directly (no mass sqrt) and solves via the Cholesky factor:
            #     x = L^{-T} w ,  L L^T = A.
            w = self.A.createVecRight()
            w.setArray(self.rng.standard_normal(w.getLocalSize()))
            # apply L^{-T}: PETSc PCApplySymmetricRight ~ L^{-T} for Cholesky
            try:
                self._chol.applySymmetricRight(w, x)
            except Exception:
                # fallback: A x = sqrt(M) w (mesh-dependent either way)
                self.Asolver.solve(b, x)

        return x.getArray().copy()

    def samples(self, n):
        return [self.sample() for _ in range(n)]

    # ------------------------------------------------------- pointwise stats
    def pointwise_variance_exact(self):
        """Diagonal of the covariance C.

        alpha=2: C = A^{-1} M A^{-1}. diag via solving against unit columns
                 of M (expensive but 'Exact', matching the tutorial).
        alpha=1: C = A^{-1}. diag via unit columns.
        Returns numpy array of nodal variances.
        """
        n = self.A.getSize()[0]
        ei = self.A.createVecRight()
        tmp = self.A.createVecRight()
        out = self.A.createVecRight()
        var = np.zeros(n)

        for i in range(n):
            ei.set(0.0)
            ei.setValue(i, 1.0)
            ei.assemble()
            if self.alpha == 2:
                self.Asolver.solve(ei, tmp)      # tmp = A^{-1} e_i
                self.M.mult(tmp, out)            # out = M A^{-1} e_i
                self.Asolver.solve(out, tmp)     # tmp = A^{-1} M A^{-1} e_i
                var[i] = tmp.getValue(i)
            else:
                self.Asolver.solve(ei, tmp)      # tmp = A^{-1} e_i
                var[i] = tmp.getValue(i)
        return var

    def pointwise_variance_randomized(self, r=200):
        """Randomized diagonal estimator (Hutchinson-style) of C.

        C = B B^T with  B = A^{-1} sqrt(M)  (alpha=2)  or  A^{-1/2} (alpha=1
        approximated by the sampler). Estimate diag(C) ~ mean over r samples
        of x_k .* x_k, where x_k are prior samples. Cheap; matches the
        'Randomized' option used for the anisotropic figure in the tutorial.
        """
        n = self.A.getSize()[0]
        acc = np.zeros(n)
        for _ in range(r):
            x = self.sample()
            acc += x * x
        return acc / r

    def correlation_structure(self, point):
        """Apply C to a (near) point source at `point`: C delta_p.
        Returns nodal values. Uses nearest dof to `point`."""
        # nearest node
        coords = self.Vh.tabulate_dof_coordinates()[:, :2]
        idx = int(np.argmin(np.sum((coords - np.asarray(point)) ** 2, axis=1)))

        rhs = self.A.createVecRight()
        rhs.set(0.0)
        rhs.setValue(idx, 1.0)
        rhs.assemble()

        out = self.A.createVecRight()
        tmp = self.A.createVecRight()
        if self.alpha == 2:
            self.Asolver.solve(rhs, tmp)
            self.M.mult(tmp, out)
            self.Asolver.solve(out, rhs)   # rhs <- C delta
            return rhs.getArray().copy()
        else:
            self.Asolver.solve(rhs, out)
            return out.getArray().copy()


# ---------------------------------------------------------------------------
# Plotting helpers (replacements for hippylib.nb.multi1_plot)
# ---------------------------------------------------------------------------
def _triang(Vh):
    import matplotlib.tri as mtri
    coords = Vh.tabulate_dof_coordinates()[:, :2]
    mesh = Vh.mesh
    mesh.topology.create_connectivity(2, 0)
    cells = mesh.geometry.dofmap  # cell -> geometry node
    # For CG1 the dof coords coincide with vertices; build triangulation from
    # the cell-vertex connectivity.
    c2v = mesh.topology.connectivity(2, 0)
    tris = np.array([c2v.links(c) for c in range(c2v.num_nodes)])
    return mtri.Triangulation(coords[:, 0], coords[:, 1], tris)


def _field_on_ax(fig, ax, f, Vh, title, cmap, vmin=None, vmax=None,
                 add_cbar=True):
    """Draw a single nodal field onto an existing axis."""
    tri = _triang(Vh)
    tpc = ax.tripcolor(tri, np.asarray(f), shading="gouraud",
                       cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    if add_cbar:
        fig.colorbar(tpc, ax=ax, fraction=0.046, pad=0.04)
    return tpc


def case_panel(meshes, mesh_labels, gamma, delta, alpha, fname,
               n_samples=3, seed=1, sample_cmap="RdBu_r", var_cmap="jet"):
    """One consolidated figure for a given alpha.

    Layout: (n_samples + 1) rows x len(meshes) columns.
      - top n_samples rows: samples, one column per mesh
      - bottom row: pointwise variance, one column per mesh (shared colorbar)
    Saved to `fname`.
    """
    ncol = len(meshes)
    nrow = n_samples + 1
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(4.2 * ncol, 3.6 * nrow))
    axes = np.atleast_2d(axes)

    kind = {1: "Laplacian (alpha=1)", 2: "BiLaplacian (alpha=2)"}[alpha]
    fig.suptitle(f"{kind}   gamma={gamma}, delta={delta}",
                 fontsize=14, y=0.995)

    # build priors once per mesh, reuse for samples + variance
    priors, Vhs = [], []
    for m in meshes:
        Vh = V(m)
        priors.append(MaternPrior(Vh, gamma, delta, alpha=alpha, seed=seed))
        Vhs.append(Vh)

    # --- sample rows: symmetric colour scale per row for readability -------
    for r in range(n_samples):
        row_fields = [p.sample() for p in priors]
        amax = max(np.abs(f).max() for f in row_fields)
        for c in range(ncol):
            title = (f"{mesh_labels[c]}\nsample {r + 1}" if r == 0
                     else f"sample {r + 1}")
            _field_on_ax(fig, axes[r, c], row_fields[c], Vhs[c], title,
                         cmap=sample_cmap, vmin=-amax, vmax=amax,
                         add_cbar=(c == ncol - 1))

    # --- variance row: shared colour scale across meshes -------------------
    var_fields = [p.pointwise_variance_exact() for p in priors]
    vmin = min(f.min() for f in var_fields)
    vmax = max(f.max() for f in var_fields)
    for c in range(ncol):
        _field_on_ax(fig, axes[n_samples, c], var_fields[c], Vhs[c],
                     f"pointwise variance\n{mesh_labels[c]}",
                     cmap=var_cmap, vmin=vmin, vmax=vmax,
                     add_cbar=(c == ncol - 1))

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"  saved {fname}")
    return fig


def multi_field_plot(fields, Vhs, titles, same_colorbar=False, cmap="viridis"):
    """Plot several nodal fields side by side."""
    k = len(fields)
    fig, axes = plt.subplots(1, k, figsize=(5 * k, 4))
    if k == 1:
        axes = [axes]

    vmin = vmax = None
    if same_colorbar:
        allv = np.concatenate([np.asarray(f) for f in fields])
        vmin, vmax = allv.min(), allv.max()

    for ax, f, Vh, t in zip(axes, fields, Vhs, titles):
        tri = _triang(Vh)
        tpc = ax.tripcolor(tri, np.asarray(f), shading="gouraud",
                           cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(t)
        ax.set_aspect("equal")
        fig.colorbar(tpc, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()


def mesh_plot(meshes, titles):
    k = len(meshes)
    fig, axes = plt.subplots(1, k, figsize=(5 * k, 4))
    if k == 1:
        axes = [axes]
    for ax, m, t in zip(axes, meshes, titles):
        m.topology.create_connectivity(2, 0)
        coords = m.geometry.x[:, :2]
        c2v = m.topology.connectivity(2, 0)
        tris = np.array([c2v.links(c) for c in range(c2v.num_nodes)])
        ax.triplot(coords[:, 0], coords[:, 1], tris, lw=0.3)
        ax.set_title(t)
        ax.set_aspect("equal")
    plt.tight_layout()


def V(mesh):
    return fem.functionspace(mesh, ("Lagrange", 1))


# ---------------------------------------------------------------------------
# Driver reproducing the tutorial sections
# ---------------------------------------------------------------------------
def main():
    gamma, delta = 1.0, 4.0
    labels = ["Coarse (16x16)", "Fine (64x64)", "Locally refined"]

    # --- meshes ------------------------------------------------------------
    print("Building meshes...")
    mesh1 = create_unit_square(MPI.COMM_WORLD, 16, 16, CellType.triangle)
    mesh2 = create_unit_square(MPI.COMM_WORLD, 64, 64, CellType.triangle)
    mesh3 = locally_refined_mesh()
    meshes = [mesh1, mesh2, mesh3]

    # 00: the three meshes
    mesh_plot(meshes, labels)
    plt.savefig("00_meshes.png", dpi=150, bbox_inches="tight")
    print("  saved 00_meshes.png")

    # 01: Laplacian (alpha=1) — samples on all meshes + pointwise variance
    print("01  Laplacian (alpha=1): samples + pointwise variance")
    case_panel(meshes, labels, gamma, delta, alpha=1,
               fname="01_laplacian.png", seed=1)

    # 02: BiLaplacian (alpha=2) — samples on all meshes + pointwise variance
    print("02  BiLaplacian (alpha=2): samples + pointwise variance")
    case_panel(meshes, labels, gamma, delta, alpha=2,
               fname="02_bilaplacian.png", seed=1)

    # 03: boundary artifacts — natural vs Robin BC
    print("03  Boundary artifacts: natural vs Robin BC")
    meshb = create_unit_square(MPI.COMM_WORLD, 32, 32, CellType.triangle)
    Vhb = V(meshb)
    p_nat = MaternPrior(Vhb, gamma, delta, alpha=2, robin_bc=False)
    p_rob = MaternPrior(Vhb, gamma, delta, alpha=2, robin_bc=True)
    multi_field_plot(
        [p_nat.pointwise_variance_exact(), p_rob.pointwise_variance_exact()],
        [Vhb, Vhb], ["Natural BC", "Robin BC"],
        same_colorbar=True, cmap="jet")
    plt.suptitle("BiLaplacian pointwise variance: boundary conditions",
                 fontsize=13)
    plt.savefig("03_boundary_conditions.png", dpi=150, bbox_inches="tight")
    print("  saved 03_boundary_conditions.png")

    # 04: anisotropic prior — 6 samples + variance + correlation
    print("04  Anisotropic prior: samples, variance, correlation")
    mesha = create_unit_square(MPI.COMM_WORLD, 64, 64, CellType.triangle)
    Vha = V(mesha)
    Theta = anis_tensor(theta0=2.0, theta1=0.5, alpha_angle=math.pi / 4)
    pa = MaternPrior(Vha, gamma, delta, alpha=2, Theta=Theta,
                     robin_bc=True, seed=3)

    ss = pa.samples(6)
    pv = pa.pointwise_variance_randomized(r=200)
    cs = pa.correlation_structure((0.5, 0.5))

    fig, axes = plt.subplots(2, 4, figsize=(4.2 * 4, 3.6 * 2))
    fig.suptitle("Anisotropic BiLaplacian prior "
                 "(theta0=2, theta1=0.5, angle=45 deg)", fontsize=14)
    amax = max(np.abs(s).max() for s in ss)
    for i in range(6):
        r, c = divmod(i, 4)
        _field_on_ax(fig, axes[r, c], ss[i], Vha, f"sample {i + 1}",
                     cmap="RdBu_r", vmin=-amax, vmax=amax,
                     add_cbar=(c == 3 or i == 5))
    _field_on_ax(fig, axes[1, 2], pv, Vha, "pointwise variance",
                 cmap="jet")
    _field_on_ax(fig, axes[1, 3], cs, Vha, "correlation structure\n(centre)",
                 cmap="jet")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("04_anisotropic.png", dpi=150, bbox_inches="tight")
    print("  saved 04_anisotropic.png")

    print("\nDone. Wrote 00_meshes.png, 01_laplacian.png, "
          "02_bilaplacian.png, 03_boundary_conditions.png, "
          "04_anisotropic.png")
    # plt.show()  # uncomment to also display interactively


if __name__ == "__main__":
    main()
