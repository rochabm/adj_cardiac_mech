"""
ex07_uq.py
----------
Bayesian UQ (Laplace approximation) for the cardiac passive-filling inverse
problem, one test problem at a time:

    python ex07_uq.py --case linear
    python ex07_uq.py --case fibrosis

Same forward model, data, noise and calibrated priors as ex06 (m = log C).
MAP by inexact Newton-CG only.

Pipeline (Bui-Thanh et al. 2013; hIPPYlib, Villa et al.):
  1. MAP point  m_MAP = argmin J(m)  (Newton-CG; or loaded with --map-from)
  2. Generalized eigenproblem of the prior-preconditioned data-misfit Hessian
         H_misfit v = lambda H_prior v ,   V^T H_prior V = I
     by a randomized double-pass algorithm (k + p Hessian applies, twice).
  3. Laplace approximation of the posterior
         Gamma_post = (H_misfit + H_prior)^{-1}
                    ~ Gamma_prior - V diag(lambda/(1+lambda)) V^T
     -> pointwise posterior variance, variance reduction.
  4. Posterior samples  m = m_MAP + (I - V diag(1 - 1/sqrt(1+lambda)) V^T H_prior) z,
     z ~ N(0, Gamma_prior); the matching prior samples m0 + z are saved too.

Discrete prior (as ex06):  H_prior = R M_L^{-1} R,  Gamma_prior = R^{-1} M_L R^{-1},
R = gamma K + delta M + beta M_bndry,  beta = sqrt(gamma delta)/1.42,  M_L lumped.

OUTPUTS in <output-dir>/<case>/:
    uq_<case>.xdmf/.h5     m and C fields: truth, MAP, prior/posterior std,
                           variance reduction, eigenvectors, prior and
                           posterior samples
    uq_<case>.npz          arrays (eigenvalues, eigenvectors, variances, samples)
    eigenvalues_<case>.txt, eigenvalues_<case>.png
    metrics_uq_<case>.json
    m_map_<case>.npy       (reusable with --map-from)

Serial only.
"""

from mpi4py import MPI
import argparse
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import petsc4py.PETSc as PETSc
from scipy.linalg import solve_triangular

import ufl
import dolfinx
import dolfinx.fem.petsc
from dolfinx import fem, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem, LinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import XDMFFile

import cardiac_geometries
from ex03_ventricle_discrete_forward import prob_ventricle_passive_filling
from cardiac_utils import select_distributed_nodes, save_selected_nodes
from hessian_ucq import HessianOperator
from newton_cg_solver import inexact_newton_cg

if MPI.COMM_WORLD.size > 1:
    raise RuntimeError("ex07 is serial-only.")

# calibrated priors (gamma, delta) per test problem -- same as the noise study
PRIOR_DEFAULTS = {"linear": (6.22, 9.85e-3), "fibrosis": (1.457, 1.441e-2)}

# =============================================================================
# command-line arguments
# =============================================================================

parser = argparse.ArgumentParser(description="ex07 -- Laplace-approximation UQ (Newton-CG MAP)")
parser.add_argument("--case", choices=["linear", "fibrosis"], required=True)
parser.add_argument("--num-nodes", type=int, default=64)
# prior (defaults: calibrated values of the chosen case)
parser.add_argument("--gamma", type=float, default=None)
parser.add_argument("--delta", type=float, default=None)
parser.add_argument("--prior-fiber-ratio", type=float, default=1.0)
parser.add_argument("--C0", type=float, default=2.0, help="Prior mean in kPa (m0 = log C0)")
# data noise and likelihood (as ex06)
parser.add_argument("--data-noise", type=float, default=0.01,
                    help="Relative data noise p (default 0.01 = 1%%)")
parser.add_argument("--data-noise-seed", type=int, default=1234)
parser.add_argument("--match-noise-std", action=argparse.BooleanOptionalAction, default=True,
                    help="Likelihood std = actual data-noise std (default on)")
parser.add_argument("--noise-std", type=float, default=1e-3,
                    help="Likelihood std if not matched (or if --data-noise 0)")
# forward problem
parser.add_argument("--load", type=float, default=-10.0)
parser.add_argument("--kappa", type=float, default=1e2)
parser.add_argument("--load-steps", type=int, default=10)
parser.add_argument("--newton-maxit", type=int, default=50)
# MAP (Newton-CG)
parser.add_argument("--map-from", type=str, default=None,
                    help="Load the MAP (m = log C, .npy) instead of solving for it")
parser.add_argument("--gtol", type=float, default=1e-6)
parser.add_argument("--ftol", type=float, default=1e-12)
parser.add_argument("--maxiter", type=int, default=100)
parser.add_argument("--cg-maxiter", type=int, default=200)
parser.add_argument("--cg-tol", type=float, default=0.1)
parser.add_argument("--C-bounds", type=float, nargs=2, default=[1.0, 16.0])
parser.add_argument("--C-init-range", type=float, nargs=2, default=[2.0, 10.0])
parser.add_argument("--init-seed", type=int, default=57)
# eigenproblem / UQ
parser.add_argument("--k-eig", type=int, default=100, help="Eigenpairs to keep")
parser.add_argument("--oversampling", type=int, default=20)
parser.add_argument("--eig-seed", type=int, default=0)
parser.add_argument("--check-eig", type=int, default=5,
                    help="Residual check of the first N eigenpairs (0 = skip)")
parser.add_argument("--eigvec-ids", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32],
                    help="Eigenvectors (1-based) written to the XDMF")
parser.add_argument("--n-samples", type=int, default=4)
parser.add_argument("--sample-seed", type=int, default=99)
parser.add_argument("--verbose-forward", action="store_true")
parser.add_argument("--output-dir", type=str, default="out_ex07")
args = parser.parse_args()

CASE = args.case
if args.gamma is None:
    args.gamma = PRIOR_DEFAULTS[CASE][0]
if args.delta is None:
    args.delta = PRIOR_DEFAULTS[CASE][1]
if args.noise_std <= 0:
    parser.error("--noise-std must be > 0")

OUT = Path(args.output_dir) / CASE
OUT.mkdir(parents=True, exist_ok=True)
LO, HI = np.log(args.C_bounds[0]), np.log(args.C_bounds[1])
M0 = float(np.log(args.C0))


# =============================================================================
# BiLaplacian prior
# =============================================================================

class BiLaplacianPrior:
    """
    H_prior = R M_L^{-1} R,   Gamma_prior = H_prior^{-1} = R^{-1} M_L R^{-1},
    R = gamma K_Theta + delta M + beta M_bndry,  beta = sqrt(gamma delta)/1.42.
    """

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
        """H_prior v = R M_L^{-1} R v."""
        return self._R_mult(self._R_mult(v) / self.M_lump)

    def solve(self, v):
        """Gamma_prior v = H_prior^{-1} v = R^{-1} M_L R^{-1} v."""
        return self._R_solve(self.M_lump * self._R_solve(v))

    def sample(self, rng):
        """z ~ N(0, Gamma_prior):  z = R^{-1} M_L^{1/2} w,  w ~ N(0, I)."""
        return self._R_solve(np.sqrt(self.M_lump) * rng.standard_normal(self.ndofs))

    def pointwise_variance(self):
        """Exact diag(Gamma_prior) = sum_k (R^{-1})_{ik}^2 M_L,k  (n back-solves)."""
        n = self.ndofs
        Y = np.empty((n, n))
        e = np.zeros(n)
        for j in range(n):
            e[j] = 1.0
            Y[:, j] = self._R_solve(e)
            e[j] = 0.0
        return (Y ** 2) @ self.M_lump


# =============================================================================
# geometry, forward model (identical to ex06)
# =============================================================================

geodir = Path("lv_ellipsoid")
try:
    geo = cardiac_geometries.geometry.Geometry.from_folder(geodir)
except Exception:
    geo = cardiac_geometries.mesh.lv_ellipsoid(
        outdir=geodir, create_fibers=True, fiber_space="DG_0",
        psize_ref=3, r_short_epi=10, fiber_angle_endo=40.0, fiber_angle_epi=-50.0)

domain = geo.mesh
dim = domain.geometry.dim
V = fem.functionspace(domain, ("Lagrange", 1, (dim,)))
Va = fem.functionspace(domain, ("Lagrange", 1))
du, v = ufl.TrialFunction(V), ufl.TestFunction(V)
uh = fem.Function(V)
facet_tags = geo.ffun
base_dofs = fem.locate_dofs_topological(V, facet_tags.dim, facet_tags.find(5))
bcs = [fem.dirichletbc(np.zeros(dim, dtype=default_scalar_type), base_dofs, V)]

selected_dofs, selected_coords = select_distributed_nodes(
    domain.geometry.x, N=args.num_nodes, seed=42)
Npoints = len(selected_dofs)
save_selected_nodes(selected_coords, selected_dofs)

# ex03 selects its sensors from the P1 *dof* coordinates and evaluates F_d in the
# first cell of each index; here they come from the *geometry* nodes. Both use the
# same cells only if the two selections give identical indices -- check it exactly.
_sel_ex03, _ = select_distributed_nodes(
    V.tabulate_dof_coordinates().reshape(-1, dim), N=args.num_nodes, min_dist=4.0, seed=42)
if not np.array_equal(np.asarray(_sel_ex03), np.asarray(selected_dofs)):
    raise RuntimeError("ex03 and ex07 select different sensor nodes (dof ordering differs "
                       "from geometry ordering) -- the data would be compared in the wrong cells.")
print(f"  sensor check: ex03 and ex07 use the same {Npoints} sensor nodes")

I = ufl.variable(ufl.Identity(dim))
F = ufl.variable(I + ufl.grad(uh))
J = ufl.variable(ufl.det(F))
e1, e2, e3 = geo.f0, geo.s0, geo.n0
bf, bt, bfs = default_scalar_type(6.6), default_scalar_type(4.0), default_scalar_type(2.6)
kappa = fem.Constant(domain, float(args.kappa))
m_fun = fem.Function(Va, name="m")
Es = 0.5 * (J ** (-2 / 3) * F.T * F - I)
E = [[ufl.inner(Es * a, b) for b in (e1, e2, e3)] for a in (e1, e2, e3)]
Q = (bf * E[0][0] ** 2
     + bt * (E[1][1] ** 2 + E[2][2] ** 2 + E[1][2] ** 2 + E[2][1] ** 2)
     + bfs * (E[0][1] ** 2 + E[1][0] ** 2 + E[0][2] ** 2 + E[2][0] ** 2))
strain_energy = ufl.exp(m_fun) / 2.0 * (ufl.exp(Q) - 1) + kappa * (J * ufl.ln(J) - J + 1)
P = ufl.diff(strain_energy, F)
p_endo = fem.Constant(domain, 0.0)
metadata = {"quadrature_degree": 4}
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags, metadata=metadata)
dx = ufl.Measure("dx", domain=domain, metadata=metadata)
N_fac = ufl.FacetNormal(domain)
Fun = (ufl.inner(P, ufl.grad(v)) * dx
       - p_endo * ufl.inner(v, J * ufl.transpose(ufl.inv(F)) * N_fac) * ds(6))


def solve_nl_prob(u_func):
    problem = NonlinearProblem(Fun, u_func, bcs)
    solver = NewtonSolver(domain.comm, problem)
    solver.atol = 1e-8
    solver.rtol = 1e-8
    solver.max_it = args.newton_maxit
    ksp = solver.krylov_solver
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    for step, load in enumerate(np.linspace(0, args.load, args.load_steps + 1)[1:]):
        p_endo.value = load
        its, converged = solver.solve(u_func)
        if not converged:
            raise RuntimeError(f"forward Newton failed at load step {step}")
        if args.verbose_forward:
            print(f"  load step {step}: load={load:.3f}, newton its {its}")


# ---- sensors, data, likelihood (pointwise sum over sensors) -----------------
node_to_cells = {}
for cell_id in range(domain.topology.index_map(domain.topology.dim).size_local):
    for node in domain.geometry.dofmap[cell_id]:
        node_to_cells.setdefault(int(node), []).append(cell_id)
sensor_cells = [node_to_cells[int(d)][0] for d in selected_dofs]
if len(set(sensor_cells)) != len(sensor_cells):
    raise RuntimeError("some sensors share a cell; reduce --num-nodes.")

TT = fem.functionspace(domain, ("DG", 0, (dim, dim)))
W0 = fem.functionspace(domain, ("DG", 0))
Fd_func = fem.Function(TT, name="Fd_observed")
bs = TT.dofmap.bs
cell_vol = fem.Function(W0)
fem.assemble_vector(cell_vol.x.array, fem.form(ufl.TestFunction(W0) * dx))
sensor_weight = fem.Function(W0)
for cell in sensor_cells:
    c = W0.dofmap.cell_dofs(cell)
    sensor_weight.x.array[c] = 1.0 / cell_vol.x.array[c]
sensor_weight.x.scatter_forward()

noise_prec = fem.Constant(domain, 1.0)          # set below to 1/sigma_lik^2
Phi = 0.5 * noise_prec * sensor_weight * ufl.inner(F - Fd_func, F - Fd_func) * dx
Phi_c = fem.form(Phi)

if args.prior_fiber_ratio != 1.0:
    _f0n = geo.f0 / ufl.sqrt(ufl.inner(geo.f0, geo.f0))
    Theta = ufl.Identity(dim) + (args.prior_fiber_ratio ** 2 - 1.0) * ufl.outer(_f0n, _f0n)
else:
    Theta = None
prior = BiLaplacianPrior(Va, args.gamma, args.delta, dx, domain, theta=Theta)
n = prior.ndofs
m0_vec = np.full(n, M0)

lu_opts = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
lmbda = fem.Function(V)
adj_problem = LinearProblem(ufl.adjoint(ufl.derivative(Fun, uh, du)),
                            -ufl.derivative(Phi, uh, v), bcs=bcs, petsc_options=lu_opts)
dFdm_c = fem.form(ufl.action(ufl.adjoint(ufl.derivative(Fun, m_fun, ufl.TrialFunction(Va))),
                             lmbda))
grad_misfit = fem.Function(Va)


class MAPObjective:
    def __init__(self):
        self.x, self.n_forward, self.n_hessvec = None, 0, 0

    def _update(self, x):
        if self.x is not None and np.array_equal(x, self.x):
            return
        self.x = None
        m_fun.x.array[:] = x
        m_fun.x.scatter_forward()
        uh.x.array[:] = 0.0
        solve_nl_prob(uh)
        self.n_forward += 1
        self.Jd = fem.assemble_scalar(Phi_c)
        lam = adj_problem.solve()
        lmbda.x.array[:] = lam.x.array
        lmbda.x.scatter_forward()
        grad_misfit.x.array[:] = 0.0
        fem.assemble_vector(grad_misfit.x.array, dFdm_c)
        dm = x - m0_vec
        Hp_dm = prior.mult(dm)
        self.Jp = 0.5 * float(dm @ Hp_dm)
        self.J = self.Jd + self.Jp
        self.g = grad_misfit.x.array.copy() + Hp_dm
        self.x = x.copy()

    def eval_J(self, x):
        self._update(x)
        return self.J

    def eval_grad(self, x):
        self._update(x)
        return self.g.copy()

    def misfit_hessian(self):
        """Data-misfit Hessian at the current point (uh, m_fun, lmbda consistent)."""
        return HessianOperator(Fun, Phi, uh, m_fun, lmbda, V, Va, facet_tags, domain)

    def build_hop(self):
        Hm, obj = self.misfit_hessian(), self

        class _Full:
            ndofs_m = Hm.ndofs_m

            def mult(self, x):
                obj.n_hessvec += 1
                return Hm.mult(x) + prior.mult(x)
        return _Full()


obj = MAPObjective()


# =============================================================================
# randomized double-pass generalized eigensolver (hIPPYlib doublePassG)
# =============================================================================

def b_orthonormalize(Y, B_mult):
    """Q = Y L^{-T} with L L^T = Y^T B Y  ->  Q^T B Q = I   (CholQR, applied twice)."""
    for _ in range(2):
        BY = np.column_stack([B_mult(Y[:, i]) for i in range(Y.shape[1])])
        G = 0.5 * (Y.T @ BY + (Y.T @ BY).T)
        L = np.linalg.cholesky(G + 1e-14 * np.trace(G) / len(G) * np.eye(len(G)))
        Y = solve_triangular(L, Y.T, lower=True).T
    return Y


def double_pass_generalized(A_mult, B_mult, Binv, n, k, p, rng):
    """
    Largest k eigenpairs of  A u = lambda B u  (A = H_misfit, B = H_prior),
    U B-orthonormal. Cost: 2 (k + p) applications of A.
    """
    Omega = rng.standard_normal((n, k + p))
    Y = np.empty_like(Omega)
    for i in range(k + p):
        Y[:, i] = Binv(A_mult(Omega[:, i]))
        if (i + 1) % 10 == 0:
            print(f"    pass 1: {i + 1}/{k + p}")
    Qm = b_orthonormalize(Y, B_mult)
    AQ = np.empty_like(Qm)
    for i in range(k + p):
        AQ[:, i] = A_mult(Qm[:, i])
        if (i + 1) % 10 == 0:
            print(f"    pass 2: {i + 1}/{k + p}")
    T = Qm.T @ AQ
    T = 0.5 * (T + T.T)
    lam, W = np.linalg.eigh(T)
    order = np.argsort(lam)[::-1][:k]
    return lam[order], Qm @ W[:, order]


# =============================================================================
# output helpers
# =============================================================================

def va_function(name, arr):
    f = fem.Function(Va, name=name)
    f.x.array[:] = arr
    f.x.scatter_forward()
    return f


def plot_eigenvalues(lam, path, title):
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "Times", "Nimbus Roman",
                                        "TeX Gyre Termes", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 10})
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    idx = np.arange(1, len(lam) + 1)
    pos = lam > 0
    ax.semilogy(idx[pos], lam[pos], "o", ms=3.5, color="#1f3b8c")
    ax.axhline(1.0, color="k", lw=1.0)
    ax.set_xlabel("index $i$")
    ax.set_ylabel(r"eigenvalue $\lambda_i$")
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, len(lam) + 1)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# =============================================================================
# 1. data + MAP
# =============================================================================

t_all = time.perf_counter()
print("\n" + "=" * 70)
print(f" ex07 UQ  |  case = {CASE}")
print(f"  prior: gamma={prior.gamma:.4g}, delta={prior.delta:.4g}, beta={prior.beta:.4g}, "
      f"m0=log({args.C0}), fibre ratio={args.prior_fiber_ratio:g}")
print("=" * 70)

ud_arr, Fd_arr, cd = prob_ventricle_passive_filling(
    geo, ndofs_data=args.num_nodes, case_type=CASE,
    target_load=args.load, kappa_bulk=args.kappa)
Fd_clean = np.asarray(Fd_arr, dtype=float)
if Fd_clean.shape[0] != Npoints:
    raise RuntimeError("ex03 and ex07 selected a different number of sensors.")
F_scale = float(np.sqrt(np.mean((Fd_clean - np.eye(dim)) ** 2)))
sigma_data = args.data_noise * F_scale
eps = np.random.default_rng(args.data_noise_seed).standard_normal(Fd_clean.shape)
Fd_obs = Fd_clean + sigma_data * eps
sigma_lik = sigma_data if (args.match_noise_std and sigma_data > 0) else args.noise_std
noise_prec.value = 1.0 / sigma_lik ** 2
Fd_func.x.array[:] = 0.0
for j, cell in enumerate(sensor_cells):
    c = TT.dofmap.cell_dofs(cell)[0]
    Fd_func.x.array[c * bs:(c + 1) * bs] = Fd_obs[j].flatten()
Fd_func.x.scatter_forward()
print(f"  data noise {100 * args.data_noise:g}% -> std {sigma_data:.3e}; "
      f"likelihood std {sigma_lik:.3e}; N_obs = {Fd_obs.size}")

C_true = cd.x.array.copy()
m_true = np.log(C_true)

if args.map_from:
    m_map = np.load(args.map_from)
    if m_map.shape != (n,):
        raise RuntimeError(f"--map-from has shape {m_map.shape}, expected ({n},)")
    print(f"  MAP loaded from {args.map_from}")
    map_info = dict(source=args.map_from)
else:
    print("\n--- MAP by inexact Newton-CG ---")
    x0 = np.clip(np.log(np.random.RandomState(args.init_seed).uniform(*args.C_init_range, n)),
                 LO, HI)
    t0 = time.perf_counter()
    m_map, nhist = inexact_newton_cg(
        obj.eval_J, obj.eval_grad, obj.build_hop, x0=x0,
        max_outer_iter=args.maxiter, grad_tol=args.gtol, ftol=args.ftol,
        cg_tol=args.cg_tol, cg_maxiter=args.cg_maxiter,
        bounds=(LO, HI), eisenstat_walker=True, verbose=False)
    map_info = dict(source="newtoncg", wall_time_s=time.perf_counter() - t0,
                    n_forward=obj.n_forward, n_hessvec=obj.n_hessvec,
                    stop_reason=nhist[-1].get("stopped_reason", "max_outer_iter"))
    print(f"  MAP: {map_info}")
np.save(OUT / f"m_map_{CASE}.npy", m_map)

J_map = obj.eval_J(m_map)          # also sets uh, m_fun, lmbda at the MAP
g_map = obj.eval_grad(m_map)
n_bound = int(np.sum((m_map <= LO + 1e-10) | (m_map >= HI - 1e-10)))
print(f"  J(MAP) = {J_map:.4e} (Phi = {obj.Jd:.4e}),  ||g||_inf = {np.abs(g_map).max():.2e},"
      f"  nodes at bounds: {n_bound}")
if n_bound:
    print("  WARNING: active bounds -> the Laplace approximation is not exact at those nodes.")

# =============================================================================
# 2. generalized eigenproblem  H_misfit v = lambda H_prior v  at the MAP
# =============================================================================

print(f"\n--- generalized eigenproblem (k = {args.k_eig}, p = {args.oversampling}) ---")
Hm = obj.misfit_hessian()
n_hv = [0]


def A_mult(x):
    n_hv[0] += 1
    return Hm.mult(x)


t0 = time.perf_counter()
k = min(args.k_eig, n - args.oversampling)
lam, U = double_pass_generalized(A_mult, prior.mult, prior.solve, n, k, args.oversampling,
                                 np.random.default_rng(args.eig_seed))
t_eig = time.perf_counter() - t0
n_neg = int(np.sum(lam < 0))
if n_neg:
    print(f"  {n_neg} negative eigenvalues (min {lam.min():.2e}) set to 0 "
          "(indefinite full misfit Hessian / round-off)")
lam = np.maximum(lam, 0.0)
print(f"  done in {t_eig:.0f} s, {n_hv[0]} Hessian applies")
print(f"  lambda_1 = {lam[0]:.3e},  lambda_k = {lam[-1]:.3e}")
print(f"  # lambda > 1: {int(np.sum(lam > 1))},  sum lambda/(1+lambda) = "
      f"{np.sum(lam / (1 + lam)):.1f}  (effective data-informed parameters)")
if lam[-1] > 0.1:
    print(f"  WARNING: lambda_k = {lam[-1]:.2f} > 0.1 -- spectrum truncated too early; "
          "increase --k-eig.")

eig_check = []
for i in range(min(args.check_eig, k)):
    Au, Bu = A_mult(U[:, i]), prior.mult(U[:, i])
    res = np.linalg.norm(Au - lam[i] * Bu) / max(np.linalg.norm(lam[i] * Bu), 1e-300)
    eig_check.append(float(res))
if eig_check:
    print(f"  eigen-residuals ||A u - lambda B u|| / ||lambda B u|| (first {len(eig_check)}): "
          + ", ".join(f"{r:.1e}" for r in eig_check))
UtBU = U.T @ np.column_stack([prior.mult(U[:, i]) for i in range(min(k, 10))])
print(f"  B-orthonormality (first 10): max|U^T B U - I| = "
      f"{np.abs(UtBU[:UtBU.shape[1]] - np.eye(UtBU.shape[1])).max():.1e}")

np.savetxt(OUT / f"eigenvalues_{CASE}.txt", lam, header="generalized eigenvalues, descending")
plot_eigenvalues(lam, OUT / f"eigenvalues_{CASE}.png",
                 f"{'Case 1 (linear)' if CASE == 'linear' else 'Case 2 (fibrosis)'}, "
                 f"{100 * args.data_noise:g}% noise")

# =============================================================================
# 3. Laplace approximation: pointwise variance, variance reduction
# =============================================================================

print("\n--- posterior (Laplace approximation) ---")
d = lam / (1.0 + lam)
prior_var = prior.pointwise_variance()
post_var = prior_var - (U ** 2) @ d
n_negvar = int(np.sum(post_var < 0))
post_var = np.maximum(post_var, 0.0)
prior_std, post_std = np.sqrt(prior_var), np.sqrt(post_var)
var_red = 1.0 - post_var / prior_var
print(f"  prior std of m:     [{prior_std.min():.3f}, {prior_std.max():.3f}]")
print(f"  posterior std of m: [{post_std.min():.3f}, {post_std.max():.3f}]"
      + (f"  ({n_negvar} negative variances clipped)" if n_negvar else ""))
print(f"  variance reduction: mean {var_red.mean():.3f}, "
      f"range [{var_red.min():.3f}, {var_red.max():.3f}]")

# calibration check: is the truth inside the pointwise 95% credible band?
z_post = np.abs(m_true - m_map) / np.maximum(post_std, 1e-12)
z_prior = np.abs(m_true - M0) / prior_std
cover_post, cover_prior = float(np.mean(z_post <= 1.96)), float(np.mean(z_prior <= 1.96))
print(f"  truth inside pointwise 95% band: posterior {100 * cover_post:.1f}% of nodes, "
      f"prior {100 * cover_prior:.1f}%")

# regional summary (fibrosis geometry from ex03)
xyz = Va.tabulate_dof_coordinates()
r = np.linalg.norm(xyz - np.array([-5.0, -2.0, -9.0]), axis=1)
regions = {"core": r <= 5.0, "transition": (r > 5.0) & (r < 10.0), "healthy": r >= 10.0}
region_stats = {name: dict(n=int(mask.sum()),
                           post_std=float(post_std[mask].mean()),
                           var_reduction=float(var_red[mask].mean()))
                for name, mask in regions.items() if mask.any()}

# =============================================================================
# 4. samples: prior z ~ N(0, Gamma_prior) and matching posterior samples
# =============================================================================

print(f"\n--- {args.n_samples} prior / posterior samples ---")
rng_s = np.random.default_rng(args.sample_seed)
dp = 1.0 - 1.0 / np.sqrt(1.0 + lam)
prior_samples, post_samples = [], []
for s in range(args.n_samples):
    z = prior.sample(rng_s)
    zp = z - U @ (dp * (U.T @ prior.mult(z)))
    prior_samples.append(m0_vec + z)
    post_samples.append(m_map + zp)
    print(f"  sample {s + 1}: posterior C in [{np.exp(post_samples[-1]).min():.2f}, "
          f"{np.exp(post_samples[-1]).max():.2f}] kPa")

# =============================================================================
# outputs
# =============================================================================

C_map = np.exp(m_map)
fields = [va_function("m_true", m_true), va_function("C_true", C_true),
          va_function("m_map", m_map), va_function("C_map", C_map),
          va_function("C_rel_error", np.abs(C_map - C_true) / C_true),
          va_function("m_prior_std", prior_std), va_function("m_post_std", post_std),
          va_function("variance_reduction", var_red),
          va_function("credible_z", z_post)]
for i in args.eigvec_ids:
    if 1 <= i <= k:
        u = U[:, i - 1]
        fields.append(va_function(f"eigvec_{i:03d}", u / np.abs(u).max()))
for s, (mp, mq) in enumerate(zip(prior_samples, post_samples), start=1):
    fields += [va_function(f"m_prior_sample_{s}", mp), va_function(f"C_prior_sample_{s}", np.exp(mp)),
               va_function(f"m_post_sample_{s}", mq), va_function(f"C_post_sample_{s}", np.exp(mq))]
with XDMFFile(MPI.COMM_WORLD, str(OUT / f"uq_{CASE}.xdmf"), "w") as xf:
    xf.write_mesh(domain)
    for f in fields:
        xf.write_function(f)

np.savez(OUT / f"uq_{CASE}.npz", eigvals=lam, eigvecs=U, prior_var=prior_var,
         post_var=post_var, m_map=m_map, m_true=m_true,
         prior_samples=np.array(prior_samples), post_samples=np.array(post_samples))

metrics = dict(
    case=CASE, settings=vars(args),
    prior=dict(gamma=prior.gamma, delta=prior.delta, beta=prior.beta, m0=M0),
    data=dict(sigma_data=sigma_data, sigma_lik=sigma_lik, n_obs=int(Fd_obs.size)),
    map=map_info | dict(J=float(J_map), Phi=float(obj.Jd), n_at_bounds=n_bound,
                        C_rel_L2_nodal=float(np.linalg.norm(C_map - C_true) /
                                             np.linalg.norm(C_true))),
    eig=dict(k=int(k), p=args.oversampling, n_hessian_applies=n_hv[0], wall_time_s=t_eig,
             lambda_first=float(lam[0]), lambda_last=float(lam[-1]),
             n_gt_1=int(np.sum(lam > 1)), n_gt_0p1=int(np.sum(lam > 0.1)),
             eff_dof=float(np.sum(d)), n_negative_clipped=n_neg, residuals=eig_check),
    posterior=dict(prior_std_range=[float(prior_std.min()), float(prior_std.max())],
                   post_std_range=[float(post_std.min()), float(post_std.max())],
                   var_reduction_mean=float(var_red.mean()),
                   coverage95_post=cover_post, coverage95_prior=cover_prior,
                   n_negative_variance_clipped=n_negvar, regions=region_stats),
    total_wall_time_s=time.perf_counter() - t_all,
)
with open(OUT / f"metrics_uq_{CASE}.json", "w") as f:
    json.dump(metrics, f, indent=2, default=float)

print(f"\nDone in {metrics['total_wall_time_s']:.0f} s. Outputs in {OUT.resolve()}")
