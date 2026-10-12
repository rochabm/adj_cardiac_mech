"""
ex06_map_comparison.py
----------------------
MAP-only comparison of L-BFGS-B and inexact Newton-CG for the cardiac
passive-filling inverse problem, linear (Case 1) and fibrosis (Case 2) cases.

PARAMETER:  log-stiffness  m = log C   (C in kPa), so the passive energy is
    W = exp(m)/2 * (exp(Q) - 1) + kappa (J ln J - J + 1)

MAP PROBLEM (negative log-posterior):
    m_MAP = argmin_m  J(m) = Phi(m) + 0.5 || A (m - m0) ||^2_{L2}

    Phi(m)  = 1/(2 sigma_noise^2) * sum_j || F_j(m) - Fd_j ||_F^2     (sensors j)
    A       = -gamma Lap + delta I,  Robin  gamma grad(m).n + beta m = 0,
              beta = sqrt(gamma delta)/1.42
    discrete: 0.5 ||A(m-m0)||^2 = 0.5 (m-m0)^T R M_L^{-1} R (m-m0),
              R = gamma K + delta M + beta M_bndry,  M_L lumped mass.

    Default prior = calibrated values for the thin LV wall
        gamma = 1.457, delta = 1.441e-2  (-> beta = 0.102),  m0 = log 2
    (sigma = log(3)/2 = 0.549, rho = 11.6 mm; see the calibration text).

SYNTHETIC DATA (--data-model):
    p2 (default): the forward problem is solved on P2 displacement with the true
        C represented in P2 (ex03_ventricle_p2_forward.py). At each sensor
        cell the observed F is obtained from the P2 solution, either
          --data-F p1interp (default): u2 interpolated to the P1 nodes of the
                    mesh, F = I + grad(I_h u2) in the sensor cell, or
          --data-F p2avg: cell average of F = I + grad(u2) over the sensor cell.
        The P1 inversion model then cannot reproduce the data exactly even at
        the true field (mitigates the inverse crime; Bui-Thanh et al. 2013).
    p1: original ex03 (same P1 model for data and inversion = inverse crime).

Both methods start from the SAME initial guess, use the same J/gradient,
the same bounds (default C in [1, 16] kPa) and the same gtol/ftol.

OUTPUTS per case in <output-dir>/<case>/:
    map_comparison_<case>.xdmf/.h5   m and C fields (true, init, L-BFGS-B,
                                     Newton-CG), errors, displacements,
                                     prior pointwise std, sensors, Fd
    metrics_<case>.json              error metrics (in C and in m) + costs
    history_<method>_<case>.txt      per-iteration history
    convergence_<case>.png
and <output-dir>/summary.csv (one row per case x method).

Serial only.

Usage:
    python ex06_map_comparison.py
    python ex06_map_comparison.py --cases fibrosis --noise-std 1e-2
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
from scipy.optimize import minimize

import ufl
import dolfinx
import dolfinx.fem.petsc
from dolfinx import fem, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem, LinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import XDMFFile

import cardiac_geometries
from ex03_ventricle_discrete_forward import prob_ventricle_passive_filling
from ex03_ventricle_p2_forward import solve_forward_p2
from cardiac_utils import select_distributed_nodes, compute_Fh, save_selected_nodes
from hessian_ucq import HessianOperator
from newton_cg_solver import inexact_newton_cg

if MPI.COMM_WORLD.size > 1:
    raise RuntimeError("ex06 is serial-only (numpy indexing assumes no ghosts).")

# =============================================================================
# command-line arguments  (physical quantities given in C = kPa, converted to m)
# =============================================================================

parser = argparse.ArgumentParser(description="ex06 — MAP, m = log C: L-BFGS-B vs Newton-CG")
parser.add_argument("--cases", nargs="+", default=["linear", "fibrosis"],
                    choices=["linear", "fibrosis"])
parser.add_argument("--methods", nargs="+", default=["lbfgsb", "newtoncg"],
                    choices=["lbfgsb", "newtoncg"])
parser.add_argument("--num-nodes", type=int, default=64)
# prior (BiLaplacian on m); defaults = calibrated values
parser.add_argument("--gamma", type=float, default=1.457)
parser.add_argument("--delta", type=float, default=1.441e-2)
parser.add_argument("--C0",    type=float, default=2.0,
                    help="Prior mean in kPa (m0 = log C0); default 2 kPa")
# likelihood
parser.add_argument("--noise-std", type=float, default=1e-3,
                    help="Std of each measured F component (pointwise likelihood)")
parser.add_argument("--misfit-weight", choices=["pointwise", "legacy"], default="pointwise",
                    help="pointwise: sum over sensors; legacy: ex05's (1/|Omega|) integral")
# synthetic data
parser.add_argument("--data-model", choices=["p2", "p1"], default="p2",
                    help="p2: data from a P2 forward solve (default); "
                         "p1: original ex03 (inverse crime)")
parser.add_argument("--data-F", choices=["p1interp", "p2avg"], default="p1interp",
                    help="How F is observed from the P2 solution at a sensor cell: "
                         "p1interp = F of the P1 nodal interpolant of u2 (default); "
                         "p2avg = cell average of F(u2)")
parser.add_argument("--data-load-steps", type=int, default=10,
                    help="Load steps of the P2 data solve")
parser.add_argument("--data-quad-degree", type=int, default=4,
                    help="Quadrature degree of the P2 data solve")
# forward problem
parser.add_argument("--load",        type=float, default=-10.0)
parser.add_argument("--kappa",       type=float, default=1e2)
parser.add_argument("--load-steps",  type=int,   default=10)
parser.add_argument("--newton-maxit", type=int,  default=50)
# optimization
parser.add_argument("--gtol",       type=float, default=1e-8)
parser.add_argument("--ftol",       type=float, default=1e-20)
parser.add_argument("--maxiter",    type=int,   default=200,
                    help="Max outer iterations for BOTH methods")
parser.add_argument("--cg-maxiter", type=int,   default=200)
parser.add_argument("--cg-tol",     type=float, default=0.1)
parser.add_argument("--C-bounds",   type=float, nargs=2, default=[1.0, 16.0],
                    help="Bounds on C in kPa (applied to m as log)")
parser.add_argument("--C-init-range", type=float, nargs=2, default=[2.0, 10.0],
                    help="Initial guess C ~ U(a, b) kPa, m_init = log C")
parser.add_argument("--init-seed",  type=int,   default=57)
parser.add_argument("--no-prior-std", action="store_true",
                    help="Skip the exact prior pointwise-std field (n back-solves)")
parser.add_argument("--verbose-forward", action="store_true")
parser.add_argument("--output-dir", type=str, default="out_ex06")
args = parser.parse_args()

OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LO, HI = np.log(args.C_bounds[0]), np.log(args.C_bounds[1])   # bounds on m
M0 = float(np.log(args.C0))


# =============================================================================
# BiLaplacian prior on m
# =============================================================================

class BiLaplacianPrior:
    """
    H_prior = R M_L^{-1} R,   C_pr = H_prior^{-1} = R^{-1} M_L R^{-1},
    R = gamma K + delta M + beta M_bndry,  beta = sqrt(gamma delta)/1.42.
    """

    def __init__(self, Va, gamma, delta, dx, domain):
        self.ndofs = Va.dofmap.index_map.size_local * Va.dofmap.index_map_bs
        self.gamma, self.delta = float(gamma), float(delta)
        self.beta = np.sqrt(self.gamma * self.delta) / 1.42
        q, dq = ufl.TrialFunction(Va), ufl.TestFunction(Va)
        ds_all = ufl.Measure("ds", domain=domain)
        R_form = (self.gamma * ufl.inner(ufl.grad(q), ufl.grad(dq)) * dx
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
        self._M_lump = np.maximum(self._x.array.copy(), 1e-30)
        M.destroy()

        self._ksp = PETSc.KSP().create(domain.comm)
        self._ksp.setOperators(self._R)
        self._ksp.setType("preonly")
        self._ksp.getPC().setType("lu")
        self._ksp.getPC().setFactorSolverType("mumps")
        self._ksp.setUp()

        print(f"  BiLaplacianPrior on m=log C: gamma={self.gamma:.4g}, "
              f"delta={self.delta:.4g}, beta={self.beta:.4g}, ndofs={self.ndofs}")

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
        return self._R_mult(self._R_mult(v) / self._M_lump)

    def pointwise_std(self, max_dofs=6000):
        """Exact sqrt(diag C_pr), C_pr = R^{-1} M_L R^{-1} (consistent with mult)."""
        n = self.ndofs
        if n > max_dofs:
            return None
        Y = np.empty((n, n))
        e = np.zeros(n)
        for j in range(n):
            e[j] = 1.0
            Y[:, j] = self._R_solve(e)
            e[j] = 0.0
        return np.sqrt(np.abs((Y ** 2) @ self._M_lump))


# =============================================================================
# geometry, spaces, BCs
# =============================================================================

geodir = Path("lv_ellipsoid")
try:
    geo = cardiac_geometries.geometry.Geometry.from_folder(geodir)
    print(f"Loaded geometry from {geodir}")
except Exception:
    geo = cardiac_geometries.mesh.lv_ellipsoid(
        outdir=geodir, create_fibers=True, fiber_space="DG_0",
        psize_ref=3, r_short_epi=10,
        fiber_angle_endo=40.0, fiber_angle_epi=-50.0)
    print(f"Generated new geometry in {geodir}")

domain = geo.mesh
dim = domain.geometry.dim

V  = fem.functionspace(domain, ("Lagrange", 1, (dim,)))
Va = fem.functionspace(domain, ("Lagrange", 1))
du = ufl.TrialFunction(V)
v  = ufl.TestFunction(V)
uh = fem.Function(V)

facet_tags = geo.ffun
base_dofs = fem.locate_dofs_topological(V, facet_tags.dim, facet_tags.find(5))
bcs = [fem.dirichletbc(np.zeros(dim, dtype=default_scalar_type), base_dofs, V)]

# =============================================================================
# measurement nodes (geometry nodes, as in ex05)
# =============================================================================

selected_dofs, selected_coords = select_distributed_nodes(
    domain.geometry.x, N=args.num_nodes, seed=42)
Npoints = len(selected_dofs)
save_selected_nodes(selected_coords, selected_dofs)

# =============================================================================
# kinematics + Holzapfel-Ogden passive model, parameterized by m = log C
# =============================================================================

I = ufl.variable(ufl.Identity(dim))
F = ufl.variable(I + ufl.grad(uh))
J = ufl.variable(ufl.det(F))
e1, e2, e3 = geo.f0, geo.s0, geo.n0

bf, bt, bfs = default_scalar_type(6.6), default_scalar_type(4.0), default_scalar_type(2.6)
kappa = fem.Constant(domain, float(args.kappa))
m_fun = fem.Function(Va, name="m")          # the parameter: m = log C

Es = 0.5 * (J ** (-2 / 3) * F.T * F - I)
E = [[ufl.inner(Es * a, b) for b in (e1, e2, e3)] for a in (e1, e2, e3)]
Q = (bf * E[0][0] ** 2
     + bt * (E[1][1] ** 2 + E[2][2] ** 2 + E[1][2] ** 2 + E[2][1] ** 2)
     + bfs * (E[0][1] ** 2 + E[1][0] ** 2 + E[0][2] ** 2 + E[2][0] ** 2))
strain_energy = (ufl.exp(m_fun) / 2.0 * (ufl.exp(Q) - 1)
                 + kappa * (J * ufl.ln(J) - J + 1))
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
        its, converged = solver.solve(u_func)   # raises RuntimeError on failure
        if not converged:
            raise RuntimeError(f"forward Newton failed at load step {step}")
        if args.verbose_forward:
            print(f"  load step {step}: load={load:.3f}, newton its {its}")


# =============================================================================
# sensors, data containers, likelihood Phi(m)
# =============================================================================

volume_mesh = fem.assemble_scalar(fem.form(fem.Constant(domain, default_scalar_type(1.0)) * dx))

node_to_cells = {}
for cell_id in range(domain.topology.index_map(domain.topology.dim).size_local):
    for node in domain.geometry.dofmap[cell_id]:
        node_to_cells.setdefault(int(node), []).append(cell_id)
sensor_cells = [node_to_cells[int(d)][0] for d in selected_dofs]

TT = fem.functionspace(domain, ("DG", 0, (dim, dim)))
W0 = fem.functionspace(domain, ("DG", 0))
Fd_func = fem.Function(TT, name="Fd_reference")
indicator = fem.Function(W0, name="measurement_indicator")
bs = TT.dofmap.bs
for cell in sensor_cells:
    indicator.x.array[W0.dofmap.cell_dofs(cell)] = 1.0
indicator.x.scatter_forward()


def set_data(Fd_arr):
    if Fd_arr.shape[0] != Npoints:
        raise RuntimeError(f"ex03 returned {Fd_arr.shape[0]} F samples but {Npoints} "
                           "sensors were selected here -- node selection must match.")
    Fd_func.x.array[:] = 0.0
    for j, cell in enumerate(sensor_cells):
        c = TT.dofmap.cell_dofs(cell)[0]
        Fd_func.x.array[c * bs:(c + 1) * bs] = Fd_arr[j].flatten()
    Fd_func.x.scatter_forward()


# F = I + grad(u) is constant on each P1 cell, so weighting sensor cell j by
# 1/|cell_j| turns  int w ||F - Fd||^2 dx  into exactly  sum_j ||F_j - Fd_j||^2.
cell_vol = fem.Function(W0)
fem.assemble_vector(cell_vol.x.array, fem.form(ufl.TestFunction(W0) * dx))
sensor_weight = fem.Function(W0)
for cell in sensor_cells:
    c = W0.dofmap.cell_dofs(cell)
    sensor_weight.x.array[c] = (1.0 / cell_vol.x.array[c]
                                if args.misfit_weight == "pointwise" else 1.0 / volume_mesh)
sensor_weight.x.scatter_forward()
sensor_vols = np.array([cell_vol.x.array[W0.dofmap.cell_dofs(c)][0] for c in sensor_cells])


# -----------------------------------------------------------------------------
# Observation operator applied to an arbitrary displacement field u (P1 or P2):
# F = I + grad(u) in each sensor cell, returned as (Npoints, dim, dim).
#   "p1interp": u -> P1 nodal interpolant, then F in the cell (constant, exact)
#   "p2avg"   : cell average (1/|K|) int_K (I + grad u) dx  (L2 projection on DG0)
# Used for the data (from the P2 solution) and for diagnostics (P1 model at
# the true field), so data and model are observed in exactly the same cells
# and with the same DG0 layout as Fd_func in Phi.
# -----------------------------------------------------------------------------
_F_dg0 = fem.Function(TT)
_tau_TT = ufl.TestFunction(TT)
_dx_obs = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 4})


def observe_F(u_fun, mode="p1interp"):
    Id = ufl.Identity(dim)
    if mode == "p1interp":
        u1 = fem.Function(V)
        u1.interpolate(u_fun)
        _F_dg0.interpolate(fem.Expression(Id + ufl.grad(u1),
                                          TT.element.interpolation_points()))
        arr, scale = _F_dg0.x.array, None
    elif mode == "p2avg":
        vec = fem.assemble_vector(fem.form(ufl.inner(Id + ufl.grad(u_fun), _tau_TT) * _dx_obs))
        arr, scale = vec.array, True
    else:
        raise ValueError(mode)
    out = np.empty((Npoints, dim, dim))
    for j, cell in enumerate(sensor_cells):
        c = TT.dofmap.cell_dofs(cell)[0]
        vals = arr[c * bs:(c + 1) * bs].copy()
        if scale:
            vals /= sensor_vols[j]
        out[j] = vals.reshape(dim, dim)
    return out

noise_prec = fem.Constant(domain, 1.0 / args.noise_std ** 2)
diffF = F - Fd_func
Phi = 0.5 * noise_prec * sensor_weight * ufl.inner(diffF, diffF) * dx
Phi_c = fem.form(Phi)

# prior
prior = BiLaplacianPrior(Va, args.gamma, args.delta, dx, domain)
m0_vec = np.full(prior.ndofs, M0)

# adjoint (direct solver -> gradient accurate to round-off)
lu_opts = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
lmbda = fem.Function(V)
adj_problem = LinearProblem(ufl.adjoint(ufl.derivative(Fun, uh, du)),
                            -ufl.derivative(Phi, uh, v),
                            bcs=bcs, petsc_options=lu_opts)
q_trial = ufl.TrialFunction(Va)
# Phi has no explicit m-dependence: dPhi/dm = <lambda, dFun/dm>
dFdm_c = fem.form(ufl.action(ufl.adjoint(ufl.derivative(Fun, m_fun, q_trial)), lmbda))
grad_misfit = fem.Function(Va)


# =============================================================================
# objective with cache + cost counters
# =============================================================================

class FullHessian:
    """H v = H_misfit v + H_prior v  (counts Hessian-vector products)."""

    def __init__(self, Hmisfit, obj):
        self._Hm, self._obj = Hmisfit, obj
        self.ndofs_m = Hmisfit.ndofs_m

    def mult(self, x):
        self._obj.n_hessvec += 1
        return self._Hm.mult(x) + prior.mult(x)


class MAPObjective:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x = None
        self.J = self.Jd = self.Jp = None
        self.g = None
        self.n_forward = 0
        self.n_hessvec = 0

    def _update(self, x):
        # exact equality (np.allclose would return stale values for small steps)
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

    def build_hop(self):
        # Newton-CG calls this right after eval_J/eval_grad at the current x,
        # so uh, m_fun and lmbda are consistent with x
        Hm = HessianOperator(Fun, Phi, uh, m_fun, lmbda, V, Va, facet_tags, domain)
        return FullHessian(Hm, self)


obj = MAPObjective()


def proj_grad_inf(x, g):
    """scipy L-BFGS-B's stopping measure: ||P(x - g) - x||_inf (in m)."""
    return float(np.abs(np.clip(x - g, LO, HI) - x).max())


# =============================================================================
# optimizers
# =============================================================================

def run_lbfgsb(x0):
    obj.reset()
    hist = []
    t0 = time.perf_counter()
    J0, g0 = obj.eval_J(x0), obj.eval_grad(x0)
    hist.append(dict(iter=0, J=J0, pgrad=proj_grad_inf(x0, g0),
                     n_forward=obj.n_forward, n_hessvec=0, cg_iters=0))

    # ------------------------------------------------------------------
    # Forward-failure guard (Newton-CG's line search has the equivalent).
    # A trial m from the line search can make the hyperelastic forward solve
    # diverge. Instead of crashing, return a large FINITE J (scipy's line
    # search mishandles inf/nan) with the last good gradient: the function
    # increase forces the line search to shrink the step.
    # ------------------------------------------------------------------
    last_good = {"J": J0, "g": g0.copy()}
    failed = {"x": None, "n": 0}

    def J_safe(x):
        if failed["x"] is not None and np.array_equal(x, failed["x"]):
            return 1e3 * (1.0 + abs(last_good["J"]))
        try:
            J_val = obj.eval_J(x)
        except RuntimeError as e:
            failed["x"], failed["n"] = x.copy(), failed["n"] + 1
            obj.x = None
            print(f"  [L-BFGS-B] forward solve failed at trial point ({e}); "
                  f"returning penalty so the line search backtracks")
            return 1e3 * (1.0 + abs(last_good["J"]))
        last_good["J"], last_good["g"] = J_val, obj.g.copy()
        return J_val

    def grad_safe(x):
        if failed["x"] is not None and np.array_equal(x, failed["x"]):
            return last_good["g"].copy()
        J_safe(x)   # (cache hit if already evaluated)
        if failed["x"] is not None and np.array_equal(x, failed["x"]):
            return last_good["g"].copy()
        return obj.eval_grad(x)

    def callback(intermediate_result):
        x = intermediate_result.x
        g = obj.eval_grad(x)
        hist.append(dict(iter=len(hist), J=float(intermediate_result.fun),
                         pgrad=proj_grad_inf(x, g), n_forward=obj.n_forward,
                         n_hessvec=0, cg_iters=0))
        print(f"  [L-BFGS-B] it {len(hist)-1:3d}  J={intermediate_result.fun:.6e}  "
              f"|Pg|inf={hist[-1]['pgrad']:.3e}  fwd={obj.n_forward}")

    res = minimize(J_safe, x0.copy(), jac=grad_safe, method="L-BFGS-B",
                   bounds=[(LO, HI)] * len(x0), callback=callback,
                   options={"ftol": args.ftol, "gtol": args.gtol,
                            "maxiter": args.maxiter, "disp": False})
    wall = time.perf_counter() - t0
    info = dict(n_iter=int(res.nit), n_forward=obj.n_forward, n_hessvec=0,
                n_forward_failed=failed["n"],
                wall_time_s=wall, stop_reason=str(res.message), success=bool(res.success))
    return res.x.copy(), hist, info


def run_newtoncg(x0):
    obj.reset()
    snapshots = []   # eval_grad is called once at the top of every outer iteration

    def grad_logged(x):
        g = obj.eval_grad(x)
        snapshots.append((obj.n_forward, obj.n_hessvec, proj_grad_inf(x, g)))
        return g

    t0 = time.perf_counter()
    x_opt, nhist = inexact_newton_cg(
        obj.eval_J, grad_logged, obj.build_hop, x0=x0.copy(),
        max_outer_iter=args.maxiter, grad_tol=args.gtol, ftol=args.ftol,
        cg_tol=args.cg_tol, cg_maxiter=args.cg_maxiter,
        bounds=(LO, HI), eisenstat_walker=True, verbose=False)
    wall = time.perf_counter() - t0

    hist = []
    for h, (nf, nh, pg) in zip(nhist, snapshots):
        hist.append(dict(iter=h["iter"], J=h["J"], pgrad=pg, n_forward=nf,
                         n_hessvec=nh, cg_iters=h["cg_iters"]))
    stop = nhist[-1].get("stopped_reason", "max_outer_iter") if nhist else "none"
    n_steps = sum(1 for h in nhist if "stopped_reason" not in h)
    info = dict(n_iter=n_steps, n_forward=obj.n_forward, n_hessvec=obj.n_hessvec,
                wall_time_s=wall, stop_reason=stop, success=stop in ("gtol", "ftol"))
    return x_opt, hist, info


# =============================================================================
# metrics
# =============================================================================

_err_fun, _ref_fun = fem.Function(Va), fem.Function(Va)
_l2_err = fem.form(_err_fun ** 2 * dx)
_l2_ref = fem.form(_ref_fun ** 2 * dx)
_uerr_fun, _uref_fun = fem.Function(V), fem.Function(V)
_l2_uerr = fem.form(ufl.inner(_uerr_fun, _uerr_fun) * dx)
_l2_uref = fem.form(ufl.inner(_uref_fun, _uref_fun) * dx)

dof_xyz = Va.tabulate_dof_coordinates()   # Va dof ordering != geometry node ordering
XC, YC, ZC, R0, R1 = -5.0, -2.0, -9.0, 5.0, 10.0   # fibrosis geometry from ex03
r_fib = np.linalg.norm(dof_xyz - np.array([XC, YC, ZC]), axis=1)
regions = {"core": r_fib <= R0, "transition": (r_fib > R0) & (r_fib < R1),
           "healthy": r_fib >= R1}


def _l2(err, ref):
    _err_fun.x.array[:] = err
    _ref_fun.x.array[:] = ref
    return (np.sqrt(fem.assemble_scalar(_l2_err)), np.sqrt(fem.assemble_scalar(_l2_ref)))


def param_metrics(m_est, m_true):
    """Errors in C = exp(m) (kPa, what is reported) and in m = log C."""
    C_est, C_true = np.exp(m_est), np.exp(m_true)
    out = {}
    # C (kPa)
    l2e, l2r = _l2(C_est - C_true, C_true)
    abs_C = np.abs(C_est - C_true)
    rel_C = abs_C / C_true
    out.update(C_rel_L2=float(l2e / l2r), C_abs_L2=float(l2e),
               C_max_abs=float(abs_C.max()), C_mean_abs=float(abs_C.mean()),
               C_max_rel=float(rel_C.max()), C_mean_rel=float(rel_C.mean()))
    # m = log C  (|dm| is the log of the C ratio)
    l2e_m, l2r_m = _l2(m_est - m_true, m_true)
    abs_m = np.abs(m_est - m_true)
    out.update(m_rel_L2=float(l2e_m / max(l2r_m, 1e-30)), m_abs_L2=float(l2e_m),
               m_max_abs=float(abs_m.max()), m_mean_abs=float(abs_m.mean()))
    # active bounds: if > 0 the MAP is a constrained minimizer, not a
    # stationary point of J (matters for the Laplace approximation)
    tol_b = 1e-10
    out.update(n_at_lower_bound=int(np.sum(m_est <= LO + tol_b)),
               n_at_upper_bound=int(np.sum(m_est >= HI - tol_b)))
    for name, mask in regions.items():
        if mask.any():
            out[f"C_mean_rel_{name}"] = float(rel_C[mask].mean())
            out[f"C_mean_est_{name}"] = float(C_est[mask].mean())
            out[f"C_mean_true_{name}"] = float(C_true[mask].mean())
    return out


def state_metrics(u_est, u_true, Fd_arr):
    _uerr_fun.x.array[:] = u_est.x.array - u_true.x.array
    _uref_fun.x.array[:] = u_true.x.array
    u_rel = np.sqrt(fem.assemble_scalar(_l2_uerr) / fem.assemble_scalar(_l2_uref))
    eF = compute_Fh(domain, u_est, selected_dofs) - Fd_arr
    return dict(u_rel_L2=float(u_rel),
                F_max_abs=float(np.abs(eF).max()), F_mean_abs=float(np.abs(eF).mean()),
                F_rel_frob=float(np.linalg.norm(eF) / np.linalg.norm(Fd_arr)))


def evaluate_point(x, label):
    """Forward/adjoint at x; returns objective values and a copy of u."""
    J_val = obj.eval_J(x)
    g = obj.eval_grad(x)
    u_copy = fem.Function(V, name=f"displacement_{label}")
    u_copy.x.array[:] = uh.x.array
    return dict(J=float(J_val), Phi=float(obj.Jd), J_prior=float(obj.Jp),
                proj_grad_inf=proj_grad_inf(x, g),
                grad_inf=float(np.abs(g).max())), u_copy


def save_history(path, hist):
    np.savetxt(path, np.array([[h["iter"], h["J"], h["pgrad"], h["n_forward"],
                                h["n_hessvec"], h["cg_iters"]] for h in hist]),
               header="iter J proj_grad_inf n_forward n_hessvec cg_iters")


def plot_convergence(path, case, hists):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    styles = {"lbfgsb": (".-", "L-BFGS-B"), "newtoncg": ("o-", "Newton-CG")}
    for method, hist in hists.items():
        st, lab = styles[method]
        it = [h["iter"] for h in hist]
        Jv = [h["J"] for h in hist]
        axes[0].semilogy(it, Jv, st, label=lab, markersize=4)
        axes[1].semilogy(it, [max(h["pgrad"], 1e-30) for h in hist], st, label=lab, markersize=4)
        axes[2].semilogy([h["n_forward"] for h in hist], Jv, st, label=lab, markersize=4)
    axes[0].set(xlabel="outer iteration", ylabel=r"$\mathcal{J}(m)$", title="cost")
    axes[1].set(xlabel="outer iteration", ylabel=r"$\|P(m-g)-m\|_\infty$",
                title="projected gradient")
    axes[2].set(xlabel="nonlinear forward solves", ylabel=r"$\mathcal{J}(m)$",
                title="cost vs forward solves")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.suptitle(f"MAP convergence ($m=\\log C$) — {case} case")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_xdmf(path, functions):
    with XDMFFile(MPI.COMM_WORLD, str(path), "w") as xdmf:
        xdmf.write_mesh(domain)
        for f in functions:
            xdmf.write_function(f)


def va_function(name, arr):
    f = fem.Function(Va, name=name)
    f.x.array[:] = arr
    f.x.scatter_forward()
    return f


def param_fields(tag, m_arr, m_true=None):
    """m and C fields (plus errors vs truth) for one estimate."""
    C_arr = np.exp(m_arr)
    out = [va_function(f"m_{tag}", m_arr), va_function(f"C_{tag}", C_arr)]
    if m_true is not None:
        C_true = np.exp(m_true)
        out += [va_function(f"C_abs_error_{tag}", np.abs(C_arr - C_true)),
                va_function(f"C_rel_error_{tag}", np.abs(C_arr - C_true) / C_true),
                va_function(f"m_abs_error_{tag}", np.abs(m_arr - m_true))]
    return out


# =============================================================================
# prior sanity check + initial guess (identical for both methods / cases)
# =============================================================================

prior_std = None if args.no_prior_std else prior.pointwise_std()
x0 = np.log(np.random.RandomState(args.init_seed).uniform(*args.C_init_range, prior.ndofs))
x0 = np.clip(x0, LO, HI)

print("\n" + "=" * 70)
print(" ex06 MAP COMPARISON   (parameter m = log C)")
print(f"  cases={args.cases}  methods={args.methods}")
print(f"  prior: m0 = log({args.C0}) = {M0:.4f}, gamma={prior.gamma:.4g}, "
      f"delta={prior.delta:.4g}, beta={prior.beta:.4g}")
if prior_std is not None:
    s_avg = np.sqrt(np.mean(prior_std ** 2))
    print(f"  prior pointwise std of m: [{prior_std.min():.3f}, {prior_std.max():.3f}], "
          f"volume-avg {s_avg:.3f}")
    print(f"  -> pointwise 95% prior band on C: [{args.C0*np.exp(-2*s_avg):.2f}, "
          f"{args.C0*np.exp(2*s_avg):.2f}] kPa")
print(f"  likelihood: {args.misfit_weight}, noise_std={args.noise_std}, sensors={Npoints}")
print(f"  |Omega|={volume_mesh:.3e}, mean sensor-cell vol={np.mean(sensor_vols):.3e}")
print(f"  forward: load={args.load}, kappa={args.kappa}, load_steps={args.load_steps}")
print(f"  data:   model={args.data_model}"
      + (f", F={args.data_F}, P2 load_steps={args.data_load_steps}, "
         f"quad={args.data_quad_degree}" if args.data_model == "p2" else " (inverse crime)"))
print(f"  bounds: C in {args.C_bounds} kPa  (m in [{LO:.4f}, {HI:.4f}])")
print(f"  init:   C ~ U{tuple(args.C_init_range)} kPa, seed={args.init_seed}")
print(f"  gtol={args.gtol}  ftol={args.ftol}  maxiter={args.maxiter}")
print("=" * 70 + "\n")

summary_rows = []

for case in args.cases:
    print("\n" + "#" * 70 + f"\n# CASE: {case}\n" + "#" * 70)
    case_dir = OUTPUT_DIR / case
    case_dir.mkdir(parents=True, exist_ok=True)

    # --- synthetic data (C in kPa) ---
    u_data_p1 = None
    if args.data_model == "p2":
        u2_true, cd, _ = solve_forward_p2(
            geo, case_type=case, target_load=args.load, kappa_bulk=args.kappa,
            load_steps=args.data_load_steps, quad_degree=args.data_quad_degree,
            newton_maxit=args.newton_maxit)
        Fd_arr = observe_F(u2_true, args.data_F)
        u_data_p1 = fem.Function(V, name="displacement_data_p2_at_vertices")
        u_data_p1.interpolate(u2_true)
        print(f"  data: P2 solve, F observed as '{args.data_F}' in {Npoints} sensor cells")
    else:
        ud_arr, Fd_arr, cd = prob_ventricle_passive_filling(
            geo, ndofs_data=args.num_nodes, case_type=case,
            target_load=args.load, kappa_bulk=args.kappa)
        Fd_arr = np.asarray(Fd_arr)
        print("  data: original P1 ex03 (inverse crime)")
    set_data(Fd_arr)
    C_true = cd.x.array.copy()
    if np.any(C_true <= 0):
        raise RuntimeError("ground-truth C has non-positive values; log C undefined")
    m_true = np.log(C_true)
    print(f"  C_true in [{C_true.min():.3f}, {C_true.max():.3f}] kPa, "
          f"max|m_true - m0| = {np.abs(m_true - M0).max():.3f}")

    # --- references: objective at the truth and at the initial guess ---
    obj.reset()
    true_obj, u_true = evaluate_point(m_true, "true")

    # --- model (discretization) gap: P1 model at the TRUE field vs the data ---
    I3 = np.eye(dim)[None, :, :]
    F_model_true = observe_F(u_true, "p1interp")
    gap = F_model_true - Fd_arr
    data_gap = dict(
        data_model=args.data_model, data_F=args.data_F,
        F_gap_rms=float(np.sqrt(np.mean(gap ** 2))),
        F_gap_max=float(np.abs(gap).max()),
        F_gap_rel_strain=float(np.linalg.norm(gap) / np.linalg.norm(Fd_arr - I3)),
        F_strain_rms=float(np.sqrt(np.mean((Fd_arr - I3) ** 2))),
        Phi_true=true_obj["Phi"], Phi_true_expected_noise_only=0.5 * Fd_arr.size,
    )
    if u_data_p1 is not None:
        _uerr_fun.x.array[:] = u_true.x.array - u_data_p1.x.array
        _uref_fun.x.array[:] = u_data_p1.x.array
        data_gap["u_gap_rel_L2_vertices"] = float(np.sqrt(
            fem.assemble_scalar(_l2_uerr) / fem.assemble_scalar(_l2_uref)))
    print(f"  P1-model vs data at the TRUE field: RMS F gap = {data_gap['F_gap_rms']:.3e} "
          f"({100*data_gap['F_gap_rel_strain']:.2f}% of ||Fd - I||), "
          f"max = {data_gap['F_gap_max']:.3e}")
    print(f"  -> compare with the likelihood std noise_std = {args.noise_std:g}: "
          f"Phi(m_true) = {true_obj['Phi']:.3e}  (would be ~{0.5*Fd_arr.size:.0f} "
          f"if the gap were pure noise of size noise_std)")
    if data_gap["F_gap_rms"] > args.noise_std:
        print("  WARNING: the model gap exceeds noise_std -- the MAP will try to fit "
              "discretization error. Consider noise_std >= RMS F gap.")

    init_obj, u_init = evaluate_point(x0, "init")
    metrics = {
        "case": case,
        "settings": vars(args) | {"m0": M0, "beta": prior.beta,
                                  "n_param_dofs": prior.ndofs, "n_sensors": Npoints,
                                  "volume_mesh": volume_mesh},
        "true": true_obj,
        "data_gap": data_gap,
        "init": init_obj | param_metrics(x0, m_true) | state_metrics(u_init, u_true, Fd_arr),
    }

    results, hists = {}, {}
    for method in args.methods:
        print(f"\n=== {case}: {method} (same initial guess) ===")
        runner = run_lbfgsb if method == "lbfgsb" else run_newtoncg
        x_opt, hist, info = runner(x0)
        hists[method] = hist
        save_history(case_dir / f"history_{method}_{case}.txt", hist)

        obj_vals, u_opt = evaluate_point(x_opt, method)
        m = info | obj_vals | param_metrics(x_opt, m_true) | state_metrics(u_opt, u_true, Fd_arr)
        metrics[method] = m
        results[method] = (x_opt, u_opt)
        summary_rows.append({"case": case, "method": method} | m)

    if len(results) == 2:
        dm_methods = results["lbfgsb"][0] - results["newtoncg"][0]
        dC_methods = np.exp(results["lbfgsb"][0]) - np.exp(results["newtoncg"][0])
        metrics["lbfgsb_vs_newtoncg"] = {
            "m_max_abs_diff": float(np.abs(dm_methods).max()),
            "C_max_abs_diff": float(np.abs(dC_methods).max()),
        }

    # --- XDMF + H5 ---
    fields = (param_fields("true", m_true)
              + param_fields("init", x0, m_true)
              + [va_function("m_prior_mean", m0_vec), u_true, indicator, Fd_func])
    if prior_std is not None:
        fields.append(va_function("m_prior_std", prior_std))
    if u_data_p1 is not None:
        fields.append(u_data_p1)
    for method, (x_opt, u_opt) in results.items():
        fields += param_fields(method, x_opt, m_true) + [u_opt]
    if len(results) == 2:
        fields += [va_function("m_diff_lbfgsb_minus_newtoncg", dm_methods),
                   va_function("C_diff_lbfgsb_minus_newtoncg", dC_methods)]
    write_xdmf(case_dir / f"map_comparison_{case}.xdmf", fields)

    with open(case_dir / f"metrics_{case}.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    plot_convergence(case_dir / f"convergence_{case}.png", case, hists)

    # --- console summary ---
    print(f"\n--- {case}: summary ---")
    print(f"  J(m_true) = {true_obj['J']:.4e}  (Phi={true_obj['Phi']:.3e}, "
          f"prior={true_obj['J_prior']:.3e})   J(m_init) = {init_obj['J']:.4e}")
    print(f"  {'metric':<16}" + "".join(f"{mm:>14}" for mm in args.methods))
    for key in ["J", "Phi", "J_prior", "proj_grad_inf", "n_iter", "n_forward",
                "n_hessvec", "wall_time_s", "C_rel_L2", "C_max_rel",
                "m_max_abs", "u_rel_L2", "F_rel_frob"]:
        print(f"  {key:<16}" + "".join(f"{metrics[mm][key]:>14.4e}" for mm in args.methods))
    for mm in args.methods:
        print(f"  {mm} stop: {metrics[mm]['stop_reason']}")

# =============================================================================
# global summary CSV
# =============================================================================

cols = ["case", "method", "stop_reason", "n_iter", "n_forward", "n_forward_failed",
        "n_hessvec", "wall_time_s",
        "J", "Phi", "J_prior", "proj_grad_inf",
        "C_rel_L2", "C_max_abs", "C_mean_abs", "C_max_rel", "C_mean_rel",
        "m_rel_L2", "m_max_abs", "m_mean_abs",
        "C_mean_rel_core", "C_mean_rel_transition", "C_mean_rel_healthy",
        "u_rel_L2", "F_max_abs", "F_mean_abs", "F_rel_frob"]
with open(OUTPUT_DIR / "summary.csv", "w") as f:
    f.write(",".join(cols) + "\n")
    for row in summary_rows:
        vals = [f'"{row.get(c, "")}"' if c == "stop_reason" else str(row.get(c, ""))
                for c in cols]
        f.write(",".join(vals) + "\n")

print(f"\nDone. Outputs in {OUTPUT_DIR.resolve()}")
