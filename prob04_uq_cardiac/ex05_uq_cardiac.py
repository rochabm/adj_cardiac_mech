"""
ex05_uq_cardiac.py
-------------------
Bayesian UQ for the cardiac passive filling inverse problem.

Two parameters are now DECOUPLED:
    alpha : regularization parameter for the MAP cost functional
            J = Jmisfit + 0.5 m^T R m  (prior provides regularization)
    gamma, delta : prior parameters for the Laplacian prior
            H_prior = gamma*K + delta*M  (controls prior shape/scale)

This follows the correct Bayesian formulation where the prior is
independent of the regularization, as in Bui-Thanh et al. (2013).

Pipeline:
    1.  Geometry + FE spaces
    2.  Constitutive model (Holzapfel-Ogden passive)
    3.  Synthetic data (from prob_ventricle_passive_filling)
    4.  Cost functional J = Jmisfit + 0.5 m^T R m
    5.  Prior H_prior = gamma*K + delta*M  (independent of alpha)
    6.  Adjoint gradient
    7.  FD gradient check
    8.  MAP via inexact Newton-CG
    9.  Hessian FD verification at MAP
   10.  Generalized eigenproblem H_misfit v = lambda H_prior v
   11.  Woodbury pointwise variance
   12.  Posterior samples
   13.  XDMF output + plots

Usage:
    python ex05_uq_cardiac.py [--case-type fibrosis] [--num-nodes 64]
                              [--alpha 1e-3] [--gamma 0.1] [--delta 0.5]
                              [--k-eig 50] [--output-dir .]
"""

from mpi4py import MPI
from dolfinx import fem, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem, LinearProblem
from dolfinx.nls.petsc import NewtonSolver
import ufl
import numpy as np
import dolfinx
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import petsc4py.PETSc as PETSc

import cardiac_geometries
from ex03_ventricle_discrete_forward import prob_ventricle_passive_filling

from cardiac_utils import (select_distributed_nodes, compute_Fh,
                            save_results_xdmf, print_F_error_statistics,
                            save_selected_nodes)
from hessian_ucq import (HessianOperator, generalized_eigensolver,
                          woodbury_pointwise_variance)
from verify_hessian import verify_hessian
from newton_cg_solver import inexact_newton_cg

# =============================================================================
# command-line arguments
# =============================================================================

parser = argparse.ArgumentParser(
    description="ex05 UQ — cardiac Bayesian inverse problem (prior=R, no alpha*Jsmooth)"
)
parser.add_argument("--case-type",  type=str,   default="fibrosis",
                    choices=["fibrosis", "linear"])
parser.add_argument("--num-nodes",  type=int,   default=64,
                    help="Number of measurement nodes (default: 64)")
parser.add_argument("--alpha",      type=float, default=1e-3,
                    help="Regularization parameter (default: 1e-3)")
parser.add_argument("--gamma",      type=float, default=0.1,
                    help="Prior stiffness (gamma*K term, default: 0.1)")
parser.add_argument("--delta",      type=float, default=0.5,
                    help="Prior mass (delta*M term, default: 0.5)")
parser.add_argument("--m0",         type=float, default=3.0,
                    help="Prior mean (healthy CC value, default: 3.0)")
parser.add_argument("--k-eig",      type=int,   default=50,
                    help="Number of eigenpairs for low-rank UQ (default: 50)")
parser.add_argument("--gtol",       type=float, default=1e-8)
parser.add_argument("--ftol",       type=float, default=1e-20)
parser.add_argument("--output-dir", type=str,   default=".")

args = parser.parse_args()

CASE_TYPE   = args.case_type
Nnodes      = args.num_nodes
alpha_value = args.alpha
gamma_pr    = args.gamma
delta_pr    = args.delta
m0_prior    = args.m0
k_eig       = args.k_eig
my_gtol     = args.gtol
my_ftol     = args.ftol
OUTPUT_DIR  = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "="*60)
print(" CARDIAC UQ PARAMETERS (ex05)")
print(f"  case_type  = {CASE_TYPE}")
print(f"  num_nodes  = {Nnodes}")
print(f"  alpha      = {alpha_value}  (regularization)")
print(f"  gamma      = {gamma_pr}  (prior stiffness)")
print(f"  delta      = {delta_pr}  (prior mass)")
print(f"  m0_prior   = {m0_prior}  (prior mean -- healthy CC)")
print(f"  k_eig      = {k_eig}")
print(f"  gtol       = {my_gtol}")
print(f"  ftol       = {my_ftol}")
print(f"  output_dir = {OUTPUT_DIR}")
print("="*60 + "\n")

# =============================================================================
# 1. Geometry and mesh
# =============================================================================

geodir = Path("lv_ellipsoid")
try:
    # try loading existing geometry first (faster)
    geo = cardiac_geometries.geometry.Geometry.from_folder(geodir)
    print(f"Loaded geometry from {geodir}")
except Exception:
    # regenerate if loading fails
    geo = cardiac_geometries.mesh.lv_ellipsoid(
        outdir=geodir, create_fibers=True, fiber_space="DG_0",
        psize_ref=3, r_short_epi=10,
        fiber_angle_endo=40.0, fiber_angle_epi=-50.0
    )
    print(f"Generated new geometry in {geodir}")

domain = geo.mesh

# =============================================================================
# 2. Synthetic data
# =============================================================================

ud_arr, Fd_arr, cd = prob_ventricle_passive_filling(
    geo, ndofs_data=Nnodes, case_type=CASE_TYPE
)

# =============================================================================
# 3. FE spaces
# =============================================================================

V  = dolfinx.fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
Va = dolfinx.fem.functionspace(domain, ("Lagrange", 1))

du = ufl.TrialFunction(V)
v  = ufl.TestFunction(V)
uh = dolfinx.fem.Function(V)

print(f"State dofs    : {V.dofmap.index_map.size_global}")
print(f"Parameter dofs: {Va.dofmap.index_map.size_global}")

dim = domain.geometry.dim

# =============================================================================
# 4. Boundary conditions (base fixed)
# =============================================================================

facet_tags = geo.ffun
u_bc      = np.array((0,) * domain.geometry.dim, dtype=default_scalar_type)
base_dofs = fem.locate_dofs_topological(V, facet_tags.dim, facet_tags.find(5))
bcs       = [fem.dirichletbc(u_bc, base_dofs, V)]

# =============================================================================
# 5. Measurement nodes
# NOTE: select from GEOMETRY node coordinates, not FE dof coordinates,
#       so that selected_dofs correctly index into node_to_cells below.
# =============================================================================

# geometry node coordinates (shape: n_geom_nodes x 3)
geom_coords = domain.geometry.x  # (n_nodes, 3)

selected_dofs, selected_coords = select_distributed_nodes(
    geom_coords, N=Nnodes, seed=42
)
Npoints = len(selected_dofs)
print(f"Measurement nodes: {Npoints}")
save_selected_nodes(selected_coords, selected_dofs)

# =============================================================================
# 6. Kinematics and constitutive model (Holzapfel-Ogden passive)
# =============================================================================

d_geo = len(uh)
I  = ufl.variable(ufl.Identity(d_geo))
F  = ufl.variable(I + ufl.grad(uh))
J  = ufl.variable(ufl.det(F))
C  = ufl.variable(F.T * F)

f0, s0, n0 = geo.f0, geo.s0, geo.n0
e1, e2, e3 = f0, s0, n0

bf    = default_scalar_type(6.6)
bt    = default_scalar_type(4.0)
bfs   = default_scalar_type(2.6)
kappa = fem.Constant(domain, 1e2)

CC = fem.Function(Va)   # parameter field (stiffness)

Cs = J**(-2/3) * F.T * F
Es = 0.5 * (Cs - I)

E11 = ufl.inner(Es*e1, e1);  E12 = ufl.inner(Es*e1, e2);  E13 = ufl.inner(Es*e1, e3)
E21 = ufl.inner(Es*e2, e1);  E22 = ufl.inner(Es*e2, e2);  E23 = ufl.inner(Es*e2, e3)
E31 = ufl.inner(Es*e3, e1);  E32 = ufl.inner(Es*e3, e2);  E33 = ufl.inner(Es*e3, e3)

Q = (bf  * E11**2
   + bt  * (E22**2 + E33**2 + E23**2 + E32**2)
   + bfs * (E12**2 + E21**2 + E13**2 + E31**2))

Wpassive      = CC / 2.0 * (ufl.exp(Q) - 1)
Wvolume       = kappa * (J * ufl.ln(J) - J + 1)
strain_energy = Wpassive + Wvolume
P             = ufl.diff(strain_energy, F)

# variational form
p_endo = fem.Constant(domain, 0.0)
metadata = {"quadrature_degree": 4}
ds = ufl.Measure('ds', domain=domain, subdomain_data=facet_tags, metadata=metadata)
dx = ufl.Measure("dx", domain=domain, metadata=metadata)
N_fac = ufl.FacetNormal(domain)
Gendo = -p_endo * ufl.inner(v, J * ufl.transpose(ufl.inv(F)) * N_fac) * ds(6)
Fun   = ufl.inner(P, ufl.grad(v)) * dx + Gendo

# =============================================================================
# 7. Forward solver (load stepping)
# =============================================================================

def solve_nl_prob(uh_func):
    forward_problem = NonlinearProblem(Fun, uh_func, bcs)
    solver = NewtonSolver(domain.comm, forward_problem)
    solver.atol = 1e-8;  solver.rtol = 1e-8
    ksp = solver.krylov_solver
    ksp.setType("preonly");  ksp.getPC().setType("lu")
    loads = np.linspace(0, -3.0, 10)
    for step, load in enumerate(loads):
        p_endo.value = load
        num_its, converged = solver.solve(uh_func)
        assert converged, f"Newton failed at load step {step}"
        print(f"  load step {step}: newton its {num_its}")

# =============================================================================
# 8. Cost functional + prior
# =============================================================================

volume_form = dolfinx.fem.form(
    dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(1.0)) * dx)
volume_mesh = dolfinx.fem.assemble_scalar(volume_form)
print(f"  volume_mesh = {volume_mesh:.4e} mm³")

# build indicator + reference Fd at measurement cells
dofmap_geo    = domain.geometry.dofmap
Ncells        = domain.topology.index_map(domain.topology.dim).size_local
node_to_cells = {}
for cell_id in range(Ncells):
    for node in dofmap_geo[cell_id]:
        node_to_cells.setdefault(int(node), []).append(cell_id)

TT        = dolfinx.fem.functionspace(domain, ("DG", 0, (dim, dim)))
Fd_func   = dolfinx.fem.Function(TT);  Fd_func.x.array[:] = 0.0
W0        = dolfinx.fem.functionspace(domain, ("DG", 0))
indicator = dolfinx.fem.Function(W0);  indicator.x.array[:] = 0.0
bs        = TT.dofmap.bs

for j, dof in enumerate(selected_dofs):
    cell_id     = node_to_cells[int(dof)][0]
    cell_dof_TT = TT.dofmap.cell_dofs(cell_id)[0]
    Fd_func.x.array[cell_dof_TT*bs : cell_dof_TT*bs + bs] = Fd_arr[j].flatten()
    indicator.x.array[W0.dofmap.cell_dofs(cell_id)] = 1.0

Fd_func.x.scatter_forward();  indicator.x.scatter_forward()

diffF       = F - Fd_func
Jdata       = 0.5 * (1.0/volume_mesh) * indicator * ufl.inner(diffF, diffF) * dx
Jsmooth     = (1.0/volume_mesh) * ufl.inner(ufl.grad(CC), ufl.grad(CC)) * dx
# Jfunctional includes alpha*Jsmooth for MAP conditioning.
# H_prior matches Hessian(alpha*Jsmooth) exactly so H_misfit = H_full - H_prior
# correctly isolates data information for the eigenproblem.
alpha_reg   = dolfinx.fem.Constant(domain, alpha_value)
Jfunctional = Jdata + alpha_reg * Jsmooth
Jh          = dolfinx.fem.form(Jfunctional)

# =============================================================================
# 9. Prior operator H_prior = gamma*K + delta*M
#    DECOUPLED from alpha (regularization parameter).
#    gamma controls stiffness (correlation length), delta controls mass (amplitude).
#    This is the correct Bayesian formulation: prior is independent of alpha.
# =============================================================================

class CardiacPrior:
    """
    Laplacian prior for the cardiac parameter field CC.

    H_prior = gamma*K + delta*M   (DECOUPLED from alpha)

    gamma controls the stiffness (spatial correlation length).
    delta controls the mass term (pointwise variance amplitude).

    These are independent of the regularization parameter alpha used
    in J = Jmisfit + 0.5 m^T R m.

    Note: The Laplacian prior is not trace-class in 3D (pointwise
    variance is mesh-dependent). For mesh-independence, the BiLaplacian
    C = R^{-1}MR^{-1} should be used, but requires careful calibration
    of gamma/delta to match H_misfit in magnitude. For a fixed mesh,
    the Laplacian prior is acceptable (Bui-Thanh et al. 2013, Sec. 6.2).

    Interface (duck-typed for generalized_eigensolver + woodbury):
        mult(v)    = H_prior * v = R v
        solve(v)   = H_prior^{-1} * v = R^{-1} v
        diag_inv() = diag(R^{-1}) via Hutchinson estimator
    """

    def __init__(self, Va, gamma, delta, dx, alpha=None, volume_mesh=None):
        n = Va.dofmap.index_map.size_local * Va.dofmap.index_map_bs
        self.ndofs = n

        q  = ufl.TrialFunction(Va)
        dq = ufl.TestFunction(Va)

        K_form = ufl.inner(ufl.grad(q), ufl.grad(dq)) * dx
        M_form = ufl.inner(q, dq) * dx
        # H_prior matches Hessian(alpha*Jsmooth) = alpha*(2/vol)*K
        # so H_misfit = H_full - H_prior correctly isolates data term
        if alpha is not None and volume_mesh is not None:
            scale    = float(alpha) * 2.0 / volume_mesh
            eps_mass = 0.01 * scale
            R_form = scale * K_form + eps_mass * M_form
        else:
            R_form = float(gamma) * K_form + float(delta) * M_form

        self._R = fem.petsc.assemble_matrix(fem.form(R_form))
        self._R.assemble()
        self._M = fem.petsc.assemble_matrix(fem.form(M_form))
        self._M.assemble()

        self._ksp = PETSc.KSP().create(Va.mesh.comm)
        self._ksp.setOperators(self._R)
        self._ksp.setType("preonly")
        self._ksp.getPC().setType("lu")
        self._ksp.getPC().setFactorSolverType("mumps")
        self._ksp.setUp()

        self._x = self._R.createVecRight()
        self._b = self._R.createVecRight()

        # lumped mass diagonal for sampling
        ones = np.ones(n)
        self._b.array[:] = ones
        self._M.mult(self._b, self._x)
        self._M_diag = np.maximum(self._x.array.copy(), 1e-30)

        print(f"  CardiacPrior: H_prior matched to alpha*Hessian(Jsmooth), ndofs={n}")

    def _R_mult(self, v):
        self._b.array[:] = v
        self._R.mult(self._b, self._x)
        return self._x.array.copy()

    def _R_solve(self, v):
        self._b.array[:] = v
        self._ksp.solve(self._b, self._x)
        return self._x.array.copy()

    def mult(self, v):   return self._R_mult(v)
    def solve(self, v):  return self._R_solve(v)

    def diag_inv(self, n_samples=300, seed=7):
        """Hutchinson estimate of diag(R^{-1}) = diag(H_prior^{-1})."""
        rng = np.random.default_rng(seed)
        diag = np.zeros(self.ndofs)
        for _ in range(n_samples):
            z = rng.choice([-1.0, 1.0], size=self.ndofs).astype(float)
            diag += z * self._R_solve(z)
        return diag / n_samples

    def __del__(self):
        try:
            self._R.destroy(); self._M.destroy()
            self._x.destroy(); self._b.destroy()
        except Exception:
            pass


prior = CardiacPrior(Va, gamma_pr, delta_pr, dx,
                     alpha=alpha_value, volume_mesh=volume_mesh)


# =============================================================================
# 10. Adjoint + gradient
# =============================================================================

lmbda    = dolfinx.fem.Function(V)
dFdu     = ufl.derivative(Fun, uh, du)
dFdu_adj = ufl.adjoint(dFdu)
dJdu     = ufl.derivative(Jfunctional, uh, v)
adj_problem = LinearProblem(dFdu_adj, -dJdu, bcs=bcs)

q_trial    = ufl.TrialFunction(Va)
dJdf       = ufl.derivative(Jfunctional, CC, q_trial)
dFdf       = ufl.action(ufl.adjoint(ufl.derivative(Fun, CC, q_trial)), lmbda)
dJdf_c     = dolfinx.fem.form(dJdf)
dFdf_c     = dolfinx.fem.form(dFdf)
dLdf       = dolfinx.fem.Function(Va)

_cache_x = None;  _cache_J = None;  _cache_g = None


def _solve_and_cache(x):
    global _cache_x, _cache_J, _cache_g
    if _cache_x is None or not np.allclose(x, _cache_x):
        CC.x.array[:] = x;  CC.x.scatter_forward()
        uh.x.array[:] = 0.0
        solve_nl_prob(uh)
        _cache_J = domain.comm.allreduce(fem.assemble_scalar(Jh), op=MPI.SUM)
        lmbda_new = adj_problem.solve()
        lmbda.x.array[:] = lmbda_new.x.array
        lmbda.x.scatter_forward()
        dLdf.x.array[:] = 0.0
        dolfinx.fem.assemble_vector(dLdf.x.array, dJdf_c)
        dolfinx.fem.assemble_vector(dLdf.x.array, dFdf_c)
        dLdf.x.scatter_forward()
        _cache_x = x.copy();  _cache_g = dLdf.x.array.copy()


def eval_J(x):
    _solve_and_cache(x);  return _cache_J

def eval_gradient(x):
    _solve_and_cache(x);  return _cache_g

# =============================================================================
# 10. FD gradient check
# =============================================================================

print("\n" + "="*60)
print("FD gradient check")
print("="*60)

rng_check = np.random.default_rng(7)
m_test = np.full(prior.ndofs, 3.0)
h_dir  = rng_check.standard_normal(prior.ndofs)
h_dir /= np.linalg.norm(h_dir)

# reset cache to force fresh evaluation
eps   = 0.1
J0, g0 = eval_J(m_test), eval_gradient(m_test)
_cache_x = None   # force re-evaluation for perturbed point
Jp     = eval_J(m_test + eps * h_dir)
dJ_fd  = (Jp - J0) / eps
dJ_adj = g0 @ h_dir
print(f"  FD  directional deriv: {dJ_fd:.6e}")
print(f"  Adj directional deriv: {dJ_adj:.6e}")
print(f"  Relative error       : {abs(dJ_adj-dJ_fd)/(abs(dJ_fd)+1e-30):.4e}")


# =============================================================================
# 11. MAP via inexact Newton-CG
# =============================================================================

print("\n" + "="*60)
print("MAP via inexact Newton-CG")
print("="*60)


def build_hop():
    """Rebuild HessianOperator at current (uh, CC, lmbda) state."""
    lmbda_cur = adj_problem.solve()
    lmbda_cur.x.scatter_forward()
    return HessianOperator(
        Fun, Jfunctional, uh, CC, lmbda_cur,
        V, Va, facet_tags, domain
    )


np.random.seed(57)
cc_init = np.random.uniform(2.0, 10.0, prior.ndofs)

m_map_arr, newton_history = inexact_newton_cg(
    eval_J, eval_gradient, build_hop,
    x0=cc_init,
    max_outer_iter=200,
    grad_tol=my_gtol,
    ftol=my_ftol,
    cg_tol=0.1,
    cg_maxiter=200,
    bounds=(1.0, 16.0),
    eisenstat_walker=True,
)

print(f"\nMAP: J = {eval_J(m_map_arr):.6e}")
g_map = eval_gradient(m_map_arr)
print(f"MAP: ||grad||_inf = {np.linalg.norm(g_map, ord=np.inf):.6e}  (gtol={my_gtol:.1e})")
print(f"MAP: ||grad||_2   = {np.linalg.norm(g_map):.6e}")
print(f"MAP: ||lmbda||    = {np.linalg.norm(lmbda.x.array):.6e}  (adjoint variable)")

CC_map = fem.Function(Va, name="c_map")
CC_map.x.array[:] = m_map_arr;  CC_map.x.scatter_forward()

# forward solve at MAP
CC.x.array[:] = m_map_arr;  CC.x.scatter_forward()
uh.x.array[:] = 0.0;        uh.x.scatter_forward()
solve_nl_prob(uh)
uh_map = fem.Function(V, name="displacement_map")
uh_map.x.array[:] = uh.x.array.copy();  uh_map.x.scatter_forward()

# save Newton-CG history
np.savetxt(
    str(OUTPUT_DIR / "out_uq_newton_cg_history.txt"),
    np.array([[h["iter"], h["J"], h["grad_norm"], h["cg_iters"]]
              for h in newton_history]),
    header="iter J grad_norm cg_iters"
)

# =============================================================================
# 12. Hessian FD verification at MAP
# =============================================================================

print("\n" + "="*60)
print("Hessian FD verification at MAP")
print("="*60)

# ensure uh/lmbda/CC consistent at MAP before building Hop
lmbda_map = adj_problem.solve();  lmbda_map.x.scatter_forward()
lmbda.x.array[:] = lmbda_map.x.array;  lmbda.x.scatter_forward()

Hop_map = HessianOperator(
    Fun, Jfunctional, uh, CC, lmbda,
    V, Va, facet_tags, domain
)

max_rel_err = verify_hessian(
    Hop_map, eval_gradient, m_map_arr,
    n_directions=3, eps=0.2, seed=0
)
print(f"Hessian verification: max rel err = {max_rel_err:.4e} "
      f"({'PASS' if max_rel_err < 5e-2 else 'FAIL'})")

# =============================================================================
# 13. Generalized eigenproblem H_misfit v = lambda H_prior v
#     Same as subsurface_bayesian.py: doublePassG equivalent
# =============================================================================

print("\n" + "="*60)
print("Generalized eigenproblem H_misfit v = λ H_prior v")
print("="*60)

# --- diagnostic: check H_full and H_prior magnitudes ---
rng_diag = np.random.default_rng(123)
v_test   = rng_diag.standard_normal(prior.ndofs)
v_test  /= np.linalg.norm(v_test)
Hv       = Hop_map.mult(v_test)
Rv       = prior.mult(v_test)
Jdata_now   = dolfinx.fem.assemble_scalar(dolfinx.fem.form(Jdata))
Jsmooth_now = dolfinx.fem.assemble_scalar(dolfinx.fem.form(Jsmooth))
print(f"  Jdata at MAP        = {Jdata_now:.4e}")
print(f"  alpha*Jsmooth at MAP= {alpha_value*Jsmooth_now:.4e}")
print(f"  Diagnostic: ||H_full·v||   = {np.linalg.norm(Hv):.4e}")
print(f"  Diagnostic: ||H_prior·v||  = {np.linalg.norm(Rv):.4e}  (R = gamma*K + delta*M)")
print(f"  Diagnostic: ratio          = {np.linalg.norm(Hv)/np.linalg.norm(Rv):.4e}")
print(f"  (ratio >> 1: data-dominated; ratio ≈ 1: balanced; ratio << 1: prior-dominated)")

p_over = 20
eigvals, eigvecs = generalized_eigensolver(
    Hop_map, prior, k=k_eig, p=p_over, seed=0
)
np.savetxt(str(OUTPUT_DIR / "out_uq_eigenvalues.txt"), eigvals)

print(f"\nTop 5 eigenvalues : {eigvals[:5]}")
print(f"Eigenvalues > 1   : {np.sum(eigvals > 1)}")
print(f"Eigenvalues > 0.1 : {np.sum(eigvals > 0.1)}")

# =============================================================================
# 14. Woodbury pointwise posterior variance
#     Same formula as subsurface_bayesian.py
# =============================================================================

print("\n" + "="*60)
print("Woodbury pointwise variance")
print("="*60)

# prior variance: diag(R^{-1}) via randomized eigenpairs of R^{-1}
# Use prior.solve(v) = R^{-1} v  (Laplacian covariance)
print("  Computing prior variance via randomized eigenpairs of R^{-1}...")
n_pr    = prior.ndofs
k_pr    = min(200, n_pr - 1)
rng_pr  = np.random.default_rng(77)
Omega_p = rng_pr.standard_normal((n_pr, k_pr + 10))
Y_p     = np.zeros_like(Omega_p)
for i in range(k_pr + 10):
    Y_p[:, i] = prior.solve(Omega_p[:, i])   # R^{-1} v
Q_p, _  = np.linalg.qr(Y_p)
RinvQ   = np.zeros_like(Q_p)
for i in range(Q_p.shape[1]):
    RinvQ[:, i] = prior.solve(Q_p[:, i])     # R^{-1} v
T_p  = 0.5 * (Q_p.T @ RinvQ + (Q_p.T @ RinvQ).T)
lp, vp = np.linalg.eigh(T_p)
order_p = np.argsort(lp)[::-1][:k_pr]
lp = lp[order_p];  Up = Q_p @ vp[:, order_p]
prior_var = np.abs((Up**2) @ lp)
print(f"  Prior variance: [{prior_var.min():.3e}, {prior_var.max():.3e}]")

# Woodbury posterior variance
_, post_var_raw, correction = woodbury_pointwise_variance(
    prior, eigvals, eigvecs, n_prior_samples=300, seed=9
)
# use smooth eigenvalue-based prior_var (more accurate than Hutchinson)
post_var = np.maximum(prior_var - correction, 0.0)

print(f"  Posterior variance: [{post_var.min():.3e}, {post_var.max():.3e}]")
print(f"  Correction        : [{correction.min():.3e}, {correction.max():.3e}]")

np.save(str(OUTPUT_DIR / "out_uq_prior_variance.npy"),    prior_var)
np.save(str(OUTPUT_DIR / "out_uq_posterior_variance.npy"), post_var)

# -----------------------------------------------------------------------
# Physical posterior stddev calibration
# -----------------------------------------------------------------------
# The raw post_var is in units of H_prior^{-1} which depends on alpha,
# volume_mesh and mesh size -- not in CC² units.
#
# Correct approach: scale so that the PRIOR std matches the expected
# physical spread of CC.  The prior should represent our uncertainty
# before seeing data, which spans roughly the full CC range / 2.
#
# Scale factor: s = CC_std_prior_physical / sqrt(mean(prior_var))
# Then: stddev_physical = sqrt(post_var) * s
#
# This gives posterior stddev in the same units as CC, directly
# comparable to the CC range [CC_min, CC_max].

cc_arr          = cd.x.array
CC_mean         = float(cc_arr.mean())
CC_std_physical = float(cc_arr.std())    # std of true CC field
if CC_std_physical < 1e-6:
    CC_std_physical = (cc_arr.max() - cc_arr.min()) / 2.0

prior_var_mean  = float(prior_var.mean())
scale_factor    = CC_std_physical / np.sqrt(prior_var_mean) if prior_var_mean > 0 else 1.0

stddev_cal      = np.sqrt(np.clip(post_var, 0, None)) * scale_factor
prior_stddev    = np.sqrt(prior_var) * scale_factor

print(f"\n  CC_true range         : [{cc_arr.min():.3f}, {cc_arr.max():.3f}]")
print(f"  CC_true std           : {CC_std_physical:.4f}  (physical scale for prior)")
print(f"  Prior stddev (physical): [{prior_stddev.min():.4f}, {prior_stddev.max():.4f}]")
print(f"  Post. stddev (physical): [{stddev_cal.min():.4f}, {stddev_cal.max():.4f}]")
print(f"  Scale factor          : {scale_factor:.4e}")

# also report variance reduction (always dimensionless and correct)
var_red_frac = np.clip(1.0 - post_var / (prior_var + 1e-30), 0, 1)
print(f"  Variance reduction    : [{var_red_frac.min():.3f}, {var_red_frac.max():.3f}]")

# wrap in fem.Function for XDMF output
prior_var_fun = fem.Function(Va, name="prior_variance")
prior_var_fun.x.array[:] = prior_var;  prior_var_fun.x.scatter_forward()

post_var_fun = fem.Function(Va, name="posterior_variance")
post_var_fun.x.array[:] = post_var;  post_var_fun.x.scatter_forward()

stddev_fun = fem.Function(Va, name="posterior_stddev_calibrated")
stddev_fun.x.array[:] = stddev_cal;  stddev_fun.x.scatter_forward()

# =============================================================================
# 15. Posterior samples
#     m_sample = m_map + z_prior - U diag(sqrt(lambda/(lambda+1))) U^T R z_prior
# =============================================================================

print("\n" + "="*60)
print("Posterior samples")
print("="*60)

nsamples  = 5
rng_s     = np.random.default_rng(99)
D_coeff   = np.sqrt(eigvals / (eigvals + 1.0))

all_samples = []
for i in range(nsamples):
    # draw from prior: z ~ N(0, R^{-1})
    w       = rng_s.standard_normal(prior.ndofs)
    z_prior = prior._R_solve(w)
    # low-rank correction
    Rz    = prior._R_mult(z_prior)
    coeff = eigvecs.T @ Rz
    delta = eigvecs @ (D_coeff * coeff)
    z_post  = z_prior - delta
    m_post  = m_map_arr + z_post
    # clip to physical bounds
    m_post  = np.clip(m_post, 1.0, 16.0)
    all_samples.append(m_post)
    print(f"  sample {i+1}: range [{m_post.min():.3f}, {m_post.max():.3f}]")

# =============================================================================
# 16. Error analysis
# =============================================================================

CC.x.array[:] = m_map_arr;  CC.x.scatter_forward()

error_map = fem.Function(Va, name="c_error_map")
abs_err   = np.abs(m_map_arr - cd.x.array[:])
denom     = np.maximum(np.abs(cd.x.array[:]), 1e-12)
error_map.x.array[:] = abs_err / denom
error_map.x.scatter_forward()

print(f"\nMAP max pointwise rel error (CC): {(abs_err/denom).max():.6e}")

# =============================================================================
# 17. XDMF output
# =============================================================================

save_results_xdmf(domain, {
    "c_true"                     : cd,
    "c_map"                      : CC_map,
    "displacement_map"           : uh_map,
    "c_error_map"                : error_map,
    "prior_variance"             : prior_var_fun,
    "posterior_variance"         : post_var_fun,
    "posterior_stddev_calibrated": stddev_fun,
    "measurement_indicator"      : indicator,
    "Fd_reference"               : Fd_func,
}, filename=str(OUTPUT_DIR / "out_uq_cardiac.xdmf"))

# --- deformation fields for regional analysis ---
# displacement magnitude (P1 scalar)
u_mag_fun = fem.Function(Va, name="displacement_magnitude")
u_mag_fun.interpolate(fem.Expression(
    ufl.sqrt(ufl.dot(uh_map, uh_map)),
    Va.element.interpolation_points()))
u_mag_fun.x.scatter_forward()

# deformation gradient Frobenius norm ||F||_F  (DG0)
W0_q = dolfinx.fem.functionspace(domain, ("DG", 0))
F_map_ufl = ufl.Identity(dim) + ufl.grad(uh_map)

F_fro_fun = fem.Function(W0_q, name="F_frobenius_map")
F_fro_fun.interpolate(fem.Expression(
    ufl.sqrt(ufl.inner(F_map_ufl, F_map_ufl)),
    W0_q.element.interpolation_points()))
F_fro_fun.x.scatter_forward()

# J = det(F) volumetric change (DG0)
J_fun = fem.Function(W0_q, name="J_det_map")
J_fun.interpolate(fem.Expression(
    ufl.det(F_map_ufl),
    W0_q.element.interpolation_points()))
J_fun.x.scatter_forward()

save_results_xdmf(domain, {
    "displacement_magnitude": u_mag_fun,
    "F_frobenius_map"       : F_fro_fun,
    "J_det_map"             : J_fun,
}, filename=str(OUTPUT_DIR / "out_uq_deformation.xdmf"))

# also save posterior samples as separate functions
sample_funs = []
for i, m_s in enumerate(all_samples):
    sf = fem.Function(Va, name=f"posterior_sample_{i+1}")
    sf.x.array[:] = m_s;  sf.x.scatter_forward()
    sample_funs.append(sf)

save_results_xdmf(domain,
    {f"posterior_sample_{i+1}": sf for i, sf in enumerate(sample_funs)},
    filename=str(OUTPUT_DIR / "out_uq_samples.xdmf"))

# save eigenvectors (first 6) as scalar fields
eig_funs = []
for i in range(min(6, k_eig)):
    ef = fem.Function(Va, name=f"eigenvector_{i}")
    v  = eigvecs[:, i]
    ef.x.array[:] = v / (np.abs(v).max() + 1e-30)
    ef.x.scatter_forward()
    eig_funs.append(ef)

save_results_xdmf(domain,
    {f"eigenvector_{i}": ef for i, ef in enumerate(eig_funs)},
    filename=str(OUTPUT_DIR / "out_uq_eigenvectors.xdmf"))

# save scalar arrays for the PyVista script
np.save(str(OUTPUT_DIR / "out_uq_eigvals.npy"),      eigvals)
np.save(str(OUTPUT_DIR / "out_uq_newton_J.npy"),
        np.array([h["J"] for h in newton_history]))
np.save(str(OUTPUT_DIR / "out_uq_newton_gnorm.npy"),
        np.array([h["grad_norm"] for h in newton_history]))

print("Saved: out_uq_cardiac.xdmf")
print("Saved: out_uq_samples.xdmf")
print("Saved: out_uq_eigenvectors.xdmf")

# =============================================================================
# 18. Plots
# =============================================================================

# convergence
newton_J = [h["J"] for h in newton_history]
fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(newton_J, "o-b", markersize=5)
ax.set_xlabel("Newton-CG iteration");  ax.set_ylabel("J")
ax.set_title("MAP convergence (Newton-CG)")
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / "fig1_MAP_convergence.png"), dpi=150)
plt.show()

# eigenvalue decay
fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(range(k_eig), eigvals, "b*", markersize=6)
ax.axhline(1.0, color="r", linestyle="-", label="λ=1")
ax.set_xlabel("index");  ax.set_ylabel("eigenvalue")
ax.set_title("Hessian misfit spectrum\nH_misfit v = λ H_prior v")
ax.legend()
plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / "fig2_eigenvalue_decay.png"), dpi=150)
plt.show()

# variance -- plot CALIBRATED physical stddev (in CC units)
coords_Va = Va.tabulate_dof_coordinates()[:, :2]
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, vals, title, cmap in [
        (axes[0], prior_stddev,  "Prior std dev (CC units)",      "inferno"),
        (axes[1], stddev_cal,    "Posterior std dev (CC units)",   "inferno"),
        (axes[2], var_red_frac,  "Variance reduction fraction",    "viridis")]:
    cf = ax.tricontourf(coords_Va[:,0], coords_Va[:,1], vals,
                        levels=30, cmap=cmap)
    plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal");  ax.set_xticks([]);  ax.set_yticks([])
plt.suptitle("Posterior UQ (calibrated to CC units)", fontsize=11)
plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / "fig3_variance.png"), dpi=150)
plt.show()

# =============================================================================
# 19. Regional analysis: fibrosis core vs healthy tissue
#     Mirrors analyze_fibrosis_region.py but inline using already-computed arrays
# =============================================================================

if CASE_TYPE == "fibrosis":
    print("\n" + "="*60)
    print("Regional analysis: fibrosis core vs healthy tissue")
    print("="*60)

    # fibrosis geometry (from ex03 c_expr)
    XC, YC, ZC = -5.0, -2.0, -9.0
    R0, R1     = 5.0, 10.0   # core radius, outer radius (mm)

    # classify geometry nodes by distance from fibrosis center
    geom_pts = domain.geometry.x
    dx_r = geom_pts[:, 0] - XC
    dy_r = geom_pts[:, 1] - YC
    dz_r = geom_pts[:, 2] - ZC
    r    = np.sqrt(dx_r**2 + dy_r**2 + dz_r**2)

    # map geometry nodes → Va dofs
    # Va is P1: dof i corresponds to geometry node i (for serial runs)
    n_va = Va.dofmap.index_map.size_local
    r_va = r[:n_va]   # trim to Va dofs if needed

    mask_core    = r_va <= R0
    mask_healthy = r_va >= R1
    mask_trans   = (~mask_core) & (~mask_healthy)

    # arrays already computed: stddev_cal, prior_stddev, var_red_frac,
    #                           cd.x.array, m_map_arr
    cc_true_arr = cd.x.array[:n_va]
    cc_map_arr  = m_map_arr[:n_va]
    rel_err_arr = np.abs(cc_map_arr - cc_true_arr) / np.maximum(np.abs(cc_true_arr), 1e-12)

    sep = "="*60
    print(f"\n  Fibrosis center: ({XC}, {YC}, {ZC}) mm")
    print(f"  Core r <= {R0} mm  |  Healthy r >= {R1} mm")
    print(f"  Nodes: core={mask_core.sum()}  transition={mask_trans.sum()}  "
          f"healthy={mask_healthy.sum()}")

    print(f"\n  {'Metric':<30} {'Core':>12} {'Transition':>12} {'Healthy':>12}")
    print("  " + "-"*68)

    for label, arr in [
        ("CC_true mean",            cc_true_arr),
        ("CC_map mean",             cc_map_arr),
        ("Rel. error mean",         rel_err_arr),
        ("Rel. error max",          rel_err_arr),
        ("Prior std dev mean (CC)", prior_stddev[:n_va]),
        ("Post. std dev mean (CC)", stddev_cal[:n_va]),
        ("Variance reduction mean", var_red_frac[:n_va]),
    ]:
        vals = []
        for mask in [mask_core, mask_trans, mask_healthy]:
            v = arr[mask]
            if len(v) == 0:
                vals.append("         N/A")
            elif "max" in label:
                vals.append(f"{v.max():>12.4f}")
            else:
                vals.append(f"{v.mean():>12.4f}")
        print(f"  {label:<30} {vals[0]} {vals[1]} {vals[2]}")

    # key question
    post_std_core    = stddev_cal[:n_va][mask_core].mean()
    post_std_healthy = stddev_cal[:n_va][mask_healthy].mean()
    vr_core          = var_red_frac[:n_va][mask_core].mean()
    vr_healthy       = var_red_frac[:n_va][mask_healthy].mean()
    er_core          = rel_err_arr[mask_core].mean()
    er_healthy       = rel_err_arr[mask_healthy].mean()

    print(f"\n  {sep}")
    print(f"  KEY QUESTION: Uncertainty larger/smaller in fibrosis core?")
    print(f"  {sep}")
    print(f"  Post. std dev -- Core   : {post_std_core:.4f} CC units")
    print(f"  Post. std dev -- Healthy: {post_std_healthy:.4f} CC units")
    if post_std_core > post_std_healthy:
        print(f"  → Uncertainty LARGER in core ({post_std_core/post_std_healthy:.2f}× higher)")
    else:
        print(f"  → Uncertainty SMALLER in core ({post_std_healthy/post_std_core:.2f}× lower)")
    print(f"\n  Variance reduction -- Core   : {vr_core:.3f}  ({100*vr_core:.1f}% reduced)")
    print(f"  Variance reduction -- Healthy: {vr_healthy:.3f}  ({100*vr_healthy:.1f}% reduced)")
    print(f"\n  MAP rel. error -- Core   : {er_core:.4f}  ({100*er_core:.1f}%)")
    print(f"  MAP rel. error -- Healthy: {er_healthy:.4f}  ({100*er_healthy:.1f}%)")

print("\n=== Done ===")
print("Output files:")
for f in ["fig1_MAP_convergence.png", "fig2_eigenvalue_decay.png",
          "fig3_variance.png", "out_uq_cardiac.xdmf",
          "out_uq_eigenvalues.txt", "out_uq_prior_variance.npy",
          "out_uq_posterior_variance.npy", "out_uq_newton_cg_history.txt"]:
    print(f"  {OUTPUT_DIR / f}")

