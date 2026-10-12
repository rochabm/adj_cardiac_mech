"""
ex05_uq_cardiac.py
-------------------
Bayesian UQ for the cardiac passive filling inverse problem.

Prior and regularization:
    gamma, delta : BiLaplacian prior parameters
            C = R^{-1} M R^{-1},   R = gamma*K + delta*M + beta*M_bndry
            precision H_prior = R M^{-1} R  (trace-class in 2D/3D)
    alpha : light regularization added to the MAP cost functional
            J = Jmisfit + alpha*Jsmooth  (conditions the Newton solve)

The BiLaplacian (squared-inverse-elliptic) covariance is the
mesh-independent, trace-class prior of Bui-Thanh et al. (2013).

Pipeline (UQ only -- MAP + posterior, no FD self-tests):
    1.  Geometry + FE spaces
    2.  Constitutive model (Holzapfel-Ogden passive)
    3.  Synthetic data (from prob_ventricle_passive_filling)
    4.  Cost functional J = Jmisfit + alpha*Jsmooth
    5.  BiLaplacian prior  C = R^{-1} M R^{-1}
    6.  Adjoint gradient
    7.  MAP via inexact Newton-CG
    8.  Hessian operator at MAP
    9.  Generalized eigenproblem H_misfit v = lambda H_prior v
   10.  Woodbury pointwise variance
   11.  Posterior samples
   12.  XDMF output + prior/posterior plots

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
from ex03_spring import prob_ventricle_passive_filling

from cardiac_utils import (select_distributed_nodes, compute_Fh,
                            save_results_xdmf, print_F_error_statistics,
                            save_selected_nodes)
from hessian_ucq import (HessianOperator, generalized_eigensolver,
                          woodbury_pointwise_variance)
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
parser.add_argument("--alpha",      type=float, default=0.0,
                    help="Optional Laplacian nugget for Newton conditioning "
                         "(default: 0.0; the BiLaplacian prior is the "
                         "regularizer)")
parser.add_argument("--gamma",      type=float, default=None,
                    help="Prior stiffness (gamma*K). If unset, derived from "
                         "--sigma and --rho.")
parser.add_argument("--delta",      type=float, default=None,
                    help="Prior mass (delta*M). If unset, derived from "
                         "--sigma and --rho.")
parser.add_argument("--sigma",      type=float, default=1.0,
                    help="Target prior marginal std sigma (CC units) used to "
                         "derive gamma/delta when they are not given "
                         "(default: 1.0)")
parser.add_argument("--rho",        type=float, default=10.0,
                    help="Target prior correlation length rho (mm) used to "
                         "derive gamma/delta when they are not given. Default "
                         "10.0 mm ~ the ground-truth fibrosis feature scale "
                         "(r1=10mm in ex03), so prior samples show similar "
                         "qualitative features to the ground truth.")
parser.add_argument("--noise-std",  type=float, default=1e-3,
                    help="Assumed measurement noise std on the deformation "
                         "gradient F, absolute (||F|| ~ O(1)). Enters the "
                         "likelihood as precision 1/noise^2. Smaller = trust "
                         "data more. ~1e-3 balances the data against a sigma=1 "
                         "prior here; 1e-2 is too weak (data ignored), 1e-4 "
                         "makes it strongly data-dominated (default: 1e-3).")
parser.add_argument("--m0",         type=float, default=3.0,
                    help="Prior mean (healthy CC value, default: 3.0)")
parser.add_argument("--load",       type=float, default=-10.0,
                    help="Endocardial target load. Larger |load| makes the "
                         "deformation depend more strongly on CC, improving "
                         "identifiability (default: -10.0; the old value -3.0 "
                         "left CC nearly unidentifiable).")
parser.add_argument("--kappa",      type=float, default=1e2,
                    help="Bulk (volumetric) penalty. If it dominates the "
                         "CC-scaled passive term the data cannot see CC; "
                         "lower it to expose CC (default: 1e2).")
parser.add_argument("--k-eig",      type=int,   default=50,
                    help="Number of eigenpairs for low-rank UQ (default: 50)")
parser.add_argument("--n-hutch",    type=int,   default=1500,
                    help="Hutchinson samples for diag(C) prior-variance "
                         "estimate. Higher = smoother variance fields and "
                         "fewer clipped-negative posterior variances "
                         "(error ~ 1/sqrt(n); default: 1500).")
parser.add_argument("--gtol",       type=float, default=1e-8)
parser.add_argument("--ftol",       type=float, default=1e-20)
parser.add_argument("--output-dir", type=str,   default=".")

args = parser.parse_args()

CASE_TYPE   = args.case_type
Nnodes      = args.num_nodes
alpha_value = args.alpha
m0_prior    = args.m0
noise_std   = args.noise_std
n_hutch     = args.n_hutch
target_load = args.load
kappa_bulk  = args.kappa
k_eig       = args.k_eig
my_gtol     = args.gtol
my_ftol     = args.ftol
OUTPUT_DIR  = Path(args.output_dir)


# -----------------------------------------------------------------------------
# BiLaplacian prior coefficients from target (sigma, rho)
#
# Covariance C = A^{-2}, A = -gamma*Laplacian + delta*I, in d dimensions.
# Standard Matern/Lindgren relations for the squared-inverse elliptic
# operator (hIPPYlib BiLaplacianComputeCoefficients), with power nu=2:
#     effective Matern smoothness  s = 2 - d/2
#     correlation length  rho = sqrt(8 s) * sqrt(gamma/delta)
#     marginal variance   sigma^2 = Gamma(s) / ( Gamma(2) (4 pi)^{d/2}
#                                    kappa^{2s} delta^2 ),  kappa^2 = delta/gamma
# Solving for (gamma, delta) given (sigma, rho):
# -----------------------------------------------------------------------------
def bilaplacian_gamma_delta(sigma, rho, d=3, nu=2):
    from math import gamma as gammafn, pi, sqrt
    s = nu - d / 2.0                      # effective smoothness (0.5 for d=3, nu=2)
    if s <= 0:
        raise ValueError("Need nu > d/2 for a trace-class BiLaplacian prior.")
    # Write A = gamma(-Lap) + delta = gamma(-Lap + kappa^2 I),  kappa^2 = delta/gamma.
    # The Whittle-Matern marginal variance for C = A^{-nu} on an unbounded
    # domain is
    #     sigma^2 = Gamma(s) / ( Gamma(nu) (4 pi)^{d/2} kappa^{2s} tau^2 ),
    # where tau is the LEADING coefficient of the operator, i.e. tau = gamma
    # (NOT delta). Using delta here understates sigma by a factor gamma/delta
    # (~25x for these values), which clamps the MAP at the prior mean.
    kappa = sqrt(8.0 * s) / rho           # rho = sqrt(8 s)/kappa
    # sigma^2 = Gamma(s) / (Gamma(nu)(4pi)^{d/2} kappa^{2s} gamma^2)
    # and kappa^2 = delta/gamma  ->  delta = gamma * kappa^2. Solve for gamma:
    const = gammafn(s) / (gammafn(nu) * (4.0 * pi) ** (d / 2.0) * kappa ** (2.0 * s))
    gamma = sqrt(const / (sigma ** 2))    # tau = gamma
    delta = gamma * kappa ** 2
    return gamma, delta

if args.gamma is not None and args.delta is not None:
    gamma_pr, delta_pr = args.gamma, args.delta
    prior_src = "explicit --gamma/--delta"
else:
    gamma_pr, delta_pr = bilaplacian_gamma_delta(args.sigma, args.rho, d=3)
    prior_src = f"derived from sigma={args.sigma}, rho={args.rho}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "="*60)
print(" CARDIAC UQ PARAMETERS (ex05)")
print(f"  case_type  = {CASE_TYPE}")
print(f"  num_nodes  = {Nnodes}")
print(f"  alpha      = {alpha_value}  (optional Laplacian nugget; 0 = prior only)")
print(f"  gamma      = {gamma_pr:.4e}  (prior stiffness)")
print(f"  delta      = {delta_pr:.4e}  (prior mass)")
print(f"  prior src  = {prior_src}")
print(f"  sigma,rho  = {args.sigma}, {args.rho}  (target prior std / corr. length)")
print(f"  m0_prior   = {m0_prior}  (prior mean -- healthy CC)")
print(f"  noise_std  = {noise_std}  (data noise; precision = 1/noise^2 = {1.0/noise_std**2:.3e})")
print(f"  load       = {target_load}  (endocardial target load)")
print(f"  kappa      = {kappa_bulk}  (bulk/volumetric penalty)")
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
    geo, ndofs_data=Nnodes, case_type=CASE_TYPE,
    target_load=target_load, kappa_bulk=kappa_bulk
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
# 4. Boundary conditions (spring/Robin on epi + base -- matches ex03_spring)
#    Base is NOT clamped. Robin springs on EPI (7) and BASE (5) are added to
#    the variational form below; an optional u_x=0 on the base removes the
#    longitudinal rigid-body mode. These MUST match ex03_spring exactly.
# =============================================================================

facet_tags  = geo.ffun
markers     = geo.markers
BASE_MARKER = markers["BASE"][0] if "BASE" in markers else 5
ENDO_MARKER = markers["ENDO"][0] if "ENDO" in markers else 6
EPI_MARKER  = markers["EPI"][0]  if "EPI"  in markers else 7

FIX_BASE_X = True
bcs = []
if FIX_BASE_X:
    base_x_dofs = fem.locate_dofs_topological(
        V.sub(0), facet_tags.dim, facet_tags.find(BASE_MARKER))
    bcs = [fem.dirichletbc(default_scalar_type(0.0), base_x_dofs, V.sub(0))]

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
kappa = fem.Constant(domain, float(kappa_bulk))

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
Gendo = -p_endo * ufl.inner(v, J * ufl.transpose(ufl.inv(F)) * N_fac) * ds(ENDO_MARKER)

# Robin springs on EPI and BASE -- must equal ex03_spring (k_epi = k_base = 1.0)
k_epi  = fem.Constant(domain, default_scalar_type(1.0))
k_base = fem.Constant(domain, default_scalar_type(1.0))
Grobin = (k_epi  * ufl.inner(uh, v) * ds(EPI_MARKER)
        + k_base * ufl.inner(uh, v) * ds(BASE_MARKER))

Fun   = ufl.inner(P, ufl.grad(v)) * dx + Gendo + Grobin

# =============================================================================
# 7. Forward solver (load stepping)
# =============================================================================

def solve_nl_prob(uh_func):
    forward_problem = NonlinearProblem(Fun, uh_func, bcs)
    solver = NewtonSolver(domain.comm, forward_problem)
    solver.atol = 1e-8;  solver.rtol = 1e-8
    ksp = solver.krylov_solver
    ksp.setType("preonly");  ksp.getPC().setType("lu")
    loads = np.linspace(0, target_load, 10)
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
# -------------------------------------------------------------------------
# Data misfit with an explicit noise model (likelihood precision).
#   Jdata = 1/(2 sigma_noise^2) * (1/vol) * indicator * ||F - Fd||^2 dx
# Without the 1/sigma_noise^2 weight the misfit is numerically tiny
# (O(1e-4)) compared with the prior penalty of moving off m0 (O(1)), so the
# MAP objective is minimized AT the prior mean even though CC IS
# identifiable (Jdata(CC_true) ~ 1e-32). The precision weight is the
# Bayesian statement "the data are trustworthy to +/- sigma_noise", and it
# is what lets the data outweigh the prior. sigma_noise is expressed as a
# fraction of the deformation-gradient scale (||F|| ~ O(1)).
# -------------------------------------------------------------------------
noise_prec  = dolfinx.fem.Constant(domain, 1.0 / (noise_std ** 2))
Jdata       = 0.5 * noise_prec * (1.0/volume_mesh) * indicator * ufl.inner(diffF, diffF) * dx
Jsmooth     = (1.0/volume_mesh) * ufl.inner(ufl.grad(CC), ufl.grad(CC)) * dx

# -------------------------------------------------------------------------
# IMPORTANT (Bayesian consistency):
#   The regularization IS the prior. We do NOT add alpha*Jsmooth (a plain
#   Laplacian) to the state/adjoint functional, because the eigenproblem
#   subtracts H_prior = R M^{-1} R and requires
#           H_full - H_prior = H_data   (pure data misfit Hessian).
#   Mixing a Laplacian regularizer into H_full would leave a spurious
#   (Laplacian - BiLaplacian) residual in H_misfit -- which is exactly
#   what produced lambda == 1 for all modes and zero variance reduction.
#
#   So the STATE/ADJOINT functional is data-only:  Jfunctional = Jdata.
#   The prior term  0.5 (m-m0)^T H_prior (m-m0)  is added to J and its
#   gradient/Hessian OUTSIDE ufl (see eval_J / eval_gradient / build_hop),
#   since H_prior = R M^{-1} R is not a single ufl form.
#
#   alpha is retained only as an OPTIONAL tiny Laplacian nugget for Newton
#   conditioning; default alpha = 0 keeps H_full = H_data exactly.
# -------------------------------------------------------------------------
alpha_reg   = dolfinx.fem.Constant(domain, alpha_value)
if alpha_value != 0.0:
    Jfunctional = Jdata + alpha_reg * Jsmooth
else:
    Jfunctional = Jdata
Jh          = dolfinx.fem.form(Jfunctional)

# =============================================================================
# 9. BiLaplacian prior  (Bui-Thanh et al. 2013 / hIPPYlib convention)
#
#    Prior covariance is the SQUARED inverse of an elliptic operator:
#
#        C_prior = A^{-2},   A = -gamma*div(grad) + delta*I    (+ Robin BC)
#
#    Discretely, with  R = gamma*K + delta*M + beta*M_bndry  and mass M:
#
#        precision   H_prior = R M^{-1} R
#        covariance  C = H_prior^{-1} = R^{-1} M R^{-1}
#
#    This is TRACE-CLASS in 2D/3D (bounded, mesh-independent pointwise
#    variance), unlike the plain Laplacian R = gamma*K + delta*M whose
#    variance blows up under refinement.
#
#    Interpretation of the two knobs (Lindgren et al. 2011):
#        correlation length  rho  ∝ sqrt(gamma/delta)
#        marginal variance   sigma^2 ∝ delta^{-2} rho^{-d}   (d = dim)
#    The Robin coefficient  beta = sqrt(gamma*delta)/1.42  minimizes the
#    boundary variance inflation (Daon & Stadler 2018).
#
#    Interface (duck-typed for generalized_eigensolver + woodbury):
#        mult(v)     = H_prior * v      = R M^{-1} R v
#        solve(v)    = H_prior^{-1} * v = R^{-1} M R^{-1} v   (= C v)
#        diag_inv()  = diag(C)          = prior pointwise variance
#        _R_solve(v) = R^{-1} v         (used to build a covariance sample)
#        _R_mult(v)  = R v
# =============================================================================

class BiLaplacianPrior:
    """
    BiLaplacian (squared-inverse-elliptic) Gaussian prior on CC.

    H_prior = R M^{-1} R,  C = H_prior^{-1} = R^{-1} M R^{-1},
    with R = gamma*K + delta*M + beta*M_bndry  (Robin boundary term).
    """

    def __init__(self, Va, gamma, delta, dx, domain, facet_tags):
        n = Va.dofmap.index_map.size_local * Va.dofmap.index_map_bs
        self.ndofs = n
        self.gamma = float(gamma)
        self.delta = float(delta)

        q  = ufl.TrialFunction(Va)
        dq = ufl.TestFunction(Va)

        # optimal Robin coefficient (Daon & Stadler 2018, hIPPYlib default)
        beta = np.sqrt(self.gamma * self.delta) / 1.42
        ds_all = ufl.Measure("ds", domain=domain)

        K_form = ufl.inner(ufl.grad(q), ufl.grad(dq)) * dx
        M_form = ufl.inner(q, dq) * dx
        Mb_form = ufl.inner(q, dq) * ds_all      # boundary mass (Robin)

        R_form = (self.gamma * K_form
                  + self.delta * M_form
                  + beta * Mb_form)

        # R : elliptic operator (precision square-root building block)
        self._R = fem.petsc.assemble_matrix(fem.form(R_form))
        self._R.assemble()
        # M : mass matrix
        self._M = fem.petsc.assemble_matrix(fem.form(M_form))
        self._M.assemble()

        # LU solver for R (reused for every R^{-1} apply)
        self._kspR = PETSc.KSP().create(Va.mesh.comm)
        self._kspR.setOperators(self._R)
        self._kspR.setType("preonly")
        self._kspR.getPC().setType("lu")
        self._kspR.getPC().setFactorSolverType("mumps")
        self._kspR.setUp()

        # scratch vectors
        self._x = self._R.createVecRight()
        self._b = self._R.createVecRight()
        self._t = self._R.createVecRight()

        # lumped mass diagonal -> cheap M^{1/2} for prior sampling
        self._b.array[:] = 1.0
        self._M.mult(self._b, self._x)
        self._M_lump = np.maximum(self._x.array.copy(), 1e-30)
        self._M_sqrt_lump = np.sqrt(self._M_lump)

        print(f"  BiLaplacianPrior: C = R^-1 M R^-1, "
              f"gamma={self.gamma:.3g}, delta={self.delta:.3g}, "
              f"beta={beta:.3g}, ndofs={n}")

    # --- low-level building blocks -----------------------------------------

    def _R_mult(self, v):
        self._b.array[:] = v
        self._R.mult(self._b, self._x)
        return self._x.array.copy()

    def _R_solve(self, v):
        self._b.array[:] = v
        self._kspR.solve(self._b, self._x)
        return self._x.array.copy()

    def _M_mult(self, v):
        self._b.array[:] = v
        self._M.mult(self._b, self._x)
        return self._x.array.copy()

    # --- operator interface ------------------------------------------------

    def mult(self, v):
        """H_prior v = R M^{-1} R v.  (M^{-1} via lumped mass.)"""
        Rv   = self._R_mult(v)
        MinvRv = Rv / self._M_lump
        return self._R_mult(MinvRv)

    def solve(self, v):
        """C v = H_prior^{-1} v = R^{-1} M R^{-1} v.  (prior covariance apply)"""
        Rinv_v  = self._R_solve(v)
        MRinv_v = self._M_mult(Rinv_v)
        return self._R_solve(MRinv_v)

    def sample(self, rng):
        """
        Draw z ~ N(0, C).  With C = R^{-1} M R^{-1} and w ~ N(0, I):
            z = R^{-1} M^{1/2} w    (lumped M^{1/2})
        so that Cov(z) = R^{-1} M^{1/2} M^{1/2} R^{-1} = R^{-1} M R^{-1}.
        """
        w = rng.standard_normal(self.ndofs)
        return self._R_solve(self._M_sqrt_lump * w)

    def diag_inv(self, n_samples=300, seed=7):
        """
        Hutchinson estimate of diag(C) = diag(H_prior^{-1}) = prior variance.
            E[ z .* (C z) ] = diag(C),   z_i = +/-1.
        """
        rng = np.random.default_rng(seed)
        diag = np.zeros(self.ndofs)
        for _ in range(n_samples):
            z = rng.choice([-1.0, 1.0], size=self.ndofs).astype(float)
            diag += z * self.solve(z)
        return diag / n_samples

    def diag_cov_exact(self):
        """
        EXACT diagonal of the prior covariance  C = R^{-1} M R^{-1}.

        For a modest mesh this is far better than Hutchinson: it has ZERO
        stochastic speckle. We form R^{-1} column by column (the LU factor
        of R is reused, so each column is one cheap back-substitution),
        then diag(C)_i = sum_k Y_ki M_kl Y_li with Y = R^{-1}.

        Concretely, with Y = R^{-1} (Y is symmetric since R is SPD):
            C = Y M Y  ->  diag(C)_i = (Y (M (Y e_i)))_i
        We assemble Y once (n back-solves) and evaluate diag(Y M Y) with
        two dense matmuls. Memory is n^2 doubles; fine for n up to a few
        thousand. Falls back to Hutchinson (diag_inv) for very large n.
        """
        n = self.ndofs
        # Y = R^{-1} : solve R Y = I, one column at a time (LU already built)
        Y = np.empty((n, n))
        e = np.zeros(n)
        for j in range(n):
            e[j] = 1.0
            Y[:, j] = self._R_solve(e)
            e[j] = 0.0
        # M as dense (small mesh) via its action on the identity
        Mdense = np.empty((n, n))
        for j in range(n):
            e[j] = 1.0
            Mdense[:, j] = self._M_mult(e)
            e[j] = 0.0
        # C = Y M Y ; we only need its diagonal
        MY = Mdense @ Y          # (n,n)
        # diag(Y @ MY)_i = sum_k Y_ik (MY)_ki
        diagC = np.einsum("ik,ki->i", Y, MY)
        return np.abs(diagC)

    def diag_cov(self, n_samples=1500, seed=7, exact_max_dofs=4000):
        """
        Prior pointwise variance diag(C). Uses the EXACT column method when
        the mesh is small enough (no speckle), otherwise Hutchinson.
        """
        if self.ndofs <= exact_max_dofs:
            return self.diag_cov_exact()
        return self.diag_inv(n_samples=n_samples, seed=seed)

    def __del__(self):
        try:
            self._R.destroy(); self._M.destroy()
            self._x.destroy(); self._b.destroy(); self._t.destroy()
        except Exception:
            pass


prior = BiLaplacianPrior(Va, gamma_pr, delta_pr, dx, domain, facet_tags)


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

# prior mean vector (healthy CC), used by the prior term of J and its gradient
m0_vec = np.full(prior.ndofs, m0_prior)


def _solve_and_cache(x):
    global _cache_x, _cache_J, _cache_g
    if _cache_x is None or not np.allclose(x, _cache_x):
        CC.x.array[:] = x;  CC.x.scatter_forward()
        uh.x.array[:] = 0.0
        solve_nl_prob(uh)
        J_data = domain.comm.allreduce(fem.assemble_scalar(Jh), op=MPI.SUM)
        lmbda_new = adj_problem.solve()
        lmbda.x.array[:] = lmbda_new.x.array
        lmbda.x.scatter_forward()
        dLdf.x.array[:] = 0.0
        dolfinx.fem.assemble_vector(dLdf.x.array, dJdf_c)
        dolfinx.fem.assemble_vector(dLdf.x.array, dFdf_c)
        dLdf.x.scatter_forward()
        g_data = dLdf.x.array.copy()

        # --- BiLaplacian prior term: 0.5 (x-m0)^T H_prior (x-m0) ---
        #     grad_prior = H_prior (x - m0)
        dm      = x - m0_vec
        Hp_dm   = prior.mult(dm)
        J_prior = 0.5 * float(dm @ Hp_dm)

        _cache_J = J_data + J_prior
        _cache_g = g_data + Hp_dm
        _cache_x = x.copy()


def eval_J(x):
    _solve_and_cache(x);  return _cache_J

def eval_gradient(x):
    _solve_and_cache(x);  return _cache_g

# =============================================================================
# 10. MAP via inexact Newton-CG
# =============================================================================

print("\n" + "="*60)
print("MAP via inexact Newton-CG")
print("="*60)


class FullHessian:
    """
    Full MAP Hessian:  H_full v = H_data v + H_prior v.

    H_data comes from the reduced (adjoint) HessianOperator built on the
    data-only Jfunctional; H_prior = R M^{-1} R is the BiLaplacian prior.
    Newton-CG needs this full operator. The generalized eigensolver later
    receives this same object and subtracts H_prior to recover H_data, so
    H_misfit = H_full - H_prior = H_data exactly (no operator mismatch).
    """
    def __init__(self, Hdata, prior):
        self._Hdata = Hdata
        self._prior = prior
        self.ndofs_m = Hdata.ndofs_m

    def mult(self, v):
        return self._Hdata.mult(v) + self._prior.mult(v)


def build_hop():
    """Rebuild the full MAP Hessian (data + BiLaplacian prior) at (uh, CC)."""
    lmbda_cur = adj_problem.solve()
    lmbda_cur.x.scatter_forward()
    Hdata = HessianOperator(
        Fun, Jfunctional, uh, CC, lmbda_cur,
        V, Va, facet_tags, domain
    )
    return FullHessian(Hdata, prior)


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

# -------------------------------------------------------------------------
# Identifiability probe: how much does the DATA misfit actually change when
# CC is perturbed? If Jdata is (nearly) flat in CC, the parameter is not
# recoverable from these measurements at this load -- the MAP will sit at
# the prior mean and posterior ~ prior regardless of prior tuning.
# -------------------------------------------------------------------------
print("\n--- identifiability probe (data sensitivity to CC) ---")
Jdata_form = dolfinx.fem.form(Jdata)
def _jdata_only(x):
    CC.x.array[:] = x;  CC.x.scatter_forward()
    uh.x.array[:] = 0.0
    solve_nl_prob(uh)
    return domain.comm.allreduce(fem.assemble_scalar(Jdata_form), op=MPI.SUM)
_rngp = np.random.default_rng(0)
base = m_map_arr.copy()
Jd0  = _jdata_only(base)
for dmag in (0.5, 2.0):
    dvec = _rngp.standard_normal(prior.ndofs); dvec /= np.linalg.norm(dvec)
    Jdp  = _jdata_only(base + dmag * dvec)
    print(f"  |dCC|={dmag:4.1f}: Jdata {Jd0:.3e} -> {Jdp:.3e} "
          f"(rel change {abs(Jdp-Jd0)/(abs(Jd0)+1e-30):.2e})")
# also compare against the true field: does CC_true fit the data better?
Jd_true = _jdata_only(cd.x.array.copy())
print(f"  Jdata(CC_true) = {Jd_true:.3e}   Jdata(CC_map) = {Jd0:.3e}")
if Jd0 <= Jd_true:
    print("  => a flat/other CC fits the data as well as the truth: "
          "parameter is NOT identifiable from these measurements.")
    print("     (Raising prior sigma will NOT help; the forward map is "
          "insensitive to CC at this load. Consider higher load, more/other "
          "measurements, or measuring stress rather than F.)")
# restore MAP state
CC.x.array[:] = m_map_arr;  CC.x.scatter_forward()

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
# 11. Build Hessian operator at MAP (for the low-rank posterior)
# =============================================================================

# ensure uh/lmbda/CC consistent at MAP before building Hop
lmbda_map = adj_problem.solve();  lmbda_map.x.scatter_forward()
lmbda.x.array[:] = lmbda_map.x.array;  lmbda.x.scatter_forward()

Hdata_map = HessianOperator(
    Fun, Jfunctional, uh, CC, lmbda,
    V, Va, facet_tags, domain
)
# Full Hessian = H_data + H_prior; the eigensolver subtracts H_prior,
# recovering H_misfit = H_data exactly.
Hop_map = FullHessian(Hdata_map, prior)

# =============================================================================
# 13. Generalized eigenproblem H_misfit v = lambda H_prior v
#     Same as subsurface_bayesian.py: doublePassG equivalent
# =============================================================================

print("\n" + "="*60)
print("Generalized eigenproblem H_misfit v = λ H_prior v")
print("="*60)

# --- diagnostic: check H_data and H_prior magnitudes ---
# The generalized eigenvalues solve  H_data v = lambda H_prior v, so the
# meaningful balance is ||H_data v|| / ||H_prior v||, NOT ||H_full v||.
# (H_full = H_data + H_prior, so its ratio to H_prior is ~1 by construction.)
rng_diag = np.random.default_rng(123)
v_test   = rng_diag.standard_normal(prior.ndofs)
v_test  /= np.linalg.norm(v_test)
Hfull_v  = Hop_map.mult(v_test)
Rv       = prior.mult(v_test)
Hdata_v  = Hfull_v - Rv                       # H_data v
Jdata_now   = dolfinx.fem.assemble_scalar(dolfinx.fem.form(Jdata))
print(f"  Jdata at MAP        = {Jdata_now:.4e}")
print(f"  Diagnostic: ||H_data·v||   = {np.linalg.norm(Hdata_v):.4e}")
print(f"  Diagnostic: ||H_prior·v||  = {np.linalg.norm(Rv):.4e}  (H_prior = R M^-1 R)")
print(f"  Diagnostic: ratio          = {np.linalg.norm(Hdata_v)/np.linalg.norm(Rv):.4e}")
print(f"  (ratio >> 1: data-dominated; ratio ≈ 1: balanced; ratio << 1: prior-dominated)")
print(f"  If ratio << 1 everywhere, ALL lambda < 1 -> little variance reduction:")
print(f"    lower delta (less prior mass) and/or gamma to let data inform more modes.")

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

# prior variance = diag(C),  C = R^{-1} M R^{-1}  (BiLaplacian covariance).
# Uses the EXACT column-wise diagonal for this mesh (no Hutchinson speckle);
# falls back to Hutchinson only for very large meshes. This removes the
# blotchy, cell-by-cell texture in the variance plots and the spurious
# negative posterior variances that came from a noisy prior baseline.
print(f"  Computing prior variance diag(C) "
      f"({'exact' if prior.ndofs <= 4000 else f'{n_hutch}-sample Hutchinson'})...")
prior_var = prior.diag_cov(n_samples=n_hutch, seed=77)
print(f"  Prior variance: [{prior_var.min():.3e}, {prior_var.max():.3e}]")

# Woodbury posterior variance. woodbury returns its own Hutchinson
# prior_var estimate first; we discard it and reuse the prior_var above so
# prior and posterior share one consistent diagonal estimate.
_, _, correction = woodbury_pointwise_variance(
    prior, eigvals, eigvecs, n_prior_samples=n_hutch, seed=9
)
post_var = np.maximum(prior_var - correction, 0.0)

print(f"  Posterior variance: [{post_var.min():.3e}, {post_var.max():.3e}]")
print(f"  Correction        : [{correction.min():.3e}, {correction.max():.3e}]")

np.save(str(OUTPUT_DIR / "out_uq_prior_variance.npy"),    prior_var)
np.save(str(OUTPUT_DIR / "out_uq_posterior_variance.npy"), post_var)

# -----------------------------------------------------------------------
# Prior-variance units
# -----------------------------------------------------------------------
# (gamma, delta) are derived from the target marginal std sigma (see top of
# file), so diag(C) is ALREADY in CC^2 units and mesh-independent -- no
# post-hoc rescaling is needed. We therefore use scale_factor = 1 and only
# REPORT how the achieved prior std compares to the true CC spread, as a
# sanity check. (If gamma/delta were given explicitly, the amplitude is
# whatever those imply; set --sigma via the derivation to control it.)

cc_arr          = cd.x.array
CC_mean         = float(cc_arr.mean())
CC_std_physical = float(cc_arr.std())    # std of true CC field
if CC_std_physical < 1e-6:
    CC_std_physical = (cc_arr.max() - cc_arr.min()) / 2.0

scale_factor    = 1.0    # prior is self-calibrated through (sigma -> gamma,delta)

stddev_cal      = np.sqrt(np.clip(post_var, 0, None)) * scale_factor
prior_stddev    = np.sqrt(prior_var) * scale_factor

achieved_prior_std = float(np.sqrt(prior_var.mean()))
print(f"\n  CC_true range          : [{cc_arr.min():.3f}, {cc_arr.max():.3f}]")
print(f"  CC_true std            : {CC_std_physical:.4f}")
print(f"  Target prior std sigma : {args.sigma:.4f}  (requested)")
print(f"  Achieved prior std     : {achieved_prior_std:.4f}  (mean sqrt(diag C))")
print(f"  Prior stddev (CC units): [{prior_stddev.min():.4f}, {prior_stddev.max():.4f}]")
print(f"  Post. stddev (CC units): [{stddev_cal.min():.4f}, {stddev_cal.max():.4f}]")

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
#     m_sample = m_map + z_prior - U diag(sqrt(lambda/(lambda+1))) U^T H_prior z_prior
#
#     z_prior ~ N(0, C)   with C = R^{-1} M R^{-1}  (BiLaplacian covariance)
#     U are H_prior-orthonormal eigenvectors (U^T H_prior U = I), so the
#     projection coefficient is  U^T (H_prior z_prior),  NOT  U^T (R z_prior).
# =============================================================================

print("\n" + "="*60)
print("Posterior samples")
print("="*60)

nsamples  = 5
rng_s     = np.random.default_rng(99)
D_coeff   = np.sqrt(eigvals / (eigvals + 1.0))

all_samples = []
prior_samples = []          # keep the raw prior draws for the prior figure
for i in range(nsamples):
    # draw from prior: z ~ N(0, C),  C = R^{-1} M R^{-1}
    z_prior = prior.sample(rng_s)
    prior_samples.append(m0_prior + z_prior)
    # low-rank data correction (H_prior-orthonormal eigenbasis)
    Hz    = prior.mult(z_prior)          # H_prior z_prior
    coeff = eigvecs.T @ Hz
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

# -----------------------------------------------------------------------
# Prior visualization  (reproduces Bui-Thanh et al. 2013, Fig 6.3):
#   column 1 : +2 sigma and -2 sigma pointwise deviation fields
#              (bounds of the 95% pointwise prior credible interval)
#   columns 2-3 : independent prior samples  m0 + z,  z ~ N(0, C)
#   column 4 : ground-truth parameter field
# All panels share ONE diverging colormap and symmetric color limits,
# so samples can be compared qualitatively against the ground truth.
#
# Because the LV is a curved 3D surface, a flat 2D tricontourf of every
# dof would overlap the front and back walls. We therefore plot a thin
# SHORT-AXIS SLICE through the fibrosis center (as the paper shows slices
# through the Earth), giving an honest planar cross-section.
# -----------------------------------------------------------------------

coords_Va3 = Va.tabulate_dof_coordinates()   # (n, 3)

# short-axis slab: keep dofs within +/- dz of the fibrosis-center z-plane
z_center = -9.0                              # fibrosis center z (see ex03)
z_half   = 3.0                               # slab half-thickness (mm)
slab     = np.abs(coords_Va3[:, 2] - z_center) <= z_half
if slab.sum() < 50:                          # fallback: widen if too thin
    z_half = (coords_Va3[:, 2].max() - coords_Va3[:, 2].min()) / 4.0
    slab   = np.abs(coords_Va3[:, 2] - z_center) <= z_half
xs = coords_Va3[slab, 0]
ys = coords_Va3[slab, 1]

# fields to show, as deviations from the prior mean m0 (so 0 = mean)
plus_2s  = ( 2.0 * prior_stddev)[slab]
minus_2s = (-2.0 * prior_stddev)[slab]
gt_dev   = (cd.x.array - m0_prior)[slab]
sample_dev = [(s - m0_prior)[slab] for s in prior_samples[:2]]

# shared symmetric color scale from the largest deviation shown
vmax = max(np.abs(plus_2s).max(),
           np.abs(gt_dev).max(),
           max(np.abs(sd).max() for sd in sample_dev))
vmin = -vmax
cmap = "RdBu_r"

fig, axes = plt.subplots(2, 3, figsize=(13, 8))
panels = [
    (axes[0, 0], plus_2s,        "+2σ prior field"),
    (axes[1, 0], minus_2s,       "-2σ prior field"),
    (axes[0, 1], sample_dev[0],  "prior sample 1"),
    (axes[1, 1], sample_dev[1],  "prior sample 2"),
    (axes[0, 2], gt_dev,         "ground truth"),
]
last_cf = None
for ax, vals, title in panels:
    last_cf = ax.tricontourf(xs, ys, vals, levels=30,
                             cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal");  ax.set_xticks([]);  ax.set_yticks([])
axes[1, 2].axis("off")   # empty slot (paper's layout is asymmetric too)

# single shared colorbar (deviation from prior mean, CC units)
cbar = fig.colorbar(last_cf, ax=axes, fraction=0.025, pad=0.02)
cbar.set_label("deviation from prior mean  (CC units)")
plt.suptitle("BiLaplacian prior: ±2σ fields, samples, and ground truth "
             "(short-axis slice)", fontsize=12)
plt.savefig(str(OUTPUT_DIR / "fig4_prior_visualization.png"), dpi=150)
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
          "fig3_variance.png", "fig4_prior_visualization.png",
          "out_uq_cardiac.xdmf",
          "out_uq_eigenvalues.txt", "out_uq_prior_variance.npy",
          "out_uq_posterior_variance.npy", "out_uq_newton_cg_history.txt"]:
    print(f"  {OUTPUT_DIR / f}")

