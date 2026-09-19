from mpi4py import MPI
from dolfinx import fem, default_scalar_type
from dolfinx.io import XDMFFile
from dolfinx.fem.petsc import NonlinearProblem, LinearProblem
from dolfinx.nls.petsc import NewtonSolver
import ufl
import numpy as np
import dolfinx
from pathlib import Path
import os
import argparse
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import cardiac_geometries
from ex03_ventricle_discrete_forward import prob_ventricle_passive_filling

from cardiac_utils import *
from hessian_ucq import HessianOperator
from verify_hessian import verify_hessian
from newton_cg_solver import inexact_newton_cg


# =============================================================================
# helper functions
# =============================================================================

def generate_cc_balaban(n_params, Pmax=6.0):
    """Gera condição inicial aleatória para CC no intervalo [1, Pmax]."""
    p = np.random.uniform(2.0, 10.0, n_params)
    return p.copy()


def solve_nl_prob(uh):
    """Resolve o problema de equilíbrio não-linear com rampa de carga."""
    forward_problem = NonlinearProblem(Fun, uh, bcs)
    solver = NewtonSolver(domain.comm, forward_problem)
    solver.atol = 1e-8
    solver.rtol = 1e-8

    ksp = solver.krylov_solver
    ksp.setType("preonly")
    ksp.getPC().setType("lu")

    load_steps  = 10
    target_load = -3.0
    loads = np.linspace(0, target_load, load_steps)

    for step in range(load_steps):
        p_endo.value = loads[step]
        num_its, converged = solver.solve(uh)
        assert converged
        print(f" load step {step} - newton its {num_its}")
    print(" ")


# =============================================================================
# geometry and mesh
# =============================================================================

geodir = Path("lv_ellipsoid")
geo = cardiac_geometries.mesh.lv_ellipsoid(
    outdir=geodir,
    create_fibers=True,
    fiber_space="DG_0",
    psize_ref=3,
    r_short_epi=10,
    fiber_angle_endo=40.0,
    fiber_angle_epi=-50.0
)

# =============================================================================
# command-line arguments
# =============================================================================

parser = argparse.ArgumentParser(
    description="ex04 — cardiac inverse problem: L-BFGS-B vs inexact Newton-CG"
)
parser.add_argument("--case-type",  type=str,   default="fibrosis",
                    choices=["fibrosis", "linear"],
                    help="True parameter field type (default: fibrosis)")
parser.add_argument("--num-nodes",  type=int,   default=64,
                    help="Number of measurement nodes (default: 64)")
parser.add_argument("--alpha",      type=float, default=100.0,
                    help="Regularization parameter (default: 100.0)")
parser.add_argument("--output-dir", type=str,   default=".",
                    help="Directory for output files (default: current dir)")
parser.add_argument("--ftol",       type=float, default=1e-20,
                    help="Relative J change tolerance for both methods (default: 1e-20)")
parser.add_argument("--gtol",       type=float, default=1e-8,
                    help="Gradient L-inf tolerance for both methods (default: 1e-8)")

args = parser.parse_args()

CASE_TYPE   = args.case_type
Nnodes      = args.num_nodes
alpha_value = args.alpha
OUTPUT_DIR  = Path(args.output_dir)
my_ftol     = args.ftol
my_gtol     = args.gtol

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n============================================================")
print(" INVERSE PROBLEM PARAMETERS")
print(f" case_type  = {CASE_TYPE}")
print(f" num_nodes  = {Nnodes}")
print(f" alpha      = {alpha_value}")
print(f" gtol       = {my_gtol}")
print(f" ftol       = {my_ftol}")
print(f" output_dir = {OUTPUT_DIR}")
print("============================================================\n")

# =============================================================================
# synthetic data (forward problem reference solution)
# =============================================================================

ud_arr, Fd_arr, cd = prob_ventricle_passive_filling(geo, ndofs_data=args.num_nodes, case_type=args.case_type)
domain = geo.mesh

# =============================================================================
# function spaces
# =============================================================================

V  = dolfinx.fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
Va = dolfinx.fem.functionspace(domain, ("Lagrange", 1))

du = ufl.TrialFunction(V)
v  = ufl.TestFunction(V)
uh = dolfinx.fem.Function(V)

# =============================================================================
# boundary conditions (base fixed)
# =============================================================================

facet_tags = geo.ffun
u_bc      = np.array((0,) * domain.geometry.dim, dtype=default_scalar_type)
base_dofs = fem.locate_dofs_topological(V, facet_tags.dim, facet_tags.find(5))
bcs       = [fem.dirichletbc(u_bc, base_dofs, V)]

# =============================================================================
# nodes selected for data measurement
# =============================================================================

dim        = domain.geometry.dim
dof_coords = V.tabulate_dof_coordinates().reshape(-1, dim)

selected_dofs, selected_coords = select_distributed_nodes(
    dof_coords, N=Nnodes, min_dist=4.0, seed=42
)

Npoints = len(selected_dofs)
print(f"Número de nodes: {Npoints}")
save_selected_nodes(selected_coords, selected_dofs)

# =============================================================================
# kinematics
# =============================================================================

d = len(uh)
I = ufl.variable(ufl.Identity(d))
F = ufl.variable(I + ufl.grad(uh))
J = ufl.variable(ufl.det(F))
C = ufl.variable(F.T * F)

f0 = geo.f0
s0 = geo.s0
n0 = geo.n0

# =============================================================================
# constitutive model (Holzapfel-Ogden passive + volumetric)
# =============================================================================

bf    = default_scalar_type(6.6)
bt    = default_scalar_type(4.0)
bfs   = default_scalar_type(2.6)
kappa = fem.Constant(domain, 1e2)

CC = fem.Function(Va)   # parameter field to be identified

e1, e2, e3 = f0, s0, n0
Cs = J**(-2/3) * F.T * F
Es = 0.5 * (Cs - I)

E11 = ufl.inner(Es*e1, e1)
E12 = ufl.inner(Es*e1, e2)
E13 = ufl.inner(Es*e1, e3)
E21 = ufl.inner(Es*e2, e1)
E22 = ufl.inner(Es*e2, e2)
E23 = ufl.inner(Es*e2, e3)
E31 = ufl.inner(Es*e3, e1)
E32 = ufl.inner(Es*e3, e2)
E33 = ufl.inner(Es*e3, e3)

Q = (
    bf  * E11**2
    + bt  * (E22**2 + E33**2 + E23**2 + E32**2)
    + bfs * (E12**2 + E21**2 + E13**2 + E31**2)
)

Wpassive      = CC / 2.0 * (ufl.exp(Q) - 1)
Wvolume       = kappa * (J * ufl.ln(J) - J + 1)
strain_energy = Wpassive + Wvolume
P             = ufl.diff(strain_energy, F)

# =============================================================================
# variational form (internal + endocardial pressure)
# =============================================================================

p_endo = fem.Constant(domain, 0.0)

metadata = {"quadrature_degree": 4}
ds = ufl.Measure('ds', domain=domain, subdomain_data=facet_tags, metadata=metadata)
dx = ufl.Measure("dx", domain=domain, metadata=metadata)

N     = ufl.FacetNormal(domain)
Gendo = -p_endo * ufl.inner(v, J * ufl.transpose(ufl.inv(F)) * N) * ds(6)
Fun   = ufl.inner(P, ufl.grad(v)) * dx + Gendo

# =============================================================================
# optimization functional (data misfit + Tikhonov regularization)
# =============================================================================

alpha = dolfinx.fem.Constant(domain, alpha_value)

volume_form = dolfinx.fem.form(
    dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(1.0)) * dx
)
volume_mesh = dolfinx.fem.assemble_scalar(volume_form)

# node -> cell map
dofmap_geo = domain.geometry.dofmap
Ncells     = domain.topology.index_map(domain.topology.dim).size_local
node_to_cells = {}
for cell_id in range(Ncells):
    for node in dofmap_geo[cell_id]:
        node_to_cells.setdefault(int(node), []).append(cell_id)

# DG_0 tensor space for reference deformation gradient Fd
TT        = dolfinx.fem.functionspace(domain, ("DG", 0, (dim, dim)))
Fd_func   = dolfinx.fem.Function(TT)
Fd_func.x.array[:] = 0.0

W0        = dolfinx.fem.functionspace(domain, ("DG", 0))
indicator = dolfinx.fem.Function(W0)
indicator.x.array[:] = 0.0

bs        = TT.dofmap.bs
dofmap_TT = TT.dofmap
dofmap_W0 = W0.dofmap

Fd = Fd_arr

for j, dof in enumerate(selected_dofs):
    cell_id     = node_to_cells[int(dof)][0]
    cell_dof_TT = dofmap_TT.cell_dofs(cell_id)[0]
    Fd_func.x.array[cell_dof_TT*bs : cell_dof_TT*bs + bs] = Fd_arr[j].flatten()
    cell_dofs_W0 = dofmap_W0.cell_dofs(cell_id)
    indicator.x.array[cell_dofs_W0] = 1.0

Fd_func.x.scatter_forward()
indicator.x.scatter_forward()

diffF       = F - Fd_func
Jdata       = 0.5 * (1.0 / volume_mesh) * indicator * ufl.inner(diffF, diffF) * dx
Jsmooth     = (1.0 / volume_mesh) * ufl.inner(ufl.grad(CC), ufl.grad(CC)) * dx
Jfunctional = Jdata + alpha * Jsmooth

# =============================================================================
# adjoint problem
# =============================================================================

lmbda    = dolfinx.fem.Function(V)
dFdu     = ufl.derivative(Fun, uh, du)
dFdu_adj = ufl.adjoint(dFdu)
dJdu     = ufl.derivative(Jfunctional, uh, v)

adj_problem = LinearProblem(dFdu_adj, -dJdu, bcs=bcs)
lmbda = adj_problem.solve()

# gradient wrt parameter CC
W    = Va
q    = ufl.TrialFunction(W)
dJdf = ufl.derivative(Jfunctional, CC, q)
dFdf = ufl.action(ufl.adjoint(ufl.derivative(Fun, CC, q)), lmbda)

dJdf_compiled = dolfinx.fem.form(dJdf)
dFdf_compiled = dolfinx.fem.form(dFdf)
dLdf = dolfinx.fem.Function(W)
Jh   = dolfinx.fem.form(Jfunctional)

# =============================================================================
# optimization -- cached forward + adjoint solve
# =============================================================================

vals_func  = []
Jits       = [0]
_last_grad = [None]
_cache_x   = None
_cache_J   = None
_cache_g   = None


def _solve_and_cache(x):
    global _cache_x, _cache_J, _cache_g
    if _cache_x is None or not np.allclose(x, _cache_x):
        CC.x.array[:] = x
        uh.x.array[:] = 0.0
        solve_nl_prob(uh)
        _cache_J = domain.comm.allreduce(fem.assemble_scalar(Jh), op=MPI.SUM)
        lmbda = adj_problem.solve()
        dLdf.x.array[:] = 0.0
        dolfinx.fem.assemble_vector(dLdf.x.array, dJdf_compiled)
        dolfinx.fem.assemble_vector(dLdf.x.array, dFdf_compiled)
        dLdf.x.scatter_forward()
        _cache_x = x.copy()
        _cache_g = dLdf.x.array.copy()
        _last_grad[0] = dLdf.x.array.copy()


def eval_J(x):
    _solve_and_cache(x)
    return _cache_J


def eval_gradient(x):
    _solve_and_cache(x)
    return _cache_g


def callback(intermediate_result):
    fval = intermediate_result.fun
    vals_func.append(fval)
    Jits[0] += 1
    gnorm = np.linalg.norm(_last_grad[0], ord=np.inf) if _last_grad[0] is not None else float('nan')
    print(f"optimization iteration {Jits[0]}")
    print(f"  J            : {fval:.6e}")
    print(f"  ||grad||_inf : {gnorm:.6e}   (gtol = {my_gtol})")
    print()


# =============================================================================
# multi-start L-BFGS-B optimization
# =============================================================================

np.random.seed(57)

CC0      = fem.Function(Va)
ne       = int(np.shape(CC.x.array)[0])
cbounds  = [(1.0, 16.0)] * ne
n_starts = 1

best_J   = np.inf
best_CC  = None

print("\nbegin L-BFGS-B optimization\n")

for k in range(n_starts):
    print(f"\n=== MULTI-START {k+1}/{n_starts} ===\n")

    cc_init = generate_cc_balaban(len(CC.x.array), Pmax=16.0)
    CC.x.array[:] = cc_init
    CC.x.scatter_forward()
    CC0.x.array[:] = cc_init
    uh.x.array[:] = 0.0

    opt_sol = minimize(
        eval_J,
        CC.x.array,
        jac=eval_gradient,
        method="L-BFGS-B",
        callback=callback,
        bounds=cbounds,
        options={"ftol": my_ftol, "gtol": my_gtol, "maxiter": 200, "disp": True},
    )



    if opt_sol.fun < best_J:
        best_J  = opt_sol.fun
        best_CC = opt_sol.x.copy()

    # print scipy's stopping reason
    print(f"L-BFGS-B J final     : {opt_sol.fun}")
    print(f"L-BFGS-B warnflag    : {opt_sol.status}")
    print(f"L-BFGS-B message     : {opt_sol.message}")
    print(f"L-BFGS-B nit         : {opt_sol.nit}")
    print(f"L-BFGS-B nfev        : {opt_sol.nfev}")
    print(f"L-BFGS-B ||grad||_inf: {np.linalg.norm(eval_gradient(best_CC), ord=np.inf):.6e}")

# =============================================================================
# post-optimization: forward solve with best L-BFGS-B parameters
# =============================================================================

print("\n=== L-BFGS-B best result ===")
print(f"best J: {best_J}")

CC.x.array[:] = best_CC
CC.x.scatter_forward()
uh.x.array[:] = 0.0
uh.x.scatter_forward()
solve_nl_prob(uh)

# save L-BFGS-B estimated field (will be written to XDMF at the end)
CC_lbfgsb = fem.Function(Va, name="c_estimated_lbfgsb")
CC_lbfgsb.x.array[:] = best_CC.copy()
CC_lbfgsb.x.scatter_forward()

uh_lbfgsb = fem.Function(V, name="displacement_lbfgsb")
uh_lbfgsb.x.array[:] = uh.x.array.copy()
uh_lbfgsb.x.scatter_forward()

# =============================================================================
# build Hessian operator at the L-BFGS-B MAP point
# =============================================================================

# recompute adjoint exactly at (CC*, uh*) -- the last _solve_and_cache call
# during optimization may not have been exactly at best_CC
lmbda = adj_problem.solve()
lmbda.x.scatter_forward()

print("\nRecomputed adjoint at the L-BFGS-B MAP point.")

Hop = HessianOperator(
    Fun, Jfunctional, uh, CC, lmbda,
    V, Va, facet_tags, domain
)

print(f"HessianOperator built. Parameter dofs: {Hop.ndofs_m}")

# =============================================================================
# verify Hessian via finite differences
# =============================================================================

CC_map = CC.x.array.copy()

max_rel_err = verify_hessian(
    Hop, eval_gradient, CC_map,
    n_directions=3, eps=0.2, seed=0
)

if max_rel_err >= 5e-2:
    raise RuntimeError(
        f"Hessian verification FAILED (max rel err = {max_rel_err:.4e}). "
        "Do not proceed with Newton-CG until hessian_ucq.py is fixed."
    )

print(f"\nHessian verification PASSED (max relative error = {max_rel_err:.4e})")

# =============================================================================
# inexact Newton-CG from the SAME initial guess as L-BFGS-B
# =============================================================================


def build_hop_at_current_state():
    """Rebuild HessianOperator at current uh/CC (updated by eval_gradient)."""
    lmbda_current = adj_problem.solve()
    lmbda_current.x.scatter_forward()
    return HessianOperator(
        Fun, Jfunctional, uh, CC, lmbda_current,
        V, Va, facet_tags, domain
    )


print("\n" + "=" * 70)
print("Inexact Newton-CG  (same initial guess as L-BFGS-B)")
print("=" * 70)

x0_newton = cc_init.copy()   # same random start as L-BFGS-B

x_newton_opt, newton_history = inexact_newton_cg(
    eval_J, eval_gradient, build_hop_at_current_state,
    x0=x0_newton,
    max_outer_iter=200,
    grad_tol=my_gtol,          # same as L-BFGS-B gtol
    ftol=my_ftol,              # same as L-BFGS-B ftol
    cg_tol=0.1,
    cg_maxiter=200,            # increased from 50 (needed near optimum)
    bounds=(1.0, 16.0),
    eisenstat_walker=True,     # adaptive CG tolerance
)

J_newton_final = eval_J(x_newton_opt)
print(f"\nNewton-CG  final J : {J_newton_final:.6e}")
print(f"L-BFGS-B   final J : {best_J:.6e}")

np.savetxt(
    str(OUTPUT_DIR / "out_ex04_newton_cg_history.txt"),
    np.array([[h["iter"], h["J"], h["grad_norm"], h["cg_iters"]]
              for h in newton_history]),
    header="iter J grad_norm cg_iters"
)

# save Newton-CG estimated field
CC_newton = fem.Function(Va, name="c_estimated_newtoncg")
CC_newton.x.array[:] = x_newton_opt.copy()
CC_newton.x.scatter_forward()

# forward solve at Newton-CG solution to get the corresponding displacement
CC.x.array[:] = x_newton_opt
CC.x.scatter_forward()
uh.x.array[:] = 0.0
uh.x.scatter_forward()
solve_nl_prob(uh)

uh_newton = fem.Function(V, name="displacement_newtoncg")
uh_newton.x.array[:] = uh.x.array.copy()
uh_newton.x.scatter_forward()

# =============================================================================
# error analysis (evaluated at the L-BFGS-B solution)
# =============================================================================

# restore to L-BFGS-B MAP for error analysis (already the "main" result)
CC.x.array[:] = best_CC
CC.x.scatter_forward()
uh.x.array[:] = uh_lbfgsb.x.array.copy()
uh.x.scatter_forward()

print("\nsummary of results (L-BFGS-B estimate):\n")

error_lbfgsb = fem.Function(Va, name="c_error_lbfgsb")
abs_err_l = np.abs(best_CC - cd.x.array[:])
# true pointwise relative error: |CC_est[i] - CC_true[i]| / |CC_true[i]|
# avoid division by zero with a small floor
denom_l   = np.maximum(np.abs(cd.x.array[:]), 1e-12)
error_lbfgsb.x.array[:] = abs_err_l / denom_l
error_lbfgsb.x.scatter_forward()

rel_error_max_l = (abs_err_l / denom_l).max()
print(f"L-BFGS-B max pointwise rel error (CC): {rel_error_max_l:.8e}")

error_newton = fem.Function(Va, name="c_error_newtoncg")
abs_err_n = np.abs(x_newton_opt - cd.x.array[:])
denom_n   = np.maximum(np.abs(cd.x.array[:]), 1e-12)
error_newton.x.array[:] = abs_err_n / denom_n
error_newton.x.scatter_forward()

rel_error_max_n = (abs_err_n / denom_n).max()
print(f"Newton-CG  max pointwise rel error (CC): {rel_error_max_n:.8e}")

Jdata_value   = dolfinx.fem.assemble_scalar(dolfinx.fem.form(Jdata))
Jsmooth_value = dolfinx.fem.assemble_scalar(dolfinx.fem.form(Jsmooth))
print(f"\nalpha (reg.)   : {float(alpha):.8e}")
print(f"J data         : {Jdata_value:.8e}")
print(f"J smooth       : {Jsmooth_value:.8e}")

np.savetxt(str(OUTPUT_DIR / "out_ex04_lbfgsb_history.txt"), vals_func)

# =============================================================================
# output XDMF -- all fields, clearly identified
# =============================================================================

save_results_xdmf(domain, {
    # true / reference
    "c_true"               : cd,
    # initial guess (same for both methods)
    "c_init"               : CC0,
    # L-BFGS-B result
    "c_estimated_lbfgsb"   : CC_lbfgsb,
    "displacement_lbfgsb"  : uh_lbfgsb,
    "c_error_lbfgsb"       : error_lbfgsb,
    # Newton-CG result
    "c_estimated_newtoncg" : CC_newton,
    "displacement_newtoncg": uh_newton,
    "c_error_newtoncg"     : error_newton,
    # measurement points: 1.0 at cells containing a sensor, 0.0 elsewhere
    "measurement_indicator": indicator,
    # reference deformation gradient at measurement cells
    "Fd_reference"         : Fd_func,
}, filename=str(OUTPUT_DIR / "out_ex04_novo.xdmf"))

# =============================================================================
# convergence comparison plot
# =============================================================================

newton_J = [h["J"] for h in newton_history]

plt.figure(figsize=(6, 5))
plt.semilogy(newton_J, "o-", label="Newton-CG")
plt.semilogy(vals_func, ".-", label="L-BFGS-B")
plt.xlabel("outer iteration")
plt.ylabel("J")
plt.title("Convergence: Newton-CG vs L-BFGS-B")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "out_ex04_convergence_comparison.png", dpi=150)
print("\nSaved: out_ex04_convergence_comparison.png")

# =============================================================================
# deformation gradient error at selected nodes (L-BFGS-B solution)
# =============================================================================

Fh_final = compute_Fh(domain, uh_lbfgsb, selected_dofs)
print("\nDeformation gradient error (L-BFGS-B):")
print_F_error_statistics(Fh_final, Fd)

print("\nend")
