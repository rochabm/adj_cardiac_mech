from mpi4py import MPI
from dolfinx import fem, default_scalar_type
from dolfinx.io import XDMFFile
from dolfinx.fem.petsc import NonlinearProblem, LinearProblem
from dolfinx.nls.petsc import NewtonSolver
import ufl
import numpy as np
import dolfinx
from pathlib import Path
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import cardiac_geometries
from ex03_ventricle_discrete import prob_ventricle_passive_filling

from cardiac_utils import *
from hessian_ucq import HessianOperator
from verify_hessian import verify_hessian, model_verify_hippylib_style
from newton_cg_solver import inexact_newton_cg


# =============================================================================
# helper functions
# =============================================================================

def generate_cc_balaban(n_params, Pmax=6.0):
    """Gera condição inicial aleatória para CC no intervalo [1, Pmax]."""
    p = np.random.uniform(1.0, Pmax, n_params)
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
# synthetic data (forward problem reference solution)
# =============================================================================

Nnodes = 32

ud_arr, Fd_arr, cd = prob_ventricle_passive_filling(geo, ndofs_data=Nnodes)
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

alpha = dolfinx.fem.Constant(domain, 1e-3)

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
my_gtol    = 1e-8


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

np.random.seed(42)

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
        options={"gtol": my_gtol, "maxiter": 200, "disp": True},
    )

    print(f"J final = {opt_sol.fun}")

    if opt_sol.fun < best_J:
        best_J  = opt_sol.fun
        best_CC = opt_sol.x.copy()

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

# -----------------------------------------------------------------------
# hIPPYlib-style full FD check at a RANDOM point (not the MAP)
# -----------------------------------------------------------------------
# hIPPYlib's modelVerify is intentionally run at a random m0 far from
# the optimum, where:
#   - ||grad_xh|| and ||H*h|| are large and well-conditioned
#   - the FD numerics are clean and the full "V shape" is visible
# Running at the MAP point (as above) compresses the V because
# grad(J) ~ 0 there, making the relative gradient error plot degenerate.

rng_check = np.random.default_rng(123)
CC_random = rng_check.uniform(1.0, 16.0, size=len(CC_map))

print("\n" + "=" * 70)
print("hIPPYlib-style Hessian FD check at a RANDOM parameter point")
print("(not the MAP -- gives a cleaner V-shape in the log-log plot)")
print("=" * 70)

# rebuild Hop at the random point (uh and lmbda must be consistent with CC_random)
eval_gradient(CC_random)                 # updates uh, lmbda internally via _solve_and_cache
lmbda_rand = adj_problem.solve()
lmbda_rand.x.scatter_forward()
Hop_rand = HessianOperator(
    Fun, Jfunctional, uh, CC, lmbda_rand,
    V, Va, facet_tags, domain
)

eps_arr, err_grad, err_H, symm_err = model_verify_hippylib_style(
    Hop_rand, eval_J, eval_gradient, CC_random,
    n_eps=32,
    max_eps=0.25,
    seed=0,
    save_plot="out_ex04_hessian_FD_check_random_point.png"
)

# restore CC/uh to the MAP point before continuing with Newton-CG
CC.x.array[:] = CC_map
CC.x.scatter_forward()
uh.x.array[:] = uh_lbfgsb.x.array.copy()
uh.x.scatter_forward()

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
    grad_tol=my_gtol,
    cg_tol=0.1,            # initial tolerance (tightened by Eisenstat-Walker)
    cg_maxiter=200,        # increased from 50 -- needed near the optimum
    bounds=(1.0, 16.0),
    eisenstat_walker=True, # adaptive inner CG tolerance
)

J_newton_final = eval_J(x_newton_opt)
print(f"\nNewton-CG  final J : {J_newton_final:.6e}")
print(f"L-BFGS-B   final J : {best_J:.6e}")

np.savetxt(
    "out_ex04_newton_cg_history.txt",
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
# Phase A: UQ via Laplace approximation at the L-BFGS-B MAP point
# =============================================================================
#
# We center the UQ at the L-BFGS-B MAP point (best_CC / uh_lbfgsb), which
# is the "main" deterministic result. The Newton-CG solution is used for
# comparison only.
#
# Step A1: rebuild Hop at the L-BFGS-B MAP point (uh and CC must be
#          consistent -- restore them first)
# Step A2: randomized low-rank eigendecomposition of H
# Step A3: pointwise posterior variance (Hutchinson + matrix-free CG)

from hessian_ucq import randomized_eigensolver, pointwise_variance

# --- restore to L-BFGS-B MAP (uh_lbfgsb / best_CC) ---
CC.x.array[:] = best_CC
CC.x.scatter_forward()
uh.x.array[:] = uh_lbfgsb.x.array.copy()
uh.x.scatter_forward()

lmbda_map = adj_problem.solve()
lmbda_map.x.scatter_forward()

Hop_map = HessianOperator(
    Fun, Jfunctional, uh, CC, lmbda_map,
    V, Va, facet_tags, domain
)

print(f"\n[Phase A/B] HessianOperator at MAP point. Parameter dofs: {Hop_map.ndofs_m}")

from hessian_ucq import (randomized_eigensolver, pointwise_variance,
                          HpriorOperator, generalized_eigensolver,
                          woodbury_pointwise_variance)

k_eig      = 50
p_oversamp = 20

# -----------------------------------------------------------------------
# Step A2 -- Phase A: standard randomized eigensolver (kept for reference)
# -----------------------------------------------------------------------

print(f"\n[Phase A] Standard eigensolver H·v = λ·v (k={k_eig}, p={p_oversamp})...")
eigvals_std, eigvecs_std = randomized_eigensolver(Hop_map, k=k_eig, p=p_oversamp, seed=0)
np.savetxt("out_ex04_eigenvalues_standard.txt", eigvals_std)
print(f"  Top 5 eigenvalues (standard, not H_prior-normalized): {eigvals_std[:5]}")

# -----------------------------------------------------------------------
# Step B1 -- Build H_prior operator (prior precision matrix)
# -----------------------------------------------------------------------

print(f"\n[Phase B] Building H_prior operator...")
Hprior = HpriorOperator(Va, alpha, volume_mesh, dx, domain)
print(f"  H_prior assembled and LU-factorized. ndofs = {Hprior.ndofs}")

# -----------------------------------------------------------------------
# Step B2 -- Generalized eigensolver: H_misfit · v = λ · H_prior · v
# -----------------------------------------------------------------------

print(f"\n[Phase B] Generalized eigensolver H_misfit·v = λ·H_prior·v "
      f"(k={k_eig}, p={p_oversamp})...")

eigvals_gen, eigvecs_gen = generalized_eigensolver(
    Hop_map, Hprior, k=k_eig, p=p_oversamp, seed=0
)
np.savetxt("out_ex04_eigenvalues_generalized.txt", eigvals_gen)

# -----------------------------------------------------------------------
# Step B3 -- Woodbury pointwise variance (fast: ~350 LU solves)
# -----------------------------------------------------------------------

print(f"\n[Phase B] Woodbury posterior variance...")
prior_var, post_var_woodbury, correction = woodbury_pointwise_variance(
    Hprior, eigvals_gen, eigvecs_gen, n_prior_samples=300, seed=9
)

# -----------------------------------------------------------------------
# Step A3 -- Phase A: Hutchinson+CG variance (kept for comparison)
# -----------------------------------------------------------------------

n_samples_var = 150
print(f"\n[Phase A] Hutchinson+CG posterior variance ({n_samples_var} samples)...")
var_array = pointwise_variance(
    Hop_map, n_samples=n_samples_var, seed=1,
    cg_tol=1e-6, cg_maxiter=300, verbose_cg=False
)

# -----------------------------------------------------------------------
# wrap all variance fields in dolfinx Functions for output
# -----------------------------------------------------------------------

# Phase A (Hutchinson+CG, raw scale)
variance_fun = fem.Function(Va, name="posterior_variance")
variance_fun.x.array[:] = var_array
variance_fun.x.scatter_forward()

stddev_fun = fem.Function(Va, name="posterior_stddev")
stddev_fun.x.array[:] = np.sqrt(np.clip(var_array, 0.0, None))
stddev_fun.x.scatter_forward()

# Phase B -- Woodbury (raw scale, H_prior-based)
prior_var_fun = fem.Function(Va, name="prior_variance")
prior_var_fun.x.array[:] = prior_var
prior_var_fun.x.scatter_forward()

post_var_wb_fun = fem.Function(Va, name="posterior_variance_woodbury")
post_var_wb_fun.x.array[:] = np.clip(post_var_woodbury, 0.0, None)
post_var_wb_fun.x.scatter_forward()

post_std_wb_fun = fem.Function(Va, name="posterior_stddev_woodbury")
post_std_wb_fun.x.array[:] = np.sqrt(np.clip(post_var_woodbury, 0.0, None))
post_std_wb_fun.x.scatter_forward()

correction_fun = fem.Function(Va, name="variance_reduction")
correction_fun.x.array[:] = correction
correction_fun.x.scatter_forward()

print(f"\n  [Phase B Woodbury] prior_var   : min={prior_var.min():.3e}, max={prior_var.max():.3e}")
print(f"  [Phase B Woodbury] post_var    : min={post_var_woodbury.min():.3e}, max={post_var_woodbury.max():.3e}")
print(f"  [Phase B Woodbury] correction  : min={correction.min():.3e}, max={correction.max():.3e}")

np.save("out_ex04_posterior_variance.npy", var_array)
np.save("out_ex04_posterior_variance_woodbury.npy", post_var_woodbury)
print("  Saved: out_ex04_posterior_variance.npy (Phase A)")
print("  Saved: out_ex04_posterior_variance_woodbury.npy (Phase B)")


# =============================================================================
# error analysis (evaluated at the L-BFGS-B solution)
# =============================================================================

# CC and uh are already at best_CC / uh_lbfgsb from the Phase A setup above

print("\nsummary of results (L-BFGS-B estimate):\n")
error_lbfgsb = fem.Function(Va, name="c_error_lbfgsb")
error_lbfgsb.x.array[:] = np.abs(best_CC - cd.x.array[:])

abs_err       = np.abs(error_lbfgsb.x.array)
i_max         = np.argmax(abs_err)
den           = abs(cd.x.array[i_max])
rel_error_max = abs_err[i_max] / den
error_lbfgsb.x.array[:] /= den

print(f"L-BFGS-B max rel error (CC): {rel_error_max:.8e}")

error_newton = fem.Function(Va, name="c_error_newtoncg")
error_newton.x.array[:] = np.abs(x_newton_opt - cd.x.array[:])
abs_err_n  = np.abs(error_newton.x.array)
i_max_n    = np.argmax(abs_err_n)
den_n      = abs(cd.x.array[i_max_n])
error_newton.x.array[:] /= den_n
print(f"Newton-CG  max rel error (CC): {abs_err_n[i_max_n]/den_n:.8e}")

Jdata_value   = dolfinx.fem.assemble_scalar(dolfinx.fem.form(Jdata))
Jsmooth_value = dolfinx.fem.assemble_scalar(dolfinx.fem.form(Jsmooth))
print(f"\nalpha (reg.)   : {float(alpha):.8e}")
print(f"J data         : {Jdata_value:.8e}")
print(f"J smooth       : {Jsmooth_value:.8e}")

# =============================================================================
# Option 1: empirical sigma from the MAP residual
# =============================================================================
#
# If the noise model is Gaussian with unknown variance sigma^2, the maximum
# likelihood estimate of sigma^2 is obtained by matching the expected residual:
#
#     2 * Jdata(CC*) = N_data / sigma^2 * sigma^2  =  N_data
#
# But our Jdata already has a 1/volume_mesh normalization, so:
#
#     Jdata = (1/volume_mesh) * (1/(2*sigma^2)) * sum_i ||F_i - Fd_i||^2
#
# =>  sigma^2 = Jdata_value * volume_mesh / (N_data * 0.5 * (1/volume_mesh))
#
# More directly: in our form Jdata = 0.5*(1/volume_mesh)*indicator*||F-Fd||^2*dx
# the "raw" misfit is  raw_misfit = 2 * Jdata_value * volume_mesh
# and sigma^2_empirical = raw_misfit / N_data
#
# where N_data = number of scalar observations
#             = Nnodes (sensor points) * 9 (components of the 3x3 tensor F)

N_data          = Nnodes * 9    # 32 sensors × 9 components of F
raw_misfit      = 2.0 * Jdata_value * volume_mesh
sigma2_empirical = raw_misfit / N_data
sigma_empirical  = np.sqrt(sigma2_empirical)

print(f"\n=== Empirical noise estimate (Option 1) ===")
print(f"  N_data (scalar observations) : {N_data}")
print(f"  raw misfit sum               : {raw_misfit:.6e}")
print(f"  sigma^2 (empirical)          : {sigma2_empirical:.6e}")
print(f"  sigma   (empirical)          : {sigma_empirical:.6e}")
print(f"  Interpretation: the MAP solution fits the data to within")
print(f"  ~{sigma_empirical:.2e} in each component of F (deformation gradient).")

# calibrated variance and stddev
var_calibrated   = var_array * sigma2_empirical
stddev_calibrated = np.sqrt(np.clip(var_calibrated, 0.0, None))

print(f"\n  Calibrated posterior stddev (in units of CC):")
print(f"    min : {stddev_calibrated.min():.4e}")
print(f"    max : {stddev_calibrated.max():.4e}")
print(f"    mean: {stddev_calibrated.mean():.4e}")
print(f"  (Compare to CC true range: {cd.x.array.min():.3f} - {cd.x.array.max():.3f})")

variance_cal_fun = fem.Function(Va, name="posterior_variance_calibrated")
variance_cal_fun.x.array[:] = var_calibrated
variance_cal_fun.x.scatter_forward()

stddev_cal_fun = fem.Function(Va, name="posterior_stddev_calibrated")
stddev_cal_fun.x.array[:] = stddev_calibrated
stddev_cal_fun.x.scatter_forward()

np.save("out_ex04_posterior_variance_calibrated.npy", var_calibrated)
print("\n  Saved: out_ex04_posterior_variance_calibrated.npy")

np.savetxt("out_ex04_lbfgsb_history.txt", vals_func)

# =============================================================================
# output XDMF -- all fields, clearly identified
# =============================================================================

save_results_xdmf(domain, {
    # true / reference
    "c_true"                        : cd,
    "c_init"                        : CC0,
    # L-BFGS-B result
    "c_estimated_lbfgsb"            : CC_lbfgsb,
    "displacement_lbfgsb"           : uh_lbfgsb,
    "c_error_lbfgsb"                : error_lbfgsb,
    # Newton-CG result
    "c_estimated_newtoncg"          : CC_newton,
    "displacement_newtoncg"         : uh_newton,
    "c_error_newtoncg"              : error_newton,
    # Phase A -- Hutchinson+CG variance (raw scale)
    "posterior_variance"            : variance_fun,
    "posterior_stddev"              : stddev_fun,
    # Phase B -- Woodbury variance (H_prior-based, generalized eigenproblem)
    "prior_variance"                : prior_var_fun,
    "posterior_variance_woodbury"   : post_var_wb_fun,
    "posterior_stddev_woodbury"     : post_std_wb_fun,
    "variance_reduction"            : correction_fun,
    # Phase A calibrated (empirical sigma)
    "posterior_variance_calibrated" : variance_cal_fun,
    "posterior_stddev_calibrated"   : stddev_cal_fun,
}, filename="out_ex04_novo.xdmf")

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
plt.savefig("out_ex04_convergence_comparison.png", dpi=150)
print("\nSaved: out_ex04_convergence_comparison.png")

# =============================================================================
# deformation gradient error at selected nodes (L-BFGS-B solution)
# =============================================================================

Fh_final = compute_Fh(domain, uh_lbfgsb, selected_dofs)
print("\nDeformation gradient error (L-BFGS-B):")
print_F_error_statistics(Fh_final, Fd)

print("\nend")
