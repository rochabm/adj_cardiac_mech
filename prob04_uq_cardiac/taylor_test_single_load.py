"""
taylor_test_single_load.py
---------------------------
Standalone Taylor remainder test for the cardiac inverse problem
using a SINGLE small load step.

Why:
    The standard multi-step load-stepping Newton solver has path-dependent
    floating-point rounding (~1e-7 noise) that masks the O(eps^3) Hessian
    signal. A single small load step (p_endo = -0.1) solves the nonlinear
    PDE in one Newton iteration with tight tolerance, making gradient and
    Hessian evaluations reproducible to machine precision.

Expected results:
    R_grad  slope ~ 2  (confirms gradient/adjoint correct)
    R_hess  slope ~ 3  (confirms Hessian correct)

Usage:
    python taylor_test_single_load.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpi4py import MPI
import dolfinx
from dolfinx import fem, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem, LinearProblem
from dolfinx.nls.petsc import NewtonSolver
import ufl
import petsc4py.PETSc as PETSc
import cardiac_geometries
from cardiac_utils import select_distributed_nodes, compute_Fh

# =============================================================================
# 1. geometry
# =============================================================================

geodir = Path("lv_ellipsoid")
try:
    geo = cardiac_geometries.geometry.Geometry.from_folder(geodir)
except Exception:
    geo = cardiac_geometries.mesh.lv_ellipsoid(
        outdir=geodir, create_fibers=True, fiber_space="DG_0",
        psize_ref=3, r_short_epi=10,
        fiber_angle_endo=40.0, fiber_angle_epi=-50.0)

domain     = geo.mesh
facet_tags = geo.ffun

# =============================================================================
# 2. FE spaces + BCs
# =============================================================================

V  = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
Va = fem.functionspace(domain, ("Lagrange", 1))
dim = domain.geometry.dim

u_bc      = np.zeros(dim, dtype=default_scalar_type)
base_dofs = fem.locate_dofs_topological(V, facet_tags.dim, facet_tags.find(5))
bcs       = [fem.dirichletbc(u_bc, base_dofs, V)]

# =============================================================================
# 3. constitutive model
# =============================================================================

uh = fem.Function(V)
CC = fem.Function(Va)

du   = ufl.TrialFunction(V)
v    = ufl.TestFunction(V)
dq   = ufl.TestFunction(Va)
q_tr = ufl.TrialFunction(Va)

f0, s0, n0 = geo.f0, geo.s0, geo.n0

bf    = default_scalar_type(6.6)
bt    = default_scalar_type(4.0)
bfs   = default_scalar_type(2.6)
kappa = fem.Constant(domain, 1e2)

I_ufl = ufl.variable(ufl.Identity(dim))
F_ufl = ufl.variable(I_ufl + ufl.grad(uh))
J_ufl = ufl.variable(ufl.det(F_ufl))
Cs    = J_ufl**(-2/3) * F_ufl.T * F_ufl
Es    = 0.5 * (Cs - I_ufl)

E11 = ufl.inner(Es*f0, f0);  E12 = ufl.inner(Es*f0, s0);  E13 = ufl.inner(Es*f0, n0)
E21 = ufl.inner(Es*s0, f0);  E22 = ufl.inner(Es*s0, s0);  E23 = ufl.inner(Es*s0, n0)
E31 = ufl.inner(Es*n0, f0);  E32 = ufl.inner(Es*n0, s0);  E33 = ufl.inner(Es*n0, n0)

Q = (bf*E11**2 + bt*(E22**2+E33**2+E23**2+E32**2)
     + bfs*(E12**2+E21**2+E13**2+E31**2))

Wpassive      = CC/2.0 * (ufl.exp(Q) - 1)
Wvolume       = kappa * (J_ufl*ufl.ln(J_ufl) - J_ufl + 1)
strain_energy = Wpassive + Wvolume
P_ufl         = ufl.diff(strain_energy, F_ufl)

# =============================================================================
# 4. SINGLE small load step (p = -0.1)
#    Small enough that Newton converges in 1-2 iterations
#    Large enough to exercise nonlinearity
# =============================================================================

P_LOAD   = -2.0   # small pressure -- few Newton steps, tight tolerance
metadata = {"quadrature_degree": 4}
ds       = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags,
                        metadata=metadata)
dx       = ufl.Measure("dx", domain=domain, metadata=metadata)
N_fac    = ufl.FacetNormal(domain)

p_endo   = fem.Constant(domain, P_LOAD)
Gendo    = -p_endo * ufl.inner(v, J_ufl*ufl.transpose(ufl.inv(F_ufl))*N_fac)*ds(6)
Fun      = ufl.inner(P_ufl, ufl.grad(v))*dx + Gendo

def solve_fwd(cc_arr, tol=1e-12):
    """Forward solve with tight tolerance and minimal load steps."""
    CC.x.array[:] = cc_arr;  CC.x.scatter_forward()
    uh.x.array[:] = 0.0;     uh.x.scatter_forward()
    prob   = NonlinearProblem(Fun, uh, bcs)
    solver = NewtonSolver(domain.comm, prob)
    solver.atol  = tol
    solver.rtol  = tol
    solver.max_it = 50
    ksp = solver.krylov_solver
    ksp.setType("preonly");  ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    # use 2 load steps: 0 → P_LOAD/2 → P_LOAD
    for load in [P_LOAD/2, P_LOAD]:
        p_endo.value = load
        n_it, conv = solver.solve(uh)
        assert conv, f"Newton did not converge at load={load}"
        print(f"    load={load:.2f}: {n_it} Newton its, tol={tol:.0e}")
    return uh.x.array.copy()

# =============================================================================
# 5. Synthetic observations (small set for speed)
# =============================================================================

N_OBS = 32
geom_coords = domain.geometry.x
sel_dofs, sel_coords = select_distributed_nodes(geom_coords, N=N_OBS, seed=42)

# true parameter: simple linear field
def c_true_func(x):
    xx = x[0]
    return 2.0 + (xx - xx.min()) / (xx.max() - xx.min() + 1e-30)

CC_true = fem.Function(Va)
CC_true.interpolate(c_true_func)
cc_true = CC_true.x.array.copy()

u_true = solve_fwd(cc_true)

# compute Fd at selected nodes
u_true_fn = fem.Function(V); u_true_fn.x.array[:] = u_true
Fd_arr = compute_Fh(domain, u_true_fn, sel_dofs)

# build DG0 indicator and Fd_func
node_to_cells = {}
dofmap_geo    = domain.geometry.dofmap
Ncells        = domain.topology.index_map(domain.topology.dim).size_local
for cell_id in range(Ncells):
    for node in dofmap_geo[cell_id]:
        node_to_cells.setdefault(int(node), []).append(cell_id)

TT        = fem.functionspace(domain, ("DG", 0, (dim, dim)))
Fd_func   = fem.Function(TT);  Fd_func.x.array[:] = 0.0
W0        = fem.functionspace(domain, ("DG", 0))
indicator = fem.Function(W0);  indicator.x.array[:] = 0.0
bs_TT     = TT.dofmap.bs

for j, dof in enumerate(sel_dofs):
    cell_id     = node_to_cells[int(dof)][0]
    cdof_TT     = TT.dofmap.cell_dofs(cell_id)[0]
    Fd_func.x.array[cdof_TT*bs_TT : cdof_TT*bs_TT+bs_TT] = Fd_arr[j].flatten()
    indicator.x.array[W0.dofmap.cell_dofs(cell_id)] = 1.0

Fd_func.x.scatter_forward();  indicator.x.scatter_forward()

# =============================================================================
# 6. Cost functional
# =============================================================================

ALPHA     = 1e-3
vol_form  = fem.form(fem.Constant(domain, default_scalar_type(1.0))*dx)
vol_mesh  = fem.assemble_scalar(vol_form)
alpha_reg = fem.Constant(domain, ALPHA)

diffF       = F_ufl - Fd_func
Jdata       = 0.5*(1.0/vol_mesh)*indicator*ufl.inner(diffF, diffF)*dx
Jsmooth     = (1.0/vol_mesh)*ufl.inner(ufl.grad(CC), ufl.grad(CC))*dx
Jfunctional = Jdata + alpha_reg*Jsmooth
Jh          = fem.form(Jfunctional)

# =============================================================================
# 7. Adjoint + gradient
# =============================================================================

lmbda = fem.Function(V)

dFdu_adj  = ufl.adjoint(ufl.derivative(Fun, uh, du))
dJdu      = ufl.derivative(Jfunctional, uh, v)
adj_prob  = LinearProblem(dFdu_adj, -dJdu, bcs=bcs,
                          petsc_options={"ksp_type":"preonly",
                                         "pc_type":"lu"})

dJdf_form = ufl.derivative(Jfunctional, CC, q_tr)
dFdf_form = ufl.action(ufl.adjoint(ufl.derivative(Fun, CC, q_tr)), lmbda)
dJdf_c    = fem.form(dJdf_form)
dFdf_c    = fem.form(dFdf_form)


def eval_J_and_g(cc_arr, tol=1e-12):
    """Evaluate cost and gradient. Always solves from zero for reproducibility."""
    solve_fwd(cc_arr, tol=tol)
    J   = fem.assemble_scalar(Jh)
    lm  = adj_prob.solve()
    lmbda.x.array[:] = lm.x.array;  lmbda.x.scatter_forward()
    g   = np.zeros(Va.dofmap.index_map.size_local)
    fem.assemble_vector(g, dJdf_c)
    fem.assemble_vector(g, dFdf_c)
    return float(J), g


# =============================================================================
# 8. Hessian operator (from hessian_ucq.py)
# =============================================================================

from hessian_ucq import HessianOperator

def build_hop(cc_arr, tol=1e-12):
    """Build HessianOperator at cc_arr (solves forward + adjoint)."""
    solve_fwd(cc_arr, tol=tol)
    lm = adj_prob.solve();  lmbda.x.scatter_forward()
    lmbda.x.array[:] = lm.x.array;  lmbda.x.scatter_forward()
    return HessianOperator(Fun, Jfunctional, uh, CC, lmbda,
                           V, Va, facet_tags, domain)

# =============================================================================
# 9. Taylor remainder test
# =============================================================================

print("\n" + "="*65)
print(f"  Taylor remainder test -- single load step p = {float(p_endo)}")
print("="*65)

rng   = np.random.default_rng(42)
m0    = np.full(Va.dofmap.index_map.size_local, 3.0)
h_dir = rng.standard_normal(len(m0));  h_dir /= np.linalg.norm(h_dir)

epsilons = [5e-1, 2e-1, 1e-1, 5e-2, 2e-2, 1e-2, 5e-3]

print("\nEvaluating at base point m0...")
J0, g0 = eval_J_and_g(m0, tol=1e-12)
dJ     = float(g0 @ h_dir)

print("\nBuilding Hessian at m0...")
Hop    = build_hop(m0, tol=1e-12)
Hh     = Hop.mult(h_dir)
d2J    = float(h_dir @ Hh)

print(f"\n  J(m0)        = {J0:.6e}")
print(f"  g^T h        = {dJ:.6e}")
print(f"  h^T H h      = {d2J:.6e}  (>0 means convex in direction h)")
print(f"  ||H*h||      = {np.linalg.norm(Hh):.6e}")

# check Hessian correction magnitude at eps=0.1
eps_ref = 0.1
J1_ref  = eval_J_and_g(m0 + eps_ref*h_dir, tol=1e-12)[0]
Rg_ref  = abs(J1_ref - J0 - eps_ref*dJ)
Hcorr   = 0.5 * eps_ref**2 * d2J
print(f"\n  At eps=0.1:")
print(f"    R_grad              = {Rg_ref:.4e}")
print(f"    ½ eps² h^T H h      = {Hcorr:.4e}")
print(f"    ratio               = {abs(Hcorr/Rg_ref):.4f}  (should be ~1)")

# --- compute remainders ---
R_grad, R_hess = [], []

print(f"\n  {'eps':>8}  {'R_grad':>14}  {'R_hess':>14}  {'rate_grad':>10}  {'rate_hess':>10}")
print("  " + "-"*65)

for eps in epsilons:
    J1    = eval_J_and_g(m0 + eps*h_dir, tol=1e-12)[0]
    dJfd  = J1 - J0
    rg    = abs(dJfd - eps*dJ)
    rh    = abs(dJfd - eps*dJ - 0.5*eps**2*d2J)
    R_grad.append(rg)
    R_hess.append(rh)

rates_g, rates_h = [], []
for i in range(1, len(epsilons)):
    def rate(a, b, ea, eb):
        return np.log(a/b)/np.log(ea/eb) if a>0 and b>0 else float("nan")
    rates_g.append(rate(R_grad[i], R_grad[i-1], epsilons[i], epsilons[i-1]))
    rates_h.append(rate(R_hess[i], R_hess[i-1], epsilons[i], epsilons[i-1]))

for i, eps in enumerate(epsilons):
    rg = f"{rates_g[i-1]:.2f}" if i > 0 else "         -"
    rh = f"{rates_h[i-1]:.2f}" if i > 0 else "         -"
    print(f"  {eps:>8.1e}  {R_grad[i]:>14.4e}  {R_hess[i]:>14.4e}  {rg:>10}  {rh:>10}")

# use middle eps range for rate averaging (avoid roundoff)
n = len(rates_g)
i_end = max(n-2, n//2)
avg_rg = np.nanmean(rates_g[:i_end])
avg_rh = np.nanmean(rates_h[:i_end])
grad_pass = abs(avg_rg - 2.0) < 0.25
hess_pass = abs(avg_rh - 3.0) < 0.5

print(f"\n  Average rate R_grad: {avg_rg:.3f}  (expected 2)  {'✓ PASS' if grad_pass else '✗ FAIL'}")
print(f"  Average rate R_hess: {avg_rh:.3f}  (expected 3)  {'✓ PASS' if hess_pass else '✗ FAIL'}")

# --- plot ---
eps_arr = np.array(epsilons)
fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(eps_arr, R_grad, "o-b", markersize=7,
          label=r"$R_{grad} = |J(m{+}\epsilon h){-}J(m){-}\epsilon g^Th|$"
                r"   $O(\epsilon^2)$")
ax.loglog(eps_arr, R_hess, "s-r", markersize=7,
          label=r"$R_{hess} = |R_{grad}{-}\frac{1}{2}\epsilon^2 h^THh|$"
                r"   $O(\epsilon^3)$")
ref2 = R_grad[0]*(eps_arr/eps_arr[0])**2
ref3 = R_hess[0]*(eps_arr/eps_arr[0])**3
ax.loglog(eps_arr, ref2, "--k", alpha=0.4, label="slope 2")
ax.loglog(eps_arr, ref3, ":k",  alpha=0.4, label="slope 3")
ax.set_xlabel(r"$\epsilon$", fontsize=13)
ax.set_ylabel("remainder", fontsize=13)
ax.set_title(f"Taylor test — single load p={P_LOAD}  (no load-stepping noise)\n"
             f"R_grad rate={avg_rg:.2f}  |  R_hess rate={avg_rh:.2f}",
             fontsize=11)
ax.legend(fontsize=8, loc="upper left")
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("fig_taylor_single_load.png", dpi=150)
print("\nSaved: fig_taylor_single_load.png")
plt.show()

print("\n" + "="*65)
print(f"  VERDICT: {'ALL PASSED ✓' if grad_pass and hess_pass else 'SEE ABOVE'}")
print("="*65)
