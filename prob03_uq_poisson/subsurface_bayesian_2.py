"""
subsurface_bayesian.py
-----------------------
Direct translation of the hIPPYlib tutorial 3_SubsurfaceBayesian into
dolfinx 0.8.0, using our own infrastructure.

We follow the tutorial EXACTLY:
  - same mesh (64x64 unit square)
  - same FE spaces (P2 state, P1 parameter)
  - same BCs: u = x[1] on top/bottom only, Neumann on left/right
  - same prior: BiLaplacian anisotropic (gamma=0.1, delta=0.5, theta0=2,
                theta1=0.5, alpha=pi/4, Robin BC)
  - same misfit: PointwiseStateObservation, 50 targets in bottom half
  - same noise: noise_std = rel_noise * max(|Bu_true|)  [linf of noiseless obs]
  - same MAP algorithm: inexact Newton-CG with Gauss-Newton warmup
  - same UQ: doublePassG generalized eigensolver, Woodbury variance,
             posterior sampling

Reference:
    https://hippylib.github.io/tutorials_v3.0.0/3_SubsurfaceBayesian/
"""

import numpy as np
import math
import ufl
import dolfinx
import dolfinx.geometry
from dolfinx import fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import petsc4py.PETSc as PETSc
from mpi4py import MPI
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from hessian_ucq import generalized_eigensolver, woodbury_pointwise_variance
from newton_cg_solver import inexact_newton_cg

# =============================================================================
# 1. Setup (mirrors tutorial section 1)
# =============================================================================

np.random.seed(seed=1)   # same seed as tutorial
comm = MPI.COMM_WORLD

# =============================================================================
# 2. Mesh and FE spaces (tutorial section 3)
#    P2 state/adjoint, P1 parameter -- exactly as hIPPYlib
# =============================================================================

nx = ny = 64
mesh = dolfinx.mesh.create_unit_square(comm, nx, ny,
                                        dolfinx.mesh.CellType.triangle)

V  = fem.functionspace(mesh, ("Lagrange", 2))   # state (P2)
Va = fem.functionspace(mesh, ("Lagrange", 1))   # parameter (P1)

print(f"Number of dofs: STATE={V.dofmap.index_map.size_global}, "
      f"PARAMETER={Va.dofmap.index_map.size_global}, "
      f"ADJOINT={V.dofmap.index_map.size_global}")

dx = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": 4})

# =============================================================================
# 3. Boundary conditions (tutorial section 4)
#    u = x[1] on TOP AND BOTTOM only  (y=0 and y=1)
#    Neumann (natural) on left/right
#    bc0 = homogeneous Dirichlet for adjoint/incremental problems
# =============================================================================

fdim = mesh.topology.dim - 1

def top_bottom(x):
    return np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0))

tb_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, top_bottom)
tb_dofs   = fem.locate_dofs_topological(V, fdim, tb_facets)

uD = fem.Function(V)
uD.interpolate(lambda x: x[1].copy())
uD.x.scatter_forward()
bc = fem.dirichletbc(uD, tb_dofs)

uD0 = fem.Function(V)
uD0.interpolate(lambda x: np.zeros_like(x[0]))
uD0.x.scatter_forward()
bc0 = fem.dirichletbc(uD0, tb_dofs)

# =============================================================================
# 4. BiLaplacian prior with anisotropic diffusion and Robin BC
#    (tutorial section 4, prior)
#
#    R = delta*M + gamma*K_aniso + sqrt(gamma*delta)*M_boundary
#    K_aniso uses tensor T = R(alpha) diag(theta0, theta1) R(alpha)^T
#    Sampling: w ~ N(0, M_lumped), m = R^{-1} w
#    H_prior = R,  H_prior^{-1} = R^{-1}
# =============================================================================

gamma  = 0.1
delta  = 0.5
theta0 = 2.0
theta1 = 0.5
alpha  = math.pi / 4

# anisotropic tensor T
ca = math.cos(alpha);  sa = math.sin(alpha)
T00 = theta0*ca**2 + theta1*sa**2
T01 = (theta0 - theta1)*ca*sa
T11 = theta0*sa**2 + theta1*ca**2


class BiLaplacianPrior:
    """
    Anisotropic BiLaplacian prior matching hIPPYlib tutorial exactly.

    R = delta*M + gamma*K_aniso + sqrt(gamma*delta)*M_bdy  (Robin BC)
    K_aniso(q,dq) = T:grad(q)⊗grad(dq)
    Covariance: Γ = R^{-1} M R^{-1}  (hIPPYlib convention)
    Precision:  A = R M^{-1} R

    For the generalized eigenproblem H_misfit v = λ R v, we use H_prior = R.
    Woodbury: Σ_post = R^{-1} - U diag(λ/(λ+1)) U^T  (U is R-orthonormal)
    """

    def __init__(self, Va, gamma, delta, T00, T01, T11, dx):
        n = Va.dofmap.index_map.size_local * Va.dofmap.index_map_bs
        self.ndofs = n

        q  = ufl.TrialFunction(Va)
        dq = ufl.TestFunction(Va)

        # anisotropic stiffness
        g_q  = ufl.grad(q);  g_dq = ufl.grad(dq)
        K_aniso = (T00*g_q[0]*g_dq[0] + T01*g_q[0]*g_dq[1]
                 + T01*g_q[1]*g_dq[0] + T11*g_q[1]*g_dq[1]) * dx
        M_form  = ufl.inner(q, dq) * dx

        # Robin BC boundary term (all boundaries)
        ds_all      = ufl.Measure("ds", domain=Va.mesh)
        robin_coeff = math.sqrt(gamma * delta)
        R_form = (delta * M_form + gamma * K_aniso
                  + robin_coeff * ufl.inner(q, dq) * ds_all)

        self._R = fem.petsc.assemble_matrix(fem.form(R_form))
        self._R.assemble()
        self._M = fem.petsc.assemble_matrix(fem.form(M_form))
        self._M.assemble()

        # LU factorize R (used for R^{-1})
        self._ksp = PETSc.KSP().create(Va.mesh.comm)
        self._ksp.setOperators(self._R)
        self._ksp.setType("preonly")
        self._ksp.getPC().setType("lu")
        self._ksp.getPC().setFactorSolverType("mumps")
        self._ksp.setUp()

        self._x = self._R.createVecRight()
        self._b = self._R.createVecRight()

        # lumped mass diagonal for sampling  w ~ N(0, M_lumped)
        ones = np.ones(n)
        self._b.array[:] = ones
        self._M.mult(self._b, self._x)
        self._M_diag = np.maximum(self._x.array.copy(), 1e-30)

    # --- core operators ---

    def _R_mult(self, v):
        self._b.array[:] = v
        self._R.mult(self._b, self._x)
        return self._x.array.copy()

    def _R_solve(self, v):
        self._b.array[:] = v
        self._ksp.solve(self._b, self._x)
        return self._x.array.copy()

    def _M_mult(self, v):
        self._b.array[:] = v
        self._M.mult(self._b, self._x)
        return self._x.array.copy()

    # --- public interface (duck-typed for generalized_eigensolver) ---

    def mult(self, v):
        """H_prior * v = R * v"""
        return self._R_mult(v)

    def solve(self, v):
        """H_prior^{-1} * v = R^{-1} * v"""
        return self._R_solve(v)

    def diag_inv(self, n_samples=300, seed=7):
        """Hutchinson estimate of diag(R^{-1}) = diag(H_prior^{-1})."""
        rng = np.random.default_rng(seed)
        diag = np.zeros(self.ndofs)
        for _ in range(n_samples):
            z = rng.choice([-1.0, 1.0], size=self.ndofs).astype(float)
            diag += z * self._R_solve(z)
        return diag / n_samples

    def sample(self, rng):
        """
        Draw m ~ N(0, R^{-1} M R^{-1}):
            w = sqrt(M_lumped) * z,  z ~ N(0,I)
            m = R^{-1} * w
        Matches hIPPYlib's prior.sample(): parRandom.normal -> R^{-1}.
        """
        z = rng.standard_normal(self.ndofs)
        w = np.sqrt(self._M_diag) * z
        return self._R_solve(w)

    def __del__(self):
        try:
            self._R.destroy(); self._M.destroy()
            self._x.destroy(); self._b.destroy()
        except Exception:
            pass


prior = BiLaplacianPrior(Va, gamma, delta, T00, T01, T11, dx)
print(f"Prior: gamma={gamma}, delta={delta}, theta0={theta0}, "
      f"theta1={theta1}, alpha=pi/4, Robin BC")
print(f"Prior ndofs: {prior.ndofs}")

# =============================================================================
# 5. True parameter: sample from prior  (tutorial: true_model(prior))
# =============================================================================

m_true = fem.Function(Va, name="m_true")
m_true.interpolate(lambda x:
    + 2.0 * np.exp(-((x[0]-0.25)**2 + (x[1]-0.3)**2) / (2*0.12**2))
    - 2.0 * np.exp(-((x[0]-0.75)**2 + (x[1]-0.3)**2) / (2*0.12**2))
)
m_true.x.scatter_forward()
m_true_arr = m_true.x.array.copy()
print(f"m_true range: [{m_true_arr.min():.3f}, {m_true_arr.max():.3f}]")
# two Gaussian bumps: +2 at (0.25, 0.3) and -2 at (0.75, 0.3)
# both in the bottom half where sensors are -> good visual test of recovery


# =============================================================================
# 6. Forward problem: -div(exp(m)*grad(u)) = 0
#    Linear in u for fixed m  =>  is_fwd_linear=True in hIPPYlib
# =============================================================================

def solve_fwd(m_fun):
    ut = ufl.TrialFunction(V)
    vt = ufl.TestFunction(V)
    a  = ufl.exp(m_fun) * ufl.inner(ufl.grad(ut), ufl.grad(vt)) * dx
    L  = fem.Constant(mesh, default_scalar_type(0.0)) * vt * dx
    return LinearProblem(a, L, bcs=[bc],
                         petsc_options={"ksp_type":"preonly",
                                        "pc_type":"lu"}).solve()

u_true = solve_fwd(m_true)
print(f"u_true range: [{u_true.x.array.min():.4f}, {u_true.x.array.max():.4f}]")

# =============================================================================
# 7. PointwiseStateObservation: targets, noiseless obs, noise, noisy data
#    (tutorial section 5)
#
#    MAX = linf norm of Bu_true  (noiseless observations)
#    noise_std_dev = rel_noise * MAX
#    d_obs = Bu_true + N(0, noise_std_dev^2)
# =============================================================================

ntargets  = 50
rel_noise = 0.01

rng = np.random.default_rng(42)   # for sensor placement, noise, gradient check

# targets in bottom half (tutorial: uniform(0.1, 0.9) x uniform(0.1, 0.5))
targets_x = rng.uniform(0.1, 0.9, ntargets)
#targets_y = rng.uniform(0.1, 0.5, ntargets)
targets_y = rng.uniform(0.1, 0.9, ntargets)
targets   = np.column_stack([targets_x, targets_y])




nx_s, ny_s = 10, 10   # 10 x 5 = 50 sensors
sx = np.linspace(0.1, 0.9, nx_s)
sy = np.linspace(0.1, 0.9, ny_s)
gx, gy = np.meshgrid(sx, sy)
targets_x = gx.ravel()
targets_y = gy.ravel()
targets   = np.column_stack([targets_x, targets_y])
ntargets  = len(targets)




print(f"Number of observation points: {ntargets}")

# --- build B matrix (observation operator) using dolfinx Function.eval ---

_bb_tree_V = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)

def _build_obs_cell_map():
    pts3 = np.column_stack([targets, np.zeros(ntargets)])
    cand = dolfinx.geometry.compute_collisions_points(_bb_tree_V, pts3)
    coll = dolfinx.geometry.compute_colliding_cells(mesh, cand, pts3)
    cells = []
    for j in range(ntargets):
        links = coll.links(j)
        cells.append(int(links[0]) if links.size else -1)
    return cells, pts3

_obs_cells, _obs_pts3 = _build_obs_cell_map()

def _build_B():
    """
    B[j, i] = phi_i(x_j)  for each obs point j and V-dof i in its cell.
    Computed using dolfinx Function.eval on unit-vector functions.
    Returns list of (cell_dofs, phi_values) per observation point.
    """
    ndofs_V = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    tmp = fem.Function(V)
    B_rows = []
    for j in range(ntargets):
        cell = _obs_cells[j]
        if cell < 0:
            B_rows.append((np.array([], dtype=int), np.array([])))
            continue
        cell_dofs = V.dofmap.cell_dofs(cell)
        x_j = _obs_pts3[j:j+1]
        phi_vals = np.zeros(len(cell_dofs))
        for li, gd in enumerate(cell_dofs):
            if gd >= ndofs_V:
                continue
            tmp.x.array[:] = 0.0
            tmp.x.array[gd] = 1.0
            tmp.x.scatter_forward()
            phi_vals[li] = tmp.eval(x_j, np.array([cell],
                                    dtype=np.int32)).item()
        B_rows.append((cell_dofs, phi_vals))

    # partition-of-unity check
    bad = sum(1 for _, phi in B_rows
              if len(phi) > 0 and abs(phi.sum() - 1.0) > 0.05)
    print(f"  B matrix: {'PoU check passed ✓' if bad==0 else f'WARNING: {bad} pts failed PoU'}")
    return B_rows

print("Building observation operator B...")
_B_rows = _build_B()

def eval_B(u_arr):
    """B*u: evaluate u at observation points."""
    ndofs_V = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    vals = np.zeros(ntargets)
    for j, (cell_dofs, phi_vals) in enumerate(_B_rows):
        if len(phi_vals) == 0:
            continue
        valid = cell_dofs < ndofs_V
        vals[j] = np.dot(u_arr[cell_dofs[valid]], phi_vals[valid])
    return vals

def BT_mult(residuals):
    """B^T*r: dual of pointwise evaluation, no mass matrix."""
    ndofs_V = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    rhs = np.zeros(ndofs_V)
    for j, (cell_dofs, phi_vals) in enumerate(_B_rows):
        if len(phi_vals) == 0:
            continue
        valid = cell_dofs < ndofs_V
        rhs[cell_dofs[valid]] += residuals[j] * phi_vals[valid]
    return rhs

# --- noiseless observations ---
u_obs_clean = eval_B(u_true.x.array)

# --- noise: tutorial uses MAX = linf of Bu_true (noiseless) ---
MAX           = np.abs(u_obs_clean).max()
noise_std_dev = rel_noise * MAX
noise_var     = noise_std_dev ** 2
d_obs         = u_obs_clean + rng.standard_normal(ntargets) * noise_std_dev
print(f"MAX (linf of Bu_true): {MAX:.4f}")
print(f"noise_std_dev: {noise_std_dev:.4e}")

# =============================================================================
# 8. Cost functional, gradient, Hessian
#    J(m) = Jmisfit(m) + Jprior(m)
#    Jmisfit = (1/(2*sigma^2)) ||Bu - d||^2
#    Jprior  = (1/2) m^T R m
# =============================================================================

m_fun = fem.Function(Va, name="m")
u_fun = fem.Function(V,  name="u")
lam   = fem.Function(V,  name="lambda")
v_test  = ufl.TestFunction(V)
dq_test = ufl.TestFunction(Va)


def assemble_K(m_f, u_f):
    """Assemble and LU-factor K = dR/du (tangent = stiffness at m_f)."""
    ut = ufl.TrialFunction(V)
    vt = ufl.TestFunction(V)
    a  = ufl.exp(m_f) * ufl.inner(ufl.grad(ut), ufl.grad(vt)) * dx
    K  = fem.petsc.assemble_matrix(fem.form(a), bcs=[bc0])
    K.assemble()
    ksp = PETSc.KSP().create(mesh.comm)
    ksp.setOperators(K)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.setUp()
    return ksp, K


def eval_cost_and_grad(m_arr, gauss_newton=False):
    """
    Returns (J, grad_J) at m_arr.
    gauss_newton=True: drop Wuu term in Hessian (not used in gradient, but
    stored for Hessian operator).
    """
    m_fun.x.array[:] = m_arr;  m_fun.x.scatter_forward()
    uh = solve_fwd(m_fun)
    u_fun.x.array[:] = uh.x.array;  u_fun.x.scatter_forward()

    u_at_obs = eval_B(u_fun.x.array)
    res      = u_at_obs - d_obs
    Jmisfit  = 0.5 * np.dot(res, res) / noise_var

    Rm       = prior._R_mult(m_arr)
    Jprior   = 0.5 * np.dot(m_arr, Rm)
    J        = Jmisfit + Jprior

    # adjoint solve: K^T lam = -B^T(res/sigma^2)
    rhs_adj = -BT_mult(res / noise_var)
    ksp_K, K = assemble_K(m_fun, u_fun)
    lam_vec  = K.createVecRight()
    b_adj    = K.createVecRight()
    b_adj.array[:] = rhs_adj
    fem.petsc.set_bc(b_adj, [bc0])
    ksp_K.solve(b_adj, lam_vec)
    lam.x.array[:] = lam_vec.array;  lam.x.scatter_forward()

    # gradient: dJmisfit/dm + R*m
    dJdm_form = (ufl.exp(m_fun) * ufl.inner(ufl.grad(u_fun), ufl.grad(lam))
                 * dq_test * dx)
    g_mis = fem.petsc.assemble_vector(fem.form(dJdm_form))
    g_mis.ghostUpdate(addv=PETSc.InsertMode.ADD,
                      mode=PETSc.ScatterMode.REVERSE)
    grad = g_mis.array + prior._R_mult(m_arr)

    K.destroy();  b_adj.destroy()
    return J, grad


# --- cache ---
_cache = {"m": None, "J": None, "g": None}

def eval_J(m_arr):
    if _cache["m"] is None or not np.allclose(m_arr, _cache["m"]):
        _cache["J"], _cache["g"] = eval_cost_and_grad(m_arr)
        _cache["m"] = m_arr.copy()
    return _cache["J"]

def eval_g(m_arr):
    if _cache["m"] is None or not np.allclose(m_arr, _cache["m"]):
        _cache["J"], _cache["g"] = eval_cost_and_grad(m_arr)
        _cache["m"] = m_arr.copy()
    return _cache["g"]


# =============================================================================
# 9. Gradient and Hessian check  (tutorial section 6: modelVerify)
# =============================================================================

print("\n" + "="*60)
print("Gradient check  (tutorial: modelVerify)")
print("="*60)

# tutorial uses m0 = sin(x[0]) as test point
m0_fun = fem.Function(Va)
m0_fun.interpolate(lambda x: np.sin(x[0]))
m0_arr = m0_fun.x.array.copy()

h_dir = rng.standard_normal(prior.ndofs)
h_dir /= np.linalg.norm(h_dir)

eps   = 1e-4
J0, g0 = eval_cost_and_grad(m0_arr)
Jp, _  = eval_cost_and_grad(m0_arr + eps * h_dir)
dJ_fd  = (Jp - J0) / eps
dJ_adj = g0 @ h_dir
print(f"  FD  directional deriv: {dJ_fd:.6e}")
print(f"  Adj directional deriv: {dJ_adj:.6e}")
print(f"  Relative error       : {abs(dJ_adj-dJ_fd)/(abs(dJ_fd)+1e-30):.4e}")

# =============================================================================
# 10. Hessian operator  (tutorial: ReducedHessian via apply_ij)
# =============================================================================

class SubsurfaceHessian:
    """
    Reduced Hessian H = H_misfit + R for the linear Poisson problem.
    Mirrors hIPPYlib's ReducedHessian / apply_ij.
    For this LINEAR PDE, K = dR/du is constant given m -- factor once.
    """

    def __init__(self, m_f, u_f, lam_f, prior_op, gauss_newton=False):
        self.m_f    = m_f
        self.u_f    = u_f
        self.lam_f  = lam_f
        self.prior  = prior_op
        self.gauss_newton = gauss_newton
        self.ndofs_m = m_f.x.array.shape[0]

        self._ksp_K, self._K = assemble_K(m_f, u_f)
        self._du_hat = fem.Function(V)
        self._dlam   = fem.Function(V)

    def mult(self, dm_arr):
        dm_f = fem.Function(Va)
        dm_f.x.array[:] = dm_arr;  dm_f.x.scatter_forward()

        # incremental forward: K du_hat = -dR/dm[dm]
        b_fwd_form = -(ufl.exp(self.m_f) * dm_f *
                       ufl.inner(ufl.grad(self.u_f), ufl.grad(v_test)) * dx)
        b_fwd = fem.petsc.assemble_vector(fem.form(b_fwd_form))
        b_fwd.ghostUpdate(addv=PETSc.InsertMode.ADD,
                          mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b_fwd, [bc0])
        du_vec = self._K.createVecRight()
        self._ksp_K.solve(b_fwd, du_vec)
        self._du_hat.x.array[:] = du_vec.array
        self._du_hat.x.scatter_forward()

        # incremental adjoint RHS
        du_at_obs = eval_B(self._du_hat.x.array)
        rhs1 = -BT_mult(du_at_obs / noise_var)

        if not self.gauss_newton:
            # Wuu term: exp(m)*dm*<grad du_hat, grad lam>
            rhs2_form = -(ufl.exp(self.m_f) * dm_f *
                          ufl.inner(ufl.grad(self._du_hat),
                                    ufl.grad(self.lam_f)) * v_test * dx)
            rhs2 = fem.petsc.assemble_vector(fem.form(rhs2_form))
            rhs2.ghostUpdate(addv=PETSc.InsertMode.ADD,
                             mode=PETSc.ScatterMode.REVERSE)
        else:
            rhs2 = None

        b_adj = self._K.createVecRight()
        n = len(b_adj.array)
        b_adj.array[:] = rhs1[:n] + (rhs2.array[:n] if rhs2 is not None else 0)
        fem.petsc.set_bc(b_adj, [bc0])
        self._ksp_K.solve(b_adj, du_vec)
        self._dlam.x.array[:] = du_vec.array
        self._dlam.x.scatter_forward()

        # Hessian action on parameter space
        Hm_form = (
            ufl.exp(self.m_f) *
            ufl.inner(ufl.grad(self.u_f), ufl.grad(self._dlam)) *
            dq_test * dx
        )
        if not self.gauss_newton:
            Hm_form = Hm_form + (
                ufl.exp(self.m_f) * dm_f *
                ufl.inner(ufl.grad(self._du_hat), ufl.grad(self.lam_f)) *
                dq_test * dx
            )
        Hm = fem.petsc.assemble_vector(fem.form(Hm_form))
        Hm.ghostUpdate(addv=PETSc.InsertMode.ADD,
                       mode=PETSc.ScatterMode.REVERSE)
        return Hm.array + self.prior._R_mult(dm_arr)

    def __del__(self):
        try: self._K.destroy()
        except Exception: pass


def build_hop(gauss_newton=False):
    """Build Hessian at current m_fun/u_fun/lam."""
    return SubsurfaceHessian(m_fun, u_fun, lam, prior, gauss_newton)

# =============================================================================
# 11. MAP via inexact Newton-CG
#     Tutorial uses GN_iter=5 (Gauss-Newton for first 5 iterations)
# =============================================================================

print("\n" + "="*60)
print("MAP via inexact Newton-CG  (tutorial: ReducedSpaceNewtonCG)")
print("="*60)

# start from prior mean = 0  (tutorial: m = prior.mean.copy())
m0 = np.zeros(prior.ndofs)

# tutorial uses GN for first 5 iterations then full Newton
GN_ITER = 5
_outer_iter = [0]

def build_hop_adaptive():
    _outer_iter[0] += 1
    gn = (_outer_iter[0] <= GN_ITER)
    return build_hop(gauss_newton=gn)

m_map_arr, history = inexact_newton_cg(
    eval_J, eval_g, build_hop_adaptive,
    x0=m0,
    max_outer_iter=25,
    grad_tol=1e-6,
    cg_tol=0.5,
    cg_maxiter=50,
    eisenstat_walker=True,
    verbose=True,
    bounds=None,
)

print(f"\nConverged in {len(history)} iterations")
print(f"Final cost: {eval_J(m_map_arr):.8e}")
print(f"Final gradient norm: {np.linalg.norm(eval_g(m_map_arr)):.6e}")

m_map = fem.Function(Va, name="m_map")
m_map.x.array[:] = m_map_arr;  m_map.x.scatter_forward()
u_map = solve_fwd(m_map)

# MAP result plot
coords_Va = Va.tabulate_dof_coordinates()[:, :2]
coords_V  = V.tabulate_dof_coordinates()[:, :2]

def plot_field(ax, coords, vals, title, cmap="viridis", sensors=None,
               vmin=None, vmax=None):
    cf = ax.tricontourf(coords[:,0], coords[:,1], vals, levels=40,
                        cmap=cmap,
                        vmin=vmin if vmin is not None else vals.min(),
                        vmax=vmax if vmax is not None else vals.max())
    plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    if sensors is not None:
        ax.scatter(sensors[:,0], sensors[:,1], c="white", s=20,
                   zorder=5, edgecolors="black", linewidths=0.4)

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
plot_field(axes[0], coords_Va, m_true.x.array, "True parameter  m", "RdBu_r")
plot_field(axes[1], coords_V,  u_true.x.array, "True state  u",     "viridis",
           sensors=targets)
plot_field(axes[2], coords_Va, m_map.x.array,  "MAP estimate  m*",  "RdBu_r")
plot_field(axes[3], coords_V,  u_map.x.array,  "State at MAP  u(m*)","viridis",
           sensors=targets)
plt.suptitle("Subsurface Bayesian Inversion — MAP point", fontsize=13)
plt.tight_layout()
plt.savefig("fig1_MAP_result.png", dpi=150, bbox_inches="tight")
print("Saved: out_subsurface_MAP_result.png")
plt.show()

# =============================================================================
# 12. Low-rank Laplace approximation
#     Tutorial: doublePassG(Hmisfit, prior.R, prior.Rsolver, Omega, k)
#     H_prior = prior.R = R  (exactly what our generalized_eigensolver uses)
# =============================================================================

print("\n" + "="*60)
print("Low-rank Laplace approximation  (tutorial: doublePassG)")
print("="*60)

# ensure m_fun/u_fun/lam at MAP
_ = eval_g(m_map_arr)
Hop_map = SubsurfaceHessian(m_fun, u_fun, lam, prior, gauss_newton=False)

k_eig = 50;  p_over = 20
print(f"Requested eigenvectors: {k_eig}; Oversampling: {p_over}")

eigvals, eigvecs = generalized_eigensolver(
    Hop_map, prior, k=k_eig, p=p_over, seed=0
)
np.savetxt("out_subsurface_eigenvalues.txt", eigvals)
print(f"\nTop 5 eigenvalues: {eigvals[:5]}")
print(f"Eigenvalues > 1: {np.sum(eigvals > 1)}")

# eigenvalue plot
fig2, ax2 = plt.subplots(figsize=(7, 4))
ax2.semilogy(range(k_eig), eigvals, "b*", markersize=6)
ax2.semilogy(range(k_eig+1), np.ones(k_eig+1), "-r", label="λ=1")
ax2.set_xlabel("number"); ax2.set_ylabel("eigenvalue")
ax2.set_title("Hessian misfit spectrum\n"
              "(tutorial: doublePassG with prior.R)")
ax2.legend()
plt.tight_layout()
plt.savefig("fig2_eigenvalue_decay.png", dpi=150)
plt.show()

# eigenvector plot
eig_indices = [0, 1, 2, 5, 10, 15]
fig3, axes3 = plt.subplots(1, len(eig_indices),
                            figsize=(3.5*len(eig_indices), 3.5))
for col, idx in enumerate(eig_indices):
    if idx >= k_eig: axes3[col].set_visible(False); continue
    v = eigvecs[:, idx]
    v_plot = v / (np.abs(v).max() + 1e-30)
    cf = axes3[col].tricontourf(coords_Va[:,0], coords_Va[:,1],
                                 v_plot, levels=40, cmap="viridis",
                                 vmin=-1, vmax=1)
    plt.colorbar(cf, ax=axes3[col], fraction=0.046, pad=0.04,
                 ticks=[-1,-0.5,0,0.5,1])
    axes3[col].set_title(f"Eigenvector {idx}", fontsize=10)
    axes3[col].set_aspect("equal")
    axes3[col].set_xticks([]); axes3[col].set_yticks([])
    axes3[col].scatter(targets[:,0], targets[:,1], c="white", s=15,
                       zorder=5, edgecolors="black", linewidths=0.4)
plt.suptitle("Generalized eigenvectors of H_misfit", fontsize=11)
plt.tight_layout()
plt.savefig("fig3_eigenvectors.png", dpi=150)
plt.show()

# =============================================================================
# 13. Pointwise variance  (tutorial section 9)
#     prior.R -> H_prior = R,  eigvecs are R-orthonormal
#     Woodbury: Σ_post = R^{-1} - U diag(λ/(λ+1)) U^T
#     prior_var = diag(R^{-1}),  correction = eigvecs^2 @ (λ/(λ+1))
# =============================================================================

print("\n" + "="*60)
print("Pointwise variance  (tutorial: posterior.pointwise_variance r=200)")
print("="*60)

# prior variance: diag(R^{-1}) via randomized eigenpairs of R^{-1}
print("  Computing prior variance via randomized eigenpairs of R^{-1}...")
n_pr = prior.ndofs
k_pr = min(200, n_pr - 1)
rng_pr = np.random.default_rng(77)
Omega_pr = rng_pr.standard_normal((n_pr, k_pr + 10))
Y_pr = np.zeros_like(Omega_pr)
for i in range(k_pr + 10):
    Y_pr[:, i] = prior._R_solve(Omega_pr[:, i])
Q_pr, _ = np.linalg.qr(Y_pr)
RinvQ = np.zeros_like(Q_pr)
for i in range(Q_pr.shape[1]):
    RinvQ[:, i] = prior._R_solve(Q_pr[:, i])
T_pr = Q_pr.T @ RinvQ
T_pr = 0.5 * (T_pr + T_pr.T)
lam_pr, vec_pr = np.linalg.eigh(T_pr)
order_pr = np.argsort(lam_pr)[::-1][:k_pr]
lam_pr = lam_pr[order_pr]
U_pr   = Q_pr @ vec_pr[:, order_pr]
prior_var = np.abs((U_pr**2) @ lam_pr)
print(f"  Prior variance: [{prior_var.min():.3e}, {prior_var.max():.3e}]")

# Woodbury posterior variance
_, post_var, correction = woodbury_pointwise_variance(
    prior, eigvals, eigvecs, n_prior_samples=200, seed=9
)
# use smooth prior_var instead of noisy Hutchinson estimate
post_var = prior_var - correction
post_var = np.maximum(post_var, 0.0)

print(f"  Posterior variance: [{post_var.min():.3e}, {post_var.max():.3e}]")
print(f"  Correction:         [{correction.min():.3e}, {correction.max():.3e}]")

np.save("out_subsurface_prior_variance.npy", prior_var)
np.save("out_subsurface_posterior_variance.npy", post_var)

# posterior trace (tutorial prints this)
post_trace  = post_var.sum()
prior_trace = prior_var.sum()
corr_trace  = correction.sum()
print(f"\n  Posterior trace {post_trace:.6e}; "
      f"Prior trace {prior_trace:.6e}; "
      f"Correction trace {corr_trace:.6e}")

# variance plot
var_max = max(prior_var.max(), post_var.max())
fig4, axes4 = plt.subplots(1, 3, figsize=(15, 4))
plot_field(axes4[0], coords_Va, prior_var, "Prior variance",
           "inferno", vmin=0, vmax=var_max, sensors=targets)
plot_field(axes4[1], coords_Va, post_var,  "Posterior variance",
           "inferno", vmin=0, vmax=var_max, sensors=targets)
var_red = np.clip(correction / (prior_var + 1e-30), 0, 1)
plot_field(axes4[2], coords_Va, var_red,
           "Variance reduction (correction/prior)", "viridis",
           vmin=0, vmax=1, sensors=targets)
plt.tight_layout()
plt.savefig("fig4_variance.png", dpi=150)
plt.show()

# =============================================================================
# 14. Posterior samples  (tutorial section 10)
#     m_sample = m_map + z_post
#     z_post = z_prior - U diag(sqrt(λ/(λ+1))) U^T R z_prior
#     z_prior ~ N(0, R^{-1})  (= prior.sample())
# =============================================================================

print("\n" + "="*60)
print("Posterior samples  (tutorial section 10)")
print("="*60)

nsamples = 5
rng_s    = np.random.default_rng(99)
D_coeff  = np.sqrt(eigvals / (eigvals + 1.0))

# generate all samples first so we can compute shared color scales
all_prior, all_post = [], []
for i in range(nsamples):
    z_prior = prior.sample(rng_s)
    Rz      = prior._R_mult(z_prior)
    coeff   = eigvecs.T @ Rz
    delta   = eigvecs @ (D_coeff * coeff)
    m_post  = m_map_arr + z_prior - delta
    all_prior.append(z_prior)
    all_post.append(m_post)
    print(f"  sample {i+1}: prior [{z_prior.min():.3f},{z_prior.max():.3f}]  "
          f"post [{m_post.min():.3f},{m_post.max():.3f}]")

# symmetric shared colorscales (RdBu_r needs symmetric range around 0)
pr_abs = max(abs(v).max() for v in all_prior)
ps_abs = max(abs(v).max() for v in all_post)

fig5, axes5 = plt.subplots(2, nsamples, figsize=(4*nsamples, 8))
for i in range(nsamples):
    for row, (arr, abs_val, title) in enumerate([
            (all_prior[i], pr_abs, f"Prior sample {i+1}"),
            (all_post[i],  ps_abs, f"Post. sample {i+1}")]):
        cf = axes5[row,i].tricontourf(
            coords_Va[:,0], coords_Va[:,1], arr,
            levels=40, cmap="RdBu_r",
            vmin=-abs_val, vmax=abs_val)
        if i == nsamples - 1:   # one colorbar per row, on rightmost panel
            plt.colorbar(cf, ax=axes5[row,i], fraction=0.046, pad=0.04)
        axes5[row,i].set_title(title, fontsize=10)
        axes5[row,i].set_aspect("equal")
        axes5[row,i].set_xticks([]); axes5[row,i].set_yticks([])

plt.suptitle("Prior vs Posterior samples  (shared colorscale per row)",
             fontsize=12)
plt.tight_layout()
plt.savefig("fig5_prior_posterior_samples.png", dpi=150)
plt.show()

# convergence plot
newton_J = [h["J"] for h in history]
fig6, ax6 = plt.subplots(figsize=(6, 4))
ax6.semilogy(newton_J, "o-b", markersize=5)
ax6.set_xlabel("Newton-CG iteration"); ax6.set_ylabel("J")
ax6.set_title("MAP convergence")
plt.tight_layout()
plt.savefig("fig6_MAP_convergence.png", dpi=150)
plt.show()

print("\n=== Done ===")
print("Output figures:")
for f in [
    "fig1_MAP_result.png          -- true m, true u, MAP m*, state at MAP",
    "fig2_eigenvalue_decay.png    -- Hessian misfit spectrum",
    "fig3_eigenvectors.png        -- generalized eigenvectors 0,1,2,5,10,15",
    "fig4_variance.png            -- prior/posterior variance + reduction",
    "fig5_prior_posterior_samples.png -- prior vs posterior samples",
    "fig6_MAP_convergence.png     -- Newton-CG convergence",
]:
    print(f"  {f}")
print("Data files:")
for f in ["out_subsurface_eigenvalues.txt",
          "out_subsurface_prior_variance.npy",
          "out_subsurface_posterior_variance.npy"]:
    print(f"  {f}")
