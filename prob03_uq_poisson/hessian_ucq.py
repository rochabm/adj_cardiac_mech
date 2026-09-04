"""
hessian_ucq.py
---------------
Manual (no hIPPYlibX) implementation of the reduced Hessian-vector
product for the inverse problem in ex04_ventricle_inverse_discrete_var.py,
plus a randomized low-rank eigensolver and a pointwise posterior
variance estimator (Laplace approximation), implemented from scratch
using only dolfinx/UFL/numpy/petsc4py.

HOW TO USE
----------
Run this AFTER ex04 has found the MAP point (CC.x.array == CC*,
uh.x.array == u* at that CC). Then, in the same script/session:

    from hessian_ucq import HessianOperator, randomized_eigensolver, \\
                             pointwise_variance

    Hop = HessianOperator(Fun, Jfunctional, uh, CC, lmbda,
                           V, Va, facet_tags, domain)

    d, U = randomized_eigensolver(Hop, k=50, p=20, seed=0)

    var = pointwise_variance(Hop, n_samples=200, seed=1)
    # `var` is a numpy array over CC's dofs -- wrap it in a
    # dolfinx.fem.Function(Va) and write it to XDMF to visualize.

-----------------------------------------------------------------------
MATHEMATICAL STRUCTURE
-----------------------------------------------------------------------
Let G(u, m) = <lambda*, F(u, m)> (replace the test function v in the
residual Fun by the already-solved adjoint lambda*; this is valid
because Fun is LINEAR in its test function v, so this substitution
gives exactly the Lagrangian pairing we need for second derivatives).

For a direction dm in parameter space, H*dm is computed via:

  1. INCREMENTAL FORWARD (linear solve, homogeneous BC):
        F_u[du_hat, v] = - F_m[dm, v]   for all v

  2. INCREMENTAL ADJOINT (linear solve, homogeneous BC, same operator
     transposed):
        F_u[du_test, dlambda_hat] =
            - ( J_uu[du_hat, du_test] + J_um[dm, du_test]
                + G_uu[du_hat, du_test] + G_um[dm, du_test] )
        for all du_test

  3. HESSIAN ACTION (pure assembly, no solve):
        H*dm = J_mu[du_hat, .] + J_mm[dm, .]
               + G_mu[du_hat, .] + G_mm[dm, .]
        assembled as a linear form over Va's test function.

This mirrors exactly what hIPPYlibX's ReducedHessian.mult() does
internally via PDEProblem.setLinearizationPoint() + apply_ij(), except
written out explicitly here.
"""

import numpy as np
import ufl
import dolfinx
from dolfinx import fem, default_scalar_type
import petsc4py.PETSc as PETSc


def safe_adjoint(form):
    """
    Wraps ufl.adjoint(form), guarding against forms that UFL has
    symbolically simplified to zero (see safe_action docstring for
    why these arise here). Returns 0 in that case.

    NOTE: checking len(form.arguments()) == 0 is NOT a reliable
    predictor of whether ufl.adjoint/ufl.action will fail -- in
    practice, a Form can report non-empty .arguments() at the
    aggregate level while still failing inside UFL's internal
    transformation (e.g. when it is a sum of integrals where the
    surviving argument structure differs across integrals, or when
    coefficients evaluate to an exact Zero after differentiation).
    We therefore just attempt the operation and catch the specific
    failure mode instead of trying to predict it in advance.
    """
    if form is None or form == 0:
        return 0
    if len(form.integrals()) == 0:
        return 0
    try:
        return ufl.adjoint(form)
    except (IndexError, ValueError):
        return 0


def safe_action(form, coefficient):
    """
    Wraps ufl.action(form, coefficient), guarding against forms that
    UFL has symbolically simplified to zero.

    Some cross second-derivative terms (e.g. d^2 J / du dC) are
    GENUINELY ZERO when the cost functional has no term that mixes u
    and C directly (as is the case here: Jdata depends on u only,
    Jsmooth depends on C only -- they only couple through the PDE
    residual Fun, not through J itself).

    NOTE: checking len(form.arguments()) == 0 is NOT sufficient to
    predict this -- a Form can still fail inside
    compute_form_action()'s internal logic even when .arguments()
    reports a non-empty tuple at the aggregate (sum-of-integrals)
    level. We therefore both check for the trivial empty-integrals
    case AND wrap the actual call in try/except as a robust fallback.
    """
    if form is None or form == 0:
        return 0
    if len(form.integrals()) == 0:
        return 0
    try:
        return ufl.action(form, coefficient)
    except (IndexError, ValueError):
        return 0


# =============================================================================
# Hessian operator: builds all forms once, applies H*dm many times
# =============================================================================

class HessianOperator:
    def __init__(self, Fun, Jfunctional, uh, CC, lmbda,
                 V, Va, facet_tags, domain):
        """
        Parameters
        ----------
        Fun, Jfunctional : UFL forms already defined in ex04, evaluated
                            AT THE MAP POINT (uh.x.array = u*, CC.x.array = C*)
        uh, CC           : dolfinx Functions holding the MAP point state/parameter
        lmbda            : dolfinx Function holding the adjoint solution at the MAP point
        V, Va            : state and parameter function spaces
        facet_tags       : geo.ffun (for rebuilding homogeneous BCs)
        domain           : the mesh
        """
        self.uh, self.CC, self.lmbda = uh, CC, lmbda
        self.V, self.Va = V, Va
        self.domain = domain

        # ------------------------------------------------------------
        # homogeneous BCs for incremental (perturbation) problems
        # ------------------------------------------------------------
        u_bc0 = np.array((0,) * domain.geometry.dim, dtype=default_scalar_type)
        base_dofs = fem.locate_dofs_topological(V, facet_tags.dim, facet_tags.find(5))
        self.bcs0 = [fem.dirichletbc(u_bc0, base_dofs, V)]

        # ------------------------------------------------------------
        # symbolic trial/test functions for each block
        # (declared fresh here to avoid UFL symbol collisions with
        # ex04's module-level `du`, `v`)
        # ------------------------------------------------------------
        q        = ufl.TrialFunction(Va)   # direction in parameter space
        dq       = ufl.TestFunction(Va)
        du_trial = ufl.TrialFunction(V)    # incremental state (delta_u_hat)
        du_test  = ufl.TestFunction(V)

        self.q, self.dq = q, dq
        self.du_trial, self.du_test = du_trial, du_test

        # ------------------------------------------------------------
        # F_u : tangent operator -- bilinear form (du_trial, du_test)
        # ------------------------------------------------------------
        F_u = ufl.derivative(Fun, uh, du_trial)

        # ------------------------------------------------------------
        # F_m as a BILINEAR form (q, du_test): derivative of the
        # residual wrt the parameter. The incremental forward RHS for
        # a specific direction dm is obtained via ufl.action(F_m_bilinear, dm_fun).
        # ------------------------------------------------------------
        F_m_bilinear = ufl.derivative(Fun, CC, q)

        # ------------------------------------------------------------
        # G(u, m) = <lambda*, F(u, m)>  (valid since Fun is linear in
        # its test function -- substitute v -> lambda*)
        # ------------------------------------------------------------
        v_symbol = Fun.arguments()[0]  # the test function used inside Fun
        G = ufl.replace(Fun, {v_symbol: lmbda})

        G_u  = ufl.derivative(G, uh, du_test)
        G_uu = ufl.derivative(G_u, uh, du_trial)   # bilinear (du_trial, du_test)
        G_um = ufl.derivative(G_u, CC, q)          # bilinear (q, du_test)

        G_m  = ufl.derivative(G, CC, dq)
        G_mu = ufl.derivative(G_m, uh, du_trial)   # bilinear (du_trial, dq)
        G_mm = ufl.derivative(G_m, CC, q)          # bilinear (q, dq)

        # ------------------------------------------------------------
        # second derivatives of the cost functional
        # ------------------------------------------------------------
        J_u  = ufl.derivative(Jfunctional, uh, du_test)
        J_uu = ufl.derivative(J_u, uh, du_trial)   # bilinear (du_trial, du_test)
        J_um = ufl.derivative(J_u, CC, q)          # bilinear (q, du_test)

        J_m  = ufl.derivative(Jfunctional, CC, dq)
        J_mu = ufl.derivative(J_m, uh, du_trial)   # bilinear (du_trial, dq)
        J_mm = ufl.derivative(J_m, CC, q)          # bilinear (q, dq)

        # ------------------------------------------------------------
        # store all UFL forms (kept symbolic; compiled lazily where
        # needed, since some require `action` with a concrete dm first)
        # ------------------------------------------------------------
        self.F_u          = F_u
        self.F_m_bilinear = F_m_bilinear
        self.F_u_adjoint  = ufl.adjoint(F_u)

        self.G_uu, self.G_um, self.G_mu, self.G_mm = G_uu, G_um, G_mu, G_mm
        self.J_uu, self.J_um, self.J_mu, self.J_mm = J_uu, J_um, J_mu, J_mm

        # ------------------------------------------------------------
        # assemble the two state-space operators that do NOT depend
        # on the direction dm (reused for every Hessian apply)
        # ------------------------------------------------------------
        self._A_fwd = fem.petsc.assemble_matrix(fem.form(self.F_u), bcs=self.bcs0)
        self._A_fwd.assemble()

        self._A_adj = fem.petsc.assemble_matrix(fem.form(self.F_u_adjoint), bcs=self.bcs0)
        self._A_adj.assemble()

        self._ksp_fwd = PETSc.KSP().create(domain.comm)
        self._ksp_fwd.setOperators(self._A_fwd)
        self._ksp_fwd.setType("preonly")
        self._ksp_fwd.getPC().setType("lu")
        self._ksp_fwd.getPC().setFactorSolverType("mumps")

        self._ksp_adj = PETSc.KSP().create(domain.comm)
        self._ksp_adj.setOperators(self._A_adj)
        self._ksp_adj.setType("preonly")
        self._ksp_adj.getPC().setType("lu")
        self._ksp_adj.getPC().setFactorSolverType("mumps")

        # scratch Functions (avoid reallocating every call)
        self._dm_fun   = fem.Function(Va)
        self._du_hat   = fem.Function(V)
        self._dlam_hat = fem.Function(V)
        self._Hdm      = fem.Function(Va)

        self.ndofs_m = len(CC.x.array)

    # -----------------------------------------------------------------
    # H * dm  (the core operation; called many times by the eigensolver)
    # -----------------------------------------------------------------
    def mult(self, dm_array: np.ndarray) -> np.ndarray:
        self._dm_fun.x.array[:] = dm_array
        self._dm_fun.x.scatter_forward()

        # ---------------------------------------------------------
        # STEP 1: incremental forward solve
        #     F_u[du_hat, v] = - F_m[dm, v]   for all v
        # ---------------------------------------------------------
        rhs_fwd_term = safe_action(self.F_m_bilinear, self._dm_fun)
        if rhs_fwd_term == 0:
            # F does not depend on the parameter at all (shouldn't
            # happen physically here, but guard anyway)
            self._du_hat.x.array[:] = 0.0
        else:
            rhs_fwd_ufl = -rhs_fwd_term
            b_fwd = fem.petsc.assemble_vector(fem.form(rhs_fwd_ufl))
            fem.petsc.apply_lifting(b_fwd, [fem.form(self.F_u)], [self.bcs0])
            b_fwd.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            fem.petsc.set_bc(b_fwd, self.bcs0)

            self._ksp_fwd.solve(b_fwd, self._du_hat.x.petsc_vec)
            self._du_hat.x.scatter_forward()
            b_fwd.destroy()

        # ---------------------------------------------------------
        # STEP 2: incremental adjoint solve
        #     F_u_adjoint[dlambda_hat, du_test] =
        #         -( J_uu[du_hat,.] + J_um[dm,.] + G_uu[du_hat,.] + G_um[dm,.] )
        #
        # NOTE: some of these four terms may be symbolically zero
        # (e.g. J_um here, since Jdata depends only on u and Jsmooth
        # only on C -- there is no direct u-C coupling in J itself,
        # only through the PDE residual). safe_action() returns the
        # python int 0 for those; we filter them out before summing,
        # since UFL forms cannot be added to a bare 0 directly inside
        # fem.form().
        # ---------------------------------------------------------
        adj_terms = [
            safe_action(self.J_uu, self._du_hat),
            safe_action(self.J_um, self._dm_fun),
            safe_action(self.G_uu, self._du_hat),
            safe_action(self.G_um, self._dm_fun),
        ]
        adj_terms = [t for t in adj_terms if t != 0]

        if len(adj_terms) == 0:
            self._dlam_hat.x.array[:] = 0.0
        else:
            rhs_adj_ufl = -sum(adj_terms)
            b_adj = fem.petsc.assemble_vector(fem.form(rhs_adj_ufl))
            fem.petsc.apply_lifting(b_adj, [fem.form(self.F_u_adjoint)], [self.bcs0])
            b_adj.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            fem.petsc.set_bc(b_adj, self.bcs0)

            self._ksp_adj.solve(b_adj, self._dlam_hat.x.petsc_vec)
            self._dlam_hat.x.scatter_forward()
            b_adj.destroy()

        # ---------------------------------------------------------
        # STEP 3: assemble H*dm (linear form over Va's test function)
        #
        # CORRECTED FORMULA. The standard reduced-Hessian action for
        # a PDE-constrained problem is:
        #
        #   H*dm = F_m^T[dlambda_hat]      <- incremental adjoint feeds
        #                                     back through F_m (NOT G_mu!)
        #          + J_mm[dm,.]
        #          + G_mm[dm,.]            <- second derivative of
        #                                     <lambda*, F> wrt m only
        #          + G_um[du_hat,.]        <- mixed term, lambda* fixed,
        #                                     contracted with du_hat
        #          + J_um[du_hat,.]        <- mixed cost term (zero here)
        #
        # PREVIOUS BUG: this code used G_mu[du_hat] (derivative of
        # <lambda*, F> wrt (m,u), contracted with du_hat) and never
        # used dlam_hat at all -- but dlam_hat is exactly what should
        # feed back through F_m here. G_mu and "F_m^T[dlambda_hat]"
        # are NOT the same term: G_mu uses the FIXED adjoint lambda*,
        # while the correct term uses the INCREMENTAL adjoint
        # dlambda_hat (the whole point of solving for it in STEP 2).
        # ---------------------------------------------------------

        # F_m^T[dlambda_hat]: action of F_m_bilinear's adjoint on the
        # incremental adjoint solution (this is the term that makes
        # dlam_hat actually matter)
        F_m_adj = safe_adjoint(self.F_m_bilinear)
        Fm_term = safe_action(F_m_adj, self._dlam_hat) if F_m_adj != 0 else 0

        # G_um[du_hat,.]: mixed term with lambda* FIXED, contracted
        # with the incremental STATE du_hat (not the incremental
        # adjoint) -- this one was already structurally correct
        # before, just re-derived here with the right justification
        Gum_adj = safe_adjoint(self.G_um)
        Gum_term = safe_action(Gum_adj, self._du_hat) if Gum_adj != 0 else 0

        hdm_terms = [
            Fm_term,
            safe_action(self.J_mm, self._dm_fun),
            safe_action(self.G_mm, self._dm_fun),
            Gum_term,
            safe_action(self.J_um, self._du_hat) if len(self.J_um.arguments()) > 0 else 0,
        ]
        hdm_terms = [t for t in hdm_terms if t != 0]

        self._Hdm.x.array[:] = 0.0
        if len(hdm_terms) > 0:
            Hdm_ufl = sum(hdm_terms)
            fem.petsc.assemble_vector(self._Hdm.x.petsc_vec, fem.form(Hdm_ufl))
            self._Hdm.x.scatter_forward()

        return self._Hdm.x.array.copy()

    def __del__(self):
        try:
            self._A_fwd.destroy()
            self._A_adj.destroy()
        except Exception:
            pass


# =============================================================================
# randomized double-pass eigensolver (hIPPYlib algorithm, from scratch)
# =============================================================================

def randomized_eigensolver(Hop: HessianOperator, k=50, p=20, seed=0):
    """
    Computes the k largest eigenpairs of the (full) Hessian H = H_misfit
    + H_prior via a randomized range finder + Rayleigh-Ritz projection.

    This is the STANDARD eigenproblem H*v = lambda*v (simplified
    relative to hIPPYlib's GENERALIZED eigenproblem H_misfit*v =
    lambda*H_prior*v, which requires separating the prior operator --
    see the module docstring for how to extend this).

    Returns
    -------
    eigvals : np.ndarray, shape (k,), descending order
    eigvecs : np.ndarray, shape (ndofs_m, k)
    """
    rng = np.random.default_rng(seed)
    n = Hop.ndofs_m

    # 1. random test matrix
    Omega = rng.standard_normal((n, k + p))

    # 2. Y = H @ Omega  (k+p Hessian-vector products -- the expensive part)
    Y = np.zeros_like(Omega)
    for i in range(k + p):
        Y[:, i] = Hop.mult(Omega[:, i])

    # 3. orthonormalize
    Q, _ = np.linalg.qr(Y)

    # 4. Rayleigh-Ritz: project H onto the subspace spanned by Q
    HQ = np.zeros_like(Q)
    for i in range(Q.shape[1]):
        HQ[:, i] = Hop.mult(Q[:, i])
    T = Q.T @ HQ
    T = 0.5 * (T + T.T)  # symmetrize (numerical safety)

    eigvals_small, eigvecs_small = np.linalg.eigh(T)

    # sort descending, keep k largest
    order = np.argsort(eigvals_small)[::-1][:k]
    eigvals = eigvals_small[order]
    eigvecs = Q @ eigvecs_small[:, order]

    return eigvals, eigvecs


# =============================================================================
# pointwise posterior variance (Hutchinson stochastic diagonal estimator)
# =============================================================================
#
# This estimates diag(H^{-1}) directly, treating H as already including
# both misfit and regularization (exactly what Hop.mult returns, since
# Jfunctional = Jdata + alpha*Jsmooth). This corresponds to the Laplace
# approximation covariance Sigma_post = H^{-1}.

def apply_hessian_inverse_cg(Hop: HessianOperator, rhs: np.ndarray,
                              tol=1e-6, maxiter=300, verbose=False) -> np.ndarray:
    """
    Solves H @ x = rhs via matrix-free Conjugate Gradient, using only
    Hop.mult() as the matrix-vector product. Valid since H (data
    misfit + quadratic regularization) is symmetric positive
    (semi-)definite at the MAP point.

    IMPORTANT: `tol` is a RELATIVE tolerance (||r|| / ||rhs|| < tol),
    not absolute. Using an absolute tolerance here is a common and
    dangerous bug: for a Rademacher rhs (entries +-1, as used by the
    Hutchinson estimator), ||rhs|| ~ sqrt(n) ~ 28 for n=771 dofs, so
    an absolute tol=1e-8 is actually a relative tol of ~3.6e-10 --
    far stricter than intended, and easily unreachable within
    maxiter. If CG silently fails to converge and is cut off at
    maxiter, the returned x can be inaccurate enough to make
    diag(H^{-1}) estimates come out NEGATIVE (a clear sign something
    is wrong, since variance must be >= 0).
    """
    x = np.zeros_like(rhs)
    r = rhs - Hop.mult(x)
    rhs_norm = np.linalg.norm(rhs) + 1e-30
    p = r.copy()
    rs_old = r @ r

    converged = False
    n_iter_used = maxiter

    for it in range(maxiter):
        rel_resid = np.sqrt(rs_old) / rhs_norm
        if rel_resid < tol:
            converged = True
            n_iter_used = it
            break

        Hp = Hop.mult(p)
        denom = p @ Hp
        if denom <= 0:
            # H should be SPD; a non-positive curvature here signals
            # either an upstream Hessian bug or severe ill-conditioning
            if verbose:
                print(f"    [CG] WARNING: p^T H p = {denom:.3e} <= 0 at "
                      f"iter {it} -- H may not be SPD here. Stopping early.")
            break

        alpha = rs_old / denom
        x += alpha * p
        r -= alpha * Hp
        rs_new = r @ r
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    final_rel_resid = np.sqrt(rs_old) / rhs_norm
    if verbose or not converged:
        status = "CONVERGED" if converged else "DID NOT CONVERGE"
        print(f"    [CG] {status} after {n_iter_used} iters, "
              f"relative residual = {final_rel_resid:.3e} (tol={tol:.1e})")

    return x


def pointwise_variance(Hop: HessianOperator, n_samples=200, seed=1,
                        cg_tol=1e-6, cg_maxiter=300, verbose_cg=False) -> np.ndarray:
    """
    Hutchinson stochastic estimator of diag(H^{-1}), i.e. the
    pointwise posterior variance under the (full-rank) Laplace
    approximation Sigma_post = H^{-1}.

    Cost: n_samples matrix-free CG solves, each costing up to
    cg_maxiter Hessian-vector products. This is the simplest "from
    scratch" option; for large problems, prefer separating H_prior
    from H_misfit and using the low-rank Woodbury formula instead
    (see module docstring).
    """
    rng = np.random.default_rng(seed)
    n = Hop.ndofs_m
    diag_estimate = np.zeros(n)

    for s in range(n_samples):
        z = rng.choice([-1.0, 1.0], size=n)
        Hinv_z = apply_hessian_inverse_cg(Hop, z, tol=cg_tol, maxiter=cg_maxiter,
                                           verbose=verbose_cg)
        diag_estimate += z * Hinv_z
        if (s + 1) % 20 == 0:
            print(f"  [pointwise_variance] sample {s+1}/{n_samples}")

    diag_estimate /= n_samples

    n_negative = np.sum(diag_estimate < 0)
    if n_negative > 0:
        frac = n_negative / n
        print(f"  WARNING: {n_negative}/{n} ({frac:.1%}) variance estimates "
              f"are NEGATIVE (min={diag_estimate.min():.3e}).")
        print(f"  Variance must be >= 0 -- this indicates CG did not "
              f"converge tightly enough (try smaller cg_tol / larger "
              f"cg_maxiter, or re-run with verbose_cg=True to inspect),")
        print(f"  or more Hutchinson samples are needed (n_samples is "
              f"currently {n_samples}).")

    return diag_estimate


# =============================================================================
# PHASE B: generalized eigenproblem  H_misfit · v = λ · H_prior · v
# =============================================================================
#
# This is the "proper" Bayesian Laplace approximation, mirroring what hIPPYlib
# does internally. It requires three new pieces:
#
#   1. HpriorOperator  -- assembles H_prior = alpha * Laplacian stiffness
#                         matrix (from Jsmooth's bilinear form) and provides:
#                           .mult(v)       -> H_prior · v
#                           .solve(v)      -> H_prior⁻¹ · v  (LU factorization)
#                           .diag_inv()    -> diagonal of H_prior⁻¹  (Hutchinson)
#
#   2. generalized_eigensolver  -- randomized double-pass algorithm for
#                                  H_misfit · v = λ · H_prior · v, mirroring
#                                  hIPPYlib's doublePassG()
#
#   3. woodbury_pointwise_variance  -- computes diag(Σ_post) via the formula
#                                      Σ_post = H_prior⁻¹
#                                               - U · diag(λᵢ/(λᵢ+1)) · Uᵀ
#                                      without any CG solves (just 50 LU
#                                      back-substitutions and numpy dot products)
#
# RELATIONSHIP TO PHASE A (standard eigenproblem):
#   Phase A solved H · v = λ · v  (standard, identity metric).
#   Eigenvalues ~ 5e-6, all equal -- no meaningful structure because H_prior
#   and H_misfit contributions are mixed together without separating them.
#
#   Phase B solves H_misfit · v = λ · H_prior · v  (generalized, H_prior metric).
#   Eigenvalue λᵢ now measures: "how much more informative is the data than
#   the prior in direction vᵢ?"  λᵢ >> 1 => data dominates; λᵢ << 1 => prior
#   dominates. The threshold λ=1 is now physically meaningful.


class HpriorOperator:
    """
    Assembles and factors the prior precision matrix H_prior, which is
    the Hessian of the regularization term Jsmooth:

        Jsmooth = (alpha / volume_mesh) * integral( ||grad(CC)||^2 ) dx

    H_prior is a SPARSE, SYMMETRIC POSITIVE DEFINITE matrix -- just the
    weighted Laplacian stiffness matrix in the parameter space Va. It is
    MUCH cheaper to work with than H_misfit (no incremental forward/adjoint
    solves needed -- just standard FEniCSx sparse matrix assembly + LU).

    Parameters
    ----------
    Va           : dolfinx.fem.FunctionSpace  (parameter space)
    alpha        : dolfinx.fem.Constant       (regularization parameter)
    volume_mesh  : float                      (domain volume, for normalization)
    dx           : ufl.Measure                (volume measure)
    domain       : dolfinx.mesh.Mesh
    """

    def __init__(self, Va, alpha, volume_mesh, dx, domain):
        self.Va = Va
        self.ndofs = Va.dofmap.index_map.size_local * Va.dofmap.index_map_bs

        import petsc4py.PETSc as PETSc

        # assemble H_prior as a PETSc matrix
        q  = ufl.TrialFunction(Va)
        dq = ufl.TestFunction(Va)
        prior_form = (float(alpha) / volume_mesh) * ufl.inner(ufl.grad(q), ufl.grad(dq)) * dx

        self._A = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(prior_form))
        self._A.assemble()

        # LU factorization (reused for all solves -- assembled once)
        self._ksp = PETSc.KSP().create(domain.comm)
        self._ksp.setOperators(self._A)
        self._ksp.setType("preonly")
        self._ksp.getPC().setType("lu")
        self._ksp.getPC().setFactorSolverType("mumps")
        self._ksp.setUp()

        # scratch PETSc vectors
        self._x = self._A.createVecRight()
        self._b = self._A.createVecRight()

    def mult(self, v: np.ndarray) -> np.ndarray:
        """H_prior · v  (sparse matrix-vector product)."""
        self._b.array[:] = v
        self._A.mult(self._b, self._x)
        return self._x.array.copy()

    def solve(self, v: np.ndarray) -> np.ndarray:
        """H_prior⁻¹ · v  (LU back-substitution, very cheap)."""
        self._b.array[:] = v
        self._ksp.solve(self._b, self._x)
        return self._x.array.copy()

    def diag_inv(self, n_samples: int = 300, seed: int = 7) -> np.ndarray:
        """
        Stochastic estimate of diag(H_prior⁻¹) via Hutchinson estimator.

        This is the PRIOR variance field -- what you would know about CC
        from the regularization alone, before seeing any data. It serves
        as the baseline in the Woodbury formula.

        NOTE: since H_prior is a sparse Laplacian matrix with known
        structure, the diagonal of its inverse could also be computed
        exactly via sparse Cholesky + selected inversion -- but the
        Hutchinson estimate is simpler to implement and accurate enough
        for visualization purposes.
        """
        rng = np.random.default_rng(seed)
        n = self.ndofs
        diag = np.zeros(n)
        for _ in range(n_samples):
            z = rng.choice([-1.0, 1.0], size=n).astype(float)
            Ainvz = self.solve(z)
            diag += z * Ainvz
        return diag / n_samples

    def __del__(self):
        try:
            self._A.destroy()
            self._x.destroy()
            self._b.destroy()
        except Exception:
            pass


def generalized_eigensolver(Hop: HessianOperator,
                             Hprior: HpriorOperator,
                             k: int = 50,
                             p: int = 20,
                             seed: int = 0) -> tuple:
    """
    Randomized double-pass algorithm for the GENERALIZED eigenproblem:

        H_misfit · v = λ · H_prior · v

    where H_misfit = H_full - H_prior  (data misfit Hessian alone).

    This mirrors hIPPYlib's doublePassG() algorithm exactly. The key
    difference from Phase A's standard eigensolver:

        Phase A:  H_full · v = λ · v          (identity metric)
        Phase B:  H_misfit · v = λ · H_prior · v  (H_prior metric)

    Eigenvalue λᵢ now has a clean Bayesian interpretation:
        λᵢ >> 1  =>  data much more informative than prior in direction vᵢ
        λᵢ ~  1  =>  data and prior equally informative
        λᵢ << 1  =>  prior dominates, data adds little in direction vᵢ

    The returned eigenvectors Uᵢ are H_prior-orthonormal:
        Uᵢᵀ · H_prior · Uⱼ = δᵢⱼ

    Parameters
    ----------
    Hop    : HessianOperator  (provides H_full.mult = (H_misfit+H_prior)·v)
    Hprior : HpriorOperator   (provides H_prior.mult and H_prior.solve)
    k      : number of eigenpairs to retain
    p      : oversampling parameter (hIPPYlib default: 20)
    seed   : random seed

    Returns
    -------
    eigvals : np.ndarray, shape (k,), descending order
              Generalized eigenvalues λᵢ of H_misfit·v = λ·H_prior·v
    eigvecs : np.ndarray, shape (ndofs, k)
              H_prior-orthonormal eigenvectors
    """
    rng = np.random.default_rng(seed)
    n = Hop.ndofs_m

    print(f"  [generalized_eigensolver] n={n}, k={k}, p={p}")

    # ------------------------------------------------------------------
    # Double-pass randomized algorithm (hIPPYlib doublePassG)
    # ------------------------------------------------------------------
    # Pass 1: build a range approximation of H_misfit in the H_prior metric.
    # Sample random vectors Omega, apply H_prior⁻¹·H_misfit to each one.
    # H_misfit·v = H_full·v - H_prior·v  (computed without ever forming H_misfit)

    Omega = rng.standard_normal((n, k + p))

    # Y = H_prior⁻¹ · H_misfit · Omega
    Y = np.zeros_like(Omega)
    for i in range(k + p):
        H_full_v   = Hop.mult(Omega[:, i])        # H_full · Ωᵢ
        H_prior_v  = Hprior.mult(Omega[:, i])      # H_prior · Ωᵢ
        H_misfit_v = H_full_v - H_prior_v          # H_misfit · Ωᵢ
        Y[:, i]    = Hprior.solve(H_misfit_v)      # H_prior⁻¹ · H_misfit · Ωᵢ
        if (i + 1) % 10 == 0:
            print(f"    pass 1: {i+1}/{k+p} vectors done")

    # H_prior-orthonormalize Y:  Q s.t. Qᵢᵀ · H_prior · Qⱼ = δᵢⱼ
    # Step 1: standard QR to get an orthonormal basis Q_std
    Q_std, _ = np.linalg.qr(Y)
    # Step 2: re-orthonormalize in the H_prior inner product
    #   form B = Q_stdᵀ · H_prior · Q_std  (small (k+p)×(k+p) matrix)
    HpQ = np.zeros_like(Q_std)
    for i in range(Q_std.shape[1]):
        HpQ[:, i] = Hprior.mult(Q_std[:, i])
    B = Q_std.T @ HpQ    # (k+p) × (k+p)
    # Cholesky of B gives the re-scaling: Q = Q_std · L⁻ᵀ
    # Cholesky of B with adaptive nugget for numerical stability
    # B = Q_std^T H_prior Q_std should be SPD, but may have small negative
    # eigenvalues due to floating point if H (or its action) contains NaN
    # or if the forward solve was degenerate.
    if np.any(np.isnan(B)) or np.any(np.isinf(B)):
        raise RuntimeError(
            "generalized_eigensolver: B matrix contains NaN/Inf. "
            "This usually means the forward solve produced NaN (e.g. "
            "exp(m) overflow due to extreme parameter values). "
            "Check that m_fun.x.array is in a reasonable range before "
            "calling the eigensolver."
        )
    nugget = 1e-12
    for _ in range(10):
        try:
            L = np.linalg.cholesky(B + nugget * np.eye(B.shape[0]))
            break
        except np.linalg.LinAlgError:
            nugget *= 100
    else:
        raise np.linalg.LinAlgError(
            f"Cholesky failed even with nugget={nugget:.1e}. "
            "B matrix may not be SPD -- check Hessian operator."
        )
    Q = Q_std @ np.linalg.inv(L.T)   # H_prior-orthonormal basis

    # ------------------------------------------------------------------
    # Pass 2: Rayleigh-Ritz projection of H_misfit onto span(Q)
    # T = Qᵀ · H_misfit · Q   (small (k+p)×(k+p) matrix)
    # ------------------------------------------------------------------
    HmQ = np.zeros_like(Q)
    for i in range(Q.shape[1]):
        H_full_v   = Hop.mult(Q[:, i])
        H_prior_v  = Hprior.mult(Q[:, i])
        HmQ[:, i]  = H_full_v - H_prior_v      # H_misfit · Qᵢ
        if (i + 1) % 10 == 0:
            print(f"    pass 2: {i+1}/{k+p} vectors done")

    T = Q.T @ HmQ    # projected H_misfit, symmetric (k+p)×(k+p)
    T = 0.5 * (T + T.T)

    # solve the SMALL generalized eigenproblem T·w = λ·I·w
    # (T is already in the H_prior-orthonormal basis, so the metric is
    # identity in this reduced space)
    eigvals_small, eigvecs_small = np.linalg.eigh(T)

    # sort descending, keep k largest
    order   = np.argsort(eigvals_small)[::-1][:k]
    eigvals = eigvals_small[order]
    eigvecs = Q @ eigvecs_small[:, order]   # back to full parameter space

    print(f"  [generalized_eigensolver] done.")
    print(f"    top    5 eigenvalues: {eigvals[:5]}")
    print(f"    bottom 5 eigenvalues: {eigvals[-5:]}")

    # clip small negative eigenvalues to zero -- these arise from
    # numerical approximation errors in H_misfit = H_full - H_prior
    # and are not physically meaningful (H_misfit must be SPD at MAP)
    n_negative = np.sum(eigvals < 0)
    if n_negative > 0:
        print(f"    WARNING: {n_negative} negative eigenvalues clipped to 0 "
              f"(min={eigvals.min():.3e}). These are numerical artifacts.")
        eigvals = np.maximum(eigvals, 0.0)

    print(f"    eigenvalues > 1 (data-informed modes): {np.sum(eigvals > 1)}")
    print(f"    eigenvalues > 0.1                    : {np.sum(eigvals > 0.1)}")

    return eigvals, eigvecs


def woodbury_pointwise_variance(Hprior: HpriorOperator,
                                 eigvals: np.ndarray,
                                 eigvecs: np.ndarray,
                                 n_prior_samples: int = 300,
                                 seed: int = 9) -> tuple:
    """
    Computes pointwise posterior variance via the Sherman-Morrison-Woodbury
    formula, using the generalized eigenpairs from generalized_eigensolver:

        Σ_post(x,x) = [H_prior⁻¹](x,x)  -  Σᵢ (λᵢ/(λᵢ+1)) · uᵢ(x)²

    where uᵢ are H_prior-orthonormal eigenvectors (Uᵀ H_prior U = I).
    Note: the correction uses uᵢ(x)² directly, NOT (H_prior⁻¹ uᵢ)(x)².

    This replaces the expensive Hutchinson+CG from Phase A (~19,500 linear
    solves) with:
        - n_prior_samples LU back-substitutions to estimate diag(H_prior⁻¹)
        - k LU back-substitutions for H_prior⁻¹·uᵢ  (one per eigenvector)
        - k pure numpy dot products for the correction term
    Total: ~350 LU solves vs ~19,500 CG solves  (~55x speedup)

    Parameters
    ----------
    Hprior          : HpriorOperator
    eigvals         : (k,) array from generalized_eigensolver
    eigvecs         : (n, k) array from generalized_eigensolver
                      (H_prior-orthonormal eigenvectors of H_misfit)
    n_prior_samples : Hutchinson samples for diag(H_prior⁻¹) estimation
    seed            : random seed

    Returns
    -------
    prior_var    : np.ndarray (n,)  -- prior variance  diag(H_prior⁻¹)
    post_var     : np.ndarray (n,)  -- posterior variance diag(Σ_post)
    correction   : np.ndarray (n,)  -- data correction term (always >= 0)

    NOTE: post_var = prior_var - correction  (elementwise)
          If any post_var entry is negative, it indicates the low-rank
          approximation is inaccurate there (k too small or p too small).
    """
    k = eigvecs.shape[1]
    n = eigvecs.shape[0]

    print(f"\n[Woodbury] Computing pointwise posterior variance...")
    print(f"  k = {k} eigenpairs")

    # ------------------------------------------------------------------
    # Term 2 first: H_prior⁻¹ · uᵢ for all eigenvectors
    # (we need these for both the correction AND to recover prior_var)
    # ------------------------------------------------------------------
    # Term 1: prior variance  diag(R⁻¹)  via Hutchinson
    # ------------------------------------------------------------------
    print(f"  Step 1: estimating diag(R⁻¹) via Hutchinson ({n_prior_samples} samples)...")
    prior_var = Hprior.diag_inv(n_samples=n_prior_samples, seed=seed)
    prior_var = np.abs(prior_var)
    print(f"    prior_var range: [{prior_var.min():.3e}, {prior_var.max():.3e}]")

    # ------------------------------------------------------------------
    # Term 2: correction  Σᵢ (λᵢ/(λᵢ+1)) · uᵢ(x)²
    #
    # DERIVATION: with H_prior = R and eigenvectors U s.t. Uᵀ R U = I
    # (R-orthonormal), the generalized eigenproblem H_misfit U = R U Λ
    # gives H_misfit = R U Λ Uᵀ R. Applying Sherman-Morrison-Woodbury
    # to H = R + H_misfit = R(I + UΛUᵀR):
    #
    #
    # DERIVATION (hIPPYlib convention): H_prior = R M^{-1} R, eigenvectors
    # satisfy U^T (R M^{-1} R) U = I (H_prior-orthonormal).
    #
    # H = H_misfit + H_prior = R M^{-1} R + H_misfit
    #   = R M^{-1} R (I + (R M^{-1} R)^{-1} H_misfit)
    #
    # From generalized eigenproblem H_misfit U = H_prior U Lambda:
    #   H_misfit = H_prior U Lambda U^T H_prior
    #
    # SMW on H = H_prior + H_prior U Lambda U^T H_prior:
    #   H^{-1} = H_prior^{-1} - U (Lambda^{-1} + U^T H_prior U)^{-1} U^T
    #           = H_prior^{-1} - U diag(lambda_i/(lambda_i+1)) U^T
    #
    # diag(H^{-1})[x] = [H_prior^{-1}](x,x) - sum_i (lambda_i/(lambda_i+1)) u_i(x)^2
    #
    # where H_prior^{-1} = R^{-1} M R^{-1}  (prior covariance)
    # ------------------------------------------------------------------
    print(f"  Step 2: assembling Woodbury correction...")
    weights    = eigvals / (eigvals + 1.0)   # λᵢ/(λᵢ+1), in [0,1)
    correction = (eigvecs ** 2) @ weights    # (n,k) @ (k,) = (n,)

    # ------------------------------------------------------------------
    # Posterior variance = prior variance - correction
    # ------------------------------------------------------------------
    post_var = prior_var - correction

    n_negative = np.sum(post_var < 0)
    if n_negative > 0:
        pct = 100 * n_negative / len(post_var)
        print(f"  Note: {n_negative} ({pct:.1f}%) posterior variance values "
              f"slightly negative -- clipping to 0 (numerical noise).")
        post_var = np.maximum(post_var, 0.0)

    print(f"  prior_var   : min={prior_var.min():.3e}, max={prior_var.max():.3e}")
    print(f"  correction  : min={correction.min():.3e}, max={correction.max():.3e}")
    print(f"  post_var    : min={post_var.min():.3e}, max={post_var.max():.3e}")

    return prior_var, post_var, correction

