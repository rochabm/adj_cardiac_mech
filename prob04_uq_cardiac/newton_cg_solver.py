"""
newton_cg_solver.py
---------------------
Inexact (truncated) Newton-CG solver for the inverse problem in ex04,
using the validated HessianOperator (hessian_ucq.py) for genuine
second-order steps -- as opposed to L-BFGS-B's quasi-Newton curvature
approximation.

WHY THIS IS A GOOD HESSIAN VALIDATION TEST
--------------------------------------------
A finite-difference check on isolated random directions can pass even
if the Hessian has subtle inconsistencies that only show up over a
full optimization trajectory. A genuine Newton-CG method using a
CORRECT Hessian should show fast (super-linear / locally quadratic)
convergence near the optimum. Slow, erratic, or divergent behavior
here is a much stronger signal of a remaining Hessian bug than a
single-point finite-difference test.

SCOPE OF THIS IMPLEMENTATION (deliberately simplified)
---------------------------------------------------------
- NO bound constraints (CC in [1, 16] is NOT enforced here). This is
  a first validation pass; box constraints would require a
  projected-Newton or trust-region-with-bounds variant, deferred
  until this simpler version is shown to work well.
- Fixed (not Eisenstat-Walker adaptive) inner CG tolerance, for
  simplicity.
- Backtracking Armijo line search (simple, robust, no bound projection).
- Negative curvature detected inside CG -- on detection, CG returns
  the best direction found so far (truncated Newton, no trust-region
  radius -- just truncation here for simplicity).

USAGE
-----
    from newton_cg_solver import inexact_newton_cg

    x_opt, history = inexact_newton_cg(
        eval_J, eval_gradient, build_hessian_operator,
        x0=CC.x.array.copy(),
        max_outer_iter=30,
        grad_tol=1e-6,
        cg_tol=0.1,
        cg_maxiter=50,
    )

`build_hessian_operator` must be a callable taking no arguments and
returning a fresh HessianOperator instance built at the CURRENT
(uh, CC) state -- since the Hessian must be re-linearized at every
outer iteration. See the integration snippet at the bottom of this
file for how to wire this into ex04.
"""

import numpy as np


def cg_steihaug(Hop_mult, g, tol=0.1, maxiter=50, verbose=False):
    """
    Truncated (inexact) CG solve of H @ p = -g, stopping early if:
      (a) the relative residual drops below `tol`, or
      (b) negative curvature is encountered (d^T H d <= 0), or
      (c) maxiter is reached.

    Parameters
    ----------
    Hop_mult : callable(v: np.ndarray) -> np.ndarray
               Hessian-vector product (Hop.mult)
    g        : np.ndarray, the gradient at the current point
    tol      : relative residual tolerance for early stopping
    maxiter  : maximum CG iterations

    Returns
    -------
    p : np.ndarray, the (possibly truncated) Newton direction
    info : dict with keys 'n_iter', 'reason', 'rel_resid'
    """
    n = len(g)
    p = np.zeros(n)
    r = -g.copy()          # residual of H@p = -g, starting at p=0
    d = r.copy()
    rs_old = r @ r
    g_norm = np.linalg.norm(g) + 1e-30

    reason = "maxiter"

    for it in range(maxiter):
        rel_resid = np.sqrt(rs_old) / g_norm
        if rel_resid < tol:
            reason = "tol_reached"
            return p, {"n_iter": it, "reason": reason, "rel_resid": rel_resid}

        Hd = Hop_mult(d)
        dHd = d @ Hd

        # NOTE: dHd is compared against a tolerance RELATIVE to the
        # scale of H (||d||^2 * ||Hd|| as a rough scale reference),
        # not a strict <= 0 check. Values like dHd ~ -1e-14 are
        # floating point noise around a true value of ~0, not genuine
        # negative curvature -- treating them as "negative curvature"
        # causes premature truncation that can lead the line search
        # into nonsensical (e.g. out-of-bounds) parameter values.
        curvature_scale = abs(d @ d) * (np.linalg.norm(Hd) + 1e-30)
        dHd_tol = 1e-10 * curvature_scale

        if dHd <= dHd_tol:
            # negative curvature: if this is the first iteration,
            # fall back to steepest descent direction; otherwise keep
            # the best p found so far (truncated Newton)
            if it == 0:
                p = -g / g_norm  # normalized steepest descent fallback
            reason = "negative_curvature"
            if verbose:
                print(f"    [CG] negative curvature detected at iter {it} "
                      f"(d^T H d = {dHd:.3e}); truncating.")
            return p, {"n_iter": it, "reason": reason,
                       "rel_resid": np.sqrt(rs_old) / g_norm}

        alpha = rs_old / dHd
        p = p + alpha * d
        r = r - alpha * Hd
        rs_new = r @ r
        d = r + (rs_new / rs_old) * d
        rs_old = rs_new

    return p, {"n_iter": maxiter, "reason": reason,
               "rel_resid": np.sqrt(rs_old) / g_norm}


def backtracking_line_search(eval_J, x, p, g, J_x=None,
                              c1=1e-4, max_backtracks=20, alpha0=1.0,
                              bounds=None, verbose_fallback=True):
    """
    Simple Armijo backtracking line search:
        J(x + alpha*p) <= J(x) + c1*alpha*(g . p)

    Parameters
    ----------
    bounds : tuple (lo, hi) or None
             If given, x_new is CLIPPED into [lo, hi] elementwise
             before being evaluated. This is a simple (non-rigorous)
             way to keep CC physically valid (e.g. CC in [1, 16]) --
             it does NOT implement a proper projected-Newton method,
             but is enough to prevent the forward solver from being
             handed nonsensical parameter values (e.g. negative
             stiffness) that make its own Newton iteration fail to
             converge with an unrelated-looking error.

    Returns
    -------
    alpha : float, the accepted step length
    J_new : float, J(x + alpha*p)
    n_backtracks : int
    p_used : np.ndarray, the direction actually used (may be -g if
             the original p was not a descent direction)
    """
    if J_x is None:
        J_x = eval_J(x)

    gp = g @ p
    if gp >= 0:
        # p is not a descent direction (can happen after truncation
        # with negative curvature) -- fall back to steepest descent
        p = -g
        gp = g @ p

    def clip(z):
        if bounds is None:
            return z
        lo, hi = bounds
        return np.clip(z, lo, hi)

    alpha = alpha0
    for k in range(max_backtracks):
        x_new = clip(x + alpha * p)
        try:
            J_new = eval_J(x_new)
        except RuntimeError as e:
            # the forward solver (solve_nl_prob's own Newton iteration)
            # failed to converge for this CC field -- this can happen
            # when a truncated/negative-curvature CG direction produces
            # a very heterogeneous CC field that, even after clipping
            # each dof into [1, 16] individually, is numerically too
            # stiff/unstable for the hyperelastic forward problem's own
            # load-stepping Newton solver. Treat this as an invalid
            # point (J = +inf) so the line search simply backtracks to
            # a smaller, hopefully smoother step, instead of crashing.
            if verbose_fallback:
                print(f"    [line search] forward solve failed at alpha={alpha:.2e} "
                      f"({e}); treating as J=+inf and backtracking.")
            J_new = np.inf
        if J_new <= J_x + c1 * alpha * gp:
            return alpha, J_new, k, p
        alpha *= 0.5

    # line search failed to find improvement; return tiny step anyway
    try:
        J_tiny = eval_J(clip(x + alpha * p))
    except RuntimeError:
        J_tiny = np.inf
    return alpha, J_tiny, max_backtracks, p


def inexact_newton_cg(eval_J, eval_gradient, build_hessian_operator,
                       x0, max_outer_iter=30, ftol=2.22e-9, grad_tol=1e-5,
                       cg_tol=0.1, cg_maxiter=50, verbose=True,
                       bounds=None, eisenstat_walker=False):
    """
    Inexact (truncated) Newton-CG with stopping criteria that EXACTLY
    match scipy L-BFGS-B, so the two methods can be compared fairly.

    scipy L-BFGS-B stops when EITHER of these is satisfied:

      1. Gradient condition (same as scipy gtol):
            max(|proj g_i|) <= grad_tol
         For unconstrained problems, proj g = g, so this is the
         L-infinity norm of the gradient. scipy default: gtol=1e-5.

      2. Relative function change (same as scipy ftol):
            (f^k - f^{k+1}) / max(|f^k|, |f^{k+1}|, 1) <= ftol
         scipy default: ftol = 2.22e-9 (= 1e7 * machine_epsilon,
         i.e. factr=1e7 in the underlying Fortran code).

    Parameters
    ----------
    eval_J                   : callable(x) -> float
    eval_gradient            : callable(x) -> np.ndarray
    build_hessian_operator   : callable() -> HessianOperator
    x0                       : np.ndarray, initial guess
    max_outer_iter           : maximum outer Newton iterations
    ftol             : relative J change tolerance  (scipy default: 2.22e-9)
                       set to 0 to disable this criterion
    grad_tol         : L-inf gradient norm tolerance (scipy default: 1e-5)
    cg_tol           : initial relative residual tolerance for inner CG
    cg_maxiter       : maximum inner CG iterations
    bounds           : tuple (lo, hi) or None
    eisenstat_walker : if True, adaptively tighten CG tolerance as
                       gradient decreases:
                           eta_k = min(cg_tol, sqrt(||g_k||/||g_{k-1}||))
                       This improves convergence near the optimum.

    Returns
    -------
    x : np.ndarray, final iterate
    history : list of dicts, one per outer iteration
    """
    x = x0.copy()
    if bounds is not None:
        x = np.clip(x, bounds[0], bounds[1])
    history = []

    J_prev    = None   # tracks previous iteration cost for ftol check
    g_norm_prev = None # tracks previous gradient norm for Eisenstat-Walker
    eta       = cg_tol # current CG tolerance

    for k in range(max_outer_iter):
        J_x = eval_J(x)
        g   = eval_gradient(x)

        # --- scipy L-BFGS-B criterion 1: L-inf gradient norm ---
        g_norm = np.linalg.norm(g, ord=np.inf)

        # --- Eisenstat-Walker adaptive CG tolerance ---
        if eisenstat_walker and g_norm_prev is not None:
            eta = min(cg_tol, np.sqrt(g_norm / g_norm_prev))
            eta = max(eta, 1e-10)
        g_norm_prev = g_norm

        print(f"\n=== Newton-CG outer iteration {k} ===")
        print(f"  J            : {J_x:.6e}")
        print(f"  ||grad||_inf : {g_norm:.6e}  (gtol = {grad_tol:.1e})")
        if eisenstat_walker:
            print(f"  CG tol       : {eta:.3e}  (Eisenstat-Walker)")

        # --- scipy L-BFGS-B criterion 2: relative J change ---
        if J_prev is not None and ftol > 0:
            rel_dJ = (J_prev - J_x) / max(abs(J_prev), abs(J_x), 1.0)
            print(f"  rel_dJ       : {rel_dJ:.6e}  (ftol = {ftol:.2e})")
            if rel_dJ <= ftol:
                print(f"  Converged (criterion 2): "
                      f"relative J change {rel_dJ:.2e} <= ftol {ftol:.2e}")
                history.append({
                    "iter": k, "J": J_x, "grad_norm": g_norm,
                    "cg_iters": 0, "cg_reason": "ftol_converged",
                    "step_len": 0.0, "n_backtracks": 0,
                    "stopped_reason": "ftol",
                })
                break

        # --- scipy L-BFGS-B criterion 1 check ---
        if g_norm <= grad_tol:
            print(f"  Converged (criterion 1): "
                  f"||grad||_inf {g_norm:.2e} <= gtol {grad_tol:.1e}")
            history.append({
                "iter": k, "J": J_x, "grad_norm": g_norm,
                "cg_iters": 0, "cg_reason": "gtol_converged",
                "step_len": 0.0, "n_backtracks": 0,
                "stopped_reason": "gtol",
            })
            break

        Hop = build_hessian_operator()

        p, cg_info = cg_steihaug(Hop.mult, g, tol=eta, maxiter=cg_maxiter,
                                  verbose=verbose)

        print(f"  CG           : {cg_info['n_iter']} iters, "
              f"reason={cg_info['reason']}, "
              f"rel_resid={cg_info['rel_resid']:.3e}")

        alpha, J_new, n_backtracks, p_used = backtracking_line_search(
            eval_J, x, p, g, J_x=J_x, bounds=bounds, verbose_fallback=verbose
        )

        print(f"  line search  : alpha={alpha:.4e}, "
              f"backtracks={n_backtracks}, J_new={J_new:.6e}")

        if not np.isfinite(J_new):
            print("  STOPPING: line search failed (all steps give J=inf).")
            history.append({
                "iter": k, "J": J_x, "grad_norm": g_norm,
                "cg_iters": cg_info["n_iter"], "cg_reason": cg_info["reason"],
                "step_len": 0.0, "n_backtracks": n_backtracks,
                "stopped_reason": "line_search_failed",
            })
            break

        J_prev = J_x   # update BEFORE moving x, so rel_dJ uses J at x^k vs x^{k+1}
        x = x + alpha * p_used
        if bounds is not None:
            x = np.clip(x, bounds[0], bounds[1])

        history.append({
            "iter": k, "J": J_x, "grad_norm": g_norm,
            "cg_iters": cg_info["n_iter"], "cg_reason": cg_info["reason"],
            "step_len": alpha, "n_backtracks": n_backtracks,
        })

    return x, history

    return x, history


# =============================================================================
# integration snippet -- how to wire this into ex04
# =============================================================================
#
# Paste this AFTER the multi-start optimization block in ex04 (i.e.
# after best_CC has been found by L-BFGS-B), to compare Newton-CG's
# trajectory against L-BFGS-B's.
#
# from newton_cg_solver import inexact_newton_cg
# from hessian_ucq import HessianOperator
#
# def build_hop():
#     # rebuild the adjoint at the CURRENT uh/CC (already updated by
#     # eval_gradient's internal _solve_and_cache call)
#     lmbda_current = adj_problem.solve()
#     lmbda_current.x.scatter_forward()
#     return HessianOperator(Fun, Jfunctional, uh, CC, lmbda_current,
#                             V, Va, facet_tags, domain)
#
# x0_newton = cc_init.copy()  # same starting point as L-BFGS-B's last
#                              # multi-start, for a fair comparison
#
# x_opt, history = inexact_newton_cg(
#     eval_J, eval_gradient, build_hop,
#     x0=x0_newton,
#     max_outer_iter=30,
#     grad_tol=my_gtol,
#     cg_tol=0.1,
#     cg_maxiter=50,
# )
#
# print("\nNewton-CG final J:", eval_J(x_opt))
# print("L-BFGS-B final J  :", best_J)
#
# import matplotlib.pyplot as plt
# newton_J = [h["J"] for h in history]
# plt.figure()
# plt.semilogy(newton_J, label="Newton-CG")
# plt.semilogy(vals_func, label="L-BFGS-B")
# plt.xlabel("outer iteration")
# plt.ylabel("J")
# plt.legend()
# plt.savefig("newton_vs_lbfgsb_convergence.png", dpi=150)
