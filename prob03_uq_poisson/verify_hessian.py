"""
verify_hessian.py
-------------------
Finite-difference sanity check for HessianOperator (hessian_ucq.py),
mirroring exactly what hIPPYlibX's modelVerify() does: confirms that
H*dm computed via the adjoint-based incremental solves matches a
direct finite-difference approximation of the gradient's directional
derivative.

WHY THIS IS ESSENTIAL
----------------------
Manually deriving second-derivative forms via repeated ufl.derivative()
is error-prone (sign conventions, which variable is "frozen", etc).
DO NOT trust HessianOperator.mult() for any UQ result until this check
passes with small relative error.

HOW TO RUN
----------
Paste/import this at the end of ex04, after the MAP point has been
found (CC.x.array == CC*, uh.x.array == u* at that point), with
`hessian_ucq.py` importable (same directory).

CHECK PERFORMED
----------------
For a random direction dm and small epsilon:

    H*dm  ?=  ( grad_J(C* + eps*dm) - grad_J(C* - eps*dm) ) / (2*eps)

where grad_J(C) is the gradient dLdf already computed in ex04's
_solve_and_cache / eval_gradient machinery (forward + adjoint solve at
the perturbed parameter).

This requires 2 EXTRA forward+adjoint solves (for C*+eps*dm and
C*-eps*dm) -- expensive, but only needs to be run ONCE to validate the
Hessian code, not as part of the main UQ pipeline.
"""

import numpy as np


def verify_hessian(Hop, eval_gradient_fn, CC_map, n_directions=3,
                    eps=0.2, seed=0, scale_to_CC=True):
    """
    Parameters
    ----------
    Hop              : HessianOperator instance, built at the MAP point
    eval_gradient_fn : callable(x: np.ndarray) -> np.ndarray
                        Returns dJ/dC evaluated at parameter x (this is
                        exactly ex04's eval_gradient(x), which already
                        does forward+adjoint solve + assembles dLdf)
    CC_map           : np.ndarray, the MAP point parameter values
                        (CC.x.array.copy() right after optimization)
    n_directions     : how many random directions to test
    eps              : finite-difference step size, interpreted as a
                        RELATIVE perturbation (eps * ||CC_map||) when
                        scale_to_CC=True (recommended), or as an
                        absolute step on a unit-norm direction
                        otherwise.
    scale_to_CC      : if True (default), the perturbation magnitude
                        is scaled to the typical magnitude of CC_map.
                        This matters because CC values here are O(1-16),
                        not O(1) -- a tiny absolute perturbation
                        (e.g. 1e-5 on a unit-norm direction spread over
                        ~770 dofs, i.e. ~3.6e-7 per dof) can be SMALLER
                        than the Newton solver's own convergence
                        tolerance (atol/rtol = 1e-8 on the RESIDUAL,
                        which does not directly bound the solution
                        perturbation -- in practice this combination
                        can make solve_nl_prob converge to numerically
                        IDENTICAL u for C_map+eps*dm and C_map-eps*dm,
                        making the finite-difference directional
                        derivative spuriously exactly zero).

    NOTE: if you still see ||FD directional deriv|| == 0.0 exactly,
    increase eps further (try the eps_scan() helper below) before
    concluding the Hessian code itself is wrong.
    """
    rng = np.random.default_rng(seed)
    n = Hop.ndofs_m
    CC_typical_scale = np.linalg.norm(CC_map) / np.sqrt(n)  # ~ typical |C| value

    print("=" * 70)
    print("Hessian finite-difference verification")
    print(f"CC_map typical magnitude per dof: {CC_typical_scale:.4e}")
    print("=" * 70)

    max_rel_err = 0.0

    for i in range(n_directions):
        dm = rng.standard_normal(n)
        dm /= np.linalg.norm(dm)  # normalize direction

        # --- adjoint-based Hessian action (always at unit-norm dm) ---
        Hdm = Hop.mult(dm)

        # --- finite-difference step: scale to CC's magnitude if requested ---
        eps_used = eps * CC_typical_scale if scale_to_CC else eps

        # --- finite-difference directional derivative of the gradient ---
        g_plus  = eval_gradient_fn(CC_map + eps_used * dm)
        g_minus = eval_gradient_fn(CC_map - eps_used * dm)
        fd_Hdm = (g_plus - g_minus) / (2.0 * eps_used)

        abs_err = np.linalg.norm(Hdm - fd_Hdm)
        rel_err = abs_err / (np.linalg.norm(fd_Hdm) + 1e-30)
        max_rel_err = max(max_rel_err, rel_err)

        print(f"direction {i}  (eps_used = {eps_used:.4e}):")
        print(f"  ||H*dm||                 : {np.linalg.norm(Hdm):.6e}")
        print(f"  ||FD directional deriv|| : {np.linalg.norm(fd_Hdm):.6e}")
        print(f"  ||H*dm - FD||            : {abs_err:.6e}")
        print(f"  relative error           : {rel_err:.6e}")
        if np.linalg.norm(fd_Hdm) == 0.0:
            print("  WARNING: FD directional derivative is EXACTLY zero.")
            print("  This usually means eps_used is too small relative to")
            print("  solve_nl_prob's own Newton tolerance (atol/rtol), so")
            print("  C_map+eps*dm and C_map-eps*dm converge to numerically")
            print("  IDENTICAL states. Try a larger eps (e.g. 1e-2 or 1e-1")
            print("  with scale_to_CC=True), or use eps_scan() below.")
        print()

    print("=" * 70)
    if max_rel_err < 5e-2:
        print(f"PASS: max relative error = {max_rel_err:.6e} (< 5e-2)")
    else:
        print(f"FAIL: max relative error = {max_rel_err:.6e} (>= 5e-2)")
        print("Do NOT trust HessianOperator.mult() until this is fixed.")
    print("=" * 70)

    return max_rel_err


def eps_scan(Hop, eval_gradient_fn, CC_map, eps_values=None, seed=0,
             scale_to_CC=True):
    """
    Diagnostic helper: tries several eps values on a SINGLE random
    direction and reports ||FD directional deriv|| for each, so you
    can identify the "good" window:

        - eps too small  -> FD deriv ~ 0 (swamped by Newton tolerance
                             / floating point noise)
        - eps too large  -> FD deriv diverges from the true derivative
                             due to second-order Taylor truncation
                             error (the curvature itself changes
                             significantly over the step)
        - eps "just right" -> FD deriv stable across a decade or so
                             of eps values; relative error vs H*dm
                             should be small there.

    Call this BEFORE verify_hessian() if you got an exact-zero FD
    derivative with the default eps.
    """
    if eps_values is None:
        eps_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    rng = np.random.default_rng(seed)
    n = Hop.ndofs_m
    CC_typical_scale = np.linalg.norm(CC_map) / np.sqrt(n)

    dm = rng.standard_normal(n)
    dm /= np.linalg.norm(dm)

    Hdm = Hop.mult(dm)
    Hdm_norm = np.linalg.norm(Hdm)

    print("=" * 70)
    print("eps scan -- single direction, varying finite-difference step")
    print(f"CC_map typical magnitude per dof: {CC_typical_scale:.4e}")
    print(f"||H*dm|| (adjoint-based, reference): {Hdm_norm:.6e}")
    print("=" * 70)
    print(f"{'eps':>12} {'eps_used':>12} {'||FD deriv||':>14} {'rel.err vs H*dm':>18}")

    for eps in eps_values:
        eps_used = eps * CC_typical_scale if scale_to_CC else eps
        g_plus  = eval_gradient_fn(CC_map + eps_used * dm)
        g_minus = eval_gradient_fn(CC_map - eps_used * dm)
        fd_Hdm = (g_plus - g_minus) / (2.0 * eps_used)
        fd_norm = np.linalg.norm(fd_Hdm)
        rel_err = np.linalg.norm(Hdm - fd_Hdm) / (fd_norm + 1e-30)
        print(f"{eps:>12.2e} {eps_used:>12.2e} {fd_norm:>14.6e} {rel_err:>18.6e}")

    print("=" * 70)
    print("Pick an eps from the window where rel.err is smallest and")
    print("roughly stable across neighboring eps values; use that eps")
    print("(the UNSCALED value in the first column) in verify_hessian().")
    print("=" * 70)


# =============================================================================
# hIPPYlib-faithful modelVerify, ported from hippylib.modeling.modelVerify
# =============================================================================
#
# This is a direct port of the algorithm in hippylib's modelVerify(), adapted
# to call our eval_J / eval_gradient / HessianOperator instead of hippylib's
# Model/ReducedHessian abstractions. The original source (hippylib legacy,
# same algorithm used conceptually in hippylibX) is:
#
#   https://hippylib.readthedocs.io/en/latest/_modules/hippylib/modeling/modelVerify.html
#
# Key differences from our simpler verify_hessian() above:
#   1. Sweeps 32 step sizes eps = 0.5**[31,...,0] (geometric, largest first)
#      instead of 1-3 manually picked values.
#   2. Uses FORWARD difference (g(x+eps*h) - g(x))/eps, not centered
#      difference -- this changes the theoretical truncation error from
#      O(eps^2) to O(eps), which is what makes the log-log slope check
#      below meaningful (see explanation in model_verify_hippylib_style's
#      docstring).
#   3. Error metric is the L-infinity norm (worst single dof), not a
#      relative L2 norm over the whole vector.
#   4. Produces a log-log plot of eps vs error, with a reference line of
#      slope 1 -- a CORRECT gradient/Hessian shows error decaying linearly
#      with eps (in log-log) until floating-point round-off takes over and
#      the curve turns back up, forming a characteristic "V" shape. A
#      genuinely WRONG gradient/Hessian instead shows a flat, eps-independent
#      error plateau (no linear decay region at all).
#   5. Adds an INDEPENDENT algebraic symmetry check: (y, H x) should equal
#      (x, H y) for any two random directions x, y, since H must be
#      symmetric. This check does NOT rely on finite differences at all,
#      so it can catch certain bugs (e.g. a mismatched i<->j term) that a
#      finite-difference check might miss if it happens to look numerically
#      similar along the tested directions.

def model_verify_hippylib_style(Hop, eval_J, eval_gradient, CC_map,
                                  n_eps=32, scale_to_CC=True, seed=0,
                                  max_eps=0.25,
                                  plot=True, save_plot="hessian_FD_check.png"):
    """
    Faithful port of hippylib.modeling.modelVerify's algorithm.

    Parameters
    ----------
    Hop            : HessianOperator instance, built at CC_map
    eval_J         : callable(x) -> float          (ex04's eval_J)
    eval_gradient  : callable(x) -> np.ndarray      (ex04's eval_gradient)
    CC_map         : np.ndarray, the point to verify at (e.g. CC* at MAP,
                      or any other point -- hippylib's modelVerify is
                      typically run at a RANDOM point, not necessarily
                      the MAP, since it's a code-correctness check, not
                      a UQ computation)
    n_eps          : number of step sizes to sweep (hippylib default: 32)
    max_eps        : caps the LARGEST unscaled eps value tested (default
                      0.25, i.e. hippylib's eps[0]=1.0 is reduced to
                      0.25). hippylib's original sweep starts at
                      eps=1.0, which for this nonlinear hyperelastic
                      problem is large enough (once scaled by
                      CC_typical_scale) to produce a CC field that
                      makes solve_nl_prob's own Newton solver fail to
                      converge. Capping the sweep avoids wasting solves
                      on points known to be unstable -- the small-eps
                      tail (what the log-log slope check actually
                      needs) is unaffected. Set to 1.0 to recover the
                      exact original hippylib behavior (with the
                      try/except safety net below still active in case
                      some intermediate eps still fails).
    scale_to_CC    : if True, the random direction h is implicitly
                      compared at a perturbation scale tied to CC_map's
                      magnitude (see eps_used below) -- same rationale
                      as in verify_hessian()/eps_scan() above: CC here
                      is O(1-16), not O(1), so unscaled tiny eps values
                      fall below solve_nl_prob's effective resolution.
    plot           : if True, produces the log-log convergence plot
    save_plot      : filename for the saved plot (None to skip saving)

    Returns
    -------
    eps       : np.ndarray, the step sizes used (largest to smallest)
    err_grad  : np.ndarray, ||FD gradient - exact directional derivative||
    err_H     : np.ndarray, ||FD Hessian-vector - H*h||_inf
    symm_err  : float, relative Hessian symmetry error (should be ~1e-10
                or smaller; hippylib warns if > 1e-10)
    """
    rng = np.random.default_rng(seed)
    n = Hop.ndofs_m
    CC_typical_scale = np.linalg.norm(CC_map) / np.sqrt(n) if scale_to_CC else 1.0

    print("=" * 70)
    print("hIPPYlib-style modelVerify (ported algorithm)")
    print(f"CC_map typical magnitude per dof: {CC_typical_scale:.4e}")
    print("=" * 70)

    # random test direction h -- NORMALIZED to unit norm so the
    # perturbation magnitude is controlled solely by eps (and
    # CC_typical_scale), not by h's random magnitude which would be
    # ~sqrt(n) ~ 28 for n=771 dofs and make all large-eps solves fail
    h = rng.standard_normal(n)
    h /= np.linalg.norm(h)

    # -------------------------------------------------------------------
    # base point: cost, gradient, and the "exact" directional derivative
    # grad_x . h  (hippylib: cx, grad_x, grad_xh = grad_x.inner(h))
    # -------------------------------------------------------------------
    cx = eval_J(CC_map)
    grad_x = eval_gradient(CC_map)
    grad_xh = grad_x @ h

    # -------------------------------------------------------------------
    # "exact" Hessian-vector product H*h (hippylib: H.mult(h, Hh))
    # -------------------------------------------------------------------
    Hh = Hop.mult(h)

    # -------------------------------------------------------------------
    # eps sweep: hippylib default is eps = 0.5**arange(32), then
    # reversed so the LARGEST step comes first. We rescale so the
    # largest value equals max_eps (default 0.25, not hippylib's 1.0 --
    # see max_eps docstring above for why).
    # -------------------------------------------------------------------
    eps = np.power(0.5, np.arange(n_eps))
    eps = eps[::-1]  # largest first, matching hippylib's convention
    eps = eps * (max_eps / eps[0])  # rescale so eps[0] == max_eps

    err_grad  = np.zeros(n_eps)
    err_H     = np.zeros(n_eps)
    my_eps_arr = np.zeros(n_eps)  # actual perturbation magnitudes used (for x-axis)
    n_failed  = 0

    for i in range(n_eps):
        my_eps = eps[i] * CC_typical_scale
        my_eps_arr[i] = my_eps

        # perturbed point (hippylib: x_plus = m0 + my_eps*h)
        CC_plus = CC_map + my_eps * h

        # forward solve + cost at the perturbed point
        #
        # NOTE: hippylib's original eps sweep starts at eps=1.0 (the
        # LARGEST step), which here corresponds to a perturbation of
        # magnitude ~CC_typical_scale added to a field already in
        # [1, 16]. For this nonlinear hyperelastic problem, large early
        # steps can produce a CC field heterogeneous enough that
        # solve_nl_prob's own Newton solver fails to converge (a
        # RuntimeError, not a numerical inaccuracy). We catch that here
        # and mark the corresponding eps as failed (NaN) rather than
        # letting it crash the whole 32-point sweep -- the SMALL-eps
        # end of the sweep (which is what actually matters for the
        # log-log slope check) is unaffected by skipping a few large,
        # unstable steps at the start.
        try:
            c_plus = eval_J(CC_plus)
            grad_xplus = eval_gradient(CC_plus)
        except RuntimeError as e:
            print(f"  [eps={eps[i]:.3e}, eps_used={my_eps:.3e}] forward "
                  f"solve failed ({e}); marking as NaN and continuing.")
            err_grad[i] = np.nan
            err_H[i] = np.nan
            n_failed += 1
            continue

        dc = c_plus - cx
        err_grad[i] = abs(dc / my_eps - grad_xh)

        # gradient at the perturbed point, for the Hessian FD check
        err_vec = (grad_xplus - grad_x) / my_eps - Hh
        err_H[i] = np.max(np.abs(err_vec))  # L-infinity norm

    if n_failed > 0:
        print(f"\n  {n_failed}/{n_eps} eps values failed (forward solver "
              f"did not converge) and were excluded from the plot/analysis.")
        print(f"  This is expected for the largest eps values on this "
              f"nonlinear problem -- only the SMALL-eps tail (where the "
              f"log-log slope check matters) needs to have succeeded.")

    # -------------------------------------------------------------------
    # independent algebraic symmetry check: (y, Hx) == (x, Hy)
    # -------------------------------------------------------------------
    xx = rng.standard_normal(n)
    yy = rng.standard_normal(n)
    Hxx = Hop.mult(xx)
    Hyy = Hop.mult(yy)
    ytHx = yy @ Hxx
    xtHy = xx @ Hyy

    denom = ytHx + xtHy
    if abs(denom) > 0:
        symm_err = 2.0 * abs(ytHx - xtHy) / abs(denom)
    else:
        symm_err = abs(ytHx - xtHy)

    print(f"\n(y, H x) - (x, H y) relative symmetry error: {symm_err:.6e}")
    if symm_err > 1e-10:
        print("  WARNING: HESSIAN IS NOT SYMMETRIC! (hippylib's own "
              "threshold is 1e-10 -- this is a strong signal of a real "
              "bug in HessianOperator, independent of any finite-")
        print("  difference comparison.)")
    else:
        print("  PASS: Hessian symmetry confirmed.")

    # -------------------------------------------------------------------
    # log-log convergence plot: a correct gradient/Hessian shows error
    # decaying linearly with eps (slope 1 in log-log) before floating-
    # point round-off causes the curve to turn back up -- a "V" shape.
    # A flat, eps-independent plateau instead indicates a genuine bug.
    # -------------------------------------------------------------------
    if plot:
        try:
            import matplotlib.pyplot as plt

            valid_grad = np.where(~np.isnan(err_grad))[0]
            valid_H    = np.where(~np.isnan(err_H))[0]

            # relative errors (normalized by the "exact" quantity being checked)
            grad_xh_norm = abs(grad_xh) + 1e-30
            Hh_norm      = np.linalg.norm(Hh) + 1e-30
            rel_err_grad = err_grad / grad_xh_norm
            rel_err_H    = err_H    / Hh_norm

            print(f"\n  ||grad_xh|| (exact directional derivative) : {grad_xh_norm:.4e}")
            print(f"  ||H*h||     (exact Hessian-vector product)  : {Hh_norm:.4e}")
            if len(valid_grad) > 0:
                print(f"  min rel err_grad : {np.nanmin(rel_err_grad):.4e}")
            if len(valid_H) > 0:
                print(f"  min rel err_H    : {np.nanmin(rel_err_H):.4e}")
                print(f"  (if min rel err_H << 1 the Hessian is correct;")
                print(f"   a flat curve in the absolute plot just means the")
                print(f"   noise floor is well below ||H*h||, which is a")
                print(f"   GOOD sign, not a bug)")

            fig, axes = plt.subplots(2, 2, figsize=(12, 9))

            # row 0: absolute error (same as before)
            axes[0, 0].loglog(my_eps_arr, err_grad, "-ob", label="FD error (abs)")
            if len(valid_grad) > 0:
                i0 = valid_grad[0]
                axes[0, 0].loglog(my_eps_arr,
                                   my_eps_arr * (err_grad[i0] / my_eps_arr[i0]),
                                   "-.k", label="slope 1 (theoretical)")
            axes[0, 0].set_xlabel("eps (actual perturbation magnitude)")
            axes[0, 0].set_ylabel("absolute error")
            axes[0, 0].set_title("FD Gradient Check (absolute)")
            axes[0, 0].legend()

            axes[0, 1].loglog(my_eps_arr, err_H, "-ob", label="FD error (abs)")
            if len(valid_H) > 0:
                i0 = valid_H[0]
                axes[0, 1].loglog(my_eps_arr,
                                   my_eps_arr * (err_H[i0] / my_eps_arr[i0]),
                                   "-.k", label="slope 1 (theoretical)")
            axes[0, 1].set_xlabel("eps (actual perturbation magnitude)")
            axes[0, 1].set_ylabel("error (L-inf norm)")
            axes[0, 1].set_title("FD Hessian Check (absolute)")
            axes[0, 1].legend()

            # row 1: relative error (normalized by ||grad_xh|| and ||H*h||)
            axes[1, 0].loglog(my_eps_arr, rel_err_grad, "-ob", label="relative FD error")
            if len(valid_grad) > 0:
                i0 = valid_grad[0]
                axes[1, 0].loglog(my_eps_arr,
                                   my_eps_arr * (rel_err_grad[i0] / my_eps_arr[i0]),
                                   "-.k", label="slope 1 (theoretical)")
            axes[1, 0].axhline(1e-2, color="r", linestyle=":", label="1% threshold")
            axes[1, 0].set_xlabel("eps (actual perturbation magnitude)")
            axes[1, 0].set_ylabel("relative error / ||grad_xh||")
            axes[1, 0].set_title("FD Gradient Check (relative)")
            axes[1, 0].legend()

            axes[1, 1].loglog(my_eps_arr, rel_err_H, "-ob", label="relative FD error")
            if len(valid_H) > 0:
                i0 = valid_H[0]
                axes[1, 1].loglog(my_eps_arr,
                                   my_eps_arr * (rel_err_H[i0] / my_eps_arr[i0]),
                                   "-.k", label="slope 1 (theoretical)")
            axes[1, 1].axhline(1e-2, color="r", linestyle=":", label="1% threshold")
            axes[1, 1].set_xlabel("eps (actual perturbation magnitude)")
            axes[1, 1].set_ylabel("relative error / ||H*h||")
            axes[1, 1].set_title("FD Hessian Check (relative)")
            axes[1, 1].legend()

            plt.tight_layout()
            if save_plot:
                plt.savefig(save_plot, dpi=150)
                print(f"\nSaved convergence plot: {save_plot}")
            plt.show()
        except Exception as e:
            print(f"\n[warning] could not generate plot: {e}")

    print("=" * 70)
    print("Interpretation:")
    print("  - In the log-log plot, look for a 'V' shape: error should")
    print("    decay along the dashed slope-1 reference line for a")
    print("    range of intermediate eps, then turn back up for very")
    print("    small eps (floating-point round-off dominates there).")
    print("  - A FLAT error curve (no decay region at all) indicates a")
    print("    genuine bug in the gradient or Hessian -- not just a")
    print("    poor choice of eps.")
    print("=" * 70)

    return my_eps_arr, err_grad, err_H, symm_err


# =============================================================================
# example usage (paste at the end of ex04, after the MAP point block)
# =============================================================================
#
# from hessian_ucq import HessianOperator
# from verify_hessian import verify_hessian, eps_scan, model_verify_hippylib_style
#
# CC_map = CC.x.array.copy()   # already holds CC* after optimization
#
# Hop = HessianOperator(Fun, Jfunctional, uh, CC, lmbda,
#                        V, Va, facet_tags, domain)
#
# # quick check (3 directions, 1 eps each):
# verify_hessian(Hop, eval_gradient, CC_map, n_directions=3, eps=0.2)
#
# # full hIPPYlib-style check (32-point eps sweep, log-log plot, symmetry test):
# eps_arr, err_grad, err_H, symm_err = model_verify_hippylib_style(
#     Hop, eval_J, eval_gradient, CC_map, n_eps=32
# )
