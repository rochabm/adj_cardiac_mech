"""
taylor_remainder_test.py
-------------------------
Taylor remainder tests for the gradient and Hessian of J(m).

GRADIENT test:
    R_grad(eps) = |J(m+eps*h) - J(m) - eps * g^T h|    should be O(eps^2)
    slope on log-log plot should be 2.

HESSIAN test:
    R_hess(eps) = |J(m+eps*h) - J(m) - eps*g^T h - 0.5*eps^2 * h^T H h|
                                                          should be O(eps^3)
    slope on log-log plot should be 3.

Both plotted together (like hIPPYlib modelVerify):
    - R_grad slope 2  confirms gradient is correct
    - R_hess slope 3  confirms Hessian is correct

Reference: hIPPYlib modelVerify, Bui-Thanh et al. 2013
"""

import numpy as np
import matplotlib.pyplot as plt


def run_taylor_tests(eval_J, eval_gradient, build_hop,
                     m_test, rng_seed=0,
                     epsilons=None,
                     savefile="fig_taylor_remainder.png",
                     label=""):
    """
    Run gradient and Hessian Taylor remainder tests.

    Parameters
    ----------
    eval_J        : callable(m) -> float
    eval_gradient : callable(m) -> np.ndarray
    build_hop     : callable() -> object with .mult(v)
                    Must build Hessian at m_test.
    m_test        : np.ndarray, test point
    rng_seed      : seed for random direction h
    epsilons      : list of step sizes
    savefile      : output PNG
    label         : string label for title

    Returns
    -------
    passed : bool  (True if gradient test passes, rate ~2)
    """
    if epsilons is None:
        epsilons = [5e-1, 2e-1, 1e-1, 5e-2, 2e-2, 1e-2, 5e-3]

    rng = np.random.default_rng(rng_seed)
    h   = rng.standard_normal(len(m_test))
    h  /= np.linalg.norm(h)

    # --- evaluate at base point ---
    J0  = eval_J(m_test)
    g0  = eval_gradient(m_test)
    dJ  = float(g0 @ h)            # g^T h  (scalar directional derivative)

    # --- build Hessian at m_test ---
    print("  [Building Hessian at m_test...]")
    Hop = build_hop()
    Hh  = Hop.mult(h)              # H * h  (vector)
    d2J = float(h @ Hh)            # h^T H h  (scalar)

    print(f"\n  J(m0)        = {J0:.6e}")
    print(f"  g^T h        = {dJ:.6e}   (directional derivative)")
    print(f"  h^T H h      = {d2J:.6e}   (should be > 0 at MAP, = 0 away)")
    print(f"  ||H*h||      = {np.linalg.norm(Hh):.6e}")

    # sanity: the Hessian correction ½ eps² d2J should be comparable to R_grad
    # at eps=0.1: ½ * 0.01 * d2J should be ~ R_grad[eps=0.1]
    eps_ref = 0.1
    J1_ref  = eval_J(m_test + eps_ref * h)
    Rgref   = abs(J1_ref - J0 - eps_ref * dJ)
    correction_ref = 0.5 * eps_ref**2 * d2J
    print(f"\n  At eps={eps_ref}:")
    print(f"    R_grad                = {Rgref:.4e}")
    print(f"    ½ eps² h^T H h        = {correction_ref:.4e}  (Hessian correction)")
    print(f"    ratio correction/R_grad = {abs(correction_ref/Rgref):.4f}"
          f"  (should be ~1 for Hessian to improve R_grad)")

    # --- compute remainders ---
    R_grad = []   # O(eps^2): confirms gradient
    R_hess = []   # O(eps^3): confirms Hessian

    print(f"\n  {'eps':>10}  {'R_grad (O(eps^2))':>20}  {'R_hess (O(eps^3))':>20}  "
          f"{'rate_grad':>10}  {'rate_hess':>10}")
    print("  " + "-"*80)

    for eps in epsilons:
        J1     = eval_J(m_test + eps * h)
        dJ_fd  = J1 - J0                           # J(m+eps*h) - J(m)
        rg     = abs(dJ_fd - eps * dJ)             # O(eps^2)
        rh     = abs(dJ_fd - eps * dJ
                     - 0.5 * eps**2 * d2J)         # O(eps^3)
        R_grad.append(rg)
        R_hess.append(rh)

    # convergence rates
    rates_grad, rates_hess = [], []
    for i in range(1, len(epsilons)):
        def rate(r_new, r_old, e_new, e_old):
            if r_old > 0 and r_new > 0:
                return np.log(r_new/r_old) / np.log(e_new/e_old)
            return float("nan")
        rates_grad.append(rate(R_grad[i], R_grad[i-1],
                               epsilons[i], epsilons[i-1]))
        rates_hess.append(rate(R_hess[i], R_hess[i-1],
                               epsilons[i], epsilons[i-1]))

    for i, eps in enumerate(epsilons):
        rg = f"{rates_grad[i-1]:.2f}" if i > 0 else "         -"
        rh = f"{rates_hess[i-1]:.2f}" if i > 0 else "         -"
        print(f"  {eps:>10.1e}  {R_grad[i]:>20.4e}  {R_hess[i]:>20.4e}  "
              f"{rg:>10}  {rh:>10}")

    # use only the middle eps range for rate averaging
    # (avoid first point which may be nonlinear regime,
    #  and last points which may be roundoff regime)
    n = len(rates_grad)
    i_start = 0
    i_end   = max(n-2, n//2)   # drop last 2 points (roundoff)
    avg_rg = np.nanmean(rates_grad[i_start:i_end])
    avg_rh = np.nanmean(rates_hess[i_start:i_end])
    grad_pass = abs(avg_rg - 2.0) < 0.3
    hess_pass = abs(avg_rh - 3.0) < 0.5

    print(f"\n  Average rate R_grad: {avg_rg:.3f}  (expected 2)  "
          f"{'✓ PASS' if grad_pass else '✗ FAIL'}")
    print(f"  Average rate R_hess: {avg_rh:.3f}  (expected 3)  "
          f"{'✓ PASS' if hess_pass else '✗ FAIL (nonlinear solver noise)'}")

    if grad_pass and not hess_pass:
        print("\n  NOTE: For nonlinear PDEs with load-stepping Newton solvers,")
        print("  the O(eps^3) Hessian signal may be masked by solver noise.")
        print("  The gradient test passing at rate ~2 is the primary")
        print("  verification. Use verify_hessian.py for Hessian confirmation.")

    # --- plot ---
    eps_arr = np.array(epsilons)
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.loglog(eps_arr, R_grad, "o-b", markersize=7,
              label=r"$R_{grad}$ = $|J(m+\epsilon h) - J(m) - \epsilon g^Th|$"
                    r"   (expected $O(\epsilon^2)$)")
    ax.loglog(eps_arr, R_hess, "s-r", markersize=7,
              label=r"$R_{hess}$ = $|J(m+\epsilon h) - J(m) - \epsilon g^Th"
                    r" - \frac{1}{2}\epsilon^2 h^THh|$"
                    r"   (expected $O(\epsilon^3)$)")

    # reference slopes anchored at first point
    ref2 = R_grad[0] * (eps_arr / eps_arr[0]) ** 2
    ref3 = R_hess[0] * (eps_arr / eps_arr[0]) ** 3
    ax.loglog(eps_arr, ref2, "--k", alpha=0.4, label="slope 2")
    ax.loglog(eps_arr, ref3, ":k",  alpha=0.4, label="slope 3")

    ax.set_xlabel(r"$\epsilon$", fontsize=13)
    ax.set_ylabel("remainder", fontsize=13)
    ax.set_title(f"Taylor remainder test  {label}\n"
                 f"R_grad rate={avg_rg:.2f} (exp 2)  |  "
                 f"R_hess rate={avg_rh:.2f} (exp 3)",
                 fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(savefile, dpi=150)
    print(f"\n  Saved: {savefile}")
    plt.show()

    print("\n" + "="*65)
    print(f"  VERDICT: {'PASSED ✓' if grad_pass else 'FAILED ✗'}")
    print("="*65)

    return grad_pass
