"""
plot_uq.py
----------
Figures for the UQ results of ex07 (same rendering and style as plot_maps.py,
fonts and line style from paper_style.py).

1) Posterior samples, one row:
       Ground truth | MAP | posterior sample 1 ... 4      (+ one colour bar)
   python plot_uq.py --case fibrosis
   python plot_uq.py --case linear --no-truth

2) Eigenvalue spectra of the prior-preconditioned data-misfit Hessian,
   all cases found in --root on one axes:
   python plot_uq.py --eig

Reads  <root>/<case>/uq_<case>.h5  and  <root>/<case>/eigenvalues_<case>.txt
written by ex07_uq.py (no dolfinx needed).
Writes <out>.pdf and <out>.png  (vector text, 300-dpi renderings).
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors, cm

from plot_maps import read_case, PyVistaPanels, MatplotlibPanels, crop_white, get_cmap, \
    CASE_LABEL
from paper_style import apply_paper_style, save_figure, COLORS, FIG_SINGLE

CASES = ["linear", "fibrosis"]
CASE_COLOR = {"linear": COLORS["blue"], "fibrosis": COLORS["orange"]}
CASE_MARKER = {"linear": "o", "fibrosis": "s"}


# =============================================================================
# 1) truth | MAP | posterior samples
# =============================================================================

def plot_samples(root, case, n_samples, with_truth, cmap, view, zoom, backend, out):
    h5 = Path(root) / case / f"uq_{case}.h5"
    names = (["C_true"] if with_truth else []) + ["C_map"] + \
            [f"C_post_sample_{s}" for s in range(1, n_samples + 1)]
    pts, tets, vals = read_case(h5, names)
    titles = (["Ground truth"] if with_truth else []) + ["MAP"] + \
             [f"Posterior sample {s}" for s in range(1, n_samples + 1)]

    # one colour scale for all panels: the full range of what is shown
    lo = min(float(vals[nm].min()) for nm in names)
    hi = max(float(vals[nm].max()) for nm in names)
    clim = (lo, hi)

    Panels = PyVistaPanels if backend == "pyvista" else MatplotlibPanels
    P = Panels(pts, tets, view, zoom)
    imgs = [crop_white(P.render(vals[nm], cmap, clim)) for nm in names]

    ncol = len(imgs)
    aspect = np.median([im.shape[0] / im.shape[1] for im in imgs])
    w_panel = 1.15                                    # inches per heart (fits 7.0 in)
    fig = plt.figure(figsize=(w_panel * ncol + 0.75, w_panel * aspect + 0.35))
    gs = fig.add_gridspec(1, ncol + 1, width_ratios=[1] * ncol + [0.16],
                          wspace=0.06, left=0.01, right=0.93, top=0.86, bottom=0.02)
    for j, (img, title) in enumerate(zip(imgs, titles)):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(img, interpolation="lanczos")
        ax.set_axis_off()
        ax.set_title(title, pad=2)

    slot = fig.add_subplot(gs[0, ncol])
    slot.set_axis_off()
    pos = slot.get_position()
    h = pos.height * 0.70
    cax = fig.add_axes([pos.x0 + 0.10 * pos.width, pos.y0 + (pos.height - h) / 2,
                        0.36 * pos.width, h])
    cb = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(*clim), cmap=cmap), cax=cax)
    cb.set_label(r"$C$ [kPa]")
    cb.ax.tick_params(direction="out", length=3, width=0.6, right=True, left=False)
    cb.ax.minorticks_off()
    cb.outline.set_linewidth(0.8)

    save_figure(fig, str(out))
    plt.close(fig)


# =============================================================================
# 2) eigenvalue spectra
# =============================================================================

def plot_eigenvalues(root, cases, out, xmax=None, ymin=None):
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    n_max = 0
    for case in cases:
        f = Path(root) / case / f"eigenvalues_{case}.txt"
        if not f.exists():
            print(f"  missing: {f}")
            continue
        lam = np.loadtxt(f)
        lam = lam[lam > 0]
        i = np.arange(1, len(lam) + 1)
        n_max = max(n_max, len(lam))
        n1 = int(np.sum(lam > 1))
        print(f"  {CASE_LABEL[case]}: {len(lam)} eigenvalues, {n1} with lambda > 1, "
              f"lambda_1 = {lam[0]:.3e}, lambda_k = {lam[-1]:.3e}")
        ax.semilogy(i, lam, linestyle="-", linewidth=1.0, color=CASE_COLOR[case],
                    marker=CASE_MARKER[case], markevery=max(1, len(lam) // 16),
                    markersize=3.5, markerfacecolor="white", markeredgewidth=0.8,
                    label=CASE_LABEL[case])
    if n_max == 0:
        raise SystemExit(f"No eigenvalues_<case>.txt found under {root}/<case>/")
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel(r"number")
    ax.set_ylabel(r"eigenvalue $\lambda_i$")
    ax.set_xlim(0, xmax or n_max + 1)
    if ymin:
        ax.set_ylim(bottom=ymin)
    ax.legend(loc="upper right")
    save_figure(fig, str(out))
    plt.close(fig)


# =============================================================================
# main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="out_ex07")
    ap.add_argument("--case", choices=CASES, help="Posterior-sample figure for this case")
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--no-truth", action="store_true", help="Omit the ground-truth panel")
    ap.add_argument("--eig", action="store_true", help="Eigenvalue figure (all cases found)")
    ap.add_argument("--eig-xmax", type=int, default=None)
    ap.add_argument("--eig-ymin", type=float, default=None)
    ap.add_argument("--cmap", default="fast")
    ap.add_argument("--view", type=float, nargs=3, default=[0.0, 0.0, -1.0])
    ap.add_argument("--zoom", type=float, default=1.15)
    ap.add_argument("--backend", choices=["auto", "pyvista", "matplotlib"], default="auto")
    ap.add_argument("--use-tex", action="store_true", help="LaTeX text (newtx), as paper_style")
    ap.add_argument("--out", default=None, help="Output stem (no extension)")
    a = ap.parse_args()
    if not a.case and not a.eig:
        ap.error("give --case linear|fibrosis (samples) and/or --eig (eigenvalues)")

    apply_paper_style(use_tex=a.use_tex)

    if a.case:
        backend = a.backend
        if backend == "auto":
            try:
                import pyvista  # noqa: F401
                backend = "pyvista"
            except ImportError:
                backend = "matplotlib"
        out = a.out or str(Path(a.root) / f"posterior_samples_{a.case}")
        plot_samples(a.root, a.case, a.n_samples, not a.no_truth, get_cmap(a.cmap),
                     a.view, a.zoom, backend, out)
    if a.eig:
        out = (a.out + "_eig") if (a.out and a.case) else (a.out or str(Path(a.root) / "eigenvalues"))
        plot_eigenvalues(a.root, CASES, out, a.eig_xmax, a.eig_ymin)


if __name__ == "__main__":
    main()
