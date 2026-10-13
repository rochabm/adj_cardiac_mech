"""
plot_scarce.py
--------------
One figure for the scarce-data scenario (Case 2, fibrosis), two rows:

  top    : ground truth | MAP | posterior samples 1..4        (maps run, e.g. 16 sensors)
           one colour scale (C [kPa])
  bottom : P(C > C_th) | std reduction 1 - s_post/s_prior (0-1, viridis, sensors as
           dots)  (maps run) | eigenvalue spectrum of the maps run
           (other runs can be overlaid with --spectra)

Reads (no dolfinx, no new computations):
    <root>/<case>/uq_<case>.h5               (maps run)
    <run>/<case>/eigenvalues_<case>.txt      (every --spectra run)

Usage:
    python plot_scarce.py --sensors 16                             # out_ex07_s16 only
    python plot_scarce.py --sensors 32                             # out_ex07_s32 only
    python plot_scarce.py --root out_ex07_s16 --n-sensors 16 \
        --spectra out_ex07:64 out_ex07_s32:32 out_ex07_s16:16      # 16 maps, three spectra
    python plot_scarce.py --clim 1 10                              # fixed C range
Writes <out>.pdf/.png (default out_compare_sensors/scarce_fibrosis_s<N>)
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from scipy.special import ndtr

from plot_maps import read_case, MatplotlibPanels, crop_white, get_cmap
from uq_regions import PyVistaPanelsSensors, sensor_coords
from paper_style import apply_paper_style, save_figure, COLORS


def add_colorbar(fig, slot_spec, clim, cmap, label, frac_w=0.13):
    slot = fig.add_subplot(slot_spec)
    slot.set_axis_off()
    pos = slot.get_position()
    h = pos.height * 0.75
    cax = fig.add_axes([pos.x0 + 0.02 * pos.width, pos.y0 + (pos.height - h) / 2,
                        frac_w * pos.width, h])
    cax.set_zorder(5)
    cb = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(*clim), cmap=cmap), cax=cax)
    cb.set_label(label)
    cb.ax.tick_params(direction="out", length=3, width=0.6, right=True, left=False)
    cb.ax.minorticks_off()
    cb.outline.set_linewidth(0.8)
    return cb


def show(ax, img, title):
    ax.imshow(img, interpolation="lanczos")
    ax.set_axis_off()
    ax.set_title(title, pad=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sensors", type=int, default=None, metavar="N",
                    help="Shortcut: maps from out_ex07_s<N> with N sensors (e.g. 16 or 32)")
    ap.add_argument("--root", "--root16", dest="root", default="out_ex07_s16",
                    help="Run whose maps are shown (default out_ex07_s16)")
    ap.add_argument("--n-sensors", "--n-scarce", dest="n_sensors", type=int, default=16,
                    help="Sensors of --root (default 16)")
    ap.add_argument("--spectra", nargs="+", default=None, metavar="ROOT:N",
                    help="Runs in the spectrum panel (default: only --root)")
    ap.add_argument("--case", default="fibrosis")
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--c-th", type=float, default=4.0)
    ap.add_argument("--clim", type=float, nargs=2, default=None,
                    help="C range of the top row (default: min/max of its panels)")
    ap.add_argument("--middle", choices=["sr", "vr"], default="sr",
                    help="Std reduction 1 - s_post/s_prior (default) or variance reduction")
    ap.add_argument("--vr-clim", type=float, nargs=2, default=[0.0, 1.0],
                    help="Colour range of that panel (default 0 1, as uq_regions.py)")
    ap.add_argument("--mid-cmap", default="viridis",
                    help="Colour map of that panel (sequential; default viridis, as uq_regions.py)")
    ap.add_argument("--no-sensors", action="store_true")
    ap.add_argument("--cmap", default="fast")
    ap.add_argument("--view", type=float, nargs=3, default=[0.0, 0.0, -1.0])
    ap.add_argument("--zoom", type=float, default=1.15)
    ap.add_argument("--backend", choices=["auto", "pyvista", "matplotlib"], default="auto")
    ap.add_argument("--use-tex", action="store_true")
    ap.add_argument("--out", default=None,
                    help="Output stem (default out_compare_sensors/scarce_<case>_s<N>)")
    a = ap.parse_args()

    apply_paper_style(use_tex=a.use_tex)
    cmap = get_cmap(a.cmap)
    backend = a.backend
    if backend == "auto":
        try:
            import pyvista  # noqa: F401
            backend = "pyvista"
        except ImportError:
            backend = "matplotlib"
    Panels = PyVistaPanelsSensors if backend == "pyvista" else MatplotlibPanels

    if a.sensors:
        a.root, a.n_sensors = f"out_ex07_s{a.sensors}", a.sensors
    case, ns = a.case, a.n_sensors
    out = a.out or f"out_compare_sensors/scarce_{case}_s{ns}"
    spectra = []
    for item in (a.spectra or [f"{a.root}:{ns}"]):
        r, n = item.rsplit(":", 1)
        spectra.append((int(n), r))
    spectra.sort(key=lambda t: -t[0])                   # most sensors first
    sample_names = [f"C_post_sample_{s}" for s in range(1, a.n_samples + 1)]
    pts, tets, v16 = read_case(Path(a.root) / case / f"uq_{case}.h5",
                               ["C_true", "C_map", "m_map", "m_prior_std", "m_post_std"]
                               + sample_names)
    P = Panels(pts, tets, a.view, a.zoom)

    def render(vals, clim, points=None, pcmap=None):
        pcmap = pcmap or cmap
        if points is not None and backend == "pyvista":
            return crop_white(P.render(vals, pcmap, clim, points=points))
        return crop_white(P.render(vals, pcmap, clim))

    # ---- top row: truth | MAP | samples ---------------------------------------
    top_names = ["C_true", "C_map"] + sample_names
    top_titles = ["Ground truth", "MAP"] + [f"Sample {s}" for s in range(1, a.n_samples + 1)]
    clim_C = tuple(a.clim) if a.clim else (
        min(float(v16[n].min()) for n in top_names), max(float(v16[n].max()) for n in top_names))
    print(f"  top row: C colour range [{clim_C[0]:.3f}, {clim_C[1]:.3f}] kPa")
    top_imgs = [render(v16[n], clim_C) for n in top_names]

    # ---- bottom row: VR | P(C>th) | spectra -----------------------------------
    ratio = v16["m_post_std"] / v16["m_prior_std"]
    if a.middle == "sr":
        vr16, mid_title, mid_label = (1.0 - ratio, "Std. reduction",
                                      r"$1 - s_{\mathrm{post}}/s_{\mathrm{prior}}$")
    else:
        vr16, mid_title, mid_label = (1.0 - ratio ** 2, "Variance reduction",
                                      r"$1 - s_{\mathrm{post}}^2/s_{\mathrm{prior}}^2$")
    clim_vr = tuple(a.vr_clim)
    mid_cmap = get_cmap(a.mid_cmap)
    print(f"  {mid_title.lower()} ({ns} sensors): range [{vr16.min():.3f}, {vr16.max():.3f}], "
          f"nodal mean {vr16.mean():.3f}; colour range {clim_vr}")
    prob16 = ndtr((v16["m_map"] - np.log(a.c_th)) / np.maximum(v16["m_post_std"], 1e-12))
    s16 = None if a.no_sensors else sensor_coords(pts, ns)
    bot_imgs = [render(vr16, clim_vr, s16, mid_cmap), render(prob16, (0.0, 1.0))]

    # ---- layout --------------------------------------------------------------
    nt = len(top_imgs)
    aspect = np.median([im.shape[0] / im.shape[1] for im in top_imgs + bot_imgs])
    w_fig = 7.2
    w_panel = (w_fig - 0.6) / nt
    h_row = w_panel * aspect + 0.30
    fig = plt.figure(figsize=(w_fig, 2 * h_row + 0.55))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1.12], hspace=0.12,
                             left=0.01, right=0.99, top=0.95, bottom=0.08)

    gt = outer[0].subgridspec(1, nt + 1, width_ratios=[1] * nt + [0.45], wspace=0.04)
    for j, (img, title) in enumerate(zip(top_imgs, top_titles)):
        show(fig.add_subplot(gt[0, j]), img, title)
    add_colorbar(fig, gt[0, nt], clim_C, cmap, r"$C$ [kPa]")

    gb = outer[1].subgridspec(1, 5, width_ratios=[1, 0.5, 1, 0.5, 2.2], wspace=0.04)
    show(fig.add_subplot(gb[0, 0]), bot_imgs[1], rf"$P(C>{a.c_th:g}$ kPa$)$")
    add_colorbar(fig, gb[0, 1], (0.0, 1.0), cmap, "probability")
    show(fig.add_subplot(gb[0, 2]), bot_imgs[0], mid_title)
    add_colorbar(fig, gb[0, 3], clim_vr, mid_cmap, mid_label)

    # spectra: an inset axes inside the last slot, leaving room for the labels
    slot = fig.add_subplot(gb[0, 4])
    slot.set_axis_off()
    pos = slot.get_position()
    ax = fig.add_axes([pos.x0 + 0.22 * pos.width, pos.y0 + 0.16 * pos.height,
                       0.62 * pos.width, 0.72 * pos.height])
    n_max = 0
    styles = [(COLORS["blue"], "o"), (COLORS["green"], "^"), (COLORS["orange"], "s"),
              (COLORS["purple"], "D")]
    for (nsens, root), (col, mk) in zip(spectra, styles):
        f = Path(root) / case / f"eigenvalues_{case}.txt"
        if not f.exists():
            print(f"  missing: {f}")
            continue
        lam = np.loadtxt(f)
        lam = lam[lam > 0]
        n_max = max(n_max, len(lam))
        print(f"  {nsens} sensors: {int(np.sum(lam > 1))} eigenvalues > 1")
        ax.semilogy(np.arange(1, len(lam) + 1), lam, color=col, lw=1.0, marker=mk,
                    markevery=max(1, len(lam) // 8), markersize=3, markerfacecolor="white",
                    markeredgewidth=0.7, label=f"{nsens} sensors")
    ax.axhline(1.0, color="black", lw=0.7, ls="--")
    ax.set_xlim(0, n_max + 1)
    ax.set_xlabel(r"number")
    ax.set_ylabel(r"$\lambda_i$")
    if len(spectra) > 1:
        ax.legend(loc="upper right", fontsize=7, handlelength=1.4, borderpad=0.3)
    ax.set_title("Spectrum" if len(spectra) > 1 else f"Spectrum, {ns} data points", pad=2)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, str(out))
    plt.close(fig)


if __name__ == "__main__":
    main()
