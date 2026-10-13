"""
uq_regions.py
-------------
Regional uncertainty summary for Case 2 (fibrosis), from the ex07 output
(no dolfinx, no rerun). Same regions as ex06/ex07 (fibrosis geometry of ex03):
    core        r <= 5 mm      transition 5 < r < 10 mm      healthy r >= 10 mm
    r = distance to (-5, -2, -9)

1) LaTeX table, one row per region (+ whole LV), volume-weighted:
     e_C^L2      relative L2 error of C_MAP in the region
     sigma_post  mean posterior std of m = log C  (~ relative uncertainty of C)
     VR          mean variance reduction 1 - sigma_post^2 / sigma_prior^2
     coverage    volume fraction where m_true lies in m_MAP +- 1.96 sigma_post
2) Figure, one row (same rendering/colour bars as plot_uq.py):
     ground truth C | P(C > C_th) | std reduction 1 - s_post/s_prior (+ sensors, 0-1, viridis)
   (--middle vr: variance reduction; --middle std: posterior std; --no-sensors: no dots)
   with the Laplace approximation  P(C > C_th) = Phi((m_MAP - log C_th) / sigma_post).

Usage:
    python uq_regions.py                      # fibrosis, C_th = 4 kPa
    python uq_regions.py --c-th 3.5 --root out_ex07

Reads  <root>/<case>/uq_<case>.h5
Writes <root>/<case>/table_regions_<case>.tex, regions_<case>.json,
       <root>/<case>/uq_regions_<case>.pdf/.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from scipy.special import ndtr

from plot_maps import read_case, PyVistaPanels, MatplotlibPanels, crop_white, get_cmap, uq_clim
from paper_style import apply_paper_style, save_figure

CENTER, R0, R1 = np.array([-5.0, -2.0, -9.0]), 5.0, 10.0      # as ex03/ex06/ex07
REGION_LABEL = {"core": "Core", "transition": "Transition", "healthy": "Healthy",
                "all": "Whole LV"}


def nodal_volumes(pts, tets):
    """Lumped (P1) volume weights: each tet gives |T|/4 to its four nodes."""
    a, b, c, d = (pts[tets[:, k]] for k in range(4))
    vol = np.abs(np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a)) / 6.0
    w = np.zeros(len(pts))
    np.add.at(w, tets.ravel(), np.repeat(vol / 4.0, 4))
    return w


def region_masks(pts):
    r = np.linalg.norm(pts - CENTER, axis=1)
    return {"core": r <= R0, "transition": (r > R0) & (r < R1), "healthy": r >= R1,
            "all": np.ones(len(pts), bool)}


def region_stats(v, w, masks, c_th):
    m_true, m_map = v["m_true"], v["m_map"]
    C_true, C_map = np.exp(m_true), np.exp(m_map)
    s_post, s_prior = v["m_post_std"], v["m_prior_std"]
    z = np.abs(m_true - m_map) / np.maximum(s_post, 1e-12)
    z_pr = np.abs(m_true - np.log(2.0)) / np.maximum(s_prior, 1e-12)
    prob = ndtr((m_map - np.log(c_th)) / np.maximum(s_post, 1e-12))
    out = {}
    for name, k in masks.items():
        if not k.any():
            continue
        wk = w[k] / w[k].sum()
        out[name] = dict(
            n_nodes=int(k.sum()), volume=float(w[k].sum()),
            C_rel_L2=float(np.sqrt(np.sum(w[k] * (C_map[k] - C_true[k]) ** 2)
                                   / np.sum(w[k] * C_true[k] ** 2))),
            post_std=float(wk @ s_post[k]), prior_std=float(wk @ s_prior[k]),
            var_reduction=float(wk @ (1.0 - s_post[k] ** 2 / s_prior[k] ** 2)),
            coverage95=float(wk @ (z[k] <= 1.96)),
            coverage95_prior=float(wk @ (z_pr[k] <= 1.96)),
            mean_P_exceed=float(wk @ prob[k]),
        )
    return out, prob


def latex_table(stats, case, noise_txt):
    rows = []
    for name in ["core", "transition", "healthy", "all"]:
        if name not in stats:
            continue
        s = stats[name]
        if name == "all":
            rows.append(r"\midrule")
        rows.append(f"{REGION_LABEL[name]:<11s} & ${100 * s['C_rel_L2']:.2f}$ & "
                    f"${s['post_std']:.3f}$ & ${s['var_reduction']:.3f}$ & "
                    f"${100 * s['coverage95']:.1f}$ \\\\")
    body = "\n".join(rows)
    return rf"""% Regional uncertainty, {case} -- requires \usepackage{{booktabs}}
\begin{{table}}[htbp]
\centering
\caption{{Regional summary of the Laplace approximation for Case~2 (fibrosis){noise_txt}:
relative $L^2$ error of $C_{{\mathrm{{MAP}}}}$, mean posterior standard deviation of $m=\log C$,
mean variance reduction $1-\sigma_{{\mathrm{{post}}}}^2/\sigma_{{\mathrm{{prior}}}}^2$, and fraction of the
region where the truth lies inside the pointwise 95\% credible interval (volume-weighted).}}
\label{{tab:uq_regions}}
\begin{{tabular}}{{lcccc}}
\toprule
Region & $e_C^{{L^2}}$ [\%] & $\bar\sigma_{{\mathrm{{post}}}}$ & Var.\ red. & Coverage [\%] \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def sensor_coords(pts, n_sensors, sensor_file=None):
    """Sensor positions: from a file (N x 3, .npy/.txt) or by repeating the ex07
    selection (farthest-point sampling on the mesh nodes, seed 42)."""
    if sensor_file:
        xs = np.load(sensor_file) if str(sensor_file).endswith(".npy") else np.loadtxt(sensor_file)
        return np.asarray(xs, float).reshape(-1, 3)
    try:
        from cardiac_utils import select_distributed_nodes
    except ImportError:
        print("  cardiac_utils not found -- sensors not shown (use --sensor-file)")
        return None
    idx, _ = select_distributed_nodes(pts, N=n_sensors, seed=42)
    return pts[np.asarray(idx, int)]


class PyVistaPanelsSensors(PyVistaPanels):
    """As plot_maps.PyVistaPanels, optionally with sensor dots (depth-tested:
    only sensors on the visible surface show)."""
    def render(self, vals, cmap, clim, points=None, point_size=9, point_color="red"):
        pl = self.pv.Plotter(off_screen=True, window_size=self.size)
        pl.set_background("white")
        g = self.surf.copy()
        g.point_data["C"] = vals[self.ids]
        pl.add_mesh(g, scalars="C", cmap=cmap, clim=clim, smooth_shading=True,
                    show_scalar_bar=False)
        if points is not None:
            pl.add_points(points, color=point_color, point_size=point_size,
                          render_points_as_spheres=True)
        pl.view_vector(self.view, viewup=(1, 0, 0))
        pl.camera.zoom(self.zoom)
        img = pl.screenshot(return_img=True)
        pl.close()
        return img


def plot_prob(pts, tets, v, prob, c_th, cmap, view, zoom, backend, out,
              middle="sr", sensors=None, c_clim=None, vr_clim=None, mid_cmap=None):
    ratio = v["m_post_std"] / v["m_prior_std"]
    if middle == "sr":                       # std reduction, 0 = prior, 1 = fully informed
        mid = ("Std. reduction", 1.0 - ratio, vr_clim or (0.0, 1.0),
               r"$1 - s_{\mathrm{post}}/s_{\mathrm{prior}}$")
    elif middle == "vr":
        mid = ("Variance reduction", 1.0 - ratio ** 2, vr_clim,
               r"$1 - s_{\mathrm{post}}^2/s_{\mathrm{prior}}^2$")
    else:
        mid = ("Posterior std", v["m_post_std"], None, r"$s_{\mathrm{post}}(m)$")
    mid_cmap = mid_cmap or cmap
    panels = [("Ground truth", np.exp(v["m_true"]), c_clim, r"$C$ [kPa]", cmap),
              (rf"$P(C > {c_th:g}$ kPa$)$", prob, (0.0, 1.0), "probability", cmap),
              mid + (mid_cmap,)]
    MID = 2                                  # column of the uncertainty panel (with sensors)
    Panels = PyVistaPanelsSensors if backend == "pyvista" else MatplotlibPanels
    if sensors is not None and backend != "pyvista":
        print("  sensor dots need the pyvista backend -- not shown")
    P = Panels(pts, tets, view, zoom)
    imgs, clims = [], []
    for j, (_, vals, clim, _, pcmap) in enumerate(panels):
        clim = clim or (float(vals.min()), float(vals.max()))
        clims.append(clim)
        if j == MID and sensors is not None and backend == "pyvista":
            img = P.render(vals, pcmap, clim, points=sensors)
        else:
            img = P.render(vals, pcmap, clim)
        imgs.append(crop_white(img))

    n = len(panels)
    aspect = np.median([im.shape[0] / im.shape[1] for im in imgs])
    w_panel = 1.45
    fig = plt.figure(figsize=((w_panel + 0.75) * n, w_panel * aspect + 0.35))
    gs = fig.add_gridspec(1, 2 * n, width_ratios=[1, 0.45] * n, wspace=0.04,
                          left=0.01, right=0.95, top=0.86, bottom=0.02)
    for j, ((title, _, _, label, pcmap), img, clim) in enumerate(zip(panels, imgs, clims)):
        ax = fig.add_subplot(gs[0, 2 * j])
        ax.imshow(img, interpolation="lanczos")
        ax.set_axis_off()
        ax.set_title(title, pad=2)
        slot = fig.add_subplot(gs[0, 2 * j + 1])
        slot.set_axis_off()
        pos = slot.get_position()
        h = pos.height * 0.70
        cax = fig.add_axes([pos.x0 + 0.02 * pos.width, pos.y0 + (pos.height - h) / 2,
                            0.13 * pos.width, h])
        cax.set_zorder(5)
        cb = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(*clim), cmap=pcmap), cax=cax)
        cb.set_label(label)
        cb.ax.tick_params(direction="out", length=3, width=0.6, right=True, left=False)
        cb.ax.minorticks_off()
        cb.outline.set_linewidth(0.8)
    save_figure(fig, str(out))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="out_ex07")
    ap.add_argument("--case", default="fibrosis", choices=["fibrosis", "linear"])
    ap.add_argument("--c-th", type=float, default=4.0,
                    help="Stiffness threshold for P(C > C_th) in kPa (default 4)")
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--cmap", default="fast")
    ap.add_argument("--view", type=float, nargs=3, default=[0.0, 0.0, -1.0])
    ap.add_argument("--zoom", type=float, default=1.15)
    ap.add_argument("--backend", choices=["auto", "pyvista", "matplotlib"], default="auto")
    ap.add_argument("--middle", choices=["sr", "vr", "std"], default="sr",
                    help="Middle panel: std reduction 1 - s_post/s_prior (default, 0-1), "
                         "variance reduction, or posterior std")
    ap.add_argument("--mid-cmap", default="viridis",
                    help="Colour map of the middle panel (sequential; default viridis)")
    ap.add_argument("--no-sensors", action="store_true", help="Do not draw the sensors")
    ap.add_argument("--vr-clim", type=float, nargs=2, default=None,
                    help="Variance-reduction colour range, e.g. 0 1 (default: min/max)")
    ap.add_argument("--num-nodes", type=int, default=64, help="Number of sensors (as ex07)")
    ap.add_argument("--sensor-file", default=None,
                    help="Sensor coordinates (N x 3, .npy/.txt) instead of re-selecting")
    ap.add_argument("--clim", type=float, nargs=2, default=None,
                    help="Colour range of the C panel in kPa (default: as plot_uq.py)")
    ap.add_argument("--truth-range", action="store_true",
                    help="C panel coloured over the ground-truth range only")
    ap.add_argument("--n-samples", type=int, default=4,
                    help="Samples in the plot_uq figure (for its colour range)")
    ap.add_argument("--use-tex", action="store_true")
    a = ap.parse_args()

    d = Path(a.root) / a.case
    pts, tets, v = read_case(d / f"uq_{a.case}.h5",
                             ["m_true", "m_map", "m_prior_std", "m_post_std"])
    w = nodal_volumes(pts, tets)
    masks = region_masks(pts)
    stats, prob = region_stats(v, w, masks, a.c_th)

    noise_txt = ""
    mfile = d / f"metrics_uq_{a.case}.json"
    if mfile.exists():
        s = json.loads(mfile.read_text()).get("settings", {})
        if "data_noise" in s:
            noise_txt = f", {100 * s['data_noise']:g}\\% noise"

    print(f"{'region':<11s} {'nodes':>6s} {'eL2[%]':>8s} {'s_prior':>8s} {'s_post':>8s} "
          f"{'VR':>6s} {'cov[%]':>7s} {'covPr[%]':>8s} {'P(C>th)':>8s}")
    for name, s in stats.items():
        print(f"{REGION_LABEL[name]:<11s} {s['n_nodes']:6d} {100 * s['C_rel_L2']:8.2f} "
              f"{s['prior_std']:8.3f} {s['post_std']:8.3f} {s['var_reduction']:6.3f} "
              f"{100 * s['coverage95']:7.1f} {100 * s['coverage95_prior']:8.1f} "
              f"{s['mean_P_exceed']:8.3f}")
    print(f"(coverage: truth inside the pointwise 95% band; covPr = same for the prior; "
          f"P(C>th) with C_th = {a.c_th:g} kPa)")

    (d / f"table_regions_{a.case}.tex").write_text(latex_table(stats, a.case, noise_txt))
    (d / f"regions_{a.case}.json").write_text(json.dumps(
        dict(case=a.case, c_th=a.c_th, regions={"center": CENTER.tolist(), "r0": R0, "r1": R1},
             stats=stats), indent=2))
    print(f"Wrote {d / f'table_regions_{a.case}.tex'} and regions_{a.case}.json")

    if not a.no_figure:
        apply_paper_style(use_tex=a.use_tex)
        backend = a.backend
        if backend == "auto":
            try:
                import pyvista  # noqa: F401
                backend = "pyvista"
            except ImportError:
                backend = "matplotlib"
        sensors = None if a.no_sensors else sensor_coords(pts, a.num_nodes, a.sensor_file)
        if sensors is not None:
            print(f"  {len(sensors)} sensors drawn on the std-reduction panel (visible surface only)")
        if a.clim:
            c_clim = tuple(a.clim)
        elif a.truth_range:
            c_clim = None
        else:                          # same colours as plot_uq.py / plot_maps --clim-from-uq
            c_clim = uq_clim(a.root, a.case, a.n_samples)
        if c_clim:
            print(f"  C panel colour range [{c_clim[0]:.3f}, {c_clim[1]:.3f}] kPa")
        plot_prob(pts, tets, v, prob, a.c_th, get_cmap(a.cmap), a.view, a.zoom, backend,
                  d / f"uq_regions_{a.case}", middle=a.middle, sensors=sensors, c_clim=c_clim,
                  vr_clim=tuple(a.vr_clim) if a.vr_clim else None,
                  mid_cmap=get_cmap(a.mid_cmap))


if __name__ == "__main__":
    main()
