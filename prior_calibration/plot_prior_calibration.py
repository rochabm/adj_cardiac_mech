"""
plot_prior_calibration.py
-------------------------
Figures for the "Prior calibration" subsection, from the files written by
prior_calibration.py (paper style).

    fig_prior_calibration   (a) correlation vs distance: measured (mean and
                                10-90 % band) vs Matern curve for the target rho
                            (b) histogram of the pointwise prior std vs target sigma
    fig_prior_samples       planar cut: m0 +- 2 std, prior samples, truth
    fig_prior_3d            Bui-Thanh et al. (2013, Fig. 6.3) style: +-2 sigma
                            fields, 4 prior samples and the ground truth on the
                            3D surface, as deviations from the prior mean, one
                            diverging colour scale

Usage:
    python plot_prior_calibration.py calib_ellipsoid
    python plot_prior_calibration.py calib_ellipsoid --plane-point -5 -2 -9 --plane-normal 0 1 0

Everything you may want to change is in the CONFIG block.
"""

import argparse
from math import gamma as gammafn, sqrt
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
from scipy.spatial import cKDTree
from scipy.special import kv

from paper_style import apply_paper_style, save_figure, COLORS, FIG_WIDE

# =============================================================================
# CONFIG
# =============================================================================
N_SAMPLES_SHOWN = 3
CMAP_FIELD = "viridis"
CMAP_DEV = "RdBu_r"               # diverging map of the Bui-Thanh style figure
VIEW = None                       # (elev, azim) in degrees; None = look at the truth
TRUTH_SCALE = 0.58                # ground-truth LV height / two-row block (0.5 = same size as a sample)
PAD = 0.008                       # gap between neighbouring LVs (figure fraction)
GRID_RES = 0.3                    # mm, resolution of the planar cut
LABEL_C = r"$C$ [kPa]"


def matern_corr(r, nu, kappa):
    r = np.asarray(r, float)
    out = np.ones_like(r)
    z = kappa * r[r > 0]
    out[r > 0] = 2 ** (1 - nu) / gammafn(nu) * z ** nu * kv(nu, z)
    return out


def plane_operator(points, cells, center, normal, res):
    normal = normal / np.linalg.norm(normal)
    u1 = np.cross(normal, [0.0, 0.0, 1.0])
    if np.linalg.norm(u1) < 0.1:
        u1 = np.cross(normal, [1.0, 0.0, 0.0])
    u1 /= np.linalg.norm(u1)
    u2 = np.cross(normal, u1)
    rel = points - center
    near = np.abs(rel @ normal) < 5.0
    gx = np.arange((rel @ u1)[near].min(), (rel @ u1)[near].max() + res, res)
    gy = np.arange((rel @ u2)[near].min(), (rel @ u2)[near].max() + res, res)
    X, Y = np.meshgrid(gx, gy)
    G = center + X.ravel()[:, None] * u1 + Y.ravel()[:, None] * u2
    xc = points[cells]
    _, cand = cKDTree(xc.mean(1)).query(G, k=min(24, len(cells)))
    idx = np.zeros((len(G), 4), dtype=np.int64)
    w = np.zeros((len(G), 4))
    inside = np.zeros(len(G), bool)
    for j in range(cand.shape[1]):
        todo = ~inside
        if not todo.any():
            break
        c = cand[todo, j]
        x0 = xc[c, 0]
        Mt = np.stack([xc[c, 1] - x0, xc[c, 2] - x0, xc[c, 3] - x0], axis=2)
        lam = np.linalg.solve(Mt, (G[todo] - x0)[..., None])[..., 0]
        bary = np.c_[1 - lam.sum(1), lam]
        ok = (bary >= -1e-9).all(1)
        rows = np.where(todo)[0][ok]
        idx[rows], w[rows], inside[rows] = cells[c[ok]], bary[ok], True
    return X, Y, idx, w, inside


def on_plane(op, vals):
    X, Y, idx, w, inside = op
    img = np.full(len(inside), np.nan)
    img[inside] = (vals[idx[inside]] * w[inside]).sum(1)
    return img.reshape(X.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("calib_dir", nargs="?", default="calib")
    ap.add_argument("--plane-point", type=float, nargs=3, default=None,
                    help="default: point of max truth (or mesh centroid)")
    ap.add_argument("--plane-normal", type=float, nargs=3, default=None,
                    help="default: 2nd principal axis (long-axis section)")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    d = Path(a.calib_dir)
    apply_paper_style()

    # ---------------- (a) correlation, (b) std histogram ----------------
    z = np.load(d / "calib_fields.npz")
    sigma, rho, order = float(z["sigma"]), float(z["rho"]), int(z["order"])
    nu = order - 1.5
    kappa_t = sqrt(8 * nu) / rho
    cr = np.loadtxt(d / "calib_corr.txt", ndmin=2)
    r, cm, p10, p90 = cr[:, 0], cr[:, 1], cr[:, 2], cr[:, 3]
    is_log = bool(z["log"])

    fig, ax = plt.subplots(1, 2, figsize=FIG_WIDE)
    ax[0].fill_between(r, p10, p90, color=COLORS["blue"], alpha=0.2, lw=0,
                       label="measured 10--90 %")
    ax[0].plot(r, cm, "-", color=COLORS["blue"], label="measured mean")
    rr = np.linspace(0, r.max(), 200)
    ax[0].plot(rr, matern_corr(rr, nu, kappa_t), "--", color=COLORS["orange"],
               label=rf"Matérn, $\nu$ = {nu:g}, $\rho$ = {rho:.3g} mm")
    tfile = d / "calib_corr_truth.txt"
    if tfile.exists():
        tc = np.loadtxt(tfile, ndmin=2)
        ax[0].plot(tc[:, 0], tc[:, 1], "-.", color="black", lw=0.9,
                   label="ground truth")
    ax[0].axvline(rho, color=COLORS["gray"], ls=":", lw=0.8)
    ax[0].axhline(0, color="black", lw=0.4)
    ax[0].set_xlabel(r"Distance $r$ [mm]")
    ax[0].set_ylabel(r"Prior correlation [$-$]")
    ax[0].legend(loc="upper right")

    std = z["std"]
    ax[1].hist(std, bins=30, color=COLORS["blue"], alpha=0.8, edgecolor="white", lw=0.3)
    ax[1].axvline(sigma, color=COLORS["orange"], ls="--",
                  label=rf"target $\sigma$ = {sigma:.3g}")
    ax[1].set_xlabel(("Pointwise prior std of $\\log C$ [$-$]" if is_log
                      else "Pointwise prior std [kPa]"))
    ax[1].set_ylabel(r"Number of nodes [$-$]")
    ax[1].legend(loc="upper left")
    fig.tight_layout()
    save_figure(fig, str(d / "fig_prior_calibration"))

    # ---------------- planar cut: +-2 std, samples, truth ----------------
    pts, cells = z["points"], z["cells"]
    m0 = float(z["m0"])
    truth = z["truth"] if z["truth"].size else None
    if a.plane_point is not None:
        center = np.array(a.plane_point)
    elif truth is not None:
        center = pts[np.argmax(truth)]
    else:
        center = pts.mean(0)
    if a.plane_normal is not None:
        normal = np.array(a.plane_normal, float)
    else:
        _, _, vt = np.linalg.svd(pts - pts.mean(0), full_matrices=False)
        normal = vt[1]
    op = plane_operator(pts, cells, center, normal, GRID_RES)

    to_C = np.exp if is_log else (lambda x: x)
    panels = [(to_C(m0 + 2 * std), r"$m_0 + 2\sigma$"),
              (to_C(m0 - 2 * std), r"$m_0 - 2\sigma$")]
    for i, s in enumerate(z["samples"][:N_SAMPLES_SHOWN]):
        panels.append((to_C(s), f"prior sample {i + 1}"))
    if truth is not None:
        panels.append((truth, "ground truth"))
    imgs = [on_plane(op, v) for v, _ in panels]
    lo = np.nanmin([np.nanmin(i) for i in imgs])
    hi = np.nanmax([np.nanmax(i) for i in imgs])
    if is_log:            # log prior -> log colour scale (C = exp(m))
        norm = LogNorm(vmin=lo, vmax=hi)
        levels = np.geomspace(lo, hi, 31)
        ticks = [t for t in (0.25, 0.5, 1, 2, 4, 8, 16, 32) if lo <= t <= hi]
    else:
        norm = None
        levels = np.linspace(lo, hi, 31)
        ticks = MaxNLocator(6).tick_values(lo, hi)
        ticks = ticks[(ticks >= lo) & (ticks <= hi)]

    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(FIG_WIDE[0], 2.2 * nrow + 0.3),
                            squeeze=False)
    X, Y = op[0], op[1]
    cf = None
    for k, (img, (_, title)) in enumerate(zip(imgs, panels)):
        axk = axs.flat[k]
        cf = axk.contourf(X, Y, img, levels=levels, cmap=CMAP_FIELD, norm=norm)
        axk.set_title(title)
        axk.set_aspect("equal")
        axk.set_xticks([])
        axk.set_yticks([])
    for k in range(len(panels), nrow * ncol):
        axs.flat[k].axis("off")
    cb = fig.colorbar(cf, ax=axs, fraction=0.025, pad=0.02, label=LABEL_C)
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{t:g}" for t in ticks])
    save_figure(fig, str(d / "fig_prior_samples"))
    # ---------------- Bui-Thanh style 3D figure ----------------
    fig3d = figure_3d(pts, cells, z, std, m0, truth, is_log)
    save_figure(fig3d, str(d / "fig_prior_3d"))
    if a.show:
        plt.show()


def _boundary_faces(cells):
    f = np.sort(np.vstack([cells[:, [1, 2, 3]], cells[:, [0, 2, 3]],
                           cells[:, [0, 1, 3]], cells[:, [0, 1, 2]]]), axis=1)
    u, c = np.unique(f, axis=0, return_counts=True)
    return u[c == 1]


def _surface(ax, pts, faces, vals, norm, cmap, view_dir, elev, azim):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    tri = pts[faces]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-30
    shade = 0.55 + 0.45 * np.abs(nrm @ view_dir)          # simple Lambert shading
    rgb = cmap(norm(vals[faces].mean(1)))[:, :3] * shade[:, None]
    coll = Poly3DCollection(tri, facecolors=np.clip(rgb, 0, 1),
                            edgecolor="none", linewidth=0)
    coll.set_clip_on(False)              # panels overlap after packing
    ax.add_collection3d(coll)
    ax.patch.set_alpha(0.0)              # transparent axes background
    ax.set_facecolor((1, 1, 1, 0))
    lo, hi = pts.min(0), pts.max(0)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def _projected_aspect(pts, elev, azim):
    """width/height of the LV as rendered with this camera."""
    f = plt.figure(figsize=(3, 3))
    ax = f.add_axes([0, 0, 1, 1], projection="3d")
    lo, hi = pts.min(0), pts.max(0)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    ax.view_init(elev=elev, azim=azim)
    f.canvas.draw()
    x0, y0, x1, y1 = _mesh_bbox_in_figure(f, ax, pts)
    plt.close(f)
    return (x1 - x0) / (y1 - y0)


def _autozoom(fig, axes, pts, fill=0.97):
    """Zoom each 3D axes so the LV fills it (no clipping, minimal margins)."""
    fig.canvas.draw()
    for ax in axes:
        x0, y0, x1, y1 = _mesh_bbox_in_figure(fig, ax, pts)
        pos = ax.get_position()
        z = fill * min(pos.width / (x1 - x0), pos.height / (y1 - y0))
        ax.set_box_aspect(np.ptp(pts, axis=0), zoom=z)
    fig.canvas.draw()


def _shift(ax, dx=0.0, dy=0.0):
    p = ax.get_position()
    ax.set_position([p.x0 + dx, p.y0 + dy, p.width, p.height])


def _pack(fig, axes, ax_t, pts, pad=0.01):
    """Slide the panels so the rendered LVs are 'pad' apart (figure fractions):
    rows together, columns together, truth right after the samples and
    vertically centred on them."""
    bb = lambda a_: _mesh_bbox_in_figure(fig, a_, pts)        # noqa: E731
    cols = [axes[i:i + 2] for i in range(0, len(axes), 2)]    # [top, bottom]
    # rows: move the top row down onto the bottom row
    gap_v = min(bb(c[0])[1] - bb(c[1])[3] for c in cols if len(c) == 2)
    for c in cols:
        _shift(c[0], dy=-(gap_v - pad))
    # columns: move each column (and everything right of it) to the left
    right = max(bb(a_)[2] for a_ in cols[0])
    for j in range(1, len(cols)):
        left = min(bb(a_)[0] for a_ in cols[j])
        dx = -(left - right - pad)
        for c in cols[j:]:
            for a_ in c:
                _shift(a_, dx=dx)
        if ax_t is not None:
            _shift(ax_t, dx=dx)
        right = max(bb(a_)[2] for a_ in cols[j])
    if ax_t is not None:
        x0, y0, x1, y1 = bb(ax_t)
        _shift(ax_t, dx=-(x0 - right - 2 * pad))
        ys = [v for a_ in axes for v in (bb(a_)[1], bb(a_)[3])]
        _shift(ax_t, dy=0.5 * (min(ys) + max(ys)) - 0.5 * (y0 + y1))
    fig.canvas.draw()


def _mesh_bbox_in_figure(fig, ax, pts):
    """Bounding box (x0, y0, x1, y1) of the rendered mesh in figure fractions."""
    from mpl_toolkits.mplot3d import proj3d
    x2, y2, _ = proj3d.proj_transform(pts[:, 0], pts[:, 1], pts[:, 2], ax.get_proj())
    disp = ax.transData.transform(np.c_[x2, y2])
    fr = fig.transFigure.inverted().transform(disp)
    return fr[:, 0].min(), fr[:, 1].min(), fr[:, 0].max(), fr[:, 1].max()


def figure_3d(pts, cells, z, std, m0, truth, is_log):
    """
    Bui-Thanh style figure, compact:  [+-2 sigma] [4 samples] [truth] [colorbar]
    deviations from the prior mean (parameter units), one diverging scale.
    The colorbar has the height of the rendered LV and is aligned with the
    ground-truth mesh.
    """
    from matplotlib.colors import Normalize
    faces = _boundary_faces(cells)
    samples = z["samples"][:4] - m0
    tdev = None
    if truth is not None:
        tdev = (np.log(truth) if is_log else truth) - m0
    fields = [2 * std, -2 * std] + list(samples)
    vmax = max(np.abs(f).max() for f in fields + ([tdev] if tdev is not None else []))
    norm = Normalize(-vmax, vmax)
    cmap = plt.get_cmap(CMAP_DEV)

    # camera: look at the largest truth deviation (or the mesh top)
    c = pts.mean(0)
    target = pts[np.argmax(np.abs(tdev))] if tdev is not None else pts[np.argmax(pts[:, 2])]
    vdir = target - c
    vdir /= np.linalg.norm(vdir) + 1e-30
    if VIEW is None:
        elev = float(np.degrees(np.arcsin(np.clip(vdir[2], -1, 1))))
        azim = float(np.degrees(np.arctan2(vdir[1], vdir[0])))
    else:
        elev, azim = VIEW
        er, ar = np.radians(elev), np.radians(azim)
        vdir = np.array([np.cos(er) * np.cos(ar), np.cos(er) * np.sin(ar), np.sin(er)])

    # ---- layout in inches, sized to the projected LV (no wasted space) ----
    asp = _projected_aspect(pts, elev, azim)            # width / height on screen
    W = FIG_WIDE[0]
    title_h, cbar_w = 0.22, 0.75                        # inches
    hs = (W - cbar_w) / (asp * (3 + 2 * TRUTH_SCALE))   # small-panel height
    ws, ht = hs * asp, 2 * hs * TRUTH_SCALE
    wt = ht * asp
    H = 2 * hs + title_h
    fig = plt.figure(figsize=(W, H))
    fx, fy = 1.0 / W, 1.0 / H                           # inch -> figure fraction
    axes = []
    for k, f in enumerate(fields):
        col, row = k // 2, k % 2
        rect = [col * ws * fx, (hs * (1 - row)) * fy, ws * fx, hs * fy]
        ax = fig.add_axes(rect, projection="3d")
        _surface(ax, pts, faces, f, norm, cmap, vdir, elev, azim)
        axes.append(ax)
    ax_t = None
    if tdev is not None:
        rect = [3 * ws * fx, (hs - ht / 2) * fy, wt * fx, ht * fy]
        ax_t = fig.add_axes(rect, projection="3d")
        _surface(ax_t, pts, faces, tdev, norm, cmap, vdir, elev, azim)
    _autozoom(fig, axes + ([ax_t] if ax_t is not None else []), pts)
    _pack(fig, axes, ax_t, pts, pad=PAD)

    # ---- titles and colorbar from the RENDERED mesh extents ----
    bb = [_mesh_bbox_in_figure(fig, a_, pts) for a_ in axes]
    ytitle = max(b[3] for b in bb) + 0.01
    fig.text(0.5 * (bb[0][0] + bb[0][2]), ytitle, r"$\pm 2\sigma$ fields",
             ha="center", va="bottom")
    fig.text(0.5 * (bb[2][0] + bb[4][2]), ytitle, "prior samples",
             ha="center", va="bottom")
    ref = ax_t if ax_t is not None else axes[-1]
    x0, y0, x1, y1 = _mesh_bbox_in_figure(fig, ref, pts)
    if ax_t is not None:
        fig.text(0.5 * (x0 + x1), ytitle, "ground truth", ha="center", va="bottom")
    cax = fig.add_axes([x1 + 0.015, y0, 0.012, y1 - y0])   # LV height, aligned
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    lab = (r"$\log C - m_0$ [$-$]" if is_log else r"$C - m_0$ [kPa]")
    fig.colorbar(sm, cax=cax, label=lab)
    return fig

if __name__ == "__main__":
    main()
