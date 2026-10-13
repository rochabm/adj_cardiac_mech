"""
plot_maps.py
------------
Visual comparison of the MAP estimates of the noise study.

Figure (2 rows x 4 columns):
    row = problem (Case 1 linear, Case 2 fibrosis)
    col = ground truth | MAP 0% | MAP 1% | MAP 3%
Each row uses ONE colour scale (the range of its ground truth), so the MAPs
are directly comparable with the truth. Same camera for all panels.
Optional second figure (--errors): |C_MAP - C_true| with one scale per row.

Reads the .h5 files written by ex06 (no dolfinx needed):
    <root>/<method>/noise_XXX/<case>/map_comparison_<case>.h5
Both methods give the same MAP, so Newton-CG is used by default.

Each heart is rendered as an image (PyVista off-screen if available, otherwise
matplotlib 3D); the figure itself (layout, Times fonts, colour bars) is built
with matplotlib, so text and colour bars are publication-style.

Usage:
    python plot_maps.py --root out_final
    python plot_maps.py --root out_final --clim-from-uq out_ex07   # same colours as plot_uq.py
    python plot_maps.py --root out_final --errors --out maps.pdf
    python plot_maps.py --root out_final --backend matplotlib
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from matplotlib.colors import LinearSegmentedColormap

# ParaView's default colour map "Fast" (K. Moreland; default since ParaView 5.11),
# control points from ParaView's ColorMaps.json preset
_FAST = [
    (0.0000, (0.0564, 0.0564, 0.4700)),
    (0.1716, (0.2430, 0.4604, 0.8100)),
    (0.2985, (0.3568, 0.7450, 0.9544)),
    (0.4321, (0.6882, 0.9300, 0.9179)),
    (0.5000, (0.8995, 0.9446, 0.7687)),
    (0.5882, (0.9571, 0.8338, 0.5089)),
    (0.7061, (0.9275, 0.6214, 0.3154)),
    (0.8476, (0.8000, 0.3520, 0.1600)),
    (1.0000, (0.5900, 0.0767, 0.1195)),
]

CASES = ["linear", "fibrosis"]
CASE_LABEL = {"linear": "Case 1 (linear)", "fibrosis": "Case 2 (fibrosis)"}
LEVELS = [("noise_000", "0%"), ("noise_001", "1%"), ("noise_003", "3%")]

# Times New Roman text; Times-like math (STIX). Falls back to the closest serif.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "Nimbus Roman No9 L",
                   "TeX Gyre Termes", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
})


def get_cmap(name):
    """'fast' (ParaView default) or any matplotlib colormap name."""
    if name.lower() in ("fast", "paraview"):
        return LinearSegmentedColormap.from_list("fast", _FAST, N=256)
    return matplotlib.colormaps[name]


# =============================================================================
# reading (dolfinx XDMF/HDF5 layout)
# =============================================================================

def _first_dataset(group):
    """First dataset inside an h5 group (dolfinx stores /Function/<name>/<time>)."""
    for key in group:
        obj = group[key]
        if isinstance(obj, h5py.Dataset):
            return obj[()]
        sub = _first_dataset(obj)
        if sub is not None:
            return sub
    return None


def read_case(h5file, fields):
    """Return (points, tets, {field: nodal values}) from an ex06 .h5 file."""
    with h5py.File(h5file, "r") as f:
        mesh_grp = f["Mesh"][next(iter(f["Mesh"]))]          # /Mesh/<mesh name>
        pts = mesh_grp["geometry"][()]
        tets = mesh_grp["topology"][()].astype(np.int64)
        vals = {}
        for name in fields:
            if name not in f["Function"]:
                raise KeyError(f"field '{name}' not in {h5file}; available: "
                               f"{list(f['Function'].keys())}")
            v = np.asarray(_first_dataset(f["Function"][name])).reshape(-1)
            if v.size != pts.shape[0]:
                raise ValueError(f"{name}: {v.size} values but {pts.shape[0]} nodes "
                                 "(expected a P1 field on the mesh nodes)")
            vals[name] = v
    return pts, tets, vals


def boundary_faces(tets):
    """Triangles that belong to exactly one tetrahedron (the outer surface)."""
    faces = np.vstack([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]],
                       tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]])
    key = np.sort(faces, axis=1)
    _, idx, counts = np.unique(key, axis=0, return_index=True, return_counts=True)
    return faces[idx[counts == 1]]


def collect(root, method):
    """{case: {"pts","tets","true", "maps": [(label, values), ...]}}"""
    data = {}
    for case in CASES:
        maps, true, mesh = [], None, None
        for folder, label in LEVELS:
            h5 = Path(root) / method / folder / case / f"map_comparison_{case}.h5"
            if not h5.exists():
                print(f"  missing: {h5}")
                continue
            pts, tets, v = read_case(h5, ["C_true", f"C_{method}"])
            mesh = mesh or (pts, tets)
            true = v["C_true"] if true is None else true
            maps.append((label, v[f"C_{method}"]))
        if maps:
            data[case] = dict(pts=mesh[0], tets=mesh[1], true=true, maps=maps)
    if not data:
        raise SystemExit(f"No results found under {root}/{method}/noise_*/<case>/")
    return data


# =============================================================================
# panel rendering: one image per heart (no text, no colour bar)
# =============================================================================

class PyVistaPanels:
    def __init__(self, pts, tets, view, zoom, size=(700, 800)):
        import pyvista as pv
        self.pv = pv
        cells = np.hstack([np.full((len(tets), 1), 4), tets]).ravel()
        self.surf = pv.UnstructuredGrid(cells, np.full(len(tets), pv.CellType.TETRA),
                                        pts).extract_surface(algorithm=None)
        self.ids = self.surf.point_data["vtkOriginalPointIds"]
        self.view, self.zoom, self.size = view, zoom, size

    def render(self, vals, cmap, clim):
        pl = self.pv.Plotter(off_screen=True, window_size=self.size)
        pl.set_background("white")
        g = self.surf.copy()
        g.point_data["C"] = vals[self.ids]
        pl.add_mesh(g, scalars="C", cmap=cmap, clim=clim, smooth_shading=True,
                    show_scalar_bar=False)
        pl.view_vector(self.view, viewup=(1, 0, 0))     # LV long axis = x (base up)
        pl.camera.zoom(self.zoom)
        img = pl.screenshot(return_img=True)
        pl.close()
        return img


class MatplotlibPanels:
    def __init__(self, pts, tets, view, zoom, size=(700, 800)):
        self.pts, self.faces = pts, boundary_faces(tets)
        self.view, self.zoom, self.size = np.asarray(view, float), zoom, size

    def render(self, vals, cmap, clim):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        dpi = 100
        fig = plt.figure(figsize=(self.size[0] / dpi, self.size[1] / dpi), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1], projection="3d")
        norm = colors.Normalize(*clim)
        fc = cmap(norm(vals[self.faces].mean(axis=1)))
        ax.add_collection3d(Poly3DCollection(self.pts[self.faces], facecolors=fc,
                                             edgecolors="none"))
        lo, hi = self.pts.min(0), self.pts.max(0)
        c, r = (lo + hi) / 2, (hi - lo).max() / 2 / self.zoom
        ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)
        v = self.view / np.linalg.norm(self.view)
        ax.view_init(elev=np.degrees(np.arcsin(v[2])), azim=np.degrees(np.arctan2(v[1], v[0])))
        ax.set_axis_off()
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return img


def crop_white(img, pad=6, thr=248):
    """Remove the white margin around a rendered heart."""
    mask = (img < thr).any(axis=2)
    if not mask.any():
        return img
    r, c = np.where(mask.any(1))[0], np.where(mask.any(0))[0]
    r0, r1 = max(r[0] - pad, 0), min(r[-1] + pad + 1, img.shape[0])
    c0, c1 = max(c[0] - pad, 0), min(c[-1] + pad + 1, img.shape[1])
    return img[r0:r1, c0:c1]


# =============================================================================
# figure (matplotlib layout, Times fonts, colour bar as in the prior figure)
# =============================================================================

def uq_clim(uq_root, case, n_samples=4):
    """Colour range of plot_uq.py's posterior-sample figure for this case:
    min/max over truth, MAP and posterior samples 1..n of uq_<case>.h5."""
    h5 = Path(uq_root) / case / f"uq_{case}.h5"
    if not h5.exists():
        print(f"  missing: {h5} -- using the ground-truth range for {case}")
        return None
    names = ["C_true", "C_map"] + [f"C_post_sample_{s}" for s in range(1, n_samples + 1)]
    _, _, v = read_case(h5, names)
    return (min(float(v[n].min()) for n in names), max(float(v[n].max()) for n in names))


def make_figure(data, out, kind, cmap, view, zoom, backend, dpi, clims=None):
    Panels = PyVistaPanels if backend == "pyvista" else MatplotlibPanels
    rows = []
    for case, d in data.items():
        P = Panels(d["pts"], d["tets"], view, zoom)
        if kind == "maps":
            clim = (clims or {}).get(case) or (float(d["true"].min()), float(d["true"].max()))
            print(f"  {case}: colour range [{clim[0]:.3f}, {clim[1]:.3f}] kPa")
            panels = [("Ground truth", d["true"])] + \
                     [(f"MAP, {lab} noise", v) for lab, v in d["maps"]]
            label = r"$C$ [kPa]"
        else:
            errs = [(f"Error, {lab} noise", np.abs(v - d["true"])) for lab, v in d["maps"]]
            clim = (0.0, max(float(e.max()) for _, e in errs))
            panels = errs
            label = r"$|C_{\mathrm{MAP}} - C_{\mathrm{true}}|$ [kPa]"
        imgs = [crop_white(P.render(vals, cmap, clim)) for _, vals in panels]
        rows.append(dict(case=case, titles=[t for t, _ in panels], imgs=imgs,
                         clim=clim, label=label))

    nrow, ncol = len(rows), max(len(r["imgs"]) for r in rows)
    aspect = np.median([im.shape[0] / im.shape[1] for r in rows for im in r["imgs"]])
    w_panel = 1.75                                   # inches per heart
    fig_w = w_panel * ncol + 1.1                     # + row label + colour bar
    fig_h = (w_panel * aspect + 0.30) * nrow
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(nrow, ncol + 1, width_ratios=[1] * ncol + [0.16],
                          wspace=0.06, hspace=0.10,
                          left=0.06, right=0.93, top=0.94, bottom=0.03)

    for i, r in enumerate(rows):
        for j, (img, title) in enumerate(zip(r["imgs"], r["titles"])):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(img, interpolation="lanczos")
            ax.set_axis_off()
            ax.set_title(title, fontsize=10, pad=2)
            if j == 0:
                ax.text(-0.06, 0.5, CASE_LABEL[r["case"]], transform=ax.transAxes,
                        rotation=90, ha="center", va="center", fontsize=11)
        # colour bar: thin, framed, ticks and label on the right, 70% of the row height
        slot = fig.add_subplot(gs[i, ncol])
        slot.set_axis_off()
        pos = slot.get_position()
        h = pos.height * 0.70
        cax = fig.add_axes([pos.x0 + 0.10 * pos.width, pos.y0 + (pos.height - h) / 2,
                            0.36 * pos.width, h])
        sm = cm.ScalarMappable(norm=colors.Normalize(*r["clim"]), cmap=cmap)
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label(r["label"], fontsize=10)
        cb.ax.tick_params(labelsize=9, direction="out", length=3, width=0.6,
                          right=True, left=False)
        cb.ax.minorticks_off()
        cb.outline.set_linewidth(0.8)

    fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Wrote {out}")


# =============================================================================
# main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="out_final")
    ap.add_argument("--method", default="newtoncg", choices=["newtoncg", "lbfgsb"])
    ap.add_argument("--out", default=None,
                    help="Output image, .png or .pdf (default <root>/maps.png)")
    ap.add_argument("--errors", action="store_true",
                    help="Also plot |C_MAP - C_true| (<out stem>_errors.<ext>)")
    ap.add_argument("--cmap", default="fast",
                    help="'fast' = ParaView default (default), or a matplotlib name")
    ap.add_argument("--view", type=float, nargs=3, default=[0.0, 0.0, -1.0],
                    help="Camera direction (default: from -z, facing the lesion)")
    ap.add_argument("--zoom", type=float, default=1.15)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--backend", choices=["auto", "pyvista", "matplotlib"], default="auto")
    ap.add_argument("--clim-from-uq", default=None, metavar="UQ_ROOT",
                    help="Use the colour range of plot_uq.py's sample figure (e.g. out_ex07)")
    ap.add_argument("--n-samples", type=int, default=4,
                    help="Samples shown in the plot_uq figure (for --clim-from-uq)")
    ap.add_argument("--clim-linear", type=float, nargs=2, default=None)
    ap.add_argument("--clim-fibrosis", type=float, nargs=2, default=None)
    a = ap.parse_args()

    data = collect(a.root, a.method)
    cmap = get_cmap(a.cmap)
    out = Path(a.out) if a.out else Path(a.root) / "maps.png"

    backend = a.backend
    if backend == "auto":
        try:
            import pyvista  # noqa: F401
            backend = "pyvista"
        except ImportError:
            backend = "matplotlib"
    print(f"Rendering with {backend}")
    clims = {}
    if a.clim_from_uq:
        clims = {c: uq_clim(a.clim_from_uq, c, a.n_samples) for c in data}
    if a.clim_linear:
        clims["linear"] = tuple(a.clim_linear)
    if a.clim_fibrosis:
        clims["fibrosis"] = tuple(a.clim_fibrosis)
    make_figure(data, out, "maps", cmap, a.view, a.zoom, backend, a.dpi, clims)
    if a.errors:
        make_figure(data, out.with_name(out.stem + "_errors" + out.suffix),
                    "errors", cmap, a.view, a.zoom, backend, a.dpi)


if __name__ == "__main__":
    main()
