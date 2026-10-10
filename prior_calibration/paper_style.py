"""
paper_style.py
--------------
One matplotlib style for all paper figures, built ON TOP of SciencePlots
(https://github.com/garrettj403/SciencePlots):

    1. plt.style.use(["science", ...])      SciencePlots base style
    2. overrides for the look of our example figure:
         - Times New Roman text + Times-compatible math (STIX, or newtx with LaTeX)
           (science + no-latex would give STIX text with Computer Modern math)
         - framed square legend (science has no legend frame)
         - MATLAB/pgfplots blue / orange colour cycle
         - vector-friendly PDF output (fonts embedded as text)

Usage (call once, BEFORE creating figures):

    from paper_style import apply_paper_style, save_figure, COLORS, FIG_SINGLE
    apply_paper_style()                      # matplotlib text, Times
    apply_paper_style(use_tex=True)          # LaTeX text (newtx = Times), slower
    apply_paper_style(extra_styles=["ieee"]) # any other SciencePlots style on top
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    ...
    save_figure(fig, "fig_name")             # fig_name.pdf + fig_name.png

If SciencePlots is not installed (pip install SciencePlots), the same look
is reproduced from plain rcParams.

Font fallback: if "Times New Roman" is not installed (common on Linux),
the next Times clone is used: Times, TeX Gyre Termes, Nimbus Roman,
Liberation Serif. On Arch: `pacman -S ttf-liberation tex-gyre-fonts`;
the real Times New Roman is in `ttf-ms-fonts` (AUR). After installing a
font, delete ~/.cache/matplotlib so matplotlib sees it.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

# colours of the example figure (MATLAB/pgfplots-like)
COLORS = {
    "blue":   "#0072BD",
    "orange": "#D95319",
    "yellow": "#EDB120",
    "purple": "#7E2F8E",
    "green":  "#77AC30",
    "gray":   "#7F7F7F",
}
COLOR_CYCLE = [COLORS[k] for k in ("blue", "orange", "yellow", "purple", "green")]

# figure sizes in inches (journal column widths; science default is 3.5 x 2.625)
FIG_SINGLE = (3.5, 2.625)   # one column
FIG_WIDE = (7.0, 2.625)     # full width, e.g. two panels side by side

TIMES_FAMILY = ["Times New Roman", "Times", "TeX Gyre Termes",
                "Nimbus Roman", "Nimbus Roman No9 L", "Liberation Serif",
                "STIXGeneral", "DejaVu Serif"]


def _first_available(families):
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for fam in families:
        if fam in installed:
            return fam
    return None


def _scienceplots_available():
    try:
        import scienceplots  # noqa: F401  (registers the styles)
        return True
    except ImportError:
        return False


def apply_paper_style(base_fontsize=10, use_tex=False, extra_styles=(),
                      legend_frame=True, minor_ticks=False):
    """
    base_fontsize : label size in pt (ticks/legend 1 pt smaller). Choose it
                    for the PRINTED width (8-10 pt for FIG_SINGLE).
    use_tex       : True -> LaTeX renders all text (Times via newtx, needs a
                    TeX installation with newtx); exact match with the paper.
    extra_styles  : more SciencePlots styles after "science", e.g. ["ieee"],
                    ["grid"], ["high-vis"]; our overrides are applied last.
    legend_frame  : framed square legend as in the example figure.
    minor_ticks   : SciencePlots shows minor ticks; the example has none.
    """
    # ---- 1. SciencePlots base ------------------------------------------
    if _scienceplots_available():
        styles = ["science", *extra_styles]
        if not use_tex:
            styles.append("no-latex")
        plt.style.use(styles)
    else:
        print("[paper_style] SciencePlots not installed "
              "(pip install SciencePlots); using plain rcParams")
        mpl.rcParams.update({
            "xtick.direction": "in", "ytick.direction": "in",
            "xtick.top": True, "ytick.right": True,
            "xtick.major.size": 3, "ytick.major.size": 3,
            "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
            "xtick.major.width": 0.5, "ytick.major.width": 0.5,
            "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
            "axes.linewidth": 0.5, "lines.linewidth": 1.0,
            "grid.linewidth": 0.5,
            "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
        })

    # ---- 2. our overrides ----------------------------------------------
    found = _first_available(TIMES_FAMILY)
    if not use_tex and found != "Times New Roman":
        print(f"[paper_style] 'Times New Roman' not installed; using '{found}'")

    rc = {
        "font.family": "serif",
        "font.serif": TIMES_FAMILY,
        "mathtext.fontset": "stix",          # Times-like math (not CM)
        "mathtext.rm": "serif",
        "font.size": base_fontsize,
        "axes.labelsize": base_fontsize,
        "axes.titlesize": base_fontsize,
        "xtick.labelsize": base_fontsize - 1,
        "ytick.labelsize": base_fontsize - 1,
        "legend.fontsize": base_fontsize - 1,
        "axes.prop_cycle": mpl.cycler(color=COLOR_CYCLE),
        "axes.formatter.use_mathtext": True,
        "figure.figsize": FIG_SINGLE,
        "xtick.minor.visible": minor_ticks,
        "ytick.minor.visible": minor_ticks,
        "lines.markersize": 4,
        "pdf.fonttype": 42,                  # editable text in PDF
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 300,
    }
    if legend_frame:
        rc.update({
            "legend.frameon": True,
            "legend.fancybox": False,
            "legend.edgecolor": "black",
            "legend.framealpha": 1.0,
            "legend.borderpad": 0.4,
            "patch.linewidth": 0.5,
        })
    if use_tex:
        rc.update({
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{newtxtext,newtxmath}",
        })
    mpl.rcParams.update(rc)


def save_figure(fig, stem, formats=("pdf", "png")):
    """Save fig as <stem>.pdf (vector, for LaTeX) and <stem>.png."""
    for ext in formats:
        fig.savefig(f"{stem}.{ext}")
    print("Saved: " + ", ".join(f"{stem}.{e}" for e in formats))
