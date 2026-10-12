"""
make_latex_tables.py
--------------------
Builds LaTeX tables from the ex06 noise study:

    <root>/noise_000/{linear,fibrosis}/metrics_<case>.json
    <root>/noise_001/...
    <root>/noise_003/...

Tables (booktabs + multirow):
    1. Data and noise:   RMS(Fd-I), sigma_data, sigma_lik, Phi and J at the truth
    2. Cost:             iterations, forward solves, Hessian-vector products,
                         wall time, final J, projected gradient, stop reason
    3. Accuracy:         errors in C (global, core, healthy), displacement,
                         deformation gradient, discrepancy, active bounds
    4. Agreement:        L-BFGS-B vs Newton-CG MAP estimates

Usage:
    python make_latex_tables.py --root out_noise_study --out out_noise_study/tables.tex \
        [--standalone out_noise_study/tables_standalone.tex]

Preamble needed:  \\usepackage{booktabs, multirow}
"""

import argparse
import json
from pathlib import Path

CASE_ORDER = ["linear", "fibrosis"]
CASE_LABEL = {"linear": "Case 1 (linear)", "fibrosis": "Case 2 (fibrosis)"}
METHOD_ORDER = ["lbfgsb", "newtoncg"]
METHOD_LABEL = {"lbfgsb": "L-BFGS-B", "newtoncg": "Newton-CG"}
DASH = "--"


# =============================================================================
# formatting helpers
# =============================================================================

def sci(x, d=2):
    """Number in LaTeX math: plain if 1e-2 <= |x| < 1e4, else a x 10^b."""
    if x is None:
        return DASH
    x = float(x)
    if x == 0.0:
        return "$0$"
    if 1e-2 <= abs(x) < 1e4:
        return f"${x:.{d}f}$" if abs(x) < 100 else f"${x:.{max(d-1,0)}f}$"
    mant, exp = f"{x:.{d}e}".split("e")
    return f"${mant}\\times10^{{{int(exp)}}}$"


def pct(x, d=2):
    return DASH if x is None else f"${100.0*float(x):.{d}f}$"


def integer(x):
    return DASH if x is None else f"${int(round(float(x)))}$"


def noise_label(level):
    return f"${100.0*float(level):g}\\%$"


def stop_label(method, reason):
    r = str(reason).upper().replace("_", " ")
    if method == "lbfgsb":
        if "PROJECTED GRADIENT" in r or "PGTOL" in r:
            return "gtol"
        if "REL_REDUCTION" in r or "REDUCTION" in r:
            return "ftol"
        if "ITERATIONS" in r or "LIMIT" in r:
            return "max it."
        if "ABNORMAL" in r:
            return "ls fail"
        return "other"
    return {"GTOL": "gtol", "FTOL": "ftol", "MAX OUTER ITER": "max it.",
            "LINE SEARCH STALL": "ls stall",
            "LINE SEARCH FAILED": "ls fail"}.get(r, str(reason).replace("_", r"\_"))


# =============================================================================
# load results
# =============================================================================

def load(root):
    """
    Collect results. Two layouts are supported (and may be mixed):
      <root>/noise_XXX/<case>/metrics_<case>.json             (both methods in one run)
      <root>/<method>/noise_XXX/<case>/metrics_<case>.json    (one method per run)
    Records of the same (noise folder, case) are merged; if both methods are
    present, their agreement is computed from the saved m_map_<method>_<case>.npy.
    """
    import numpy as np
    root = Path(root)
    merged = {}          # (noise folder name, case) -> record
    mpaths = {}          # (noise folder name, case) -> {method: npy path}
    files = list(root.glob("noise_*/*/metrics_*.json")) + \
            list(root.glob("*/noise_*/*/metrics_*.json"))
    for f in sorted(files, key=lambda p: p.stat().st_mtime):   # newest wins
        case = f.parent.name
        if case not in CASE_ORDER:
            continue
        key = (f.parent.parent.name, case)
        with open(f) as fh:
            r = json.load(fh)
        rec = merged.setdefault(key, r)
        for m in METHOD_ORDER:
            if m in r:
                rec[m] = r[m]
                npy = f.parent / f"m_map_{m}_{case}.npy"
                if npy.exists():
                    mpaths.setdefault(key, {})[m] = npy
        if "lbfgsb_vs_newtoncg" in r:
            rec["lbfgsb_vs_newtoncg"] = r["lbfgsb_vs_newtoncg"]

    for key, rec in merged.items():
        p = mpaths.get(key, {})
        if "lbfgsb_vs_newtoncg" not in rec and "lbfgsb" in p and "newtoncg" in p:
            ma, mb = np.load(p["lbfgsb"]), np.load(p["newtoncg"])
            rec["lbfgsb_vs_newtoncg"] = {
                "m_max_abs_diff": float(np.abs(ma - mb).max()),
                "C_max_abs_diff": float(np.abs(np.exp(ma) - np.exp(mb)).max())}

    runs = list(merged.values())
    if not runs:
        raise SystemExit(f"No metrics_*.json found under {root}/[<method>/]noise_*/<case>/")
    runs.sort(key=lambda r: (CASE_ORDER.index(r["case"]), noise(r)["level"]))
    return runs


def noise(r):
    """Noise block of the current ex06 json ('data_noise'), normalized."""
    n = r.get("data_noise", {})
    n_obs = 9 * int(r.get("settings", {}).get("n_sensors", 64))
    return dict(level=float(n.get("level", 0.0)),
                F_scale_rms=n.get("F_scale_rms"),
                sigma_data=n.get("sigma_data"),
                sigma_lik=n.get("noise_std_likelihood"),
                n_obs=n_obs)


def resid_over_sigma(r, x):
    """RMS(F_model - F_obs)/sigma_lik = sqrt(2 Phi / N_obs) (exact for the pointwise likelihood)."""
    if x.get("Phi") is None:
        return None
    return (2.0 * float(x["Phi"]) / noise(r)["n_obs"]) ** 0.5


def by_case(runs):
    out = {}
    for r in runs:
        out.setdefault(r["case"], []).append(r)
    return [(c, out[c]) for c in CASE_ORDER if c in out]


# =============================================================================
# generic grouped table: case (multirow) > noise (multirow) > method rows
# =============================================================================

def table(caption, label, colspec, header, groups, note=None):
    """
    groups: [(case_label, [(noise_label, [row_cells, ...]), ...]), ...]
    row_cells exclude the case and noise columns.
    """
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\setlength{\tabcolsep}{4pt}",
         f"\\caption{{{caption}}}", f"\\label{{{label}}}",
         f"\\begin{{tabular}}{{{colspec}}}", r"\toprule", header + r" \\", r"\midrule"]
    for gi, (case_lab, noise_groups) in enumerate(groups):
        n_case = sum(len(rows) for _, rows in noise_groups)
        first_case = True
        for ni, (nlab, rows) in enumerate(noise_groups):
            for ri, cells in enumerate(rows):
                c0 = f"\\multirow{{{n_case}}}{{*}}{{{case_lab}}}" if first_case else ""
                c1 = (f"\\multirow{{{len(rows)}}}{{*}}{{{nlab}}}"
                      if ri == 0 and len(rows) > 1 else (nlab if ri == 0 else ""))
                L.append(" & ".join([c0, c1] + cells) + r" \\")
                first_case = False
            if ni < len(noise_groups) - 1:
                ncols = 2 + len(noise_groups[0][1][0])
                L.append(f"\\cmidrule(l){{2-{ncols}}}")
        if gi < len(groups) - 1:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}"]
    if note:
        L += [r"\par\smallskip", r"\begin{minipage}{0.95\linewidth}\footnotesize",
              note, r"\end{minipage}"]
    L.append(r"\end{table}")
    return "\n".join(L)


# =============================================================================
# the four tables
# =============================================================================

def table_data(runs):
    groups = []
    for case, rs in by_case(runs):
        ng = []
        for r in rs:
            n = noise(r)
            ng.append((noise_label(n["level"]), [[
                sci(n["F_scale_rms"]), sci(n["sigma_data"]), sci(n["sigma_lik"]),
                integer(n["n_obs"]),
                sci(r["true"]["Phi"]), sci(r["true"]["J"]), sci(r["init"]["J"]),
            ]]))
        groups.append((CASE_LABEL[case], ng))
    header = (r"Case & Noise & RMS$(\mathbf{F}_d-\mathbf{I})$ & $\sigma_{\mathrm{data}}$ "
              r"& $\sigma_{\mathrm{lik}}$ & $N_{\mathrm{obs}}$ & $\Phi(m_{\mathrm{true}})$ "
              r"& $\mathcal{J}(m_{\mathrm{true}})$ & $\mathcal{J}(m_{\mathrm{init}})$")
    note = (r"Noise is added to each component of $\mathbf{F}$ at the sensors as "
            r"$\varepsilon\sim\mathcal{N}(0,\sigma_{\mathrm{data}}^2)$, "
            r"$\sigma_{\mathrm{data}} = p\,\mathrm{RMS}(\mathbf{F}_d-\mathbf{I})$. "
            r"For $p>0$ the likelihood uses $\sigma_{\mathrm{lik}}=\sigma_{\mathrm{data}}$; "
            r"for noise-free data ($p=0$) a fixed $\sigma_{\mathrm{lik}}$ is assumed. "
            r"With matched noise, $\Phi(m_{\mathrm{true}})\approx N_{\mathrm{obs}}/2$.")
    return table("Synthetic data, noise levels and reference values of the "
                 "negative log-posterior.", "tab:map_data",
                 "llccccccc", header, groups, note)


def table_cost(runs):
    groups = []
    for case, rs in by_case(runs):
        ng = []
        for r in rs:
            rows = []
            for m in METHOD_ORDER:
                if m not in r:
                    continue
                x = r[m]
                rows.append([METHOD_LABEL[m], integer(x["n_iter"]), integer(x["n_forward"]),
                             integer(x["n_hessvec"]) if m == "newtoncg" else DASH,
                             f"${x['wall_time_s']:.0f}$", sci(x["J"], 4),
                             sci(x["proj_grad_inf"]), stop_label(m, x["stop_reason"])])
            ng.append((noise_label(noise(r)["level"]), rows))
        groups.append((CASE_LABEL[case], ng))
    header = (r"Case & Noise & Method & Iter. & Fwd. & $H\hat{m}$ & Time [s] "
              r"& $\mathcal{J}(m_{\mathrm{MAP}})$ & $\|P(m-g)-m\|_\infty$ & Stop")
    note = (r"Fwd.: nonlinear forward solves (one adjoint each). "
            r"$H\hat{m}$: Hessian-vector products (two linearized solves each). "
            r"Both methods start from the same initial guess and use the same "
            r"bounds and tolerances.")
    return table("Computational cost of the MAP estimate: L-BFGS-B vs.\\ inexact Newton-CG.",
                 "tab:map_cost", "lllrrrrccc", header, groups, note)


def table_accuracy(runs):
    groups = []
    for case, rs in by_case(runs):
        ng = []
        for r in rs:
            rows = []
            for m in METHOD_ORDER:
                if m not in r:
                    continue
                x = r[m]
                core = x.get("C_mean_rel_core") if case == "fibrosis" else None
                healthy = x.get("C_mean_rel_healthy") if case == "fibrosis" else None
                bnd = f"${x.get('n_at_lower_bound', 0)}/{x.get('n_at_upper_bound', 0)}$"
                rows.append([METHOD_LABEL[m], pct(x["C_rel_L2"]), pct(x["C_max_rel"]),
                             pct(core), pct(healthy), pct(x["u_rel_L2"], 3),
                             pct(x.get("F_rel_frob"), 3), sci(resid_over_sigma(r, x)), bnd])
            ng.append((noise_label(noise(r)["level"]), rows))
        groups.append((CASE_LABEL[case], ng))
    header = (r"Case & Noise & Method & $e_C^{L^2}$ [\%] & $e_C^{\max}$ [\%] "
              r"& core [\%] & healthy [\%] & $e_u^{L^2}$ [\%] & $e_F$ [\%] "
              r"& $r/\sigma_{\mathrm{lik}}$ & bnd.")
    note = (r"$e_C^{L^2}=\|C_{\mathrm{MAP}}-C_{\mathrm{true}}\|_{L^2}/\|C_{\mathrm{true}}\|_{L^2}$; "
            r"$e_C^{\max}$: maximum nodal relative error; core/healthy: mean nodal relative "
            r"error for $r\le5$~mm / $r\ge10$~mm from the fibrosis centre. "
            r"$e_u^{L^2}$: displacement error. $e_F$: Frobenius error of $\mathbf{F}$ at the sensors "
            r"w.r.t.\ the noise-free data, relative to $\|\mathbf{F}_d\|$. "
            r"$r/\sigma_{\mathrm{lik}}$: RMS residual w.r.t.\ the noisy data over "
            r"$\sigma_{\mathrm{lik}}$ ($\approx1$: fit at the noise level). "
            r"bnd.: nodes at the lower/upper bound.")
    return table("Accuracy of the MAP estimates with respect to the ground truth.",
                 "tab:map_accuracy", "lllcccccccc", header, groups, note)


def table_agreement(runs):
    groups = []
    for case, rs in by_case(runs):
        ng = []
        for r in rs:
            if "lbfgsb" not in r or "newtoncg" not in r:
                continue
            a, b = r["lbfgsb"], r["newtoncg"]
            d = r.get("lbfgsb_vs_newtoncg", {})
            ng.append((noise_label(noise(r)["level"]), [[
                sci(d.get("C_max_abs_diff")), sci(d.get("m_max_abs_diff")),
                sci(abs(a["J"] - b["J"]) / max(abs(b["J"]), 1e-300)),
                sci(a["n_forward"] / max(b["n_forward"], 1)),
                sci(a["wall_time_s"] / max(b["wall_time_s"], 1e-12)),
            ]]))
        if ng:
            groups.append((CASE_LABEL[case], ng))
    header = (r"Case & Noise & $\max|\Delta C|$ [kPa] & $\max|\Delta m|$ "
              r"& $|\Delta\mathcal{J}|/\mathcal{J}$ & Fwd.\ ratio & Time ratio")
    note = (r"$\Delta$: L-BFGS-B minus Newton-CG MAP. Ratios are L-BFGS-B / Newton-CG.")
    return table("Agreement between the L-BFGS-B and Newton-CG MAP estimates.",
                 "tab:map_agreement", "llccccc", header, groups, note)


# =============================================================================
# main
# =============================================================================

def table_paper(runs):
    """
    Single summary table: problems x noise x methods.
    Cost per method row; accuracy once per (case, noise) since both methods
    converge to the same MAP (max |dm| reported in the note).
    """
    ncol = 13
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\setlength{\tabcolsep}{3pt}",
         r"\caption{MAP estimation with L-BFGS-B and inexact Newton-CG for both test "
         r"problems and three noise levels: computational cost and accuracy.}",
         r"\label{tab:map_summary}",
         r"\begin{tabular}{lllrrrrlccccc}", r"\toprule",
         r" & & & \multicolumn{5}{c}{Computational cost} & \multicolumn{4}{c}{Accuracy of $m_{\mathrm{MAP}}$} & \\",
         r"\cmidrule(lr){4-8}\cmidrule(lr){9-12}",
         r"Problem & Noise & Method & Iter. & Fwd. & $H\hat{m}$ & Time [s] & Stop "
         r"& $e_C^{L^2}$ [\%] & $e_C^{\max}$ [\%] & $e_C^{\mathrm{core}}$ [\%] "
         r"& $r/\sigma_{\mathrm{lik}}$ & Speed-up \\",
         r"\midrule"]
    dm_max = 0.0
    groups = by_case(runs)
    for gi, (case, rs) in enumerate(groups):
        n_case_rows = sum(len([m for m in METHOD_ORDER if m in r]) for r in rs)
        first_case = True
        for ri, r in enumerate(rs):
            meths = [m for m in METHOD_ORDER if m in r]
            k = len(meths)
            ref = r.get("newtoncg") or r.get(meths[0])
            ag = r.get("lbfgsb_vs_newtoncg", {})
            if ag.get("m_max_abs_diff") is not None:
                dm_max = max(dm_max, ag["m_max_abs_diff"])
            if k == 2:
                a, b = r["lbfgsb"], r["newtoncg"]
                speed = (f"${a['n_forward']/max(b['n_forward'],1):.1f}\\,/\\,"
                         f"{a['wall_time_s']/max(b['wall_time_s'],1e-12):.1f}$")
            else:
                speed = DASH
            core = ref.get("C_mean_rel_core") if case == "fibrosis" else None
            acc = [pct(ref.get("C_rel_L2")), pct(ref.get("C_max_rel")), pct(core),
                   f"${resid_over_sigma(r, ref):.2f}$" if ref.get("Phi") is not None else DASH,
                   speed]
            for mi, m in enumerate(meths):
                x = r[m]
                c0 = (f"\\multirow{{{n_case_rows}}}{{*}}{{{CASE_LABEL[case]}}}"
                      if first_case else "")
                first_case = False
                c1 = ((f"\\multirow{{{k}}}{{*}}{{{noise_label(noise(r)['level'])}}}"
                       if k > 1 else noise_label(noise(r)["level"])) if mi == 0 else "")
                cost = [METHOD_LABEL[m], integer(x.get("n_iter")), integer(x.get("n_forward")),
                        integer(x.get("n_hessvec")) if m == "newtoncg" else DASH,
                        f"${x['wall_time_s']:.0f}$" if x.get("wall_time_s") is not None else DASH,
                        stop_label(m, x.get("stop_reason", ""))]
                if mi == 0:
                    tail = [(f"\\multirow{{{k}}}{{*}}{{{v}}}" if k > 1 else v) for v in acc]
                else:
                    tail = [""] * len(acc)
                L.append(" & ".join([c0, c1] + cost + tail) + r" \\")
            if ri < len(rs) - 1:
                L.append(f"\\cmidrule(l){{2-{ncol}}}")
        if gi < len(groups) - 1:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
          r"\begin{minipage}{0.97\linewidth}\footnotesize",
          r"Noise: Gaussian, std $p\,\mathrm{RMS}(\mathbf{F}_d-\mathbf{I})$ added to each "
          r"component of $\mathbf{F}$ at the 64 sensors; for $p>0$ the likelihood uses the "
          r"same std, for $p=0$ an assumed $\sigma_{\mathrm{lik}}=10^{-3}$. "
          r"Fwd.: nonlinear forward solves (one adjoint each); $H\hat{m}$: Hessian-vector "
          r"products. Both methods start from the same initial guess with the same bounds "
          r"and tolerances, and converge to the same MAP point "
          + (f"(max.\\ nodal $|\\Delta m| = {sci(dm_max)[1:-1]}$)" if dm_max > 0 else "")
          + r"; the accuracy columns therefore apply to both. "
          r"$e_C^{L^2}=\|C_{\mathrm{MAP}}-C_{\mathrm{true}}\|_{L^2}/\|C_{\mathrm{true}}\|_{L^2}$; "
          r"$e_C^{\max}$: maximum nodal relative error; $e_C^{\mathrm{core}}$: mean nodal "
          r"relative error in the fibrotic core ($r\le5$~mm). "
          r"$r/\sigma_{\mathrm{lik}}$: RMS data residual over $\sigma_{\mathrm{lik}}$. "
          r"Speed-up: L-BFGS-B / Newton-CG in forward solves / wall time.",
          r"\end{minipage}", r"\end{table}"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="out_noise_study")
    ap.add_argument("--out", default=None, help="Output .tex (default <root>/tables.tex)")
    ap.add_argument("--paper", action="store_true",
                    help="Write only the single summary table (problems x noise x methods)")
    ap.add_argument("--standalone", default=None,
                    help="Also write a compilable document with the tables")
    a = ap.parse_args()

    runs = load(a.root)
    tables = ([table_paper(runs)] if a.paper else
              [table_data(runs), table_cost(runs), table_accuracy(runs), table_agreement(runs)])
    body = "\n\n".join(
        ["% Generated by make_latex_tables.py -- requires \\usepackage{booktabs, multirow}"]
        + tables) + "\n"

    out = Path(a.out) if a.out else Path(a.root) / "tables.tex"
    out.write_text(body)
    print(f"Wrote {out}  ({len(runs)} case/noise runs)")

    if a.standalone:
        doc = ("\\documentclass[10pt]{article}\n"
               "\\usepackage[margin=1.5cm,landscape]{geometry}\n"
               "\\usepackage{booktabs, multirow, amsmath, amssymb}\n"
               "\\begin{document}\n" + body + "\\end{document}\n")
        Path(a.standalone).write_text(doc)
        print(f"Wrote {a.standalone}")


if __name__ == "__main__":
    main()
