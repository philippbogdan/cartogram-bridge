"""Render the endpoint figures and world-population movie from included data.

Read supporting inputs from figure_data.py and world maps from data/hero/.
Write PNG and SVG supporting figures, an MP4 and PNG poster for the world maps,
and a JSON manifest with input hashes, fit checks and display choices. The clock export
marks errors below 2e-15, including zeros, with downward triangles; numerical
precision limits and the plotted encodings are described in docs/figures.md.
"""
import argparse
import hashlib
import io
import json
import platform
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, to_rgb
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter, NullLocator
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

DPI = 300
WIDTH_PX = 3320
DISPLAY_PX = 830
SCALE = WIDTH_PX / DISPLAY_PX
PT = SCALE * 72 / DPI

INK = "#222222"
MUTED = "#555555"
GRID = "#E9E9E9"
BACKGROUND = "#FFFFFF"

PLANE_CASES = [
    ("smooth", "Smooth", "#0072B2", "o"),
    ("mixed", "Mixed", "#E66100", "s"),
    ("large", "Large", "#5D3A9B", "^"),
]
SPHERE_FLOW = ("#00876C", "o", "-")
SPHERE_FRECHET = ("#3A3A3A", "D", (0, (4.0, 2.2)))
REFERENCE = "#8A8A8A"
REFERENCE_SCALE = 1.0
PLANE_REFERENCE_GAMMA = (1.5e-4, 1.5e-3)
SPHERE_REFERENCE_GAMMA = (3.0e-3, 1.2e-2)

CLOCK_CMAP_NAME = "viridis"
CLOCK_CMAP_RANGE = (0.0, 0.59)
CLOCK_TAU = 2e-15
CLOCK_YLIM = (1e-15, 10.0)

FLOOR_MARKER = 4.0 * PT

TRANSPORT_SIZE_PX = (3320, 1840)
CLOCKS_SIZE_PX = (3320, 2560)


def px(n):
    return n * PT


def rc():
    return {
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "font.size": px(12.5),
        "axes.labelsize": px(13.5),
        "axes.titlesize": px(14),
        "xtick.labelsize": px(12),
        "ytick.labelsize": px(12),
        "legend.fontsize": px(12.5),
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "axes.linewidth": px(0.9),
        "xtick.major.width": px(0.9),
        "ytick.major.width": px(0.9),
        "xtick.minor.width": px(0.7),
        "ytick.minor.width": px(0.7),
        "xtick.major.size": px(4.5),
        "ytick.major.size": px(4.5),
        "xtick.minor.size": px(2.5),
        "ytick.minor.size": px(2.5),
        "xtick.major.pad": px(3.5),
        "ytick.major.pad": px(3.5),
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": BACKGROUND,
        "axes.facecolor": BACKGROUND,
        "savefig.facecolor": BACKGROUND,
        "savefig.transparent": False,
        "svg.fonttype": "path",
        "svg.hashsalt": "render_figures",
        "path.simplify": False,
        "legend.frameon": False,
    }


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def mix_with_white(colour, weight):
    r, g, b = to_rgb(colour)
    return (weight * r + (1 - weight), weight * g + (1 - weight), weight * b + (1 - weight))


def axes_at(fig, left, bottom, width, height, size_px):
    w, h = size_px[0] / SCALE, size_px[1] / SCALE
    return fig.add_axes([left / w, bottom / h, width / w, height / h])


def decimal_label(value, _pos=None):
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text


def ols(x, y, columns):
    a = np.column_stack(columns)
    coef = np.linalg.lstsq(a, y, rcond=None)[0]
    return coef


def check_transport(data):
    report = {}
    for key, _, _, _ in PLANE_CASES:
        case = data["plane"][key]
        g = np.asarray(case["gamma"], float)
        e = np.asarray(case["flow_max"], float)
        r = np.asarray(case["refinement_change"], float)
        m = np.asarray(case["fit_mask"], bool)
        rule = (g <= 0.02) & (e > 100 * r)
        coef = ols(np.log(g[m]), np.log(e[m]), [np.ones(m.sum()), np.log(g[m])])
        fit = case["fit"]
        report[key] = {
            "n_observations": int(len(g)),
            "n_fit_mask": int(m.sum()),
            "fit_n": int(fit["n"]),
            "mask_equals_rule_gamma_le_0.02_and_error_gt_100x_refinement": bool(np.array_equal(rule, m)),
            "fit_gamma_range_equals_mask_extent": bool(np.isclose(g[m].min(), fit["gamma_range"][0]) and np.isclose(g[m].max(), fit["gamma_range"][1])),
            "refit_exponent": float(coef[1]),
            "stored_exponent": float(fit["exponent"]),
            "refit_prefactor": float(np.exp(coef[0])),
            "stored_prefactor": float(fit["prefactor"]),
            "excluded_gamma": g[~m].tolist(),
            "all_positive_finite": bool(np.all(np.isfinite(e)) and np.all(e > 0) and np.all(np.isfinite(r)) and np.all(r > 0)),
        }
    s = data["sphere"]
    g = np.asarray(s["gamma"], float)
    e = np.asarray(s["flow_max"], float)
    b = np.asarray(s["bary_max"], float)
    m = np.asarray(s["fit_mask"], bool)
    two = s["two_term_fit"]
    plain = s["plain_fit"]
    bary = s["barycentric_fit"]
    bm = (g >= bary["gamma_range"][0]) & (g <= bary["gamma_range"][1])
    c2 = ols(np.log(g[m]), np.log(e[m]), [np.ones(m.sum()), np.log(g[m]), g[m]])
    c1 = ols(np.log(g[m]), np.log(e[m]), [np.ones(m.sum()), np.log(g[m])])
    cb = ols(np.log(g[bm]), np.log(b[bm]), [np.ones(bm.sum()), np.log(g[bm])])
    report["sphere"] = {
        "n_flow": int(len(e)),
        "n_static_frechet": int(len(b)),
        "n_fit_mask": int(m.sum()),
        "fit_mask_gamma": g[m].tolist(),
        "two_term_gamma_range": two["gamma_range"],
        "refit_two_term": [float(np.exp(c2[0])), float(c2[1]), float(c2[2])],
        "stored_two_term": [two["prefactor"], two["exponent"], two["linear_correction"]],
        "refit_plain_exponent": float(c1[1]),
        "stored_plain_exponent": float(plain["exponent"]),
        "static_frechet_fit_mask_from_gamma_range": bm.tolist(),
        "static_frechet_fit_n": int(bary["n"]),
        "static_frechet_mask_count_equals_fit_n": bool(bm.sum() == bary["n"]),
        "refit_static_frechet_exponent": float(cb[1]),
        "stored_static_frechet_exponent": float(bary["exponent"]),
        "all_positive_finite": bool(np.all(np.isfinite(e)) and np.all(e > 0) and np.all(np.isfinite(b)) and np.all(b > 0)),
    }
    return report, bm


def render_transport(data, out):
    checks, frechet_mask = check_transport(data)
    w, h = TRANSPORT_SIZE_PX[0] / SCALE, TRANSPORT_SIZE_PX[1] / SCALE
    left, right, top, bottom, gap = 80.0, 14.0, 34.0, 54.0, 26.0
    plane_xlim = (7.5e-5, 0.4)
    sphere_xlim = (1.9e-3, 5.3e-2)
    ylim = (1e-15, 1e-1)
    dp = np.log10(plane_xlim[1] / plane_xlim[0])
    ds = np.log10(sphere_xlim[1] / sphere_xlim[0])
    total = w - left - right - gap
    wp = total * dp / (dp + ds)
    wsph = total - wp
    height = h - top - bottom

    with mpl.rc_context(rc()):
        fig = plt.figure(figsize=(TRANSPORT_SIZE_PX[0] / DPI, TRANSPORT_SIZE_PX[1] / DPI), dpi=DPI)
        axp = axes_at(fig, left, bottom, wp, height, TRANSPORT_SIZE_PX)
        axs = axes_at(fig, left + wp + gap, bottom, wsph, height, TRANSPORT_SIZE_PX)

        for ax, xlim in ((axp, plane_xlim), (axs, sphere_xlim)):
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.yaxis.set_major_locator(FixedLocator([10.0 ** k for k in range(-15, 0, 3)]))
            ax.yaxis.set_minor_locator(FixedLocator([10.0 ** k for k in range(-15, 0)]))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.grid(True, which="major", axis="both", color=GRID, linewidth=px(0.8), zorder=0)
            ax.set_axisbelow(True)
            ax.set_xlabel(r"$\gamma$")
        axp.xaxis.set_major_locator(FixedLocator([1e-4, 1e-3, 1e-2, 1e-1]))
        axp.xaxis.set_major_formatter(FuncFormatter(decimal_label))
        axp.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
        axp.xaxis.set_minor_formatter(NullFormatter())
        axs.xaxis.set_major_locator(FixedLocator([0.0025, 0.01, 0.04]))
        axs.xaxis.set_major_formatter(FuncFormatter(decimal_label))
        axs.xaxis.set_minor_locator(FixedLocator([0.005, 0.02]))
        axs.xaxis.set_minor_formatter(NullFormatter())
        axs.tick_params(axis="y", labelleft=False)
        axp.set_ylabel("Maximum map error / diameter")
        axp.set_title("Plane", loc="left", fontweight="bold", pad=px(8))
        axs.set_title("Sphere", loc="left", fontweight="bold", pad=px(8))

        line_w = px(1.3)
        marker = px(7.0)
        edge = px(1.2)
        halo_w = px(7.0)
        handles = []
        fits = {}
        for key, label, colour, mk in PLANE_CASES:
            case = data["plane"][key]
            g = np.asarray(case["gamma"], float)
            e = np.asarray(case["flow_max"], float)
            r = np.asarray(case["refinement_change"], float)
            m = np.asarray(case["fit_mask"], bool)
            order = np.argsort(g)
            g, e, r, m = g[order], e[order], r[order], m[order]
            fit = case["fit"]
            gf = np.geomspace(fit["gamma_range"][0], fit["gamma_range"][1], 200)
            ef = fit["prefactor"] * gf ** fit["exponent"]
            axp.plot(gf, ef, color=mix_with_white(colour, 0.28), linewidth=halo_w, solid_capstyle="butt", zorder=2)
            axp.plot(g, r, color=colour, linewidth=px(1.1), linestyle=(0, (1.0, 1.6)), dash_capstyle="round", zorder=3)
            axp.plot(g, e, color=colour, linewidth=line_w, zorder=4)
            axp.plot(g[m], e[m], linestyle="none", marker=mk, markersize=marker, markerfacecolor=colour, markeredgecolor=colour, markeredgewidth=edge, zorder=5)
            axp.plot(g[~m], e[~m], linestyle="none", marker=mk, markersize=marker, markerfacecolor=BACKGROUND, markeredgecolor=colour, markeredgewidth=edge, zorder=5)
            handles.append(mpl.lines.Line2D([], [], color=colour, linewidth=line_w, marker=mk, markersize=marker, markerfacecolor=colour, markeredgecolor=colour, label=label))
            fits[key] = {"model": "prefactor * gamma ** exponent", "gamma_range": fit["gamma_range"], "prefactor": fit["prefactor"], "exponent": fit["exponent"]}

        axp.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 0.24), handlelength=2.2, handletextpad=0.6, labelspacing=0.45, borderaxespad=0.2)
        axp.text(0.03, 3.2e-14, "Refinement change", color=MUTED, fontsize=px(12), ha="left", va="bottom", zorder=6)

        s = data["sphere"]
        g = np.asarray(s["gamma"], float)
        e = np.asarray(s["flow_max"], float)
        b = np.asarray(s["bary_max"], float)
        m = np.asarray(s["fit_mask"], bool)
        order = np.argsort(g)
        g, e, b, m, bm = g[order], e[order], b[order], m[order], frechet_mask[order]
        two = s["two_term_fit"]
        gf = np.geomspace(0.0025, 0.01, 200)
        ef = two["prefactor"] * gf ** two["exponent"] * np.exp(two["linear_correction"] * gf)
        fc, fmk, fls = SPHERE_FLOW
        bc, bmk, bls = SPHERE_FRECHET
        axs.plot(gf, ef, color=mix_with_white(fc, 0.28), linewidth=halo_w, solid_capstyle="butt", zorder=2)
        axs.plot(g, e, color=fc, linewidth=line_w, linestyle=fls, zorder=4)
        axs.plot(g[m], e[m], linestyle="none", marker=fmk, markersize=marker, markerfacecolor=fc, markeredgecolor=fc, markeredgewidth=edge, zorder=5)
        axs.plot(g[~m], e[~m], linestyle="none", marker=fmk, markersize=marker, markerfacecolor=BACKGROUND, markeredgecolor=fc, markeredgewidth=edge, zorder=5)
        axs.plot(g, b, color=bc, linewidth=line_w, linestyle=bls, zorder=4)
        axs.plot(g[bm], b[bm], linestyle="none", marker=bmk, markersize=marker * 0.9, markerfacecolor=bc, markeredgecolor=bc, markeredgewidth=edge, zorder=5)
        axs.plot(g[~bm], b[~bm], linestyle="none", marker=bmk, markersize=marker * 0.9, markerfacecolor=BACKGROUND, markeredgecolor=bc, markeredgewidth=edge, zorder=5)
        axs.text(2.05e-3, 5.0e-3, "Static Fréchet", color=bc, ha="left", va="bottom", zorder=6)
        axs.text(0.05, 4.0e-6, "Flow", color=fc, ha="right", va="top", zorder=6)
        fits["sphere_two_term"] = {"model": "prefactor * gamma ** exponent * exp(linear_correction * gamma)", "gamma_range": [0.0025, 0.01], "prefactor": two["prefactor"], "exponent": two["exponent"], "linear_correction": two["linear_correction"]}

        references = {}
        for ax, (g0, g1), name in ((axp, PLANE_REFERENCE_GAMMA, "plane"), (axs, SPHERE_REFERENCE_GAMMA, "sphere")):
            gr = np.array([g0, g1])
            ax.plot(gr, REFERENCE_SCALE * gr ** 2, color=REFERENCE, linewidth=px(1.4), solid_capstyle="butt", zorder=3)
            gm = np.sqrt(g0 * g1)
            ax.text(gm / 1.12, REFERENCE_SCALE * gm ** 2 * 1.9, r"$\gamma^2$", color=MUTED, ha="right", va="bottom", fontsize=px(13), zorder=6)
            references[name] = {"model": "scale * gamma ** 2", "scale": REFERENCE_SCALE, "gamma_range": [g0, g1]}

        fig.canvas.draw()
        geometry = {}
        for name, ax, xl in (("plane", axp, plane_xlim), ("sphere", axs, sphere_xlim)):
            bb = ax.get_window_extent()
            geometry[name] = {
                "axes_px_export": [round(bb.x0, 2), round(TRANSPORT_SIZE_PX[1] - bb.y1, 2), round(bb.width, 2), round(bb.height, 2)],
                "x_px_per_decade_display": round(bb.width / SCALE / np.log10(xl[1] / xl[0]), 3),
                "y_px_per_decade_display": round(bb.height / SCALE / np.log10(ylim[1] / ylim[0]), 3),
                "xlim": list(xl),
                "ylim": list(ylim),
            }
        files = save(fig, out, "transport-rate", TRANSPORT_SIZE_PX)
        plt.close(fig)

    record = {
        "inputs": ["transport.json"],
        "checks": checks,
        "encoding": {
            "plane": {k: {"label": lab, "colour": c, "marker": mk} for k, lab, c, mk in PLANE_CASES},
            "sphere_flow": {"colour": SPHERE_FLOW[0], "marker": SPHERE_FLOW[1], "linestyle": "solid"},
            "sphere_static_frechet": {"colour": SPHERE_FRECHET[0], "marker": SPHERE_FRECHET[1], "linestyle": "dashed"},
            "filled_marker": "fit_mask true",
            "open_marker": "fit_mask false",
            "static_frechet_filled": "gamma within barycentric_fit.gamma_range",
            "refinement_change": "dotted, matching colour",
            "fit_curve": "wide pale line, admitted gamma range",
            "sample_line": "thin line through all samples",
            "reference_colour": REFERENCE,
            "error_bars": None,
        },
        "fits_drawn": fits,
        "reference_segments": references,
        "geometry": geometry,
        "labels": ["Plane", "Sphere", "γ", "Maximum map error / diameter", "Smooth", "Mixed", "Large", "Flow", "Static Fréchet", "Refinement change", "γ²"],
        "outputs": files,
    }
    return record


def load_curves(path, gamma_json):
    z = np.load(path, allow_pickle=False)
    g = np.asarray(z["gamma"], float)
    n = len(g)
    expected = {"gamma"} | {f"{p}_{i}" for p in ("s", "linear", "canonical") for i in range(n)}
    keys = set(z.files)
    if keys != expected:
        raise ValueError(f"{path.name}: unexpected or missing curve arrays: {sorted(keys ^ expected)}")
    if not np.array_equal(g, np.asarray(gamma_json, float)):
        raise ValueError(f"{path.name}: noise levels differ from clocks.json.")
    if not np.all(np.diff(g) > 0):
        raise ValueError(f"{path.name}: noise levels must be strictly increasing.")
    curves = []
    for i in range(n):
        s = np.asarray(z[f"s_{i}"], float)
        t = 2 * s / g[i]
        lin = np.asarray(z[f"linear_{i}"], float)
        can = np.asarray(z[f"canonical_{i}"], float)
        if lin.shape != (len(s), 4) or can.shape != (len(s), 4):
            raise ValueError(f"{path.name}, level {i}: each clock array must have one row per time and four columns.")
        if not np.all(np.diff(t) > 0):
            raise ValueError(f"{path.name}, level {i}: bridge times must be strictly increasing.")
        curves.append({"gamma": g[i], "t": t, "natural": lin[:, 0], "eigenvalue": can[:, 0]})
    return g, curves


def clock_cmap():
    base = mpl.colormaps[CLOCK_CMAP_NAME]
    a, b = CLOCK_CMAP_RANGE
    return LinearSegmentedColormap.from_list(f"{CLOCK_CMAP_NAME}_{a:g}_{b:g}", base(np.linspace(a, b, 256)))


def render_clocks(data_dir, out):
    clocks = json.loads((data_dir / "clocks.json").read_text())
    domains = [("sphere", "Sphere", "sph_generic"), ("plane", "Plane", "pl_generic")]
    loaded = {}
    for name, _, key in domains:
        rec = clocks[key]
        if rec["lambda_domain"] != rec["lambda_clock"]:
            raise ValueError(f"{key}: the plotted clock must use the domain's first positive eigenvalue.")
        g, curves = load_curves(data_dir / f"{key}_curves.npz", rec["gamma"])
        loaded[name] = (rec["lambda_domain"], g, curves)

    lg_all = np.concatenate([lam * g for lam, g, _ in loaded.values()])
    vmin, vmax = float(lg_all.min()), float(lg_all.max())
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap = clock_cmap()

    tau = CLOCK_TAU
    ylim = CLOCK_YLIM
    stats = {}
    for name, (lam, g, curves) in loaded.items():
        for clock in ("natural", "eigenvalue"):
            per_level = []
            totals = dict.fromkeys(["samples", "retained", "below", "zero", "positive_below", "invalid", "negative", "nonfinite", "endpoints_below", "endpoints_invalid", "isolated_retained"], 0)
            for i, c in enumerate(curves):
                e = c[clock]
                finite = np.isfinite(e)
                valid = finite & (e >= 0)
                keep = valid & (e >= tau)
                below = valid & (e < tau)
                iso = keep & ~np.r_[False, keep[:-1]] & ~np.r_[keep[1:], False]
                row = {
                    "index": i,
                    "gamma": float(c["gamma"]),
                    "lambda1_gamma": float(lam * c["gamma"]),
                    "samples": int(e.size),
                    "retained": int(keep.sum()),
                    "below": int(below.sum()),
                    "zero": int(np.sum(valid & (e == 0))),
                    "positive_below": int(np.sum(valid & (e > 0) & (e < tau))),
                    "invalid": int(np.sum(~valid)),
                    "negative": int(np.sum(finite & (e < 0))),
                    "nonfinite": int(np.sum(~finite)),
                    "endpoints_below": int(below[-1]),
                    "endpoints_invalid": int(not valid[-1]),
                    "isolated_retained": int(iso.sum()),
                    "min_positive": float(e[valid & (e > 0)].min()) if np.any(valid & (e > 0)) else None,
                }
                if row["retained"] + row["below"] + row["invalid"] != row["samples"]:
                    raise ValueError(name, clock, i, row)
                for k in totals:
                    totals[k] += row[k]
                per_level.append(row)
            vals = np.concatenate([c[clock] for c in curves])
            pos = vals[np.isfinite(vals) & (vals > 0)]
            stats[f"{name}_{clock}"] = {
                **totals,
                "accounting_closes": totals["retained"] + totals["below"] + totals["invalid"] == totals["samples"],
                "curves": len(curves),
                "samples_per_curve": sorted({len(c["t"]) for c in curves}),
                "raw_min_positive": float(pos.min()),
                "raw_max": float(pos.max()),
                "terminal_circles_drawn": int(sum(1 for c in curves if np.isfinite(c[clock][-1]) and c[clock][-1] >= tau)),
                "lambda1": float(lam),
                "lambda1_gamma_range": [float(lam * g.min()), float(lam * g.max())],
                "per_level": per_level,
            }
            if not stats[f"{name}_{clock}"]["accounting_closes"]:
                raise ValueError(name, clock, totals)

    w, h = CLOCKS_SIZE_PX[0] / SCALE, CLOCKS_SIZE_PX[1] / SCALE
    left, top, bottom = 90.0, 34.0, 56.0
    col_gap, row_gap = 20.0, 18.0
    row_label_w, cbar_gap, cbar_w, cbar_right = 26.0, 22.0, 14.0, 52.0
    pw = (w - left - col_gap - row_label_w - cbar_gap - cbar_w - cbar_right) / 2
    ph = (h - top - bottom - row_gap) / 2

    with mpl.rc_context(rc()):
        fig = plt.figure(figsize=(CLOCKS_SIZE_PX[0] / DPI, CLOCKS_SIZE_PX[1] / DPI), dpi=DPI)
        axes = {}
        for r, (name, label, _) in enumerate(domains):
            y0 = bottom + (1 - r) * (ph + row_gap)
            for c, clock in enumerate(("natural", "eigenvalue")):
                x0 = left + c * (pw + col_gap)
                axes[(name, clock)] = axes_at(fig, x0, y0, pw, ph, CLOCKS_SIZE_PX)

        line_w = px(1.1)
        for (name, clock), ax in axes.items():
            lam, g, curves = loaded[name]
            ax.set_yscale("log")
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(*ylim)
            decades = range(int(round(np.log10(ylim[0]))), int(round(np.log10(ylim[1]))) + 1)
            ax.yaxis.set_major_locator(FixedLocator([10.0 ** k for k in decades if k % 5 == 0]))
            ax.yaxis.set_minor_locator(FixedLocator([10.0 ** k for k in decades]))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.xaxis.set_major_locator(FixedLocator([0, 0.2, 0.4, 0.6, 0.8, 1.0]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
            ax.grid(True, which="major", axis="both", color=GRID, linewidth=px(0.8), zorder=0)
            ax.set_axisbelow(True)
            ax.axhline(tau, color=REFERENCE, linewidth=px(0.9), linestyle=(0, (1.0, 1.8)), dash_capstyle="round", zorder=2)
            for cv in curves:
                e = cv[clock]
                valid = np.isfinite(e) & (e >= 0)
                keep = valid & (e >= tau)
                below = valid & (e < tau)
                y = np.where(keep, e, np.nan)
                colour = cmap(norm(lam * cv["gamma"]))
                ax.plot(cv["t"], y, color=colour, linewidth=line_w, solid_joinstyle="round", zorder=3)
                iso = keep & ~np.r_[False, keep[:-1]] & ~np.r_[keep[1:], False]
                if iso.any():
                    ax.plot(cv["t"][iso], y[iso], linestyle="none", marker="o", markersize=px(2.2), markerfacecolor=colour, markeredgewidth=0, zorder=3)
                if below.any():
                    ax.plot(cv["t"][below], np.full(int(below.sum()), tau), linestyle="none", marker="v", markersize=FLOOR_MARKER, markerfacecolor=colour, markeredgewidth=0, zorder=4, clip_on=False)
            for cv in curves:
                if np.isfinite(cv[clock][-1]) and cv[clock][-1] >= tau:
                    colour = cmap(norm(lam * cv["gamma"]))
                    ax.plot([cv["t"][-1]], [cv[clock][-1]], linestyle="none", marker="o", markersize=px(4.2), markerfacecolor=colour, markeredgewidth=0, zorder=4, clip_on=False)
            if clock == "eigenvalue":
                ax.tick_params(axis="y", labelleft=False)
            if name == "sphere":
                ax.tick_params(axis="x", labelbottom=False)

        floor_handle = mpl.lines.Line2D([], [], linestyle="none", marker="v", markersize=FLOOR_MARKER * 1.6, markerfacecolor=MUTED, markeredgewidth=0, label="Below display floor")
        axes[("sphere", "natural")].legend(handles=[floor_handle], loc="lower right", bbox_to_anchor=(1.0, 0.07), handletextpad=0.3, borderaxespad=0.3, handlelength=1.0)
        axes[("sphere", "natural")].set_title("Natural clock", fontweight="bold", pad=px(8))
        axes[("sphere", "eigenvalue")].set_title("Eigenvalue clock", fontweight="bold", pad=px(8))
        for name, label, _ in domains:
            ax = axes[(name, "eigenvalue")]
            ax.text(1.0 + (row_label_w * 0.55) / pw, 0.5, label, transform=ax.transAxes, rotation=270, ha="center", va="center", fontweight="bold", fontsize=px(14))

        grid_left = left / w
        grid_right = (left + 2 * pw + col_gap) / w
        fig.text((grid_left + grid_right) / 2, 12.0 / h, r"Bridge time $t$", ha="center", va="bottom", fontsize=px(13.5))
        fig.text(16.0 / w, (bottom + ph + row_gap / 2) / h, r"Relative $L^2$ velocity error", rotation=90, ha="left", va="center", fontsize=px(13.5))

        cax = axes_at(fig, left + 2 * pw + col_gap + row_label_w + cbar_gap, bottom + ph * 0.35, cbar_w, ph * 1.3 + row_gap, CLOCKS_SIZE_PX)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, cax=cax)
        cb.outline.set_linewidth(px(0.7))
        ticks = [t for t in (0.5, 1, 2, 5, 10, 20, 50) if vmin <= t <= vmax]
        cb.set_ticks(ticks)
        cb.set_ticklabels([f"{t:g}" for t in ticks])
        cb.ax.yaxis.set_minor_locator(NullLocator())
        cb.ax.tick_params(labelsize=px(12), width=px(0.8), length=px(3.5))
        cb.ax.set_title(r"$\lambda_1\gamma$", fontsize=px(13.5), pad=px(8), loc="left")

        fig.canvas.draw()
        geometry = {}
        for (name, clock), ax in axes.items():
            bb = ax.get_window_extent()
            geometry[f"{name}_{clock}"] = [round(bb.x0, 2), round(CLOCKS_SIZE_PX[1] - bb.y1, 2), round(bb.width, 2), round(bb.height, 2)]
        files = save(fig, out, "clocks", CLOCKS_SIZE_PX)
        plt.close(fig)

    record = {
        "inputs": ["clocks.json", "sph_generic_curves.npz", "pl_generic_curves.npz"],
        "error_column": "E_area (column 0)",
        "bridge_time": "t = 2 * s_i / gamma[i]",
        "time_samples": "all stored",
        "display_floor": {
            "tau": tau,
            "retained": "isfinite(E) and E >= tau, drawn at E",
            "below": "isfinite(E) and 0 <= E < tau, downward triangle at (t, tau), no line",
            "invalid": "nonfinite or E < 0, not drawn, counted",
            "line": "joins adjacent retained samples only",
            "isolated_retained": "dot",
            "terminal_circle": "endpoint retained",
            "terminal_below": "downward triangle at t = 1",
            "reference_line": "dotted at tau",
            "legend": "Below display floor",
        },
        "ylim": list(ylim),
        "colour": {"quantity": "lambda_domain * gamma", "norm": "LogNorm", "vmin": vmin, "vmax": vmax, "cmap": CLOCK_CMAP_NAME, "cmap_range": list(CLOCK_CMAP_RANGE)},
        "draw_order": "increasing lambda1 gamma",
        "stats": stats,
        "geometry_axes_px_export": geometry,
        "labels": ["Sphere", "Plane", "Natural clock", "Eigenvalue clock", "Bridge time t", "Relative L² velocity error", "λ₁γ", "Below display floor"],
        "outputs": files,
    }
    return record


def save(fig, out, stem, size_px):
    out.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor=BACKGROUND, transparent=False)
    buf.seek(0)
    img = Image.open(buf)
    img.load()
    if img.size != tuple(size_px):
        raise ValueError(f"{stem}: exported size {img.size} differs from requested size {size_px}.")
    if img.mode == "RGBA":
        alpha = np.asarray(img)[..., 3]
        if alpha.min() != 255:
            raise ValueError(f"{stem}: the exported figure must have an opaque background.")
        img = img.convert("RGB")
    png = out / f"{stem}.png"
    img.save(png, format="PNG", dpi=(DPI, DPI), optimize=True)
    svg = out / f"{stem}.svg"
    fig.savefig(svg, format="svg", dpi=DPI, facecolor=BACKGROUND, transparent=False, metadata={"Date": None, "Creator": None})
    files = {}
    for p in (png, svg):
        entry = {"bytes": p.stat().st_size, "sha256": sha256(p)}
        if p.suffix == ".png":
            with Image.open(p) as im:
                entry.update({"pixels": list(im.size), "mode": im.mode, "dpi": [round(v, 2) for v in im.info.get("dpi", (0, 0))]})
        else:
            entry.update({"inches": [size_px[0] / DPI, size_px[1] / DPI], "fonts": "glyph paths"})
        files[p.name] = entry
    return files


def versions():
    import PIL
    return {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": mpl.__version__, "pillow": PIL.__version__}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, help="Directory written by scripts/figure_data.py.")
    p.add_argument("--hero", type=Path, help="World-map directory containing levels.json and its NPZ files, such as data/hero/.")
    p.add_argument("--ffmpeg", default="ffmpeg", help="FFmpeg executable with libx264 support, used for the hero MP4 (default: ffmpeg on PATH).")
    p.add_argument("--out", type=Path, default=ROOT / "figures", help="Output directory for images, video and render-manifest.json (default: figures/). Selected exports are replaced; other figure entries are retained.")
    p.add_argument("--only", choices=["transport", "clocks", "hero"], action="append", help="Select an output; repeat for several. By default render both supporting figures, plus the hero when --hero is supplied.")
    a = p.parse_args()
    only = set(a.only or ["transport", "clocks"] + (["hero"] if a.hero else []))
    if {"transport", "clocks"} & only and a.data is None:
        p.error("--data is required for transport or clocks; first run scripts/figure_data.py, or select --only hero.")
    if "hero" in only and a.hero is None:
        p.error("--hero is required for the movie and poster; use --hero data/hero with the included maps.")
    manifest_path = a.out / "render-manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest["canvas"] = {"png_width_px": WIDTH_PX, "display_width_px": DISPLAY_PX, "dpi": DPI, "background": BACKGROUND, "font": "DejaVu Sans", "mathtext": "dejavusans"}
    manifest["packages"] = versions()
    manifest.setdefault("figures", {})
    hashes = {f.name: sha256(f) for f in sorted(a.data.iterdir()) if f.suffix in (".json", ".npz") and f.name != "SHA256SUMS.json"} if a.data else {}
    if "transport" in only:
        rec = render_transport(json.loads((a.data / "transport.json").read_text()), a.out)
        rec["input_sha256"] = {k: hashes[k] for k in rec["inputs"]}
        manifest["figures"]["transport-rate"] = rec
    if "clocks" in only:
        rec = render_clocks(a.data, a.out)
        rec["input_sha256"] = {k: hashes[k] for k in rec["inputs"]}
        manifest["figures"]["clocks"] = rec
    if "hero" in only:
        import world_hero
        manifest["figures"]["world-bridge"] = world_hero.render(a.hero, a.out, a.ffmpeg)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
