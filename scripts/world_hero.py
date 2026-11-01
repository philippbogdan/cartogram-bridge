"""Pack and render the coarse world-population map family.

The command packs world_family.py outputs: invariant arrays remain lossless in
common.npz, mapped coordinates become float32, and levels.json retains solver
diagnostics, source hashes and maximum rounding errors. Rendering colours each
pixel from the nearest mapped cell centre, using original population counts and
land categories from all centres. Coastlines and graticules retain their mapped
coordinates and polyline breaks. The movie and poster share a fixed colour scale;
the nearest-centre regions are display samples, not transported cell polygons.
See docs/world-map.md for the frame schedule and the illustration's scope.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, to_rgb
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullLocator
from PIL import Image
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]

STATIC = ("cell_input", "population", "land_fraction", "mass", "coast_input", "graticule_input")
TEXT = ("licence", "attribution", "changes")
PAIRS = (("cell_input", "cell_mapped"), ("coast_input", "coast_mapped"), ("graticule_input", "graticule_mapped"))
MAPPED = tuple(m for _, m in PAIRS)

FPS = 24
HERO_CRF = 21
VIDEO_PX = (1920, 960)
POSTER_PX = (3320, 1660)
SCHEDULE = {"geographic": 24, "first_hold": 48, "slow": 12, "fast": 3, "slow_max_gamma": 0.3, "join_hold": 12, "last_hold": 24, "diffusion_hold": 72}

BG = "#0B0C0E"
OCEAN = BG
ZERO_LAND = "#33363B"
UNDER = "#4E4272"
COAST = "#E4E7EB"
KEYLINE = BG
GRATICULE = (0.478, 0.498, 0.529, 0.5)
TEXT_COLOUR = "#D9DCE0"
SWATCH_EDGE = "#8A8F96"
CMAP_NAME = "inferno"
CMAP_LUT = 256
CONTRAST_MIN = 3.0
VMIN = 1.0
TICKS = (1.0, 1e2, 1e4, 1e6)
NEAR_TIE_RTOL = 1e-12
CHECK_PIXELS = 2000
CHECK_SEED = 20261101
LAYOUT = {"map": (96, 12, 1728, 864), "label_xy": (96, 936), "label_px": 36, "cbar": (1404, 893, 420, 13), "cbar_extendfrac": 0.15, "under_label_gap": 10, "cbar_tick_px": 26, "cbar_label_px": 27, "cbar_label_gap": 22, "swatch_px": 22, "swatch_label_px": 26, "swatches": ((560, 900), (740, 900)), "coast_px": 1.8, "keyline_px": 1.4, "graticule_px": 1.0}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_family(path):
    path = Path(path)
    meta = json.loads((path / "levels.json").read_text())
    records = meta["levels"]
    indices = [r["index"] for r in records]
    if sorted(indices) != list(range(len(records))):
        raise ValueError(f"Level indices must be unique and consecutive from zero; found {indices}.")
    files = [r["file"] for r in records]
    if len(set(files)) != len(files):
        raise ValueError(f"Each level must reference a different NPZ file; found {files}.")
    nx, ny = meta["grid"]
    width, height = meta["rectangle"][0][1] - meta["rectangle"][0][0], meta["rectangle"][1][1] - meta["rectangle"][1][0]
    common = None
    if "common" in meta:
        with np.load(path / meta["common"], allow_pickle=False) as z:
            common = {k: z[k] for k in z.files}
    hashes = {"levels.json": sha256(path / "levels.json")}
    if common is not None:
        hashes[meta["common"]] = sha256(path / meta["common"])
    static = None
    levels = []
    for r in sorted(records, key=lambda r: r["index"]):
        f = path / r["file"]
        hashes[r["file"]] = sha256(f)
        with np.load(f, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        if common is not None:
            for k in STATIC + TEXT:
                if k in arrays and not np.array_equal(arrays[k], common[k], equal_nan=arrays[k].dtype.kind == "f"):
                    raise ValueError(f"{r['file']}: {k} differs from common.npz.")
                arrays.setdefault(k, common[k])
        if static is None:
            static = {k: arrays[k] for k in STATIC + TEXT}
        for k in STATIC + TEXT:
            if not np.array_equal(arrays[k], static[k], equal_nan=static[k].dtype.kind == "f"):
                raise ValueError(f"{r['file']}: invariant field {k} differs between levels.")
        mapped = {}
        for a, m in PAIRS:
            x = np.asarray(arrays[m], dtype=np.float64)
            if x.shape != static[a].shape or x.ndim != 2 or x.shape[1] != 2:
                raise ValueError(f"{r['file']}: {m} has shape {x.shape}; expected the matching input's two-coordinate shape {static[a].shape}.")
            if not np.array_equal(np.isnan(x).any(axis=1), np.isnan(static[a]).any(axis=1)):
                raise ValueError(f"{r['file']}: {m} does not preserve the input's NaN separator rows.")
            if np.isnan(x).any(axis=1).sum() != np.isnan(x).all(axis=1).sum():
                raise ValueError(f"{r['file']}: each NaN separator in {m} must occupy both coordinates.")
            fin = np.isfinite(x).all(axis=1)
            if np.any(x[fin, 0] < -1e-6) or np.any(x[fin, 0] > width + 1e-6) or np.any(x[fin, 1] < -1e-6) or np.any(x[fin, 1] > height + 1e-6):
                raise ValueError(f"{r['file']}: {m} extends outside the map rectangle by more than 1e-6 radians.")
            mapped[m] = x
        if not np.isfinite(mapped["cell_mapped"]).all():
            raise ValueError(f"{r['file']}: cell_mapped must contain only finite coordinates.")
        levels.append({"record": r, "mapped": mapped, "dtype": str(arrays["cell_mapped"].dtype)})
    ci = static["cell_input"]
    if ci.shape != (nx * ny, 2) or static["population"].shape != (nx * ny,):
        raise ValueError(f"The {nx} by {ny} grid requires {nx * ny} centres and population values; found shapes {ci.shape} and {static['population'].shape}.")
    grid_x = ci[:, 0].reshape(ny, nx)
    grid_y = ci[:, 1].reshape(ny, nx)
    if not (np.all(np.diff(grid_x, axis=1) > 0) and np.all(np.diff(grid_y, axis=0) > 0) and np.ptp(grid_y, axis=1).max() == 0):
        raise ValueError("cell_input must run west to east within rows and south to north between rows, with constant y in each row.")
    bridges = [lv for lv in levels if lv["record"]["kind"] == "bridge"]
    diffusion = [lv for lv in levels if lv["record"]["kind"] == "diffusion"]
    if len(diffusion) != 1 or diffusion[0]["record"]["gamma"] is not None:
        raise ValueError(f"Expected one diffusion level with gamma=null; found diffusion indices {[lv['record']['index'] for lv in diffusion]}.")
    if len(bridges) + 1 != len(levels):
        raise ValueError(f"Every other level must have kind='bridge'; found {[lv['record']['kind'] for lv in levels]}.")
    for lv in bridges:
        g = lv["record"]["gamma"]
        if g is None or not np.isfinite(g) or g <= 0:
            raise ValueError(f"Bridge level {lv['record']['index']} needs finite positive gamma; found {g}.")
    bridges.sort(key=lambda lv: lv["record"]["gamma"])
    gam = np.array([lv["record"]["gamma"] for lv in bridges])
    if not np.all(np.diff(gam) > 0):
        raise ValueError(f"Bridge noise levels must be distinct and increasing; found {gam.tolist()}.")
    return {"meta": meta, "static": static, "bridges": bridges, "diffusion": diffusion[0], "hashes": hashes}


def pack(source, out):
    fam = load_family(source)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "common.npz", **fam["static"])
    meta = {k: v for k, v in fam["meta"].items() if k != "levels"}
    meta["common"] = "common.npz"
    meta["mapped_dtype"] = "float32"
    records = []
    for lv in sorted(fam["bridges"] + [fam["diffusion"]], key=lambda lv: lv["record"]["index"]):
        r = dict(lv["record"])
        m32 = {k: v.astype(np.float32) for k, v in lv["mapped"].items()}
        np.savez_compressed(out / r["file"], **m32)
        r["source_sha256"] = fam["hashes"][r["file"]]
        r["float32_max_abs_rounding"] = float(max(np.nanmax(np.abs(m32[k].astype(np.float64) - lv["mapped"][k])) for k in MAPPED))
        records.append(r)
    meta["levels"] = records
    meta["source_levels_json_sha256"] = fam["hashes"]["levels.json"]
    (out / "levels.json").write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n")
    shutil.copyfile(ROOT / "LICENSES/CC-BY-4.0.md", out / "LICENSE.md")
    (out / "ATTRIBUTION.md").write_text((ROOT / "data/hero/ATTRIBUTION.md").read_text())
    return out


def adjacent_displacement(fam):
    rect = fam["meta"]["rectangle"]
    diameter = float(np.hypot(rect[0][1] - rect[0][0], rect[1][1] - rect[1][0]))
    static = fam["static"]
    fin = np.isfinite(static["coast_input"]).all(axis=1)
    seq = [("geographic", None, {"cell_mapped": static["cell_input"], "coast_mapped": static["coast_input"]})]
    seq += [(lv["record"]["index"], lv["record"], lv["mapped"]) for lv in fam["bridges"]]
    seq += [(fam["diffusion"]["record"]["index"], fam["diffusion"]["record"], fam["diffusion"]["mapped"])]
    rows = []
    for (ia, ra, a), (ib, rb, b) in zip(seq[:-1], seq[1:]):
        dc = np.hypot(*(b["cell_mapped"] - a["cell_mapped"]).T) / diameter
        dk = np.hypot(*(b["coast_mapped"][fin] - a["coast_mapped"][fin]).T) / diameter
        rows.append({
            "from": ia, "to": ib,
            "gamma_from": None if ra is None else ra["gamma"], "gamma_to": rb["gamma"],
            "solver_from": None if ra is None else ra["solver"], "solver_to": rb["solver"],
            "cells_max": float(dc.max()), "cells_p95": float(np.percentile(dc, 95)),
            "coast_max": float(dk.max()), "coast_p95": float(np.percentile(dk, 95)),
            "cells_argmax": int(dc.argmax()),
        })
    return {"diameter": diameter, "rows": rows}


def build_schedule(fam):
    s = SCHEDULE
    br = fam["bridges"]
    g = [lv["record"]["gamma"] for lv in br]
    idx = [lv["record"]["index"] for lv in br]
    frames = []
    segments = []

    def add(spec_kind, a, b, fracs, labels, note, shown):
        start = len(frames)
        for f, lab in zip(fracs, labels):
            frames.append((spec_kind, a, b, f, lab))
        segments.append({"frames": [start, len(frames) - 1], "count": len(fracs), "treatment": note, "levels": shown})

    add("geographic", None, None, [0.0] * s["geographic"], [None] * s["geographic"], "hold", ["geographic"])
    add("level", 0, 0, [0.0] * s["first_hold"], [g[0]] * s["first_hold"], "cut, hold", [idx[0]])
    for i in range(len(br) - 1):
        if br[i]["record"]["solver"] != br[i + 1]["record"]["solver"]:
            add("level", i, i, [0.0] * s["join_hold"], [g[i]] * s["join_hold"], "hold", [idx[i]])
            add("level", i + 1, i + 1, [0.0] * s["join_hold"], [g[i + 1]] * s["join_hold"], "cut, hold", [idx[i + 1]])
            continue
        m = s["slow"] if g[i + 1] <= s["slow_max_gamma"] else s["fast"]
        fr = [k / m for k in range(1, m + 1)]
        add("interp", i, i + 1, fr, [float(np.exp((1 - f) * np.log(g[i]) + f * np.log(g[i + 1]))) for f in fr], f"interpolation k/{m}, k = 1..{m}", [idx[i], idx[i + 1]])
    n = len(br) - 1
    add("level", n, n, [0.0] * s["last_hold"], [g[n]] * s["last_hold"], "hold", [idx[n]])
    add("diffusion", None, None, [0.0] * s["diffusion_hold"], [None] * s["diffusion_hold"], "cut, hold", [fam["diffusion"]["record"]["index"]])
    return frames, segments


def state(fam, spec):
    kind, a, b, f, lab = spec
    st = fam["static"]
    if kind == "geographic":
        return st["cell_input"], st["coast_input"], st["graticule_input"], "Geographic"
    if kind == "diffusion":
        m = fam["diffusion"]["mapped"]
        return m["cell_mapped"], m["coast_mapped"], m["graticule_mapped"], "Diffusion"
    A = fam["bridges"][a]["mapped"]
    if kind == "level":
        return A["cell_mapped"], A["coast_mapped"], A["graticule_mapped"], gamma_label(lab)
    B = fam["bridges"][b]["mapped"]
    out = [(1.0 - f) * A[k] + f * B[k] for k in MAPPED]
    return out[0], out[1], out[2], gamma_label(lab)


def gamma_label(g):
    return r"$\gamma$ = " + f"{g:.3g}"


def relative_luminance(rgb):
    c = np.asarray(rgb, float)
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])


def contrast(a, b):
    la, lb = relative_luminance(a), relative_luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def palette():
    base = mpl.colormaps[CMAP_NAME].resampled(CMAP_LUT)
    bg = to_rgb(BG)
    start = next(i for i in range(CMAP_LUT) if contrast(base(i)[:3], bg) >= CONTRAST_MIN)
    lo = start / (CMAP_LUT - 1)
    cmap = LinearSegmentedColormap.from_list(f"{CMAP_NAME}_{lo:.5f}_1", base(np.arange(start, CMAP_LUT)), N=CMAP_LUT - start)
    cmap.set_under(UNDER)
    return cmap, {"lut_size": CMAP_LUT, "lut_index": start, "position": lo, "colour": mpl.colors.to_hex(base(start)), "contrast": contrast(base(start)[:3], bg), "previous_contrast": contrast(base(start - 1)[:3], bg) if start else None}


def categories(static):
    pop, land = static["population"], static["land_fraction"]
    if not (np.all(np.isfinite(pop)) and np.all(pop >= 0) and np.all(np.isfinite(land)) and np.all(land >= 0)):
        raise ValueError("population and land_fraction must contain only finite, nonnegative values before assigning display categories.")
    cat = np.full(pop.size, 3, dtype=np.int8)
    cat[(pop == 0) & (land > 0)] = 2
    cat[(pop > 0) & (pop < VMIN)] = 1
    cat[pop >= VMIN] = 0
    return cat


def cell_colours(static):
    pop = static["population"]
    cmap, info = palette()
    norm = LogNorm(vmin=VMIN, vmax=float(pop.max()))
    cat = categories(static)
    rgb = np.empty((pop.size, 3))
    rgb[cat == 3] = to_rgb(OCEAN)
    rgb[cat == 2] = to_rgb(ZERO_LAND)
    rgb[cat == 1] = to_rgb(UNDER)
    rgb[cat == 0] = cmap(norm(pop[cat == 0]))[:, :3]
    rgb8 = np.round(rgb * 255).astype(np.uint8)
    return rgb8, cat, norm, cmap, info


class Field:
    def __init__(self, rect, w, h):
        x0, x1 = rect[0]
        y0, y1 = rect[1]
        xs = x0 + (np.arange(w) + 0.5) * (x1 - x0) / w
        ys = y1 - (np.arange(h) + 0.5) * (y1 - y0) / h
        gx, gy = np.meshgrid(xs, ys)
        self.q = np.column_stack([gx.ravel(), gy.ravel()])
        self.w, self.h = w, h

    def nearest(self, cells):
        q = self.q
        _, j = cKDTree(cells).query(q, k=2, workers=-1)
        d0 = ((q - cells[j[:, 0]]) ** 2).sum(axis=1)
        d1 = ((q - cells[j[:, 1]]) ** 2).sum(axis=1)
        pick = np.where((d1 < d0) | ((d1 == d0) & (j[:, 1] < j[:, 0])), j[:, 1], j[:, 0])
        near = np.flatnonzero(np.abs(d1 - d0) <= NEAR_TIE_RTOL * np.minimum(d0, d1))
        ties = 0
        for k in near:
            dd = ((q[k] - cells) ** 2).sum(axis=1)
            cand = np.flatnonzero(dd == dd.min())
            pick[k] = cand.min()
            ties += int(cand.size > 1)
        return pick, {"near_ties_checked": int(near.size), "exact_ties": ties}


def brute_nearest(q, cells):
    out = np.empty(len(q), dtype=np.int64)
    for k in range(len(q)):
        dd = ((q[k] - cells) ** 2).sum(axis=1)
        out[k] = np.flatnonzero(dd == dd.min()).min()
    return out


def duplicate_centres(cells):
    _, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    return int(np.sum(counts[counts > 1])), int(np.sum(counts > 1))


class Scene:
    def __init__(self, fam, size_px):
        self.w, self.h = size_px
        s = self.w / VIDEO_PX[0]
        self.s = s
        L = LAYOUT
        self.rgb8, self.cat, self.norm, self.cmap, self.palette = cell_colours(fam["static"])
        rect = fam["meta"]["rectangle"]
        x0, y0, mw, mh = (int(round(v * s)) for v in L["map"])
        if mw != 2 * mh:
            raise ValueError(f"The map viewport must have aspect ratio 2:1; found {mw} by {mh} pixels.")
        self.box = (x0, y0, mw, mh)
        self.field = Field(rect, mw, mh)
        self.fig = plt.figure(figsize=(self.w / 100, self.h / 100), dpi=100, facecolor=BG)
        self.img = self.fig.figimage(np.zeros((mh, mw, 3), dtype=np.uint8), xo=x0, yo=self.h - y0 - mh, origin="upper", zorder=0, resample=False)
        self.ax = self.fig.add_axes([x0 / self.w, 1 - (y0 + mh) / self.h, mw / self.w, mh / self.h], zorder=1)
        self.ax.set_xlim(*rect[0])
        self.ax.set_ylim(*rect[1])
        self.ax.set_axis_off()
        self.ax.patch.set_visible(False)
        lw = lambda p: p * s * 0.72
        (self.grat,) = self.ax.plot([], [], color=GRATICULE, linewidth=lw(L["graticule_px"]), solid_joinstyle="round", zorder=1)
        (self.key,) = self.ax.plot([], [], color=KEYLINE, linewidth=lw(L["coast_px"] + L["keyline_px"]), solid_joinstyle="round", solid_capstyle="round", zorder=2)
        (self.coast,) = self.ax.plot([], [], color=COAST, linewidth=lw(L["coast_px"]), solid_joinstyle="round", solid_capstyle="round", zorder=3)
        lx, ly = (v * s for v in L["label_xy"])
        self.label = self.fig.text(lx / self.w, 1 - ly / self.h, "", color=TEXT_COLOUR, fontsize=L["label_px"] * s * 0.72, family="DejaVu Sans", math_fontfamily="dejavusans", ha="left", va="baseline")
        sw = L["swatch_px"] * s
        for (sx, sy), colour, text in zip(L["swatches"], (OCEAN, ZERO_LAND), ("Ocean", "0 (land)")):
            sx, sy = sx * s, sy * s
            self.fig.add_artist(Rectangle(((sx) / self.w, 1 - (sy + sw) / self.h), sw / self.w, sw / self.h, transform=self.fig.transFigure, facecolor=colour, edgecolor=SWATCH_EDGE, linewidth=lw(0.9)))
            self.fig.text((sx + sw + 9 * s) / self.w, 1 - (sy + sw / 2) / self.h, text, color=TEXT_COLOUR, fontsize=L["swatch_label_px"] * s * 0.72, family="DejaVu Sans", ha="left", va="center_baseline")
        cx, cy, cw, ch = (v * s for v in L["cbar"])
        cax = self.fig.add_axes([cx / self.w, 1 - (cy + ch) / self.h, cw / self.w, ch / self.h])
        cb = self.fig.colorbar(mpl.cm.ScalarMappable(norm=self.norm, cmap=self.cmap), cax=cax, orientation="horizontal", extend="min", extendfrac=L["cbar_extendfrac"])
        cb.outline.set_edgecolor(TEXT_COLOUR)
        cb.outline.set_linewidth(lw(0.8))
        ticks = [t for t in TICKS if self.norm.vmin <= t <= self.norm.vmax]
        cb.set_ticks(ticks)
        cb.set_ticklabels([f"{t:g}" if t < 10 else f"$10^{{{int(round(np.log10(t)))}}}$" for t in ticks])
        cax.xaxis.set_minor_locator(NullLocator())
        cax.tick_params(colors=TEXT_COLOUR, labelcolor=TEXT_COLOUR, labelsize=L["cbar_tick_px"] * s * 0.72, width=lw(0.8), length=4 * s, pad=3 * s)
        self.fig.canvas.draw()
        inner = cb.ax.get_window_extent()
        tb = cax.get_xticklabels()[0].get_window_extent()
        self.fig.text((tb.x0 - L["under_label_gap"] * s) / self.w, tb.y0 / self.h, "<1", color=TEXT_COLOUR, fontsize=L["cbar_tick_px"] * s * 0.72, family="DejaVu Sans", ha="right", va="bottom")
        if tb.x0 - L["under_label_gap"] * s <= cx:
            raise ValueError(f"The <1 label does not fit beside the colour scale: first tick starts at {tb.x0:.2f} px, colour-bar box at {cx:.2f} px.")
        self.fig.text((cx - L["cbar_label_gap"] * s) / self.w, 1 - (cy + ch / 2) / self.h, "People per input cell", color=TEXT_COLOUR, fontsize=L["cbar_label_px"] * s * 0.72, family="DejaVu Sans", ha="right", va="center")
        self.ticks = ticks

    def fill(self, cells):
        pick, stats = self.field.nearest(np.asarray(cells, dtype=np.float64))
        self.img.set_data(self.rgb8[pick].reshape(self.field.h, self.field.w, 3))
        return pick, stats

    def draw(self, cells, coast, grat, label):
        pick, stats = self.fill(cells)
        self.coast.set_data(coast[:, 0], coast[:, 1])
        self.key.set_data(coast[:, 0], coast[:, 1])
        self.grat.set_data(grat[:, 0], grat[:, 1])
        self.label.set_text(label)
        self.fig.canvas.draw()
        self.last = (pick, stats)
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3]

    def check(self, cells, pick, n=CHECK_PIXELS, seed=CHECK_SEED):
        rng = np.random.default_rng(seed)
        k = np.sort(rng.choice(len(self.field.q), size=min(n, len(self.field.q)), replace=False))
        ref = brute_nearest(self.field.q[k], np.asarray(cells, dtype=np.float64))
        return {"pixels": int(k.size), "seed": seed, "mismatches": int(np.sum(ref != pick[k]))}

    def close(self):
        plt.close(self.fig)


def ffmpeg_version(ffmpeg):
    out = subprocess.run([ffmpeg, "-hide_banner", "-version"], capture_output=True, text=True, check=True).stdout.splitlines()
    conf = [l for l in out if l.startswith("configuration:")]
    return {"version": out[0], "gpl": bool(conf and "--enable-gpl" in conf[0]), "libx264": bool(conf and "--enable-libx264" in conf[0])}


def frame_stats(scene, cells, pick, stats):
    if len(cells) != len(scene.cat):
        raise ValueError(f"Nearest-centre lookup needs one centre per population category; found {len(cells)} centres and {len(scene.cat)} categories.")
    rep = np.zeros(len(cells), dtype=bool)
    rep[pick] = True
    omitted = ~rep
    dup_rows, dup_groups = duplicate_centres(np.asarray(cells, dtype=np.float64))
    return rep, {"exact_ties": stats["exact_ties"], "near_ties_checked": stats["near_ties_checked"], "omitted": int(omitted.sum()), "omitted_by_category": [int(np.sum(omitted & (scene.cat == c))) for c in range(4)], "duplicate_rows": dup_rows, "duplicate_groups": dup_groups}


def check_frames(segments):
    wanted = []
    for seg in segments:
        if seg["treatment"] != "hold" or seg["levels"] == ["geographic"]:
            wanted.append(seg["frames"][0] + (seg["count"] // 2 if seg["treatment"].startswith("interpolation") else 0))
    return sorted(set(wanted))


def render_movie(fam, path, ffmpeg, crf):
    frames, segments = build_schedule(fam)
    w, h = VIDEO_PX
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
           "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
           "-movflags", "+faststart", "-map_metadata", "-1", "-metadata", "creation_time=", "-fflags", "+bitexact", "-flags:v", "+bitexact", str(path)]
    scene = Scene(fam, VIDEO_PX)
    checks = set(check_frames(segments))
    ever = np.zeros(len(fam["static"]["cell_input"]), dtype=bool)
    per_frame, verified = [], {}
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for i, spec in enumerate(frames):
            cells, coast, grat, label = state(fam, spec)
            img = scene.draw(cells, coast, grat, label)
            if img.shape != (h, w, 3):
                raise ValueError(f"Movie frame has shape {img.shape}; expected {(h, w, 3)} RGB pixels.")
            proc.stdin.write(np.ascontiguousarray(img).tobytes())
            pick, stats = scene.last
            rep, row = frame_stats(scene, cells, pick, stats)
            ever |= rep
            per_frame.append(row)
            if i in checks:
                verified[str(i)] = scene.check(cells, pick)
    finally:
        proc.stdin.close()
        rc = proc.wait()
        info = {"norm": [scene.norm.vmin, scene.norm.vmax], "colourbar_ticks": scene.ticks, "palette": scene.palette, "map_box_px": list(scene.box), "categories": scene.cat}
        scene.close()
    if rc != 0:
        raise RuntimeError(f"FFmpeg exited with status {rc}; inspect its preceding error output.")
    info["never_represented"] = [int(np.sum(~ever & (info["categories"] == c))) for c in range(4)]
    return frames, segments, info, per_frame, verified


def render_still(fam, spec, size_px):
    scene = Scene(fam, size_px)
    cells, coast, grat, label = state(fam, spec)
    img = scene.draw(cells, coast, grat, label)
    pick, stats = scene.last
    _, row = frame_stats(scene, cells, pick, stats)
    row["check"] = scene.check(cells, pick)
    row["map_box_px"] = list(scene.box)
    row["pixels_by_category"] = [int(np.sum(scene.cat[pick] == c)) for c in range(4)]
    scene.close()
    return img, row


def diagnostics(fam):
    recs = [lv["record"] for lv in fam["bridges"]] + [fam["diffusion"]["record"]]
    out = {}
    for key in ("largest_cell_density_deviation", "continuity", "minimum_density", "sinkhorn_error", "tracking_refine"):
        vals = [r.get(key) for r in recs if r.get(key) is not None]
        out[key] = {"min": min(vals), "max": max(vals), "missing": [r["index"] for r in recs if r.get(key) is None]} if vals else None
    solvers = {}
    for lv in fam["bridges"]:
        r = lv["record"]
        solvers.setdefault(r["solver"], []).append(r)
    out["solver_split"] = {k: {"count": len(v), "indices": [v[0]["index"], v[-1]["index"]], "gamma": [v[0]["gamma"], v[-1]["gamma"]]} for k, v in solvers.items()}
    out["diffusion_solver"] = fam["diffusion"]["record"]["solver"]
    return out


def counts(static):
    pop, land = static["population"], static["land_fraction"]
    cat = categories(static)
    return {"cells": int(pop.size), "population_at_least_1": int(np.sum(cat == 0)), "population_positive_below_1": int(np.sum(cat == 1)), "zero_population_land": int(np.sum(cat == 2)), "zero_population_ocean": int(np.sum(cat == 3)), "positive_population_zero_land_fraction": int(np.sum((land == 0) & (pop > 0))), "population_negative": int(np.sum(pop < 0)), "population_nonfinite": int(np.sum(~np.isfinite(pop))), "population_min_positive": float(pop[pop > 0].min()), "population_max": float(pop.max()), "population_total": float(pop.sum())}


def save_png(img, path, dpi):
    im = Image.fromarray(np.ascontiguousarray(img), mode="RGB")
    im.save(path, format="PNG", dpi=(dpi, dpi), optimize=True)
    return {"bytes": path.stat().st_size, "sha256": sha256(path), "pixels": list(im.size), "mode": "RGB", "dpi": [dpi, dpi]}


def render(hero_dir, out, ffmpeg, crf=HERO_CRF):
    fam = load_family(hero_dir)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    movie = out / "world-bridge.mp4"
    frames, segments, info, per_frame, verified = render_movie(fam, movie, ffmpeg, crf)
    first = fam["bridges"][0]
    poster_spec = ("level", 0, 0, 0.0, first["record"]["gamma"])
    poster = out / "world-bridge-poster.png"
    poster_img, poster_stats = render_still(fam, poster_spec, POSTER_PX)
    poster_entry = save_png(poster_img, poster, 300)
    category_names = ["population_at_least_1", "population_positive_below_1", "zero_population_land", "zero_population_ocean"]
    record = {
        "inputs": fam["hashes"],
        "source": {k: v for k, v in fam["meta"].items() if k != "levels"},
        "mapped_dtype": sorted({lv["dtype"] for lv in fam["bridges"] + [fam["diffusion"]]}),
        "levels": {"bridge_count": len(fam["bridges"]), "gamma_min": fam["bridges"][0]["record"]["gamma"], "gamma_max": fam["bridges"][-1]["record"]["gamma"], "diffusion_index": fam["diffusion"]["record"]["index"], "order": [lv["record"]["index"] for lv in fam["bridges"]]},
        "diagnostics": diagnostics(fam),
        "adjacent_displacement": adjacent_displacement(fam),
        "schedule": {"rules": SCHEDULE, "fps": FPS, "frames": len(frames), "seconds": len(frames) / FPS, "segments": segments, "interpolation": "linear in mapped coordinates for cells, coastline and graticule vertices; gamma label log-linear with the same fraction", "per_frame": [[k, None if a is None else fam["bridges"][a]["record"]["index"], None if b is None else fam["bridges"][b]["record"]["index"], f, lab] for k, a, b, f, lab in frames]},
        "fill": {
            "field": "nearest mapped cell centre, original population and land category",
            "metric": "squared Euclidean distance in plane radians, dx*dx + dy*dy in float64",
            "lookup": "all centres, including ocean; cKDTree k=2, both candidates rechecked exactly; near ties within relative 1e-12 resolved by brute force over all centres",
            "tie_rule": "lowest original row-major index",
            "lookup_centres_every_frame": int(len(fam["static"]["cell_input"])),
            "sampling": "one sample at each map pixel centre, x = (j + 0.5) / width, y = 1 - (i + 0.5) / height of the rectangle; evaluated anew every frame",
            "pixel_coverage": "none; no antialiasing, blur, smoothing or sharpening of the fill",
            "boundaries_drawn": False,
            "category_order": category_names,
            "category_colours": {"population_positive_below_1": UNDER, "zero_population_land": ZERO_LAND, "zero_population_ocean": OCEAN},
            "palette": {"cmap": CMAP_NAME, **info["palette"], "upper_position": 1.0, "norm": "LogNorm", "vmin": info["norm"][0], "vmax": info["norm"][1], "under": UNDER, "background": BG, "contrast_rule": f"lowest LUT entry with contrast >= {CONTRAST_MIN} against background"},
            "per_frame": {k: [row[k] for row in per_frame] for k in ("exact_ties", "near_ties_checked", "omitted", "duplicate_rows", "duplicate_groups")},
            "per_frame_omitted_by_category": [row["omitted_by_category"] for row in per_frame],
            "never_represented_in_movie_by_category": info["never_represented"],
            "brute_force_checks": verified,
            "map_box_px_video": info["map_box_px"],
        },
        "display": {"video_px": list(VIDEO_PX), "poster_px": list(POSTER_PX), "layout_px_at_1920": LAYOUT, "map_aspect": 2.0, "background": BG, "coast": COAST, "keyline": KEYLINE, "graticule_rgba": list(GRATICULE), "text": TEXT_COLOUR, "swatch_edge": SWATCH_EDGE, "colourbar_ticks": info["colourbar_ticks"], "draw_order": "fill, graticule, keyline, coastline", "font": "DejaVu Sans"},
        "counts": counts(fam["static"]),
        "poster": {"index": first["record"]["index"], "gamma": first["record"]["gamma"], "solver": first["record"]["solver"], "spec": "computed level, no interpolation", **poster_stats},
        "encoder": {**ffmpeg_version(ffmpeg), "codec": "libx264", "preset": "slow", "crf": crf, "pix_fmt": "yuv420p", "colour": "bt709 limited range"},
        "outputs": {movie.name: {"bytes": movie.stat().st_size, "sha256": sha256(movie), "pixels": list(VIDEO_PX), "fps": FPS, "frames": len(frames)}, poster.name: poster_entry},
        "labels": ["Geographic", "γ = <value>", "Diffusion", "People per input cell", "<1", "0 (land)", "Ocean"],
    }
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True, help="Completed world_family.py output containing levels.json and the referenced NPZ files.")
    p.add_argument("--out", type=Path, required=True, help="Directory for common.npz, float32 level files, levels.json and notices. Existing matching files are replaced; use a separate directory to preserve the included pack.")
    a = p.parse_args()
    pack(a.source, a.out)


if __name__ == "__main__":
    main()
