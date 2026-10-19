"""Run the whole measurement and write docs/metrics.json.

Everything reported comes from this script: real installed TrueType fonts,
real glyph outlines, real pixel grids, no synthetic geometry anywhere.
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np

from scanflat.bench import TEXT, TOLS, corpus, run_case
from scanflat.flatten import flatten_moment, flatten_uniform, segment_count
from scanflat.outline import basename, scaled_glyph
from scanflat.raster import alpha8, coverage, fill, pixel_box

PPEMS = [11.0, 14.0, 18.0, 25.0, 36.0]


def first_exact(rows):
    """First ladder row whose 8 bit output already matches exact coverage."""
    for r in rows:
        if r["wrong8"] == 0:
            return r
    return None


def interp_err(rows, segs):
    """Baseline max error at a given segment budget, log-linear in segments."""
    xs = [r["segs"] for r in rows]
    ys = [max(r["maxerr"], 1e-18) for r in rows]
    if segs <= xs[0]:
        return ys[0]
    if segs >= xs[-1]:
        return ys[-1]
    return float(np.exp(np.interp(segs, xs, np.log(ys))))


def agg_anchor(n_fonts=3, chars="aeoBQ8", ppems=(16.0, 28.0)):
    """A real shipped rasterizer (matplotlib's Agg) fed three different paths."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    def to_path(contours):
        verts, codes = [], []
        for c in contours:
            verts.append(c[0][1])
            codes.append(Path.MOVETO)
            for s in c:
                if s[0] == "L":
                    verts.append(s[2])
                    codes.append(Path.LINETO)
                else:
                    verts.append(s[2])
                    codes.append(Path.CURVE3)
                    verts.append(s[3])
                    codes.append(Path.CURVE3)
            verts.append(c[0][1])
            codes.append(Path.CLOSEPOLY)
        return Path(np.array(verts), np.array(codes))

    def render(contours, box, dpi=20):
        x0, y0, w, h = box
        fig = Figure(figsize=(w / dpi, h / dpi), dpi=dpi)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(x0, x0 + w)
        ax.set_ylim(y0, y0 + h)
        ax.set_axis_off()
        fig.patch.set_facecolor("white")
        ax.add_patch(PathPatch(to_path(contours), facecolor="black",
                               edgecolor="none", antialiased=True, snap=False))
        canvas.draw()
        buf = np.asarray(canvas.buffer_rgba())[:, :, 0].astype(float)
        return (1.0 - buf / 255.0)[::-1]

    rows = []
    picked = corpus(n_fonts=n_fonts, ppems=[16.0], offsets=None, text=chars, seed=11)
    seen = set()
    for path, font, ch, nm, _ in picked:
        for ppem in ppems:
            key = (path, ch, ppem)
            if key in seen:
                continue
            seen.add(key)
            g = scaled_glyph(font, nm, ppem, 0.37, 0.61)
            if segment_count(g) < 4:
                continue
            box = pixel_box(g)
            ref = fill(coverage(g, box))
            if render(g, box).shape != ref.shape:
                continue
            edge = int(np.count_nonzero((ref > 1e-9) & (ref < 1.0 - 1e-9)))
            rec = dict(font=basename(path), ch=ch, ppem=ppem, edge_px=edge)
            variants = (("curve", g), ("moment", flatten_moment(g)),
                        ("fine", flatten_uniform(g, 1e-4)))
            for label, pth in variants:
                img = render(pth, box)
                err = np.abs(img - ref)
                rec[label] = dict(mae=float(err.mean()), maxerr=float(err.max()),
                                  segs=segment_count(pth),
                                  wrong8=int(np.count_nonzero(alpha8(img) != alpha8(ref))))
            rows.append(rec)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonts", type=int, default=30)
    ap.add_argument("--extras", type=int, default=150)
    ap.add_argument("--out", default="docs/metrics.json")
    args = ap.parse_args()

    rng = random.Random(17)
    cases = corpus(n_fonts=args.fonts, ppems=PPEMS, offsets=None, seed=5)
    print("corpus: %d (font, glyph, size) cases" % len(cases), flush=True)
    every = max(1, len(cases) // max(1, args.extras))
    t0 = time.time()
    results, fonts_used = [], set()
    for k, (path, font, ch, nm, ppem) in enumerate(cases):
        off = (rng.random(), rng.random())
        try:
            r = run_case(font, nm, ppem, off, with_extras=(k % every == 0))
        except Exception as exc:
            print("  skip %s %r %s: %s" % (basename(path), ch, ppem, exc), flush=True)
            continue
        if r is None:
            continue
        r["font"] = basename(path)
        r["ch"] = ch
        fonts_used.add(r["font"])
        results.append(r)
        if (k + 1) % 200 == 0:
            print("  %d/%d  %.0fs" % (k + 1, len(cases), time.time() - t0), flush=True)
    print("evaluated %d cases over %d fonts in %.0f s"
          % (len(results), len(fonts_used), time.time() - t0), flush=True)

    edge_total = sum(r["edge_px"] for r in results)
    out = dict(corpus=dict(cases=len(results), fonts=len(fonts_used), glyphs=TEXT,
                           ppems=PPEMS, tols=TOLS,
                           total_px=sum(r["px"] for r in results),
                           edge_px=edge_total))
    out["exactness"] = dict(
        worst_maxerr=max(r["moment"]["maxerr"] for r in results),
        worst_mae=max(r["moment"]["mae"] for r in results),
        wrong8_total=sum(r["moment"]["wrong8"] for r in results),
        segs=float(np.mean([r["moment"]["segs"] for r in results])),
        cells=float(np.mean([r["moment"]["cells"] for r in results])),
        true_segs=float(np.mean([r["true_segs"] for r in results])),
        true_cells=float(np.mean([r["true_cells"] for r in results])))
    out["ablation_cells"] = dict(
        maxerr=float(np.mean([r["cells"]["maxerr"] for r in results])),
        mae=float(np.mean([r["cells"]["mae"] for r in results])),
        segs=float(np.mean([r["cells"]["segs"] for r in results])),
        wrong8_frac=float(sum(r["cells"]["wrong8"] for r in results) / max(1, edge_total)))

    for rule in ("uniform", "recursive"):
        curve = []
        for i, tol in enumerate(TOLS):
            rows = [r[rule][i] for r in results if len(r[rule]) > i]
            if len(rows) < 0.6 * len(results):
                break
            edges = sum(r["edge_px"] for r in results if len(r[rule]) > i)
            curve.append(dict(tol=tol, n=len(rows),
                              segs=float(np.mean([x["segs"] for x in rows])),
                              maxerr=float(np.mean([x["maxerr"] for x in rows])),
                              mae=float(np.mean([x["mae"] for x in rows])),
                              cells=float(np.mean([x["cells"] for x in rows])),
                              wrong8_frac=float(sum(x["wrong8"] for x in rows)
                                                / max(1, edges))))
        out[rule + "_curve"] = curve
        per_ppem, matched, censored = {}, [], 0
        for r in results:
            fe = first_exact(r[rule])
            if fe is None:
                censored += 1
                continue
            per_ppem.setdefault(r["ppem"], []).append(
                (fe["segs"], r["moment"]["segs"], fe["cells"], r["moment"]["cells"]))
            matched.append(interp_err(r[rule], r["moment"]["segs"]))
        pairs = [v for vals in per_ppem.values() for v in vals]
        out[rule + "_exact8"] = dict(
            censored=censored, paired=len(pairs),
            overall_baseline=float(np.mean([a for a, _, _, _ in pairs])),
            overall_moment=float(np.mean([b for _, b, _, _ in pairs])),
            overall_baseline_cells=float(np.mean([c for _, _, c, _ in pairs])),
            overall_moment_cells=float(np.mean([d for _, _, _, d in pairs])),
            by_ppem={str(k): dict(baseline=float(np.mean([a for a, _, _, _ in v])),
                                  moment=float(np.mean([b for _, b, _, _ in v])),
                                  baseline_cells=float(np.mean([c for _, _, c, _ in v])),
                                  moment_cells=float(np.mean([d for _, _, _, d in v])),
                                  n=len(v))
                     for k, v in sorted(per_ppem.items())})
        out[rule + "_matched_budget_maxerr"] = float(np.mean(matched))

    ex = [r for r in results if "shifted" in r]
    out["limits"] = dict(
        n=len(ex),
        dev_moment=float(np.mean([r["dev_moment"] for r in ex])),
        dev_cells=float(np.mean([r["dev_cells"] for r in ex])),
        dev_uniform8=float(np.mean([r["dev_uniform8"] for r in ex])),
        shifted_maxerr=float(np.mean([r["shifted"]["maxerr"] for r in ex])),
        shifted_mae=float(np.mean([r["shifted"]["mae"] for r in ex])),
        supersample_maxerr=float(np.mean([r["supersample"]["maxerr"] for r in ex])),
        supersample_mae=float(np.mean([r["supersample"]["mae"] for r in ex])))

    print("agg anchor ...", flush=True)
    rows = agg_anchor()
    out["agg"] = dict(n=len(rows), cases=rows, mean={
        label: dict(mae=float(np.mean([r[label]["mae"] for r in rows])),
                    maxerr=float(np.mean([r[label]["maxerr"] for r in rows])),
                    worst_maxerr=float(np.max([r[label]["maxerr"] for r in rows])),
                    segs=float(np.mean([r[label]["segs"] for r in rows])),
                    wrong8_frac=float(sum(r[label]["wrong8"] for r in rows)
                                      / max(1, sum(r["edge_px"] for r in rows))))
        for label in ("curve", "moment", "fine")})
    out["by_ppem_moment"] = {
        str(p): dict(segs=float(np.mean([r["moment"]["segs"] for r in results if r["ppem"] == p])),
                     cells=float(np.mean([r["moment"]["cells"] for r in results if r["ppem"] == p])))
        for p in PPEMS}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", args.out, flush=True)
    e = out["exactness"]
    print("  worst coverage error %.2e, wrong 8 bit pixels %d of %d edge pixels"
          % (e["worst_maxerr"], e["wrong8_total"], edge_total), flush=True)
    print("  segments per glyph for 8 bit exact: uniform %.1f, recursive %.1f, scanflat %.1f"
          % (out["uniform_exact8"]["overall_baseline"],
             out["recursive_exact8"]["overall_baseline"],
             out["uniform_exact8"]["overall_moment"]), flush=True)


if __name__ == "__main__":
    sys.exit(main())
