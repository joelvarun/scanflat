"""Corpus sweep: real installed fonts, real glyphs, real pixel grids."""
import math
import random

import numpy as np

from .flatten import (cell_crossings, chord_deviation, flatten_cells,
                      flatten_moment, flatten_recursive, flatten_uniform,
                      segment_count)
from .geom import eval_seg
from .outline import basename, font_files, load_contours, open_font, scaled_glyph, transform
from .raster import alpha8, coverage, coverage_supersample, fill, pixel_box

TOLS = [2.0 ** -k for k in range(1, 18)]
TEXT = "aegosnBRQS28"


def error_stats(ref, cov, ref8=None):
    err = cov - ref
    a_ref = ref8 if ref8 is not None else alpha8(ref)
    wrong = int(np.count_nonzero(alpha8(cov) != a_ref))
    return dict(mae=float(np.abs(err).mean()), maxerr=float(np.abs(err).max()),
                bias=float(err.mean()), wrong8=wrong)


def max_deviation(contours, poly, per_seg=17):
    """Largest distance from the true outline to the flattened polyline."""
    ax, ay, bx, by = [], [], [], []
    for contour in poly:
        for seg in contour:
            ax.append(seg[1][0]); ay.append(seg[1][1])
            bx.append(seg[-1][0]); by.append(seg[-1][1])
    ax = np.array(ax); ay = np.array(ay)
    dx = np.array(bx) - ax; dy = np.array(by) - ay
    den = dx * dx + dy * dy
    den[den == 0.0] = 1.0
    pts = []
    for contour in contours:
        for seg in contour:
            if seg[0] == "L":
                continue
            for k in range(per_seg):
                pts.append(eval_seg(seg, k / (per_seg - 1.0)))
    if not pts:
        return 0.0
    px = np.array([p[0] for p in pts])[:, None]
    py = np.array([p[1] for p in pts])[:, None]
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / den, 0.0, 1.0)
    d = np.hypot(px - (ax + t * dx), py - (ay + t * dy))
    return float(d.min(axis=1).max())


def ladder(contours, box, ref, ref8, rule):
    """Error against segment count for one tolerance rule."""
    rows = []
    done = 0
    for tol in TOLS:
        poly = rule(contours, tol)
        st = error_stats(ref, fill(coverage(poly, box)), ref8)
        st.update(tol=tol, segs=segment_count(poly), cells=cell_crossings(poly))
        rows.append(st)
        if st["wrong8"] == 0:
            done += 1
            if done >= 2:
                break
    return rows


def run_case(font, name, ppem, off, with_extras=False):
    g = scaled_glyph(font, name, ppem, off[0], off[1])
    if segment_count(g) < 4:
        return None
    box = pixel_box(g)
    if box[2] * box[3] > 40000:
        return None
    ref = fill(coverage(g, box))
    ref8 = alpha8(ref)
    out = dict(ppem=ppem, ink=float(ref.sum()), px=int(box[2] * box[3]),
               true_segs=segment_count(g), true_cells=cell_crossings(g),
               edge_px=int(np.count_nonzero((ref > 1e-9) & (ref < 1.0 - 1e-9))))
    for key, poly in (("moment", flatten_moment(g)), ("cells", flatten_cells(g))):
        st = error_stats(ref, fill(coverage(poly, box)), ref8)
        st.update(segs=segment_count(poly), cells=cell_crossings(poly))
        out[key] = st
    out["uniform"] = ladder(g, box, ref, ref8, flatten_uniform)
    out["recursive"] = ladder(g, box, ref, ref8, flatten_recursive)
    if with_extras:
        mom = flatten_moment(g)
        out["dev_moment"] = max_deviation(g, mom)
        out["dev_cells"] = max_deviation(g, flatten_cells(g))
        ref_u = [r for r in out["uniform"] if r["wrong8"] == 0]
        tol_u = ref_u[0]["tol"] if ref_u else TOLS[-1]
        out["dev_uniform8"] = max_deviation(g, flatten_uniform(g, tol_u))
        # reuse the same polyline on a grid shifted by half a pixel
        sg = transform(g, 1.0, 0.5, 0.5)
        sm = transform(mom, 1.0, 0.5, 0.5)
        sbox = pixel_box(sg)
        out["shifted"] = error_stats(fill(coverage(sg, sbox)),
                                     fill(coverage(sm, sbox)))
        # the other rasterizer model: n by n point samples per pixel
        ss_ref = coverage_supersample(g, box, 4)
        out["supersample"] = error_stats(ss_ref, coverage_supersample(mom, box, 4))
    return out


def corpus(n_fonts, ppems, offsets, text=TEXT, seed=0, limit=None):
    rng = random.Random(seed)
    files = font_files()
    rng.shuffle(files)
    picked, cases = 0, []
    for path in files:
        if n_fonts is not None and picked >= n_fonts:
            break
        try:
            font = open_font(path)
            names = [(ch, nm) for ch, nm in
                     [(c, font.getBestCmap().get(ord(c))) for c in text] if nm]
            if not names:
                continue
            load_contours(font, names[0][1])
        except Exception:
            continue
        picked += 1
        for ch, nm in names:
            for ppem in ppems:
                cases.append((path, font, ch, nm, ppem))
    if limit:
        cases = cases[:limit]
    return cases
