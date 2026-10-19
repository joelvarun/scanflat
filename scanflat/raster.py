"""Rasterizers.

`coverage` is the exact-area scanline model that FreeType's smooth renderer and
Anti-Grain Geometry both implement: the boundary is cut at every pixel cell
edge, and each cell receives two numbers from the pieces inside it,

    D(i, j)  = sum of dy over boundary pieces in the cell
    M(i, j)  = sum of integral x dy over those pieces

after which the coverage of the cell follows in closed form.  Everything here
is analytic, so the result is exact for curved input and is the ground truth
the flatteners in `flatten.py` are measured against.

`coverage_supersample` is the other common rasterizer model, an n by n grid of
point samples per pixel.  It is used only to show which of the two models the
moment construction is exact for.
"""
import math

import numpy as np

from .geom import cell_pieces, moments


def pixel_box(contours, margin=1):
    """Integer pixel box (x0, y0, width, height) containing all the geometry."""
    xs, ys = [], []
    for contour in contours:
        for seg in contour:
            for p in seg[1:]:
                xs.append(p[0])
                ys.append(p[1])
    if not xs:
        return (0, 0, 1, 1)
    x0 = int(math.floor(min(xs))) - margin
    y0 = int(math.floor(min(ys))) - margin
    x1 = int(math.ceil(max(xs))) + margin
    y1 = int(math.ceil(max(ys))) + margin
    return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))


def coverage(contours, box):
    """Exact per-pixel accumulation under the nonzero winding rule.

    The result is signed: its sign follows the orientation of the outer
    contour, and a point covered twice accumulates twice, exactly as in the
    integer accumulator of a real rasterizer.  Pass it through `fill` to get
    what a renderer stores.  Raises if the box does not contain the geometry,
    since pieces outside it would be miscounted.
    """
    x0, y0, w, h = box
    D = np.zeros((h, w))
    M = np.zeros((h, w))
    for contour in contours:
        for seg in contour:
            for piece in cell_pieces(seg):
                dy, xdy = moments(piece)
                if dy == 0.0:
                    continue
                mx = 0.5 * (piece[1][0] + piece[-1][0])
                my = 0.5 * (piece[1][1] + piece[-1][1])
                i = int(math.floor(mx)) - x0
                j = int(math.floor(my)) - y0
                if not (0 <= i < w and 0 <= j < h):
                    raise ValueError("geometry outside the pixel box")
                D[j, i] += dy
                M[j, i] += (x0 + i + 1.0) * dy - xdy
    left = np.cumsum(D, axis=1) - D
    return left + M


def fill(cov):
    """Accumulation to coverage in [0, 1].

    FreeType's smooth renderer does exactly this: take the absolute value of
    the accumulated area so orientation does not matter, then clamp so that
    overlapping contours saturate instead of overflowing.
    """
    return np.clip(np.abs(cov), 0.0, 1.0)


def alpha8(cov):
    """Coverage to the 8 bit alpha a renderer would actually store."""
    return np.rint(fill(cov) * 255.0).astype(np.int16)


def _row_crossings(contours, ys):
    """Crossings of every segment with the horizontal lines `ys`.

    Returns (row, x, direction) arrays, one entry per crossing.
    """
    rows, xx, dd = [], [], []
    for contour in contours:
        for seg in contour:
            if seg[0] == "L":
                (ax, ay), (bx, by) = seg[1], seg[2]
                if ay == by:
                    continue
                m = (ys >= min(ay, by)) & (ys < max(ay, by))
                if not m.any():
                    continue
                idx = np.nonzero(m)[0]
                t = (ys[m] - ay) / (by - ay)
                rows.append(idx)
                xx.append(ax + t * (bx - ax))
                dd.append(np.full(idx.shape, 1.0 if by > ay else -1.0))
                continue
            (ax, ay), (cx, cy), (bx, by) = seg[1], seg[2], seg[3]
            a = ay - 2.0 * cy + by
            b = 2.0 * (cy - ay)
            lo, hi = min(ay, by), max(ay, by)
            if abs(a) > 1e-12:
                te = (ay - cy) / a
                if 0.0 < te < 1.0:
                    u = 1.0 - te
                    ve = u * u * ay + 2.0 * te * u * cy + te * te * by
                    lo, hi = min(lo, ve), max(hi, ve)
            m = (ys >= lo) & (ys < hi)
            if not m.any():
                continue
            idx = np.nonzero(m)[0]
            yv = ys[m]
            if abs(a) < 1e-12:
                if abs(b) < 1e-15:
                    continue
                roots = [(yv - ay) / b]
            else:
                disc = b * b - 4.0 * a * (ay - yv)
                ok = disc >= 0.0
                r = np.sqrt(np.where(ok, disc, 0.0))
                roots = [np.where(ok, (-b + r) / (2.0 * a), -1.0),
                         np.where(ok, (-b - r) / (2.0 * a), -1.0)]
            for t in roots:
                g = (t >= 0.0) & (t <= 1.0)
                if not g.any():
                    continue
                tg = t[g]
                u = 1.0 - tg
                xv = u * u * ax + 2.0 * tg * u * cx + tg * tg * bx
                dy = 2.0 * (u * (cy - ay) + tg * (by - cy))
                nz = dy != 0.0
                rows.append(idx[g][nz])
                xx.append(xv[nz])
                dd.append(np.where(dy[nz] > 0.0, 1.0, -1.0))
    if not rows:
        return (np.zeros(0, dtype=int), np.zeros(0), np.zeros(0))
    return np.concatenate(rows), np.concatenate(xx), np.concatenate(dd)


def coverage_supersample(contours, box, samples=4):
    """Coverage from an n by n grid of point samples per pixel."""
    x0, y0, w, h = box
    n = samples
    ys = y0 + (np.arange(h * n) + 0.5) / n
    xs = x0 + (np.arange(w * n) + 0.5) / n
    row, xx, dd = _row_crossings(contours, ys)
    if row.size == 0:
        return np.zeros((h, w))
    order = np.lexsort((xx, row))
    row, xx, dd = row[order], xx[order], dd[order]
    wind = np.cumsum(dd)
    first = np.nonzero(np.diff(row, prepend=-1) != 0)[0]
    group = np.cumsum(np.diff(row, prepend=-1) != 0) - 1
    wind = wind - (wind - dd)[first][group]
    hit = np.zeros((h * n, w * n))
    for r in np.unique(row):
        m = row == r
        k = np.searchsorted(xx[m], xs, side="right") - 1
        inside = np.where(k >= 0, np.abs(wind[m][np.clip(k, 0, None)]) > 1e-9, False)
        hit[r] = inside
    return hit.reshape(h, n, w, n).mean(axis=(1, 3))
