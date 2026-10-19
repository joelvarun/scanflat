"""Flatteners: the two rules renderers ship, one ablation, and the method.

Every shipped flattener bounds a geometric quantity (the distance between the
curve and its chord) and so converges to the right picture as the tolerance
falls.  `flatten_moment` instead reproduces the two numbers the exact-area
rasterizer reads out of each pixel cell, which makes the rendered result exact
rather than close.
"""
import math

from .geom import cell_pieces, moments, eval_seg, split_quad

TINY = 1e-12


def chord_deviation(seg):
    """Maximum distance between a quadratic and its chord."""
    if seg[0] == "L":
        return 0.0
    dx = seg[1][0] - 2.0 * seg[2][0] + seg[3][0]
    dy = seg[1][1] - 2.0 * seg[2][1] + seg[3][1]
    return 0.25 * math.hypot(dx, dy)


def flatten_uniform(contours, tol):
    """Uniform split count from the second-difference bound.

    The deviation of a quadratic from its chord is |p0 - 2 p1 + p2| / 4 and
    falls as 1 / n^2 under n uniform splits, so n = ceil(sqrt(dev / tol)).
    This is the closed-form rule used by Skia and several other renderers.
    """
    out = []
    for contour in contours:
        flat = []
        for seg in contour:
            if seg[0] == "L":
                flat.append(seg)
                continue
            dev = chord_deviation(seg)
            n = max(1, int(math.ceil(math.sqrt(dev / tol)))) if dev > 0.0 else 1
            pts = [eval_seg(seg, k / n) for k in range(n + 1)]
            flat.extend(("L", pts[k], pts[k + 1]) for k in range(n))
        out.append(flat)
    return out


def flatten_recursive(contours, tol):
    """Recursive halving while the chord deviation exceeds the tolerance.

    The structure FreeType's smooth renderer and stb_truetype use.
    """
    out = []
    for contour in contours:
        flat = []
        for seg in contour:
            if seg[0] == "L":
                flat.append(seg)
                continue
            stack = [seg]
            while stack:
                q = stack.pop()
                if chord_deviation(q) <= tol:
                    flat.append(("L", q[1], q[3]))
                else:
                    a, b = split_quad(q, 0.5)
                    stack.append(b)
                    stack.append(a)
        out.append(flat)
    return out


def flatten_cells(contours):
    """Ablation: split at every pixel cell edge, then take the chord.

    Isolates how much of the method comes from aligning the splits to the
    raster grid and how much comes from matching the moments.
    """
    out = []
    for contour in contours:
        flat = []
        for seg in contour:
            for piece in cell_pieces(seg):
                flat.append(("L", piece[1], piece[-1]))
        out.append(flat)
    return out


def _emit_matched(piece, out, depth, max_depth):
    """Two segments whose dy and integral x dy equal the piece's, in-cell."""
    v0, v1 = piece[1], piece[-1]
    h = v1[1] - v0[1]
    if abs(h) < TINY:
        if v0 != v1:
            out.append(("L", v0, v1))
        return
    _, xdy = moments(piece)
    base = 2.0 * xdy / h
    col = math.floor(0.5 * (v0[0] + v1[0]))
    lo, hi = col + 1e-9, col + 1.0 - 1e-9
    # x of the inserted vertex when it sits a fraction s of the way up in y
    span = v1[0] - v0[0]
    s = 0.5
    xm = base - v1[0] + s * span
    if xm < lo or xm > hi:
        target = min(max(xm, lo), hi)
        s = (target - base + v1[0]) / span if abs(span) > 1e-9 else -1.0
        if 0.02 <= s <= 0.98:
            xm = base - v1[0] + s * span
        elif depth < max_depth and piece[0] == "Q":
            a, b = split_quad(piece, 0.5)
            _emit_matched(a, out, depth + 1, max_depth)
            _emit_matched(b, out, depth + 1, max_depth)
            return
        else:
            s = 0.5
            xm = base - v1[0] + s * span
    mid = (xm, v0[1] + s * h)
    out.append(("L", v0, mid))
    out.append(("L", mid, v1))


def flatten_moment(contours, max_depth=6):
    """Cut at the cell edges, then match both moments inside every cell."""
    out = []
    for contour in contours:
        flat = []
        for seg in contour:
            for piece in cell_pieces(seg):
                _emit_matched(piece, flat, 0, max_depth)
        out.append(flat)
    return out


def segment_count(contours):
    return sum(len(c) for c in contours)


def cell_crossings(contours):
    """Pieces a scanline rasterizer has to account for, the other cost unit."""
    return sum(len(cell_pieces(s)) for c in contours for s in c)
