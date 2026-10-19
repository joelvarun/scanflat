"""Line and quadratic Bezier primitives, exact moments, grid-critical parameters.

A segment is a tuple: ("L", p0, p1) or ("Q", p0, p1, p2) with points as
(x, y) float pairs. Quadratics are what the TrueType glyf table stores.
"""
import math

EPS = 1e-12
# two-point Gauss-Legendre nodes on [0, 1]; exact for polynomials up to degree 3
_G = 0.5 / math.sqrt(3.0)
_NODES = (0.5 - _G, 0.5 + _G)


def eval_seg(s, t):
    if s[0] == "L":
        (ax, ay), (bx, by) = s[1], s[2]
        return (ax + t * (bx - ax), ay + t * (by - ay))
    (ax, ay), (cx, cy), (bx, by) = s[1], s[2], s[3]
    u = 1.0 - t
    return (u * u * ax + 2.0 * t * u * cx + t * t * bx,
            u * u * ay + 2.0 * t * u * cy + t * t * by)


def split_quad(s, t):
    """de Casteljau: split a quadratic at t into two quadratics."""
    (ax, ay), (cx, cy), (bx, by) = s[1], s[2], s[3]
    p01 = (ax + t * (cx - ax), ay + t * (cy - ay))
    p12 = (cx + t * (bx - cx), cy + t * (by - cy))
    mid = (p01[0] + t * (p12[0] - p01[0]), p01[1] + t * (p12[1] - p01[1]))
    return ("Q", s[1], p01, mid), ("Q", mid, p12, s[3])


def sub_seg(s, t0, t1):
    """The piece of s over the parameter interval [t0, t1], same degree."""
    if s[0] == "L":
        return ("L", eval_seg(s, t0), eval_seg(s, t1))
    if t1 < 1.0:
        s = split_quad(s, t1)[0]
        t0 = t0 / t1 if t1 > 0.0 else 0.0
    if t0 > 0.0:
        s = split_quad(s, t0)[1]
    return s


def moments(s):
    """Return (dy, xdy) where dy = y1 - y0 and xdy = integral of x dy.

    Both are exact: the integrand x(t) y'(t) is cubic in t for a quadratic
    Bezier, and two-point Gauss-Legendre is exact to degree three.
    """
    if s[0] == "L":
        (ax, ay), (bx, by) = s[1], s[2]
        return by - ay, 0.5 * (ax + bx) * (by - ay)
    (ax, ay), (cx, cy), (bx, by) = s[1], s[2], s[3]
    total = 0.0
    for t in _NODES:
        u = 1.0 - t
        x = u * u * ax + 2.0 * t * u * cx + t * t * bx
        dy = 2.0 * (u * (cy - ay) + t * (by - cy))
        total += 0.5 * x * dy
    return by - ay, total


def extremum_t(s, axis):
    """Parameter of the single interior extremum in the given axis, or None."""
    if s[0] == "L":
        return None
    p0, p1, p2 = s[1][axis], s[2][axis], s[3][axis]
    a = p0 - 2.0 * p1 + p2
    if abs(a) < EPS:
        return None
    t = (p0 - p1) / a
    return t if EPS < t < 1.0 - EPS else None


def integer_crossings(s, axis):
    """Parameters where coordinate `axis` crosses an integer (pixel grid line)."""
    out = []
    if s[0] == "L":
        a, b = s[1][axis], s[2][axis]
        if a == b:
            return out
        for k in range(int(math.floor(min(a, b))) + 1, int(math.ceil(max(a, b)))):
            out.append((k - a) / (b - a))
        return [t for t in out if EPS < t < 1.0 - EPS]
    p0, p1, p2 = s[1][axis], s[2][axis], s[3][axis]
    a = p0 - 2.0 * p1 + p2
    b = 2.0 * (p1 - p0)
    lo = min(p0, p2)
    hi = max(p0, p2)
    te = extremum_t(s, axis)
    if te is not None:
        ve = eval_seg(s, te)[axis]
        lo = min(lo, ve)
        hi = max(hi, ve)
    for k in range(int(math.floor(lo)) + 1, int(math.ceil(hi))):
        c = p0 - k
        if abs(a) < EPS:
            if abs(b) > EPS:
                out.append(-c / b)
        else:
            disc = b * b - 4.0 * a * c
            if disc >= 0.0:
                r = math.sqrt(disc)
                out.append((-b + r) / (2.0 * a))
                out.append((-b - r) / (2.0 * a))
    return sorted(t for t in out if EPS < t < 1.0 - EPS)


def cell_pieces(s, axes=(0, 1)):
    """Split s so every piece lies inside one pixel cell.

    Splitting at the integer crossings of both axes plus the extremum in each
    axis makes every piece monotone in both coordinates and therefore confined
    to a single cell of the unit grid.
    """
    ts = set()
    for axis in axes:
        te = extremum_t(s, axis)
        if te is not None:
            ts.add(te)
        ts.update(integer_crossings(s, axis))
    bounds = [0.0] + sorted(ts) + [1.0]
    pieces = []
    for i in range(len(bounds) - 1):
        t0, t1 = bounds[i], bounds[i + 1]
        if t1 - t0 < 1e-11:
            continue
        pieces.append(sub_seg(s, t0, t1))
    return pieces


def cell_of(s):
    """Integer cell (column, row) holding the piece, from its chord midpoint."""
    mx = 0.5 * (s[1][0] + s[-1][0])
    my = 0.5 * (s[1][1] + s[-1][1])
    return int(math.floor(mx)), int(math.floor(my))
