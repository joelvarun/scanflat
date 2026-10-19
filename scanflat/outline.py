"""TrueType outline extraction.

The corpus is the fonts already installed on the machine.  Only the `glyf`
table is used, so every outline is a chain of lines and quadratic Beziers,
which is what `geom.py` handles.  No hinting is applied: the point of the
measurement is the flattening step, and hinting would move the outline before
the flattener ever sees it.
"""
import glob
import os

from fontTools.ttLib import TTFont


def font_files(pattern="C:/Windows/Fonts/*.ttf"):
    return sorted(glob.glob(pattern))


def open_font(path):
    return TTFont(path, fontNumber=0, lazy=True)


def glyph_names(font, text):
    cmap = font.getBestCmap()
    return [(ch, cmap.get(ord(ch))) for ch in text]


def load_contours(font, name):
    """Contours in font units: each a list of ("L", p0, p1) / ("Q", p0, p1, p2).

    Implied on-curve midpoints between consecutive control points are expanded
    here, the same reconstruction a rasterizer does when reading `glyf`.
    """
    from fontTools.pens.recordingPen import RecordingPen

    pen = RecordingPen()
    font.getGlyphSet()[name].draw(pen)
    contours, current, start, last = [], [], None, None
    for op, args in pen.value:
        if op == "moveTo":
            if current:
                contours.append(current)
            current = []
            start = last = args[0]
        elif op == "lineTo":
            current.append(("L", last, args[0]))
            last = args[0]
        elif op == "qCurveTo":
            pts = list(args)
            if pts and pts[-1] is None:
                pts = pts[:-1]
                mid = (0.5 * (pts[0][0] + pts[-1][0]), 0.5 * (pts[0][1] + pts[-1][1]))
                start = last = mid
            on, offs = pts[-1], pts[:-1]
            for i, ctrl in enumerate(offs):
                if i == len(offs) - 1:
                    nxt = on
                else:
                    nxt = (0.5 * (ctrl[0] + offs[i + 1][0]),
                           0.5 * (ctrl[1] + offs[i + 1][1]))
                current.append(("Q", last, ctrl, nxt))
                last = nxt
        elif op == "curveTo":
            raise ValueError("cubic outline (CFF), not a glyf font")
        elif op == "closePath":
            if last != start:
                current.append(("L", last, start))
            contours.append(current)
            current = []
    if current:
        contours.append(current)
    return [c for c in contours if c]


def transform(contours, scale, dx=0.0, dy=0.0):
    out = []
    for contour in contours:
        out.append([(seg[0],) + tuple((p[0] * scale + dx, p[1] * scale + dy)
                                      for p in seg[1:]) for seg in contour])
    return out


def scaled_glyph(font, name, ppem, dx=0.0, dy=0.0):
    """Outline placed on the pixel grid at `ppem` pixels per em."""
    upem = font["head"].unitsPerEm
    return transform(load_contours(font, name), ppem / float(upem), dx, dy)


def basename(path):
    return os.path.basename(path)
