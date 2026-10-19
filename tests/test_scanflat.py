"""Correctness gates. Run with: python -m unittest discover -s tests -v"""
import math
import os
import random
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scanflat.flatten import (cell_crossings, chord_deviation, flatten_cells,
                              flatten_moment, flatten_recursive, flatten_uniform,
                              segment_count)
from scanflat.geom import cell_pieces, eval_seg, moments, sub_seg
from scanflat.raster import (alpha8, coverage, coverage_supersample, fill,
                             pixel_box)


def random_blob(rng, n=7, radius=4.0, cx=5.3, cy=5.7):
    """A closed contour of quadratics with a random star-shaped outline."""
    pts = []
    for k in range(n):
        a = 2.0 * math.pi * k / n
        r = radius * (0.55 + 0.45 * rng.random())
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    contour = []
    for k in range(n):
        p0 = pts[k]
        p2 = pts[(k + 1) % n]
        mid = (0.5 * (p0[0] + p2[0]), 0.5 * (p0[1] + p2[1]))
        ctrl = (mid[0] + 1.6 * (rng.random() - 0.5), mid[1] + 1.6 * (rng.random() - 0.5))
        on0 = (0.5 * (p0[0] + mid[0]), 0.5 * (p0[1] + mid[1]))
        contour.append(("Q", on0 if False else p0, ctrl, p2))
    return [contour]


class TestMoments(unittest.TestCase):
    def test_gauss_is_exact(self):
        """Two-point Gauss matches a fine Riemann sum for x dy on quadratics."""
        rng = random.Random(1)
        for _ in range(200):
            s = ("Q", (rng.uniform(-3, 3), rng.uniform(-3, 3)),
                 (rng.uniform(-3, 3), rng.uniform(-3, 3)),
                 (rng.uniform(-3, 3), rng.uniform(-3, 3)))
            dy, xdy = moments(s)
            n = 20000
            t = (np.arange(n) + 0.5) / n
            (ax, ay), (cx, cy), (bx, by) = s[1], s[2], s[3]
            u = 1.0 - t
            x = u * u * ax + 2 * t * u * cx + t * t * bx
            dydt = 2.0 * (u * (cy - ay) + t * (by - cy))
            self.assertAlmostEqual(xdy, float((x * dydt).sum() / n), places=7)
            self.assertAlmostEqual(dy, by - ay, places=12)

    def test_subdivision_preserves_moments(self):
        rng = random.Random(2)
        for _ in range(100):
            s = ("Q", (rng.uniform(0, 9), rng.uniform(0, 9)),
                 (rng.uniform(0, 9), rng.uniform(0, 9)),
                 (rng.uniform(0, 9), rng.uniform(0, 9)))
            whole = moments(s)
            parts = [moments(p) for p in cell_pieces(s)]
            self.assertAlmostEqual(whole[0], sum(p[0] for p in parts), places=10)
            self.assertAlmostEqual(whole[1], sum(p[1] for p in parts), places=9)


class TestCellPieces(unittest.TestCase):
    def test_every_piece_lies_in_one_cell(self):
        rng = random.Random(3)
        for _ in range(300):
            s = ("Q", (rng.uniform(0, 7), rng.uniform(0, 7)),
                 (rng.uniform(-2, 9), rng.uniform(-2, 9)),
                 (rng.uniform(0, 7), rng.uniform(0, 7)))
            for piece in cell_pieces(s):
                xs, ys = [], []
                for k in range(21):
                    px, py = eval_seg(piece, k / 20.0)
                    xs.append(px)
                    ys.append(py)
                self.assertLessEqual(math.floor(max(xs) - 1e-9) - math.floor(min(xs) + 1e-9), 0)
                self.assertLessEqual(math.floor(max(ys) - 1e-9) - math.floor(min(ys) + 1e-9), 0)


class TestRasterizer(unittest.TestCase):
    def test_square_is_fully_covered(self):
        sq = [[("L", (2.0, 3.0), (2.0, 7.0)), ("L", (2.0, 7.0), (6.0, 7.0)),
               ("L", (6.0, 7.0), (6.0, 3.0)), ("L", (6.0, 3.0), (2.0, 3.0))]]
        cov = fill(coverage(sq, (0, 0, 9, 10)))
        self.assertAlmostEqual(float(cov.sum()), 16.0, places=10)
        self.assertAlmostEqual(float(cov[3:7, 2:6].min()), 1.0, places=10)
        self.assertAlmostEqual(float(cov[0, 0]), 0.0, places=10)

    def test_total_coverage_equals_green_area(self):
        rng = random.Random(4)
        for _ in range(25):
            blob = random_blob(rng)
            area = sum(moments(s)[1] for c in blob for s in c)
            cov = fill(coverage(blob, pixel_box(blob)))
            self.assertAlmostEqual(float(cov.sum()), abs(area), places=9)

    def test_supersampler_converges_to_exact(self):
        rng = random.Random(5)
        blob = random_blob(rng)
        box = pixel_box(blob)
        exact = fill(coverage(blob, box))
        prev = None
        for n in (16, 64, 256):
            err = float(np.abs(coverage_supersample(blob, box, n) - exact).max())
            if prev is not None:
                self.assertLess(err, prev)
            prev = err
        self.assertLess(prev, 5e-3)


class TestFlatteners(unittest.TestCase):
    def test_uniform_rule_respects_its_tolerance(self):
        rng = random.Random(6)
        for _ in range(60):
            s = ("Q", (rng.uniform(0, 9), rng.uniform(0, 9)),
                 (rng.uniform(0, 9), rng.uniform(0, 9)),
                 (rng.uniform(0, 9), rng.uniform(0, 9)))
            for tol in (0.2, 0.02, 0.002):
                flat = flatten_uniform([[s]], tol)[0]
                n = len(flat)
                worst = max(chord_deviation(sub_seg(s, i / n, (i + 1) / n))
                            for i in range(n))
                self.assertLessEqual(worst, tol * 1.0001)

    def test_moment_flatten_is_exact_on_blobs(self):
        rng = random.Random(7)
        for _ in range(40):
            blob = random_blob(rng, n=rng.choice([5, 6, 8]), radius=rng.uniform(2.0, 6.0),
                               cx=5.0 + rng.random(), cy=5.0 + rng.random())
            box = pixel_box(blob)
            exact = fill(coverage(blob, box))
            got = fill(coverage(flatten_moment(blob), box))
            self.assertLess(float(np.abs(got - exact).max()), 1e-11)
            self.assertEqual(int(np.count_nonzero(alpha8(got) != alpha8(exact))), 0)

    def test_moment_flatten_beats_the_chord_ablation(self):
        rng = random.Random(8)
        blob = random_blob(rng, n=6, radius=5.0)
        box = pixel_box(blob)
        exact = fill(coverage(blob, box))
        chord = float(np.abs(fill(coverage(flatten_cells(blob), box)) - exact).max())
        moment = float(np.abs(fill(coverage(flatten_moment(blob), box)) - exact).max())
        self.assertGreater(chord, 1e-3)
        self.assertLess(moment, chord * 1e-6)

    def test_output_is_a_closed_polyline(self):
        rng = random.Random(9)
        blob = random_blob(rng)
        for flat in flatten_moment(blob):
            for a, b in zip(flat, flat[1:]):
                self.assertAlmostEqual(a[-1][0], b[1][0], places=12)
                self.assertAlmostEqual(a[-1][1], b[1][1], places=12)
            self.assertAlmostEqual(flat[0][1][0], flat[-1][-1][0], places=12)
            self.assertAlmostEqual(flat[0][1][1], flat[-1][-1][1], places=12)

    def test_recursive_rule_terminates_and_is_monotone(self):
        rng = random.Random(10)
        blob = random_blob(rng)
        box = pixel_box(blob)
        exact = fill(coverage(blob, box))
        last = None
        for tol in (0.5, 0.1, 0.02, 0.004):
            poly = flatten_recursive(blob, tol)
            err = float(np.abs(fill(coverage(poly, box)) - exact).max())
            self.assertGreater(segment_count(poly), 0)
            if last is not None:
                self.assertLessEqual(err, last + 1e-12)
            last = err


class TestFonts(unittest.TestCase):
    def test_exact_on_installed_fonts(self):
        from scanflat.outline import font_files, open_font, scaled_glyph
        files = font_files()
        if not files:
            self.skipTest("no TrueType fonts found")
        checked = 0
        for path in files[:6]:
            try:
                font = open_font(path)
                cmap = font.getBestCmap()
            except Exception:
                continue
            for ch in "aeB2":
                name = cmap.get(ord(ch))
                if name is None:
                    continue
                for ppem in (12.0, 19.0):
                    g = scaled_glyph(font, name, ppem, 0.23, 0.71)
                    if segment_count(g) < 4:
                        continue
                    box = pixel_box(g)
                    exact = fill(coverage(g, box))
                    poly = flatten_moment(g)
                    got = fill(coverage(poly, box))
                    self.assertLess(float(np.abs(got - exact).max()), 1e-10)
                    self.assertEqual(
                        int(np.count_nonzero(alpha8(got) != alpha8(exact))), 0)
                    self.assertEqual(segment_count(poly), cell_crossings(poly))
                    checked += 1
        self.assertGreater(checked, 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
