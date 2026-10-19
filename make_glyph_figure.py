"""Render one glyph three ways and show where the coverage error lands."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scanflat.flatten import (flatten_cells, flatten_moment, flatten_uniform,
                              segment_count)
from scanflat.outline import open_font, scaled_glyph
from scanflat.raster import coverage, fill, pixel_box


def main(font_path="C:/Windows/Fonts/georgia.ttf", char="a", ppem=18.0,
         out="docs/glyph_error.png"):
    font = open_font(font_path)
    name = font.getBestCmap()[ord(char)]
    g = scaled_glyph(font, name, float(ppem), 0.37, 0.61)
    box = pixel_box(g)
    ref = fill(coverage(g, box))
    moment = flatten_moment(g)
    budget = segment_count(moment)
    # give the tolerance rule the same segment budget
    tol, best = 0.5, None
    for k in range(1, 20):
        cand = 2.0 ** -k
        n = segment_count(flatten_uniform(g, cand))
        if best is None or abs(n - budget) < abs(best - budget):
            best, tol = n, cand
    variants = [("uniform rule, %d segments" % best, fill(coverage(flatten_uniform(g, tol), box))),
                ("cell aligned chords, %d segments" % segment_count(flatten_cells(g)),
                 fill(coverage(flatten_cells(g), box))),
                ("scanflat, %d segments" % budget, fill(coverage(moment, box)))]
    # scale the colour map to the matched-budget baseline, which is the
    # comparison that matters; the ablation panel saturates and says so
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.8))
    axes[0].imshow(ref[::-1], cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title("exact coverage of the curve", fontsize=9.5)
    scale = float(np.abs(variants[0][1] - ref).max())
    for ax, (label, cov) in zip(axes[1:], variants):
        err = (cov - ref)[::-1]
        im = ax.imshow(err, cmap="RdBu_r", vmin=-scale, vmax=scale, interpolation="nearest")
        sat = " (colour scale saturates)" if np.abs(err).max() > scale * 1.05 else ""
        ax.set_title("%s\nworst error %.1e%s"
                     % (label, np.abs(err).max(), sat), fontsize=9.5)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.01,
                 label="coverage error (signed)")
    fig.suptitle("%s %r at %g pixels per em: coverage error per pixel"
                 % (font_path.split("/")[-1], char, float(ppem)), fontsize=11)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
