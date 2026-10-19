"""Build docs/before_after.png from docs/metrics.json."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK = "#1b1b1b"
BASE = "#c2452d"
BASE2 = "#e08a2e"
OURS = "#1f6f8b"
GREY = "#8a8a8a"
EIGHT_BIT = 1.0 / 512.0


def panel_accuracy(ax, d):
    for rule, colour, label in (("uniform", BASE, "uniform split count rule"),
                                ("recursive", BASE2, "recursive halving rule")):
        curve = d[rule + "_curve"]
        xs = [c["segs"] for c in curve]
        ys = [c["maxerr"] for c in curve]
        ax.plot(xs, ys, "o-", color=colour, ms=3.5, lw=1.4, label=label)
    cells = d["ablation_cells"]
    ax.plot([cells["segs"]], [cells["maxerr"]], "s", color=GREY, ms=7,
            label="ablation: cell-aligned splits only")
    ex = d["exactness"]
    ax.plot([ex["segs"]], [max(ex["worst_maxerr"], 1e-15)], "*", color=OURS, ms=18,
            zorder=5)
    ax.axhline(EIGHT_BIT, color=INK, ls=":", lw=1.1)
    ax.text(0.02, EIGHT_BIT * 1.5, "half an 8 bit alpha step", ha="left",
            va="bottom", fontsize=7.5, color=INK, transform=ax.get_yaxis_transform())
    floor = max(ex["worst_maxerr"], 1e-15)
    ax.annotate("scanflat: exact to float precision",
                xy=(ex["segs"], floor), xytext=(ex["segs"] * 1.35, floor * 60),
                fontsize=8.5, color=OURS, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=OURS, lw=0.9))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("polyline segments per glyph")
    ax.set_ylabel("worst per-pixel coverage error")
    ax.set_title("Accuracy against flattening cost", fontsize=10)
    ax.set_ylim(floor * 0.12, 3.0)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.95)
    ax.grid(alpha=0.25, which="both", lw=0.4)


def panel_segments(ax, d):
    ppems = list(d["uniform_exact8"]["by_ppem"].keys())
    x = np.arange(len(ppems))
    uni = [d["uniform_exact8"]["by_ppem"][p]["baseline"] for p in ppems]
    rec = [d["recursive_exact8"]["by_ppem"][p]["baseline"] for p in ppems]
    mom = [d["uniform_exact8"]["by_ppem"][p]["moment"] for p in ppems]
    w = 0.27
    ax.bar(x - w, uni, w, color=BASE, label="uniform split count rule")
    ax.bar(x, rec, w, color=BASE2, label="recursive halving rule")
    ax.bar(x + w, mom, w, color=OURS, label="scanflat")
    for xi, (a, b) in enumerate(zip(uni, mom)):
        ax.text(xi + w, b * 1.08, "%.0fx" % (a / b), ha="center", fontsize=8,
                color=OURS, fontweight="bold")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([("%g" % float(p)) for p in ppems])
    ax.set_xlabel("pixels per em")
    ax.set_ylabel("segments per glyph")
    ax.set_title("Cost of 8 bit exact coverage", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.95)
    ax.grid(alpha=0.25, axis="y", which="both", lw=0.4)


def panel_agg(ax, d):
    m = d["agg"]["mean"]
    labels = ["Agg flattens\nthe curve itself", "Agg fed the\nmoment polyline",
              "Agg fed a very fine\npolyline (its floor)"]
    vals = [m["curve"]["maxerr"], m["moment"]["maxerr"], m["fine"]["maxerr"]]
    segs = [None, m["moment"]["segs"], m["fine"]["segs"]]
    cols = [BASE, OURS, GREY]
    bars = ax.bar(np.arange(3), vals, 0.6, color=cols)
    for i, (b, v) in enumerate(zip(bars, vals)):
        txt = "%.3f" % v
        if segs[i] is not None:
            txt += "\n%.0f segments" % segs[i]
        ax.text(b.get_x() + b.get_width() / 2, v * 1.12, txt, ha="center",
                fontsize=8, color=INK)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("worst per-pixel coverage error")
    ax.set_title("Through a shipped rasterizer (matplotlib Agg)", fontsize=10)
    ax.set_ylim(min(vals) * 0.35, max(vals) * 3.2)
    ax.grid(alpha=0.25, axis="y", which="both", lw=0.4)


def main(path="docs/metrics.json", out="docs/before_after.png"):
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    panel_accuracy(axes[0], d)
    panel_segments(axes[1], d)
    panel_agg(axes[2], d)
    c = d["corpus"]
    fig.suptitle("scanflat: moment matched flattening makes anti-aliased glyph "
                 "coverage exact, not merely close", fontsize=12.5, y=0.99)
    fig.text(0.5, 0.005,
             "%d (font, glyph, size) cases from %d installed TrueType fonts, "
             "%s at %s pixels per em, random subpixel offsets"
             % (c["cases"], c["fonts"], c["glyphs"],
                "/".join("%g" % p for p in c["ppems"])),
             ha="center", fontsize=8, color=GREY)
    fig.tight_layout(rect=(0, 0.025, 1, 0.955))
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
