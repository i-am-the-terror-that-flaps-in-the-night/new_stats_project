import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from analysis.paths import ROOT

OUTPUT = ROOT / "output" / "figure_B12.svg"

# A diagram of the biology, not a result: nothing here comes from the data.
# Each box: centre (x, y), width, height, text, what measured it (None if we
# didn't), and fill colour. The main chain runs left to right and darkens
# toward the liver; the end point is orange.
EDGE, DARK_BLUE, GREY = "#1f4e79", "#1f4e79", "#444444"
BOXES = {
    "sugar": ((1.2, 3.6), 2.0, 1.4, "Sugar eaten\n(half of table sugar\nis fructose)", "diet recall", "#deebf7"),
    "liver": ((3.7, 3.6), 2.0, 1.4, "Fructose goes\nto the liver", None, "#c6dbef"),
    "dnl": ((6.2, 3.6), 2.0, 1.4, "Liver turns extra\nfructose into fat", None, "#9ecae1"),
    "fat": ((8.9, 3.6), 2.0, 1.4, "Fat builds up\nin the liver", "FibroScan CAP", "#f4b183"),
    "tg": ((6.2, 5.35), 2.4, 0.75, "Some of the fat leaves\nas triglycerides", None, "#f2f2f2"),
    "uric": ((4.9, 1.45), 1.9, 1.05, "Uric acid is made\nalong the way", "blood test", "#f2f2f2"),
    "insulin": ((7.3, 1.45), 1.9, 1.05, "Insulin resistance\nmakes it worse", "HOMA-IR", "#f2f2f2"),
    "belly": ((9.65, 1.45), 1.8, 1.05, "Belly fat sends fatty\nacids to the liver", "DXA scan", "#f2f2f2"),
}

# Arrows: from (x, y) to (x, y)
ARROWS = [
    ((2.2, 3.6), (2.7, 3.6)),  # sugar -> liver
    ((4.7, 3.6), (5.2, 3.6)),  # liver -> fat made
    ((7.2, 3.6), (7.9, 3.6)),  # fat made -> fat builds up
    ((6.2, 4.3), (6.2, 4.97)),  # fat made -> triglycerides
    ((5.75, 2.9), (5.25, 1.98)),  # fat made -> uric acid
    ((7.45, 1.98), (7.55, 3.64)),  # insulin resistance -> the build-up step
    ((9.5, 1.98), (9.3, 2.9)),  # belly fat -> liver fat
]


def box(ax, centre, width, height, text, measured, colour):
    (x, y) = centre
    ax.add_patch(FancyBboxPatch((x - width / 2, y - height / 2), width, height,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=colour, edgecolor=EDGE,
                                linewidth=2.2 if measured else 1.2))
    if measured:
        # Measured boxes get a thick edge and a line saying how
        ax.text(x, y + 0.12, text, ha="center", va="center", fontsize=9.5)
        ax.text(x, y - height / 2 + 0.13, f"measured: {measured}", ha="center", va="bottom",
                fontsize=8, style="italic", color=DARK_BLUE)
    else:
        ax.text(x, y, text, ha="center", va="center", fontsize=9.5)


def figure_b12():
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=200)
    for centre, width, height, text, measured, colour in BOXES.values():
        box(ax, centre, width, height, text, measured, colour)
    for start, end in ARROWS:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                                     color=DARK_BLUE, linewidth=1.8))

    ax.text(5.35, 0.45, "We measured the start (sugar), the middle (insulin resistance, belly fat, "
            "uric acid) and the end (liver fat).", ha="center", fontsize=10, weight="bold", color=DARK_BLUE)
    ax.text(5.35, 0.02, "Thick edge = measured in this study. A diagram of the proposed biology, "
            "not a result.", ha="center", fontsize=9, style="italic", color=GREY)
    ax.set_xlim(0, 10.7)
    ax.set_ylim(-0.2, 5.9)
    ax.axis("off")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    fig = figure_b12()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT)
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")
