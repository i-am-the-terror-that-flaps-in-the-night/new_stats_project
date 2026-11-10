import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from analysis.engine.table_B2 import examined_teens
from analysis.paths import FIGURES_DIR, ROOT
from analysis.utilities.cleaning import data_cutting

OUTPUT = FIGURES_DIR / "figure_A3.svg"

# Medium blue for the steps on the way, navy for the two samples the results use
STEP, SAMPLE, EDGE, GREY = "#2e75b6", "#1f4e79", "#1f4e79", "#444444"
WIDTH, HEIGHT = 2.0, 1.1


def sample_flow(samples=None):
    # Teens left at each step. The screening and fasting samples both come from
    # the scan sample, on separate branches: the fasting sample is cut from the
    # diet sample, not from the screening sample.
    if samples is None:
        samples = data_cutting()
    return {
        "examined": len(examined_teens()),
        "scan": len(samples["scan"]),
        "screening": len(samples["screening"]),
        "diet": len(samples["diet"]),
        "fasting": len(samples["fasting"]),
    }


def box(ax, x, y, text, colour, bold=False):
    ax.add_patch(FancyBboxPatch((x - WIDTH / 2, y - HEIGHT / 2), WIDTH, HEIGHT,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor=colour, edgecolor=EDGE, linewidth=1.2))
    ax.text(x, y, text, ha="center", va="center", fontsize=9, color="white",
            weight="bold" if bold else "normal")


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                                 color="#7f7f7f", linewidth=1.4))


def figure_a3(samples=None):
    n = sample_flow(samples)

    fig, ax = plt.subplots(figsize=(10, 3.4), dpi=200)
    top, bottom = 2.5, 0.95
    box(ax, 1.1, top, f"Teens aged 12–17\nexamined\nn = {n['examined']}", STEP)
    box(ax, 3.6, top, f"Complete, reliable\nFibroScan\nn = {n['scan']}", STEP)
    box(ax, 6.1, top, f"Screening sample\n+ ALT, BMI, waist\nn = {n['screening']}", SAMPLE, bold=True)
    box(ax, 6.1, bottom, f"Diet sample\n+ reliable diet recall\nn = {n['diet']}", STEP)
    box(ax, 8.6, bottom, f"Fasting sample\n+ insulin, glucose\nn = {n['fasting']}", SAMPLE, bold=True)
    arrow(ax, (2.15, top), (2.55, top))
    arrow(ax, (4.65, top), (5.05, top))
    arrow(ax, (4.2, top - HEIGHT / 2 - 0.02), (5.05, bottom + 0.1))
    arrow(ax, (7.15, bottom), (7.55, bottom))

    ax.text(4.85, 0.05,
            f"Dropped: {n['examined'] - n['scan']} without a usable scan (or with hepatitis B or C); "
            f"{n['scan'] - n['screening']} missing ALT, BMI or waist.\nThe fasting sample is smaller "
            f"because only some teens are asked to fast.",
            ha="center", va="center", fontsize=8.5, style="italic", color=GREY)
    ax.set_xlim(0, 9.7)
    ax.set_ylim(-0.35, 3.2)
    ax.axis("off")
    fig.tight_layout()
    return fig, n


if __name__ == "__main__":
    fig, n = figure_a3()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT)
    print(n)
    print(f"\nWrote {OUTPUT.relative_to(ROOT)}")
