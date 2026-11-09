import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from analysis.paths import ROOT

OUTPUT = ROOT / "output" / "figure_A2.svg"

# A diagram of how NHANES works, not a result: nothing here comes from the data.
# Each step: heading, what happens, and a fill colour (blues darkening toward
# the exam, which is orange, as it is where our measurements come from).
EDGE, DARK_BLUE, GREY = "#1f4e79", "#1f4e79", "#444444"
STEPS = [
    ("1. Counties chosen", "The CDC picks\ncounties across\nthe U.S.", "#deebf7"),
    ("2. Households picked", "Neighborhoods and\nhomes are chosen;\nsome groups are\nover-sampled", "#c6dbef"),
    ("3. Home interview", "Questions about\nhealth, diet,\nactivity", "#9ecae1"),
    ("4. Mobile exam center", "Trailers with\nclinics travel\nto each county", "#6baed6"),
    ("5. Exams and tests", "Height, weight, waist\nBlood tests\nFibroScan (liver)\nDXA (body fat)", "#f4b183"),
]

WIDTH, HEIGHT, GAP = 2.3, 1.35, 0.3


def figure_a2():
    fig, ax = plt.subplots(figsize=(10, 2.8), dpi=200)
    for i, (heading, text, colour) in enumerate(STEPS):
        left = i * (WIDTH + GAP)
        ax.add_patch(FancyBboxPatch((left, 0.5), WIDTH, HEIGHT,
                                    boxstyle="round,pad=0.02,rounding_size=0.06",
                                    facecolor=colour, edgecolor=EDGE, linewidth=1.2))
        ax.text(left + WIDTH / 2, 0.5 + HEIGHT - 0.2, heading, ha="center", va="center",
                fontsize=8, weight="bold")
        ax.text(left + WIDTH / 2, 0.5 + (HEIGHT - 0.35) / 2, text, ha="center", va="center", fontsize=8.5)
        if i < len(STEPS) - 1:
            ax.add_patch(FancyArrowPatch((left + WIDTH + 0.04, 0.5 + HEIGHT / 2),
                                         (left + WIDTH + GAP - 0.04, 0.5 + HEIGHT / 2),
                                         arrowstyle="-|>", mutation_scale=12, color=DARK_BLUE, linewidth=1.5))

    total = len(STEPS) * WIDTH + (len(STEPS) - 1) * GAP
    ax.text(total / 2, 0.15, "About 10,000 people every two years. Each person is given a \"weight\" saying "
            "how many Americans they stand for, so results apply to the whole country.",
            ha="center", va="center", fontsize=8.5, style="italic", color=GREY)
    ax.set_xlim(-0.1, total + 0.1)
    ax.set_ylim(0, 1.95)
    ax.axis("off")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    fig = figure_a2()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT)
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")
