from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


PART_COUNTS = [4, 5, 6, 7, 8, 9, 10]
BELL_NUMBERS = [15, 52, 203, 877, 4140, 21147, 115975]
FEASIBLE_SOLUTIONS = [12, 32.8, 99.2, 397.6, 1521.2, 7214, 37845.8]


def main() -> None:
    output_path = Path(__file__).with_name("bell_vs_feasible.png")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        PART_COUNTS,
        BELL_NUMBERS,
        marker="o",
        linewidth=2,
        label="Bell number",
    )
    ax.plot(
        PART_COUNTS,
        FEASIBLE_SOLUTIONS,
        marker="s",
        linewidth=2,
        label="Feasible solutions",
    )

    ax.set_title("Bell Number vs Feasible Solution Count")
    ax.set_xlabel("Number of parts")
    ax.set_ylabel("Count")
    ax.set_xticks(PART_COUNTS)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # If you want to compare the curves more clearly despite the large scale gap,
    # uncomment the next line.
    # ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
