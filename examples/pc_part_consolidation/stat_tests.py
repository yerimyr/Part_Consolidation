from __future__ import annotations

import argparse
import math
from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy.stats import friedmanchisquare
from scipy.stats import wilcoxon


DEFAULT_METHODS = ["CPCCD", "GA", "NCO"]
DEFAULT_METRICS = ["score", "groups", "time"]


def holm_correction(p_values: list[float]) -> list[float]:
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m

    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = (m - rank) * p
        running_max = max(running_max, adj)
        adjusted[orig_idx] = min(running_max, 1.0)
    return adjusted


def rank_biserial_from_wilcoxon(x: pd.Series, y: pd.Series) -> float:
    diffs = (x - y).astype(float)
    diffs = diffs[diffs != 0.0]
    n = len(diffs)
    if n == 0:
        return 0.0

    abs_ranks = diffs.abs().rank(method="average")
    pos_rank_sum = float(abs_ranks[diffs > 0].sum())
    neg_rank_sum = float(abs_ranks[diffs < 0].sum())
    total_rank_sum = n * (n + 1) / 2.0
    if total_rank_sum == 0:
        return 0.0
    return (pos_rank_sum - neg_rank_sum) / total_rank_sum


def describe_metric(pivot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in pivot.columns:
        values = pivot[method].astype(float)
        rows.append(
            {
                "method": method,
                "mean": values.mean(),
                "std": values.std(ddof=1),
                "median": values.median(),
                "q1": values.quantile(0.25),
                "q3": values.quantile(0.75),
                "min": values.min(),
                "max": values.max(),
                "n": len(values),
            }
        )
    return pd.DataFrame(rows)


def build_pivot(df: pd.DataFrame, metric: str, methods: list[str]) -> pd.DataFrame:
    pivot = df.pivot(index="instance_id", columns="method", values=metric)
    pivot = pivot[methods].dropna()
    if len(pivot) == 0:
        raise ValueError(f"No complete paired rows found for metric '{metric}' and methods {methods}.")
    return pivot


def save_metric_boxplot(pivot: pd.DataFrame, metric: str, output_path: Path) -> bool:
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        print(f"Skip boxplot for '{metric}': {exc}")
        return False

    matplotlib.use("Agg")
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        print(f"Skip boxplot for '{metric}': {exc}")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)

    methods = list(pivot.columns)
    data = [pivot[m].astype(float).values for m in methods]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, patch_artist=True, labels=methods)
    colors = ["#9ecae1", "#fdae6b", "#a1d99b"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    means = [float(pivot[m].astype(float).mean()) for m in methods]
    ax.scatter(range(1, len(methods) + 1), means, color="#d62728", marker="D", s=36, label="Mean")
    ax.set_title(f"{metric.capitalize()} Distribution by Method")
    ax.set_xlabel("Method")
    ax.set_ylabel(metric.capitalize())
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def run_friedman_and_posthoc(df: pd.DataFrame, metric: str, methods: list[str]) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pivot = build_pivot(df, metric, methods)

    arrays = [pivot[m].astype(float).values for m in methods]
    friedman_stat, friedman_p = friedmanchisquare(*arrays)
    kendalls_w = friedman_stat / (len(pivot) * (len(methods) - 1))

    pair_rows = []
    raw_ps = []
    pair_specs = list(combinations(methods, 2))
    for left, right in pair_specs:
        x = pivot[left].astype(float)
        y = pivot[right].astype(float)
        diff = x - y
        if bool((diff == 0.0).all()):
            stat = 0.0
            p = 1.0
        else:
            stat, p = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided", mode="auto")
        pair_rows.append(
            {
                "comparison": f"{left} vs {right}",
                "mean_diff": diff.mean(),
                "median_diff": diff.median(),
                "wilcoxon_stat": float(stat),
                "p_value": float(p),
                "rank_biserial": rank_biserial_from_wilcoxon(x, y),
            }
        )
        raw_ps.append(float(p))

    adjusted = holm_correction(raw_ps)
    for row, adj_p in zip(pair_rows, adjusted):
        row["holm_p_value"] = adj_p

    overall = {
        "metric": metric,
        "n_instances": int(len(pivot)),
        "friedman_stat": float(friedman_stat),
        "friedman_p_value": float(friedman_p),
        "kendalls_w": float(kendalls_w),
    }
    return overall, describe_metric(pivot), pd.DataFrame(pair_rows), pivot


def print_block(title: str, df: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Run Friedman and post-hoc Wilcoxon tests on PC experiment CSV results.")
    parser.add_argument("--csv", type=str, default="generalization_results.csv")
    parser.add_argument("--instance-type", type=str, default="generalization")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--output-dir", type=str, default="stat_test_results")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df[df["instance_type"] == args.instance_type].copy()
    if len(df) == 0:
        raise ValueError(f"No rows found for instance_type='{args.instance_type}' in {csv_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input CSV: {csv_path}")
    print(f"Instance type: {args.instance_type}")
    print(f"Methods: {args.methods}")
    print(f"Metrics: {args.metrics}")
    print(f"Rows: {len(df)}")

    summary_rows = []
    for metric in args.metrics:
        overall, descriptives, pairwise, pivot = run_friedman_and_posthoc(df, metric, args.methods)

        print(f"\n\n##### METRIC: {metric} #####")
        print(
            "Friedman: "
            f"stat={overall['friedman_stat']:.6f}, "
            f"p={overall['friedman_p_value']:.6g}, "
            f"Kendall_W={overall['kendalls_w']:.6f}, "
            f"n={overall['n_instances']}"
        )
        print_block(f"{metric} descriptives", descriptives)
        print_block(f"{metric} pairwise Wilcoxon + Holm", pairwise)

        summary_rows.append(overall)
        descriptives.to_csv(output_dir / f"{metric}_descriptives.csv", index=False)
        pairwise.to_csv(output_dir / f"{metric}_pairwise.csv", index=False)
        plot_path = output_dir / f"{metric}_boxplot.png"
        if save_metric_boxplot(pivot, metric, plot_path):
            print(f"Saved boxplot: {plot_path.resolve()}")

    pd.DataFrame(summary_rows).to_csv(output_dir / "friedman_summary.csv", index=False)
    print(f"\nSaved results to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
