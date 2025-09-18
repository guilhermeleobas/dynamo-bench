import pandas as pd
import rich
import statistics
from dataclasses import dataclass
from rich.table import Table


@dataclass(frozen=True)
class SummaryStats:
    n_tests: int
    n_faster: int
    n_slower: int
    n_same: int
    geomean: float
    best_speedup: float
    best_speedup_test: str
    best_speedup_test_file: str
    worst_slowdown: float
    worst_slowdown_test: str
    worst_slowdown_test_file: str
    geomean_str: str


def build_metadata_table(metadata1, metadata2):
    tab = Table(title="Metadata Comparison")
    tab.add_column("Field", style="cyan", no_wrap=True)
    tab.add_column("Baseline", style="white")
    tab.add_column("New", style="white")

    def add_row(baseline, new, key):
        color = "green" if baseline == new else "red"
        tab.add_row(key, str(baseline), f"[{color}]{new}[/{color}]")

    def walk(d):
        for k, v in d.items():
            if isinstance(v, dict):
                yield from walk(v)
            else:
                yield k, v

    baseline = dict(walk(metadata1))
    new = dict(walk(metadata2))

    all_keys = set(baseline.keys()).union(set(new.keys()))
    all_keys.discard("os_version")
    for key in sorted(all_keys):
        val1 = baseline.get(key, "N/A")
        val2 = new.get(key, "N/A")
        add_row(val1, val2, key)
    return tab


def build_compare_table(df):
    tab = Table(title="Comparison Results")
    tab.add_column("Benchmark", style="cyan", no_wrap=True)
    tab.add_column("Baseline Time", style="white")
    tab.add_column("New Time", style="white")
    tab.add_column("Speedup/Slowdown", justify="right", style="white")

    idx = 0
    df = pd.concat([df.head(10), df.tail(10)]) if len(df) > 20 else df
    for entry in df.itertuples():
        benchmark = entry.test
        baseline_time = f"{entry.dynamo_time_mean_1:.4f}s"
        new_time = f"{entry.dynamo_time_mean_2:.4f}s"
        color = "green" if entry.faster else "red"
        ratio = f"[{color}]{entry.mean_ratio:.2f}x [/{color}]"
        tab.add_row(benchmark, baseline_time, new_time, ratio)
        idx += 1
        if idx == 10:
            tab.add_row("...", "...", "...", "...")
    return tab


def merge_dataframes(df1, df2, verbose=False):
    # Merge on test_file and test columns
    merged = pd.merge(df1, df2, on=["test_file", "test"], suffixes=("_1", "_2"))

    # Calculate ratios
    merged["mean_ratio"] = merged["dynamo_time_mean_2"] / merged["dynamo_time_mean_1"]
    merged["faster"] = merged["mean_ratio"] < 1.0

    if verbose:
        rich.print(
            merged[
                [
                    "test_file",
                    "test",
                    "dynamo_time_mean_1",
                    "dynamo_time_mean_2",
                    "mean_ratio",
                    "faster",
                ]
            ]
        )

    merged = merged[merged.mean_ratio.isna() == False]
    merged.sort_values(by="mean_ratio", ascending=True, inplace=True)
    return merged


def compute_statistics(df) -> SummaryStats:
    geomean = statistics.geometric_mean(df["mean_ratio"].dropna())
    slow_msg = "Tests are slower on average"
    fast_msg = "Tests are faster on average"
    no_change_msg = "No overall change in speed"
    geomean_str = (
        fast_msg if geomean < 1.0 else slow_msg if geomean > 1.0 else no_change_msg
    )

    stats = SummaryStats(
        n_tests=len(df),
        n_faster=(df["mean_ratio"] < 1.0).sum(),
        n_slower=(df["mean_ratio"] > 1.0).sum(),
        n_same=(df["mean_ratio"] == 1.0).sum(),
        geomean=geomean,
        best_speedup=df.mean_ratio.min(),
        best_speedup_test=df.loc[df.mean_ratio.idxmin()]["test"],
        best_speedup_test_file=df.loc[df.mean_ratio.idxmin()]["test_file"],
        worst_slowdown=df.mean_ratio.max(),
        worst_slowdown_test=df.loc[df.mean_ratio.idxmax()]["test"],
        worst_slowdown_test_file=df.loc[df.mean_ratio.idxmax()]["test_file"],
        geomean_str=geomean_str,
    )
    return stats


def print_summary(compare_table, metadata_table, stats: SummaryStats):
    rich.print(compare_table)
    rich.print(metadata_table)

    rich.print(
        f"""
Summary:
    {stats.n_tests} benchmarks run
    {stats.n_faster} faster, {stats.n_slower} slower, {stats.n_same} same
    Geomean ratio = {stats.geomean:.2f}x
    Best speedup = {stats.best_speedup:.2f}x ({stats.best_speedup_test_file} {stats.best_speedup_test})
    Worst slowdown = {stats.worst_slowdown:.2f}x ({stats.worst_slowdown_test_file} {stats.worst_slowdown_test})

    {stats.geomean_str}
"""
    )


def compare_dataframes(df1, df2, meta1: dict, meta2: dict, verbose=False):
    merged = merge_dataframes(df1, df2, verbose=verbose)
    stats = compute_statistics(merged)
    compare_table = build_compare_table(merged)

    metadata_table = build_metadata_table(meta1, meta2)
    print_summary(compare_table, metadata_table, stats)
