#!/usr/bin/env python3
import argparse
import warnings
import itertools
import collections
import statistics
import re
import os
import sys
import subprocess
import unittest
import pathlib
import pandas as pd
import rich
import tempfile

from datetime import date
from rich.table import Table


version = sys.version_info
PY_VER = f"{version.major}_{version.minor}"
PYVER = f"{version.major}{version.minor}"
PATH_TO_PYTORCH = pathlib.Path("../pytorch").resolve()
PATH_TO_DYNAMO_FAILURES = PATH_TO_PYTORCH / "test" / "dynamo_expected_failures"
PATH_TO_CPYTHON_TESTS = (
    PATH_TO_PYTORCH / "test" / "dynamo" / "cpython" / PY_VER
).resolve()


def discover_tests(test_file):
    """Return fully-qualified unittest test names in test_file."""
    loader = unittest.TestLoader()
    suite = loader.discover(
        start_dir=os.path.dirname(test_file) or ".", pattern=os.path.basename(test_file)
    )
    tests = []

    def collect(s):
        for t in s:
            if isinstance(t, unittest.TestSuite):
                collect(t)
            else:
                tests.append(f"{t.__class__.__name__}.{t._testMethodName}")

    collect(suite)
    return tests


def compute_min_max_mean(data: list[float]) -> tuple[float, float, float]:
    if all(r == 0.0 for r in data):
        return 0.0, 0.0, 0.0

    min_ = min(data)
    mean, stddev = (
        statistics.mean(data),
        statistics.stdev(data) if len(data) > 1 else 0.0,
    )
    max_ = max(data)

    cv = stddev / mean
    if cv > 0.03:
        warnings.warn(f"High coefficient of variation detected: {mean=}, {stddev=}, {cv=}")
    return min_, mean, max_


def run_bash_script(script: str, verbose=False, env_extra=None) -> list[float]:
    if verbose:
        print(script)

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sh") as f:
        f.write(script)

    env = os.environ.copy()
    env.update(env_extra or {})

    try:
        out = subprocess.run(
            ["bash", f.name], env=env, check=True, text=True, stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        print(e)
        raise
    matches = re.findall(r"Ran \d+ test.* in ([0-9.]+)s", out.stderr)
    runtimes = [float(m) for m in matches]
    return runtimes


def run_tests(
    test_file: str,
    test_names: list[str],
    *,
    env_extra=None,
    warmup: int,
    runs: int,
    verbose: bool,
) -> dict[str, list[float]]:
    assert isinstance(test_names, list)
    assert isinstance(warmup, int)
    assert isinstance(runs, int)

    test_file_path = os.path.join(PATH_TO_CPYTHON_TESTS, test_file)
    total = len(test_names)
    results = collections.defaultdict(list)

    non_skipped_tests = []

    bash_script = f"""
#!/bin/bash

set -e
"""

    for idx, test_name in enumerate(test_names):
        if idx > 3:
            break

        results[test_name] = [0.0, 0.0, 0.0]
        if skip_test(test_file, test_name):
            print(f"Skipping test {test_name}...")
            continue
        else:
            non_skipped_tests.append(test_name)

        bash_script += f"""
echo "[{idx}/{total}] Running test {test_name}..."
for i in $(seq 1 {warmup + runs}); do
    python {test_file_path} {test_name}
done
"""
    runtimes = run_bash_script(bash_script, verbose=verbose, env_extra=env_extra)
    batched = itertools.batched(runtimes, warmup + runs)
    for test_name, batch in zip(non_skipped_tests, batched):
        assert len(batch) == warmup + runs

        batch = batch[warmup:]  # Skip warmup runs
        min_, mean, max_ = compute_min_max_mean(batch)
        results[test_name] = [min_, mean, max_]
    return results


def run_cpython(*args, **kwargs):
    return run_tests(*args, **kwargs)


def run_dynamo(*args, **kwargs):
    return run_tests(*args, env_extra={"PYTORCH_TEST_WITH_DYNAMO": "1"}, **kwargs)


def skip_test(test_file: str, test_name: str):
    module = os.path.basename(test_file).strip(".py")
    file = PATH_TO_DYNAMO_FAILURES / f"CPython{PYVER}-{module}-{test_name}"
    return file.exists()


def get_commit_hash():
    return subprocess.check_output(
        ["git", "log", "-1", "--pretty=%h"],  # %h = short hash
        cwd=PATH_TO_PYTORCH,
        text=True,
    ).strip()


def main(test_files: list[str], *, save: str | None, compare_with: str | None, **kwargs):
    results = []

    for test_file in test_files:
        abs_path = os.path.join(PATH_TO_CPYTHON_TESTS, test_file)
        tests = discover_tests(abs_path)

        print(f"===== Test {test_file} =====")
        cpython_results = run_cpython(test_file, tests, **kwargs)
        dynamo_results = run_dynamo(test_file, tests, **kwargs)

        assert len(cpython_results) == len(dynamo_results)

        for test in cpython_results.keys():
            base_min, base_mean, base_max = cpython_results[test]
            dynamo_min, dynamo_mean, dynamo_max = dynamo_results[test]
            results.append(
                {
                    "test_file": test_file,
                    "test": test,
                    "base_time_min": base_min,
                    "base_time_mean": base_mean,
                    "base_time_max": base_max,
                    "dynamo_time_min": dynamo_min,
                    "dynamo_time_mean": dynamo_mean,
                    "dynamo_time_max": dynamo_max,
                    "ratio": dynamo_mean / base_mean
                    if base_mean and dynamo_mean
                    else None,
                },
            )

    df = pd.DataFrame(results)

    if save:
        # prepend name by git commit and user date
        commit = get_commit_hash()
        today = date.today()
        d = f"{today.year}-{today.month}-{today.day}"
        name = f"result_{d}_{commit}.csv"
        p = (pathlib.Path("results") / name).resolve()
        df.sort_values(by="ratio", ascending=False).to_csv(p, index=False)

    if compare_with:
        baseline = pd.read_csv(compare_with)
        compare_dataframes(baseline, df, verbose=kwargs.get("verbose", False))


def get_test_files() -> list[str]:
    tests = []
    for _, _, test_files in PATH_TO_CPYTHON_TESTS.walk():
        for test_file in test_files:
            if test_file.startswith("test_") and test_file.endswith(".py"):
                tests.append(test_file)
    return tests


def list_tests() -> None:
    tests = get_test_files()
    for test_file in tests:
        print(test_file)


def compare_files(baseline, file2, verbose=False):
    """
    Compare the execution time of two result CSV files.
    Returns a DataFrame with timing and ratio information.
    """
    df1 = pd.read_csv(baseline)[["test_file", "test", "dynamo_time_mean"]]
    df2 = pd.read_csv(file2)[["test_file", "test", "dynamo_time_mean"]]
    return compare_dataframes(df1, df2, verbose=verbose)


def compare_dataframes(df1, df2, verbose=False):

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

    merged.sort_values(by="mean_ratio", ascending=True, inplace=True)

    n_tests = len(merged)
    n_same = (merged["mean_ratio"] == 1.0).sum()
    n_faster = merged["faster"].sum()
    n_slower = n_tests - n_faster - n_same

    geomean = statistics.geometric_mean(merged["mean_ratio"].dropna())

    best_speedup = merged.mean_ratio.min()
    best_speedup_test = merged.loc[merged.mean_ratio.idxmin()]["test"]
    best_speedup_test_file = merged.loc[merged.mean_ratio.idxmin()]["test_file"]

    worst_slowdown = merged.mean_ratio.max()
    worst_slowdown_test = merged.loc[merged.mean_ratio.idxmax()]["test"]
    worst_slowdown_test_file = merged.loc[merged.mean_ratio.idxmax()]["test_file"]

    threshold = 0.01
    if geomean > 1.0 + threshold:
        geomean_str = "Tests are slower on average"
    elif geomean < 1.0 - threshold:
        geomean_str = "Tests are faster on average"
    else:
        geomean_str = "No overall change in speed"

    tab = Table(title="Comparison Results")
    tab.add_column("Benchmark", style="cyan", no_wrap=True)
    tab.add_column("Baseline Time", style="white")
    tab.add_column("New Time", style="white")
    tab.add_column("Speedup/Slowdown", justify="right", style="white")

    idx = 0
    for entry in pd.concat([merged.head(10), merged.tail(10)]).itertuples():
        benchmark = entry.test
        baseline_time = f"{entry.dynamo_time_mean_1:.4f}s"
        new_time = f"{entry.dynamo_time_mean_2:.4f}s"
        color = "green" if entry.faster else "red"
        ratio = f"[{color}]{entry.mean_ratio:.2f}x [/{color}]"
        tab.add_row(benchmark, baseline_time, new_time, ratio)
        idx += 1
        if idx == 10:
            tab.add_row("...", "...", "...", "...")

    rich.print(tab)

    rich.print(
        f"""
Summary:
    {n_tests} benchmarks run
    {n_faster} faster, {n_slower} slower, {n_same} same
    Geomean ratio = {geomean:.2f}x
    Best speedup = {best_speedup:.2f}x ({best_speedup_test_file} {best_speedup_test})
    Worst slowdown = {worst_slowdown:.2f}x ({worst_slowdown_test_file} {worst_slowdown_test})

    {geomean_str}
"""
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run dynamo-bench tests.")
    parser.add_argument(
        "--all", action="store_true", default=False, help="Run all tests"
    )
    parser.add_argument("--single", type=str, help="Run a single test file")
    parser.add_argument(
        "--list",
        action="store_true",
        default=False,
        help="list all test files available",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="Enable verbose output"
    )
    parser.add_argument(
        "--warmup", type=int, default=2, help="Number of warmup runs for hyperfine"
    )
    parser.add_argument(
        "--runs", type=int, default=3, help="Number of runs for hyperfine"
    )
    parser.add_argument(
        "--save", action="store_true", default=False, help="Save the result"
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("FILE1", "FILE2"),
        help="Compare two result files",
    )
    parser.add_argument(
        "--compare_with",
        type=str,
        help="Compare the latest result with a given result file",
    )
    args = parser.parse_args()

    if args.list:
        list_tests()
    elif args.single:
        assert args.single in get_test_files(), args.single
        main(
            [args.single],
            warmup=args.warmup,
            runs=args.runs,
            verbose=args.verbose,
            save=args.save,
            compare_with=args.compare_with,
        )
    elif args.all:
        main(
            get_test_files(),
            warmup=args.warmup,
            runs=args.runs,
            verbose=args.verbose,
            save=args.save,
            compare_with=args.compare_with,
        )
    elif args.compare:
        compare_files(args.compare[0], args.compare[1], verbose=args.verbose)
    else:
        print("invalid call")
        sys.exit(1)

    # main()
