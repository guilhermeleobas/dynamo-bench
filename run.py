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
import tempfile
import platform
import psutil
import torch
import datetime
import json

from datetime import date
from dataclasses import dataclass, field, asdict
from compare import compare_dataframes


version = sys.version_info
PY_VER = f"{version.major}_{version.minor}"
PYVER = f"{version.major}{version.minor}"
PATH_TO_PYTORCH = pathlib.Path("../pytorch").resolve()
PATH_TO_DYNAMO_FAILURES = PATH_TO_PYTORCH / "test" / "dynamo_expected_failures"
PATH_TO_CPYTHON_TESTS = (
    PATH_TO_PYTORCH / "test" / "dynamo" / "cpython" / PY_VER
).resolve()


def get_commit_hash():
    return subprocess.check_output(
        ["git", "log", "-1", "--pretty=%h"],  # %h = short hash
        cwd=PATH_TO_PYTORCH,
        text=True,
    ).strip()


@dataclass(frozen=True)
class Hardware:
    cpu_model: str = platform.processor()
    num_cores: int = psutil.cpu_count(logical=False)
    num_threads: int = psutil.cpu_count(logical=True)
    ram_gb: float = round(psutil.virtual_memory().total / 1e9, 2)


@dataclass(frozen=True)
class System:
    os_name: str = platform.system()
    os_version: str = platform.version()


@dataclass(frozen=True)
class Software:
    python_version: str = platform.python_version()
    torch_version: str = torch.__version__


@dataclass(frozen=True)
class BenchmarkParameters:
    warmup: int
    runs: int


@dataclass(frozen=True)
class Execution:
    benchmark_parameters: BenchmarkParameters
    run_date: str = datetime.datetime.utcnow().isoformat() + "Z"
    commit_hash: str = get_commit_hash()


@dataclass(frozen=True)
class Metadata:
    execution: Execution
    hardware: Hardware = field(default_factory=Hardware)
    system: System = field(default_factory=System)
    software: Software = field(default_factory=Software)


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


def compute_run_stats(data: list[float]) -> tuple[float, float, float]:
    if all(r == 0.0 for r in data):
        return 0.0, 0.0, 0.0

    min_ = min(data)
    mean, stddev = (
        statistics.mean(data),
        statistics.stdev(data) if len(data) > 1 else 0.0,
    )
    max_ = max(data)

    threshold = 0.03
    cv = stddev / mean
    if cv > threshold:
        warnings.warn(
            f"High coefficient of variation detected: {mean=}, {stddev=}, {cv=}"
        )
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
    results = collections.defaultdict(list)
    non_skipped_tests = []

    bash_script = f"""
#!/bin/bash

set -e
"""

    for idx, test_name in enumerate(test_names):
        results[test_name] = [0.0, 0.0, 0.0]
        if skip_test(test_file, test_name):
            if verbose:
                print(f"Skipping test {test_name}")
            continue
        else:
            non_skipped_tests.append(test_name)

    n_tests = len(non_skipped_tests)
    for idx, test_name in enumerate(non_skipped_tests):
        bash_script += f"""
echo "[{idx}/{n_tests}] Running test {test_name}"
for i in $(seq 1 {warmup + runs}); do
    python {test_file_path} {test_name}
done
"""

    runtimes = run_bash_script(bash_script, verbose=verbose, env_extra=env_extra)
    batched = itertools.batched(runtimes, warmup + runs)
    for test_name, batch in zip(non_skipped_tests, batched):
        assert len(batch) == warmup + runs

        batch = batch[warmup:]  # Skip warmup runs
        min_, mean, max_ = compute_run_stats(batch)
        results[test_name] = [min_, mean, max_]
    return results


def dump_result(df: pd.DataFrame):
    commit = get_commit_hash()
    today = date.today()
    d = f"{today.year}-{today.month}-{today.day}"
    name = f"result_{d}_{commit}.csv"
    p = (pathlib.Path("results") / name).resolve()
    df.sort_values(by="ratio", ascending=False).to_csv(p, index=False)


def dump_metadata(*, warmup: int, runs: int, **kwargs):
    params = BenchmarkParameters(warmup, runs)
    execution = Execution(benchmark_parameters=params)
    metadata = Metadata(execution=execution)
    metadata_json = json.dumps(asdict(metadata), indent=2)
    fname = f"{get_commit_hash()}_metadata.json"
    with open(fname, "w") as f:
        f.write(metadata_json)


def run_cpython(*args, **kwargs):
    return run_tests(*args, **kwargs)


def run_dynamo(*args, **kwargs):
    return run_tests(*args, env_extra={"PYTORCH_TEST_WITH_DYNAMO": "1"}, **kwargs)


def skip_test(test_file: str, test_name: str):
    module = os.path.basename(test_file).strip(".py")
    file = PATH_TO_DYNAMO_FAILURES / f"CPython{PYVER}-{module}-{test_name}"
    return file.exists()


def main(
    test_files: list[str], *, save: str | None, compare_with: str | None, **kwargs
):
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
        # save result.csv
        dump_result(df)
        # save metadata
        dump_metadata(**kwargs)

    if compare_with:
        baseline = pd.read_csv(compare_with)
        hash_baseline = compare_with.split("_")[-1].strip(".csv")
        compare_dataframes(
            baseline,
            df,
            hash_baseline,
            get_commit_hash(),
            verbose=kwargs.get("verbose", False),
        )


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
    baseline_hash = baseline.split("_")[-1].strip(".csv")
    new_hash = file2.split("_")[-1].strip(".csv")
    return compare_dataframes(df1, df2, baseline_hash, new_hash, verbose=verbose)


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
