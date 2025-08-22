#!/usr/bin/env python3
import argparse
import warnings
import statistics
import re
import os
import sys
import subprocess
import unittest
import pathlib
import pandas as pd
import rich

from datetime import date


version = sys.version_info
PY_VER = f"{version.major}_{version.minor}"
PYVER = f"{version.major}{version.minor}"
PATH_TO_PYTORCH = pathlib.Path("../pytorch").resolve()
PATH_TO_DYNAMO_FAILURES = PATH_TO_PYTORCH / "test" / "dynamo_expected_failures"
PATH_TO_CPYTHON_TESTS = (PATH_TO_PYTORCH / "test" / "dynamo" / "cpython" / PY_VER).resolve()


def discover_tests(test_file):
    """Return fully-qualified unittest test names in test_file."""
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=os.path.dirname(test_file) or ".",
                            pattern=os.path.basename(test_file))
    tests = []
    def collect(s):
        for t in s:
            if isinstance(t, unittest.TestSuite):
                collect(t)
            else:
                tests.append(f"{t.__class__.__name__}.{t._testMethodName}")
    collect(suite)
    return tests


def run_test(test_file: str, test_name: str, *, env_extra=None, warmup: int, runs: int, verbose: bool) -> float:
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)

    env["TEST_FILE"] = test_file
    env["TEST_CASE"] = test_name
    env["RUNS"] = str(runs + warmup)

    cmd = [
        "bash",
        "run.bash",
    ]

    out = subprocess.run(cmd, env=env, check=True, text=True, stderr=subprocess.PIPE).stderr
    matches = re.findall(r"Ran \d+ test.* in ([0-9.]+)s", out)
    runtimes = [float(m) for m in matches][warmup:]  # Skip warmup runs
    if verbose:
        print(runtimes)

    if all(r == 0.0 for r in runtimes):
        return 0.0, 0.0, 0.0

    min_ = min(runtimes)
    mean, stddev = statistics.mean(runtimes), statistics.stdev(runtimes) if len(runtimes) > 1 else 0.0
    max_ = max(runtimes)
    
    cv = stddev / mean
    if cv > 0.03:
        warnings.warn(f"High coefficient of variation detected: {mean=}, {stddev=}, {cv=}")
    return min_, mean, max_


def run_cpython(*args, **kwargs):
    return run_test(*args, **kwargs)


def run_dynamo(*args, **kwargs):
    return run_test(*args, env_extra={"PYTORCH_TEST_WITH_DYNAMO": "1"}, **kwargs)


def skip_test(test_file: str, test_name: str):
    module = os.path.basename(test_file).strip(".py")
    file = PATH_TO_DYNAMO_FAILURES / f"CPython{PYVER}-{module}-{test_name}"
    return file.exists()

def get_commit_hash():
    return subprocess.check_output(
        ["git", "log", "-1", "--pretty=%h"],  # %h = short hash
        cwd=PATH_TO_PYTORCH,
        text=True
    ).strip()

def main(test_files: list[str], *, save: str | None, **kwargs):
    results = []

    for test_file in test_files:
        abs_path = os.path.join(PATH_TO_CPYTHON_TESTS, test_file)
        tests = discover_tests(abs_path)

        print(f"===== Test {test_file} =====")

        for idx, test in enumerate(tests):
            if skip_test(test_file, test):
                print(f"[{idx}/{len(tests)}] Skipping test {test}...")
                continue

            print(f"[{idx}/{len(tests)}] Running {test}...")
            base_min, base_mean, base_max = run_cpython(abs_path, test, **kwargs)
            dynamo_min, dynamo_mean, dynamo_max = run_dynamo(abs_path, test, **kwargs)

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
                    "ratio": dynamo_mean / base_mean if base_mean and dynamo_mean else None
                },
            )

        print()

    df = pd.DataFrame(results)
    print(df)

    if save:
        # prepend name by git commit and user date
        commit = get_commit_hash()
        today = date.today()
        d = f"{today.year}-{today.month}-{today.day}"
        name = f"result_{d}_{commit}.csv"
        p = (pathlib.Path("results") / name).resolve()
        df.sort_values(by="ratio", ascending=False).to_csv(p, index=False)


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run dynamo-bench tests.")
    parser.add_argument("--all", action="store_true", default=False, help="Run all tests")
    parser.add_argument("--single", type=str, help="Run a single test file")
    parser.add_argument("--list", action="store_true", default=False, help="list all test files available")
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose output")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup runs for hyperfine")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs for hyperfine")
    parser.add_argument("--save", action="store_true", default=False, help="Save the result")
    args = parser.parse_args()

    if args.list:
        list_tests()
    elif args.single:
        assert args.single in get_test_files(), args.single
        main([args.single], warmup=args.warmup, runs=args.runs, verbose=args.verbose, save=args.save)
    elif args.all:
        main(get_test_files(), warmup=args.warmup, runs=args.runs, verbose=args.verbose, save=args.save)
    else:
        print("invalid call")
        sys.exit(1)

    # main()
