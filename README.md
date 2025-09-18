# dynamo-bench

A collection of scripts to run CPython test suite and compute the Dynamo overhead.

# Usage: `run.py`

The `run.py` script is the main entry point for running and benchmarking tests in this repository. Below are the available command-line options:

```bash
python run.py [OPTIONS]
```

## Options

```bash
$ python run.py --help
options:
  -h, --help            show this help message and exit
  --testfile TESTFILE   Run a single test file
  --testcase TESTCASE   Run a single test
  --list                list all test files available
  --verbose             Enable verbose output
  --warmup WARMUP       Number of warmup runs for hyperfine
  --runs RUNS           Number of runs for hyperfine
  --save                Save the result
  --compare FILE1 FILE2
                        Compare two result files
  --compare_with COMPARE_WITH
                        Compare the latest result with a given result file
  --run_only_dynamo     Run only dynamo tests
```

## Examples

- **List all test files:**
	```bash
	python run.py --list
	```

- **Run all tests (default):**
	```bash
	python run.py
	```

- **Run a specific test file:**
	```bash
	python run.py --testfile test_list.py
	```

- **Run a specific test case in a file:**
	```bash
	python run.py --testfile test_set.py --testcase TestSetSubclass.test_sub
	```

- **Save results and metadata:**
	```bash
	python run.py --save
	```

- **Compare two result files:**
	```bash
	python run.py --compare results/result_2025-9-13_25b273bfb00.csv results/result_2025-9-17_fa919feab6a.csv
	```

- **Compare latest result with a baseline:**
	```bash
	python run.py --compare_with results/result_2025-9-13_25b273bfb00.csv
	```

- **Run only dynamo tests:**
	```bash
	python run.py --run_only_dynamo
	```

For more details, see the source code or run with `--help`.


