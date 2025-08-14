#!/bin/bash

set -e

# Command to run
CMD="python ${TEST_FILE} ${TEST_CASE}"

for i in $(seq 1 $RUNS); do
    $CMD
done

