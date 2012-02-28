#!/bin/bash

THREAD_COUNTS="1 3 7 19"
for i in ${THREAD_COUNTS}; do
    export OMP_NUM_THREADS=${i}

    echo "Threads available: ${i}"
    make run-test
    make run-test
    make run-test
    make run-test
    make run-test
    echo " "
done