#!/bin/bash

THREAD_COUNTS="1 3 7 19 21 57 131 133 393 399 917 2489 2751 7467 17423 52269"

for i in ${THREAD_COUNTS}; do
    export OMP_NUM_THREADS=${i}
    echo "Maximum thread count: ${i}"
    make run-full
    make run-full
    make run-full
    make run-full
    make run-full
    echo " "
done
