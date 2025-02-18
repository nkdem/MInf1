#!/bin/bash
# This script takes one argument: the number of experiments to run.
# It loops through and runs full_adam.py with a different experiment number each time.

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <number_of_experiments>"
    exit 1
fi

NUM_EXPERIMENTS="$1"

for (( i=1; i<=NUM_EXPERIMENTS; i++ ))
do
    echo "Starting full-adam experiment run $i..."
    python full-adam.py --experiment_no "$i" --cuda 
done