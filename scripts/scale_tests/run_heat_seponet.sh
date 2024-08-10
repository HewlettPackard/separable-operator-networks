#!/bin/bash

# Define the sets of parameters
# ncs=(128)
# batches=(5 10 20 50)

# ncs=(8 16 32 64 128)
# batches=(100)

ncs=(128)
batches=(100)
rs=(1 5 10 15 20 30 35 40 45 50)


# Generate the arguments list and run the script for each combination

for nc in "${ncs[@]}"; do
    for batch in "${batches[@]}"; do
        for r in "${rs[@]}"; do
            python ../main_scripts/heat.py --model_name "SepONet" --device_name 1 --nc "$nc" --batch "$batch" --hidden 50 --r "$r"
        done
    done
done