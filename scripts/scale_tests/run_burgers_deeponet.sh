#!/bin/bash

# Define the sets of parameters
# ncs=(8 16 32 64 128)
# batches=(100)

ncs=(128)
batches=(5 10 20 50)

# Generate the arguments list and run the script for each combination

for nc in "${ncs[@]}"; do
    for batch in "${batches[@]}"; do
        python ../main_scripts/burgers.py --model_name "DeepONet" --device_name 3 --nc "$nc" --batch "$batch" --hidden 100 --r 100
    done
done