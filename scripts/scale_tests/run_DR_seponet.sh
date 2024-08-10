#!/bin/bash

# Define the sets of parameters
ncs=(8 16 32 64 128)
batches=(5 10 20 50 100)

# Generate the arguments list and run the script for each combination

for nc in "${ncs[@]}"; do
    for batch in "${batches[@]}"; do
        python ../main_scripts/diffusion_reaction.py --model_name "SepONet" --device_name 1 --nc "$nc" --batch "$batch" --hidden 50 --r 50
    done
done