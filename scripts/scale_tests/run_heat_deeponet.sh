#!/bin/bash

# Define the sets of parameters
# ncs=(128)
# batches=(5 10 20 50)

# ncs=(8 16 32 64 128)
# batches=(100)

ncs=(128)
batches=(100)
#rs=(1 5 10)
#rs=(15 20 30)
#rs=(35 40)
rs=(45 50)


#rs=(1 5 10 15 20 30 35 40 45 50)


# Generate the arguments list and run the script for each combination

# for nc in "${ncs[@]}"; do
#     for batch in "${batches[@]}"; do
#         python heat.py --model_name "DEEPOP" --device_name 0 --nc "$nc" --batch "$batch" --hidden 50 --r 50
#     done
# done

for nc in "${ncs[@]}"; do
    for batch in "${batches[@]}"; do
        for r in "${rs[@]}"; do
            python ../main_scripts/heat.py --model_name "DeepONet" --device_name 3 --nc "$nc" --batch "$batch" --hidden 50 --r "$r"
        done
    done
done
