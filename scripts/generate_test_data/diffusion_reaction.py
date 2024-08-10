#!/usr/bin/python3

# usage: python diffusion_reaction.py

import os
import argparse
import jax
import jax.numpy as jnp
from separable_operator_networks.pde import diffusion_reaction

__author__ = "Sean Hooten, Xinling Yu"
__email__ = "sean.hooten@hpe.com, xyu644@ucsb.edu"
__copyright__ = "Copyright 2024 Hewlett Packard Enterprise Development LP."
__license__ = "MIT"
__version__ = "0.0.1"

parser = argparse.ArgumentParser(
    description="Generate test data for diffusion reaction"
)
parser.add_argument(
    "--num_train", type=int, default=10000, help="Number of training samples"
)
parser.add_argument(
    "--num_test", type=int, default=100, help="Number of testing samples"
)
parser.add_argument("--seed1", type=int, default=1234, help="Random seed")
parser.add_argument("--seed2", type=int, default=4231, help="Random seed")
parser.add_argument(
    "--save_dir",
    type=str,
    default="../../data/diffusion_reaction/",
    help="Directory to save data",
)
args = parser.parse_args()

key1 = jax.random.key(args.seed1)
key2 = jax.random.key(args.seed2)

os.makedirs(args.save_dir, exist_ok=True)
fs_train = diffusion_reaction.generate_training_data_DR(args.num_train, key1)
fs_test, u_test = diffusion_reaction.generate_test_data_DR(args.num_test, key2)
jnp.save(args.save_dir + "fs_train.npy", jnp.float32(fs_train))
jnp.save(args.save_dir + "fs_test.npy", jnp.float32(fs_test))
jnp.save(args.save_dir + "u_test.npy", jnp.float32(u_test))
