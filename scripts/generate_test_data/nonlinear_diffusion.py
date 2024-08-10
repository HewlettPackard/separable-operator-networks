#!/usr/bin/python3

# usage: python advection.py

import os
import numpy as np
from scipy.integrate import odeint
import argparse

__author__ = "Sean Hooten, Xinling Yu"
__email__ = "sean.hooten@hpe.com, xyu644@ucsb.edu"
__copyright__ = "Copyright 2024 Hewlett Packard Enterprise Development LP."
__license__ = "MIT"
__version__ = "0.0.1"

parser = argparse.ArgumentParser(
    description="Generate test data for nonlinear diffusion"
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
    default="../../data/nonlinear_diffusion/",
    help="Directory to save data",
)
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)

np.random.seed(args.seed1)


def generate_grf(x, y, num_gaussians=3):
    field = np.zeros_like(x)
    for _ in range(num_gaussians):
        amplitude = np.random.uniform(0.2, 0.5)
        x_center = np.random.uniform(-0.5, 0.5)
        y_center = np.random.uniform(-0.5, 0.5)
        width = np.random.uniform(10, 20)
        field += amplitude * np.exp(
            -width * ((x - x_center) ** 2 + (y - y_center) ** 2)
        )
    return field


# Set up the grid
nx, ny = 101, 101
x = np.linspace(-1, 1, nx)
y = np.linspace(-1, 1, ny)
X, Y = np.meshgrid(x, y, indexing="ij")

# Generate GRFs
n_train = args.num_train
n_test = args.num_test
n_total = n_train + n_test

all_fields = np.zeros((n_total, nx, ny))
for i in range(n_total):
    all_fields[i] = generate_grf(X, Y)

# Save training data
np.save(args.save_dir + "fs_train.npy", all_fields[:n_train])
print("fs_train shape:", all_fields[:n_train].shape)

# Save test data
np.save(args.save_dir + "fs_test.npy", all_fields[n_train:])
print("fs_test shape:", all_fields[n_train:].shape)


def pde_rhs(u, t, X, Y, alpha):
    u = u.reshape((nx, ny))  # Reshape to (nx, ny)

    ux = np.gradient(u, X[:, 0], axis=0)
    uy = np.gradient(u, Y[0, :], axis=1)
    uxx = np.gradient(ux, X[:, 0], axis=0)
    uyy = np.gradient(uy, Y[0, :], axis=1)

    laplacian = uxx + uyy
    grad_u_squared = ux**2 + uy**2

    du_dt = alpha * (grad_u_squared + u * laplacian)

    du_dt[0, :] = du_dt[-1, :] = du_dt[:, 0] = du_dt[:, -1] = 0

    return du_dt.flatten()


# Set up the grid
nx, ny = 101, 101
x = np.linspace(-1, 1, nx)
y = np.linspace(-1, 1, ny)
X, Y = np.meshgrid(x, y, indexing="ij")

# Time settings
nt = 101
t = np.linspace(0, 1, nt)
alpha = 0.05

# Solve PDE for test data
u_test = np.zeros((n_test, nt, nx, ny))
for i in range(n_test):
    u0 = all_fields[n_train + i]  # Initial condition (nx, ny)
    solution = odeint(pde_rhs, u0.flatten(), t, args=(X, Y, alpha))
    u_test[i] = solution.reshape((nt, nx, ny))

# Save test solutions
np.save(args.save_dir + "u_test.npy", u_test)
print("u_test shape:", u_test.shape)
