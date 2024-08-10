__author__ = "Xinling Yu, Sean Hooten"
__email__ = "xyu644@ucsb.edu, sean.hooten@hpe.com"
__copyright__ = "Copyright 2024 Hewlett Packard Enterprise Development LP."
__license__ = "MIT"
__version__ = "0.0.1"

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import numpy as np
import jax
import jax.numpy as jnp
import argparse
import optax
import equinox as eqx

from separable_operator_networks.pde import advection
from separable_operator_networks.train import train_loop, update
from separable_operator_networks.eval import eval_ADV, pred_ADV
from separable_operator_networks.models import DeepONet, SepONet

# config
parser = argparse.ArgumentParser(description="Training configurations")
parser.add_argument(
    "--data-dir",
    type=str,
    default="../../data/advection/",
    help="Data directory. Make sure data has been generated first.",
)

parser.add_argument(
    "--results-dir",
    type=str,
    default="../../results/advection/",
    help="Results directory.",
)

parser.add_argument(
    "--model_name",
    type=str,
    default="SepONet",
    choices=["DeepONet", "SepONet"],
    help="model name (DeepONet; SepONet)",
)
parser.add_argument(
    "--device_name",
    type=int,
    default=1,
    choices=[0, 1, 2, 3],
    help="GPU device",
)

# training data settings
parser.add_argument(
    "--nc",
    type=int,
    default=128,
    help="the number of input points for each axis",
)
parser.add_argument(
    "--batch", type=int, default=100, help="the number of train functions"
)

# training settings
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument(
    "--epochs", type=int, default=120000, help="training epochs"
)
parser.add_argument(
    "--log_epoch",
    type=int,
    default=100,
    help="log the loss every chosen epochs",
)

# model settings
parser.add_argument("--dim", type=int, default=2, help="the input size")
parser.add_argument(
    "--branch_dim",
    type=int,
    default=128,
    help="the number of sensors for indentifying an input function",
)
parser.add_argument(
    "--field_dim",
    type=int,
    default=1,
    help="the dimension of the output field",
)
parser.add_argument(
    "--depth",
    type=int,
    default=6,
    help="the number of hidden layers, including the output layer",
)
parser.add_argument(
    "--hidden", type=int, default=100, help="the size of each hidden layer"
)
parser.add_argument(
    "--r",
    type=int,
    default=100,
    help="rank*field_dim equals the output size",
)

args = parser.parse_args()

# set up the device
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_name)

fs_train = jnp.load(args.data_dir + "fs_train.npy")
fs_test = jnp.load(args.data_dir + "fs_test.npy")
u_test = jnp.load(args.data_dir + "u_test.npy")

root_dir = os.path.join(args.results_dir, "results_ADV", args.model_name)

result_dir = os.path.join(
    root_dir, "nf" + str(args.batch) + "_nc" + str(args.nc)
)

# make dir
os.makedirs(result_dir, exist_ok=True)

# logs
if os.path.exists(os.path.join(result_dir, "log (loss).csv")):
    os.remove(os.path.join(result_dir, "log (loss).csv"))

if os.path.exists(os.path.join(result_dir, "log (rel_l2, rmse).csv")):
    os.remove(os.path.join(result_dir, "log (rel_l2, rmse).csv"))

if os.path.exists(os.path.join(result_dir, "total runtime (sec).csv")):
    os.remove(os.path.join(result_dir, "total runtime (sec).csv"))

if os.path.exists(os.path.join(result_dir, "memory usage (mb).csv")):
    os.remove(os.path.join(result_dir, "memory usage (mb).csv"))

# update function
update_fn = update

# define the optimizer
schedule = optax.exponential_decay(args.lr, 1000, 0.9)
optimizer = optax.adam(schedule)
# optimizer = optax.adam(args.lr)

# random key
key = jax.random.PRNGKey(args.seed)
key, subkey = jax.random.split(key, 2)

# init model
if args.model_name == "DeepONet":
    model = eqx.filter_jit(
        eqx.filter_vmap(
            DeepONet(
                dim=args.dim,
                branch_dim=args.branch_dim,
                field_dim=args.field_dim,
                depth=args.depth,
                hidden=args.hidden,
                rank=args.r,
                key=subkey,
            )
        )
    )
elif args.model_name == "SepONet":
    model = eqx.filter_jit(
        SepONet(
            dim=args.dim,
            branch_dim=args.branch_dim,
            field_dim=args.field_dim,
            depth=args.depth,
            hidden=args.hidden,
            rank=args.r,
            key=subkey,
        )
    )

# init state
key, subkey = jax.random.split(key)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

# train/test generator
if args.model_name == "DeepONet":
    train_generator = jax.jit(
        lambda key: advection.DeepONet_train_generator_ADV(
            fs_train, args.batch, args.nc, key
        )
    )
    test_generator = jax.jit(advection.DeepONet_test_generator_ADV)
    loss_fn = advection.apply_model_DeepONet
elif args.model_name == "SepONet":
    train_generator = jax.jit(
        lambda key: advection.SepONet_train_generator_ADV(
            fs_train, args.batch, args.nc, key
        )
    )
    test_generator = jax.jit(advection.SepONet_test_generator_ADV)
    loss_fn = advection.apply_model_SepONet

# train the model
model, optimizer, opt_state, runtime = train_loop(
    model,
    optimizer,
    opt_state,
    update_fn,
    train_generator,
    loss_fn,
    args.epochs,
    args.log_epoch,
    result_dir,
    args.device_name,
    subkey,
)

# eval the trained model
rel_l2_mean, rel_l2_std, rmse_mean, rmse_std = eval_ADV(
    model, test_generator, fs_test, u_test
)
print(
    f"Runtime --> total: {runtime:.2f}sec ({(runtime/(args.epochs-1)*1000):.2f}ms/iter.)"
)
print(f"rel_l2 --> mean: {rel_l2_mean:.8f} (std: {rel_l2_std: 8f})")
print(f"rmse --> mean: {rmse_mean:.8f} (std: {rmse_std: 8f})")

# save runtime and eval metrics
runtime = np.array([runtime])
np.savetxt(
    os.path.join(result_dir, "total runtime (sec).csv"),
    runtime,
    delimiter=",",
)
with open(os.path.join(result_dir, "log (rel_l2, rmse).csv"), "a") as f:
    f.write(f"rel_l2_mean: {rel_l2_mean}\n")
    f.write(f"rel_l2_std: {rel_l2_std}\n")
    f.write(f"rmse_mean: {rmse_mean}\n")
    f.write(f"rmse_std: {rmse_std}\n")

basis_ts, basis_b, u_pred = pred_ADV(
    model, test_generator, fs_test, u_test, result_dir
)
