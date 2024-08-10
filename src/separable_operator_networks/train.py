"""
This module provides some simple training loops for eqx.Module models.
"""

__author__ = "Sean Hooten, Xinling Yu"
__email__ = "sean.hooten@hpe.com, xyu644@ucsb.edu"
__copyright__ = "Copyright 2024 Hewlett Packard Enterprise Development LP."
__license__ = "MIT"
__version__ = "0.0.1"

import jax
import equinox as eqx
import time
import os
from . import utils


# Define your update function
@eqx.filter_jit
def update(grads, optimizer, opt_state, model):
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state


# Define your training loop
def train_loop(
    model,
    optimizer,
    opt_state,
    update_fn,
    train_generator,
    loss_fn,
    num_epochs,
    log_epoch,
    result_dir,
    device_name,
    key,
):
    for epoch in range(num_epochs):
        if epoch % 100 == 0:
            key, subkey = jax.random.split(key)
            inputs = train_generator(subkey)

        loss, grads = loss_fn(model, *inputs)
        model, opt_state = update_fn(grads, optimizer, opt_state, model)

        if epoch == 1:
            gpu_memory = utils.get_gpu_memory(device_name)
            with open(
                os.path.join(result_dir, "memory usage (mb).csv"), "a"
            ) as f:
                f.write(f"{gpu_memory}\n")
            start = time.time()

        if epoch % log_epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss}")
            with open(os.path.join(result_dir, "log (loss).csv"), "a") as f:
                f.write(f"{loss}\n")
    runtime = time.time() - start

    return model, optimizer, opt_state, runtime
