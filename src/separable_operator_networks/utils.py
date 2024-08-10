__author__ = "Xinling Yu"
__email__ = "xyu644@ucsb.edu"
__copyright__ = "Copyright 2024 Hewlett Packard Enterprise Development LP."
__license__ = "MIT"
__version__ = "0.0.1"

import jax
import jax.numpy as jnp
import GPUtil


def get_gpu_memory(device_name):
    gpus = GPUtil.getGPUs()
    if gpus:
        return gpus[device_name].memoryUsed
    return None


@jax.jit
def sine(x):
    return jnp.sin(x)


@jax.jit
def identity(x):
    return x


@jax.jit
def create_mesh(ti_batch, xi_batch):
    return jnp.meshgrid(ti_batch.ravel(), xi_batch.ravel(), indexing="ij")
