__author__ = "Xinling Yu, Sean Hooten"
__email__ = "xyu644@ucsb.edu, sean.hooten@hpe.com"
__copyright__ = "Copyright 2024 Hewlett Packard Enterprise Development LP."
__license__ = "MIT"
__version__ = "0.0.1"

import jax
import jax.numpy as jnp
from jax import vmap


# use this for PI-DeepONet when eval if having OOM issue
def process_batch(model, batch):
    (t, x, y), f = batch
    return model(((t, x, y), f))


def process_all_data(model, data, num_batches):
    (t, x, y), f = data
    batch_size = 100 // num_batches
    results = []
    for i in range(num_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        batch = ((t[start:end], x[start:end], y[start:end]), f[start:end])
        result = process_batch(model, batch)
        results.append(result)
    return jnp.concatenate(results, axis=0)


@jax.jit
def rel_l2(u, u_pred):
    u_norm = jnp.linalg.norm(u.reshape(-1, 1))
    diff_norm = jnp.linalg.norm(u.reshape(-1, 1) - u_pred.reshape(-1, 1))
    return diff_norm / u_norm


@jax.jit
def rmse(u, u_pred):
    return jnp.sqrt(jnp.mean((u_pred.reshape(-1, 1) - u.reshape(-1, 1)) ** 2))


def pred_heat(model, test_generator, fs, u, result_dir):
    t, x, f, u = test_generator(fs, u)
    basis_ts, basis_b, u_pred = model(((t, x), f), return_basis=True)
    r = basis_b.shape[-1]
    jnp.save(result_dir + "/f_heat.npy", f)
    jnp.save(result_dir + "/basis_t_heat_r" + str(r) + ".npy", basis_ts[0])
    jnp.save(result_dir + "/basis_x_heat_r" + str(r) + ".npy", basis_ts[1])
    jnp.save(result_dir + "/basis_b_heat_r" + str(r) + ".npy", basis_b)
    jnp.save(result_dir + "/u_pred_heat.npy", u_pred)
    return basis_ts, basis_b, u_pred


def pred_nonlinear_diffusion(model, test_generator, fs, u, result_dir):
    t, x, y, f, u = test_generator(fs, u)
    basis_ts, basis_b, u_pred = model(((t, x, y), f), return_basis=True)
    jnp.save(result_dir + "/f_heat2d.npy", f)
    jnp.save(result_dir + "/basis_t_heat.npy", basis_ts[0])
    jnp.save(result_dir + "/basis_x_heat.npy", basis_ts[1])
    jnp.save(result_dir + "/basis_y_heat.npy", basis_ts[2])
    jnp.save(result_dir + "/basis_b_heat.npy", basis_b)
    jnp.save(result_dir + "/u_pred_heat2d.npy", u_pred)
    return basis_ts, basis_b, u_pred


def pred_DR(model, test_generator, fs, u, result_dir):
    t, x, f, u = test_generator(fs, u)
    basis_ts, basis_b, u_pred = model(((t, x), f), return_basis=True)
    jnp.save(result_dir + "/f_DR.npy", f)
    jnp.save(result_dir + "/basis_t_DR.npy", basis_ts[0])
    jnp.save(result_dir + "/basis_x_DR.npy", basis_ts[1])
    jnp.save(result_dir + "/basis_b_DR.npy", basis_b)
    jnp.save(result_dir + "/u_pred_DR.npy", u_pred)
    return basis_ts, basis_b, u_pred


def pred_ADV(model, test_generator, fs, u, result_dir):
    t, x, f, u = test_generator(fs, u)
    basis_ts, basis_b, u_pred = model(((t, x), f), return_basis=True)
    jnp.save(result_dir + "/f_ADV.npy", f)
    jnp.save(result_dir + "/basis_t_ADV.npy", basis_ts[0])
    jnp.save(result_dir + "/basis_x_ADV.npy", basis_ts[1])
    jnp.save(result_dir + "/basis_b_ADV.npy", basis_b)
    jnp.save(result_dir + "/u_pred_ADV.npy", u_pred)
    return basis_ts, basis_b, u_pred


def pred_Burgers(model, test_generator, fs, u, result_dir):
    t, x, f, u = test_generator(fs, u)
    basis_ts, basis_b, u_pred = model(((t, x), f), return_basis=True)
    jnp.save(result_dir + "/f_Burgers.npy", f)
    jnp.save(result_dir + "/basis_t_Burgers.npy", basis_ts[0])
    jnp.save(result_dir + "/basis_x_Burgers.npy", basis_ts[1])
    jnp.save(result_dir + "/basis_b_Burgers.npy", basis_b)
    jnp.save(result_dir + "/u_pred_Burgers.npy", u_pred)
    return basis_ts, basis_b, u_pred


# can be customized
def eval_heat(model, test_generator, fs, u):
    t, x, f, u = test_generator(fs, u)
    u_pred = model(((t, x), f))
    rel_l2_u = vmap(rel_l2, in_axes=(0, 0))(u, u_pred)
    rmse_u = vmap(rmse, in_axes=(0, 0))(u, u_pred)
    return (
        jnp.mean(rel_l2_u),
        jnp.std(rel_l2_u),
        jnp.mean(rmse_u),
        jnp.std(rmse_u),
    )


def eval_nonlinear_diffusion(model, test_generator, fs, u):
    t, x, y, f, u = test_generator(fs, u)
    u_pred = model(((t, x, y), f))
    rel_l2_u = vmap(rel_l2, in_axes=(0, 0))(u, u_pred)
    rmse_u = vmap(rmse, in_axes=(0, 0))(u, u_pred)
    return (
        jnp.mean(rel_l2_u),
        jnp.std(rel_l2_u),
        jnp.mean(rmse_u),
        jnp.std(rmse_u),
    )


def eval_nonlinear_diffusion_v2(model, test_generator, fs, u):
    t, x, y, f, u = test_generator(fs, u)
    u_pred = process_all_data(model, ((t, x, y), f), 20)
    rel_l2_u = vmap(rel_l2, in_axes=(0, 0))(u, u_pred)
    rmse_u = vmap(rmse, in_axes=(0, 0))(u, u_pred)
    return (
        jnp.mean(rel_l2_u),
        jnp.std(rel_l2_u),
        jnp.mean(rmse_u),
        jnp.std(rmse_u),
    )


def eval_DR(model, test_generator, fs, u):
    t, x, f, u = test_generator(fs, u)
    u_pred = model(((t, x), f))
    rel_l2_u = vmap(rel_l2, in_axes=(0, 0))(u, u_pred)
    rmse_u = vmap(rmse, in_axes=(0, 0))(u, u_pred)
    return (
        jnp.mean(rel_l2_u),
        jnp.std(rel_l2_u),
        jnp.mean(rmse_u),
        jnp.std(rmse_u),
    )


def eval_ADV(model, test_generator, fs, u):
    t, x, f, u = test_generator(fs, u)
    u_pred = model(((t, x), f))
    rel_l2_u = vmap(rel_l2, in_axes=(0, 0))(u, u_pred)
    rmse_u = vmap(rmse, in_axes=(0, 0))(u, u_pred)
    return (
        jnp.mean(rel_l2_u),
        jnp.std(rel_l2_u),
        jnp.mean(rmse_u),
        jnp.std(rmse_u),
    )


def eval_Burgers(model, test_generator, fs, u):
    t, x, f, u = test_generator(fs, u)
    u_pred = model(((t, x), f))
    rel_l2_u = vmap(rel_l2, in_axes=(0, 0))(u, u_pred)
    rmse_u = vmap(rmse, in_axes=(0, 0))(u, u_pred)
    return (
        jnp.mean(rel_l2_u),
        jnp.std(rel_l2_u),
        jnp.mean(rmse_u),
        jnp.std(rmse_u),
    )
