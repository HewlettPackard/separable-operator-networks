__author__ = "Xinling Yu, Sean Hooten"
__email__ = "xyu644@ucsb.edu, sean.hooten@hpe.com"
__copyright__ = "Copyright 2024 Hewlett Packard Enterprise Development LP."
__license__ = "MIT"
__version__ = "0.0.1"

from functools import partial
import jax
import jax.numpy as jnp
from jax import vmap, vjp, jvp
import equinox as eqx

from ..hvp import hvp_fwdfwd, hvp_fwdrev
from ..utils import create_mesh

#########################################################################
# Loss functions for DeepONet/SepONet Models
#########################################################################


# DeepONet PI Loss
@eqx.filter_jit
def apply_model_DeepONet(
    model, tc, xc, tb, xb, ti, xi, fc, lam_b=1.0, lam_i=20.0
):
    def residual_loss(model, t, x, f):
        # compute u
        u = model(((t, x), f))
        # tangent vector du/du
        v = jnp.ones(u.shape)
        # 1st, 2nd derivatives of u
        ut = vjp(lambda t: model(((t, x), f)), t)[1](v)[0]
        ux, uxx = hvp_fwdrev(lambda x: model(((t, x), f)), (x,), (v,), True)
        return jnp.mean((ut + u * ux - 0.01 * uxx) ** 2)

    def boundary_loss(model, t, x, f):
        # compute u
        u0 = model(((t[0], x[0]), f))
        u1 = model(((t[1], x[1]), f))
        # compute bcs
        loss_bc1 = jnp.mean((u0 - u1) ** 2)
        # tangent vector du/du
        v = jnp.ones(u0.shape)
        # 1st derivatives of u
        ux0 = vjp(lambda x: model(((t[0], x), f)), x[0])[1](v)[0]
        ux1 = vjp(lambda x: model(((t[1], x), f)), x[1])[1](v)[0]
        loss_bc2 = jnp.mean((ux0 - ux1) ** 2)
        loss = loss_bc1 + loss_bc2
        return loss

    def initial_loss(model, t, x, f):
        # interpolation
        nf = f.shape[0]
        f_fn = lambda x, f: jnp.interp(x.flatten(), jnp.linspace(0, 1, 101), f)  # noqa: E731
        f_x = vmap(f_fn, in_axes=(0, 0))(x, f)

        return jnp.mean((model(((t, x), f)) - f_x.reshape(nf, -1, 1)) ** 2)

    loss_fn = (  # noqa: E731
        lambda model: residual_loss(model, tc, xc, fc)
        + lam_b * boundary_loss(model, tb, xb, fc)
        + lam_i * initial_loss(model, ti, xi, fc)
    )

    loss, gradient = eqx.filter_value_and_grad(loss_fn)(model)

    return loss, gradient


# SepONet PI Loss
@eqx.filter_jit
def apply_model_SepONet(
    model, tc, xc, tb, xb, ti, xi, fc, lam_b=1.0, lam_i=20.0
):
    def residual_loss(model, t, x, f):
        # compute u
        u = model(((t, x), f))
        # tangent vector dx/dx
        v_t = jnp.ones(t.shape)
        v_x = jnp.ones(x.shape)
        # 1st, 2nd derivatives of u
        ut = jvp(lambda t: model(((t, x), f)), (t,), (v_t,))[1]
        ux, uxx = hvp_fwdfwd(lambda x: model(((t, x), f)), (x,), (v_x,), True)
        return jnp.mean((ut + u * ux - 0.01 * uxx) ** 2)

    def boundary_loss(model, t, x, f):
        # compute u
        u0 = model(((t[0], x[0]), f))
        u1 = model(((t[1], x[1]), f))
        # compute bcs
        loss_bc1 = jnp.mean((u0 - u1) ** 2)
        # tangent vector dx/dx
        v_x = jnp.ones(x[0].shape)
        # 1st derivatives of u
        ux0 = jvp(lambda x: model(((t[0], x), f)), (x[0],), (v_x,))[1]
        ux1 = jvp(lambda x: model(((t[1], x), f)), (x[1],), (v_x,))[1]
        loss_bc2 = jnp.mean((ux0 - ux1) ** 2)
        loss = loss_bc1 + loss_bc2
        return loss

    def initial_loss(model, t, x, f):
        nf = f.shape[0]
        # interpolation
        f_fn = lambda x, f: jnp.interp(x.flatten(), jnp.linspace(0, 1, 101), f)  # noqa: E731
        f_x = vmap(f_fn, in_axes=(None, 0))(x, f)
        return jnp.mean((model(((t, x), f)) - f_x.reshape(nf, 1, -1, 1)) ** 2)

    ## debug and check magnitudes of different loss terms
    # jax.debug.print("residual_loss: {}",residual_loss(model, tc, xc, fc))
    # jax.debug.print("boundary_loss: {}",boundary_loss(model, tb, xb, fc))
    # jax.debug.print("initial_loss: {}",initial_loss(model, ti, xi, ui, fc))

    loss_fn = (  # noqa: E731
        lambda model: residual_loss(model, tc, xc, fc)
        + lam_b * boundary_loss(model, tb, xb, fc)
        + lam_i * initial_loss(model, ti, xi, fc)
    )

    loss, gradient = eqx.filter_value_and_grad(loss_fn)(model)

    return loss, gradient


#########################################################################
# Train generators for DeepONet and SepONet
#########################################################################


@partial(jax.jit, static_argnums=(1, 2))
def DeepONet_train_generator_Burgers(fs, batch, nc, key):
    nb = nc
    ni = nc
    key, subkey = jax.random.split(key)
    idx = jax.random.choice(key, fs.shape[0], (batch,), replace=False)
    fc = fs[idx, :]
    keys = jax.random.split(subkey, 4)
    tc = jnp.stack(
        [jax.random.uniform(keys[0], (nc**2, 1), minval=0.0, maxval=1.0)]
        * batch
    )
    xc = jnp.stack(
        [jax.random.uniform(keys[1], (nc**2, 1), minval=0.0, maxval=1.0)]
        * batch
    )
    tb0 = jnp.stack(
        [jax.random.uniform(keys[2], (nb, 1), minval=0.0, maxval=1.0)] * batch
    )
    tb1 = tb0
    tb = [tb0, tb1]
    xb = [jnp.array([[[0.0]] * nb] * batch), jnp.array([[[1.0]] * nb] * batch)]
    ti = jnp.array([[[0.0]] * ni] * batch)
    xi = jnp.stack(
        [jax.random.uniform(keys[3], (ni, 1), minval=0.0, maxval=1.0)] * batch
    )
    return tc, xc, tb, xb, ti, xi, fc


@partial(jax.jit, static_argnums=(1, 2))
def SepONet_train_generator_Burgers(fs, batch, nc, key):
    key, subkey = jax.random.split(key)
    idx = jax.random.choice(key, fs.shape[0], (batch,), replace=False)
    fc = fs[idx, :]
    keys = jax.random.split(subkey, 2)
    tc = jax.random.uniform(keys[0], (nc, 1), minval=0.0, maxval=1.0)
    xc = jax.random.uniform(keys[1], (nc, 1), minval=0.0, maxval=1.0)
    tb = [tc, tc]
    xb = [jnp.array([[0.0]]), jnp.array([[1.0]])]
    ti = jnp.array([[0.0]])
    xi = xc
    return tc, xc, tb, xb, ti, xi, fc


#########################################################################
# Test generators (random functions and collocation ppints) for DeepONet and SepONet
#########################################################################


@jax.jit
def DeepONet_test_generator_Burgers(fs, u):
    nf = fs.shape[0]
    t = jnp.stack([jnp.linspace(0, 1, 101).reshape(-1, 1)] * nf)
    x = jnp.stack([jnp.linspace(0, 1, 101).reshape(-1, 1)] * nf)
    t_mesh, x_mesh = vmap(create_mesh, in_axes=(0, 0))(t, x)
    t = t_mesh.reshape(nf, 101**2, 1)
    x = x_mesh.reshape(nf, 101**2, 1)
    u = u.reshape(nf, 101**2, 1)
    return t, x, fs, u


@jax.jit
def SepONet_test_generator_Burgers(fs, u):
    t = jnp.linspace(0, 1, 101).reshape(-1, 1)
    x = jnp.linspace(0, 1, 101).reshape(-1, 1)
    return t, x, fs, u
