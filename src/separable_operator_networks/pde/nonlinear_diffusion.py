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


def interpolate2d(x, y, values, return_mesh):
    points = (jnp.linspace(-1, 1, 101), jnp.linspace(-1, 1, 101))
    interpolate = jax.scipy.interpolate.RegularGridInterpolator(points, values)
    if return_mesh:
        x_mesh, y_mesh = jnp.meshgrid(x.ravel(), y.ravel(), indexing="ij")
        query_points = jnp.concat(
            (x_mesh.reshape(-1, 1), y_mesh.reshape(-1, 1)), axis=1
        )
        return interpolate(query_points).reshape(x.shape[0], y.shape[0])
    else:
        query_points = jnp.concat((x, y), axis=1)
        return interpolate(query_points)


#########################################################################
# Loss functions for DeepONet/SepONet Models
#########################################################################


# Deep Operator PI Loss
@eqx.filter_jit
def apply_model_DeepONet(
    model, tc, xc, yc, tb, xb, yb, ti, xi, yi, fc, lam_b=1.0, lam_i=20.0
):
    def residual_loss(model, t, x, y, f):
        # compute u
        u = model(((t, x, y), f))
        # tangent vector du/du
        v = jnp.ones(u.shape)
        # 1st, 2nd derivatives of u
        ut = vjp(lambda t: model(((t, x, y), f)), t)[1](v)[0]
        ux, uxx = hvp_fwdrev(lambda x: model(((t, x, y), f)), (x,), (v,), True)
        uy, uyy = hvp_fwdrev(lambda y: model(((t, x, y), f)), (y,), (v,), True)
        grad_u_squared = ux**2 + uy**2
        laplacian = uxx + uyy

        return jnp.mean((ut - 0.05 * (grad_u_squared + u * laplacian)) ** 2)

    def boundary_loss(model, t, x, y, f):
        loss = 0
        for i in range(4):
            loss += jnp.mean(model(((t[i], x[i], y[i]), f)) ** 2)
        return loss

    def initial_loss(model, t, x, y, f):
        # interpolation
        nf = f.shape[0]
        f_xy = vmap(interpolate2d, in_axes=(0, 0, 0, None))(
            x, y, f.reshape(nf, 101, 101), False
        )

        return jnp.mean((model(((t, x, y), f)) - f_xy.reshape(nf, -1, 1)) ** 2)

    loss_fn = (  # noqa: E731
        lambda model: residual_loss(model, tc, xc, yc, fc)
        + lam_b * boundary_loss(model, tb, xb, yb, fc)
        + lam_i * initial_loss(model, ti, xi, yi, fc)
    )

    loss, gradient = eqx.filter_value_and_grad(loss_fn)(model)

    return loss, gradient


# Separable Deep Operator PI Loss
@eqx.filter_jit
def apply_model_SepONet(
    model, tc, xc, yc, tb, xb, yb, ti, xi, yi, fc, lam_b=1.0, lam_i=20.0
):
    def residual_loss(model, t, x, y, f):
        # compute u
        u = model(((t, x, y), f))

        # tangent vector dt/dt dx/dx dy/dy
        v_t = jnp.ones(t.shape)
        v_x = jnp.ones(x.shape)
        v_y = jnp.ones(y.shape)

        # 1st, 2nd derivatives of u
        ut = jvp(lambda t: model(((t, x, y), f)), (t,), (v_t,))[1]
        ux, uxx = hvp_fwdfwd(
            lambda x: model(((t, x, y), f)), (x,), (v_x,), True
        )
        uy, uyy = hvp_fwdfwd(
            lambda y: model(((t, x, y), f)), (y,), (v_y,), True
        )
        grad_u_squared = ux**2 + uy**2
        laplacian = uxx + uyy

        return jnp.mean((ut - 0.05 * (grad_u_squared + u * laplacian)) ** 2)

    def boundary_loss(model, t, x, y, f):
        loss = 0
        for i in range(4):
            loss += jnp.mean(model(((t[i], x[i], y[i]), f)) ** 2)
        return loss

    def initial_loss(model, t, x, y, f):
        nf = f.shape[0]
        nx = x.shape[0]
        ny = y.shape[0]

        # interpolation
        f_xy = vmap(interpolate2d, in_axes=(None, None, 0, None))(
            x, y, f.reshape(nf, 101, 101), True
        )

        return jnp.mean(
            (model(((t, x, y), f)) - f_xy.reshape(nf, 1, nx, ny, 1)) ** 2
        )

    ## debug and check magnitudes of different loss terms
    # jax.debug.print("residual_loss: {}",residual_loss(model, tc, xc, fc))
    # jax.debug.print("boundary_loss: {}",boundary_loss(model, tb, xb, fc))
    # jax.debug.print("initial_loss: {}",initial_loss(model, ti, xi, ui, fc))

    loss_fn = (  # noqa: E731
        lambda model: residual_loss(model, tc, xc, yc, fc)
        + lam_b * boundary_loss(model, tb, xb, yb, fc)
        + lam_i * initial_loss(model, ti, xi, yi, fc)
    )

    loss, gradient = eqx.filter_value_and_grad(loss_fn)(model)

    return loss, gradient


#########################################################################
# Train generators for DeepONet and SepONet
#########################################################################
@partial(jax.jit, static_argnums=(1, 2))
def DeepONet_train_generator_nonlinear_diffusion(fs, batch, nc, key):
    nb = nc**2
    ni = nc**2
    key, subkey = jax.random.split(key)
    idx = jax.random.choice(key, fs.shape[0], (batch,), replace=False)
    fc = fs[idx, :]
    keys = jax.random.split(subkey, 13)
    tc = jnp.stack(
        [jax.random.uniform(keys[0], (nc**3, 1), minval=0.0, maxval=1.0)]
        * batch
    )
    xc = jnp.stack(
        [jax.random.uniform(keys[1], (nc**3, 1), minval=-1.0, maxval=1.0)]
        * batch
    )
    yc = jnp.stack(
        [jax.random.uniform(keys[2], (nc**3, 1), minval=-1.0, maxval=1.0)]
        * batch
    )
    tb = [
        jnp.stack(
            [jax.random.uniform(keys[3], (nb, 1), minval=0.0, maxval=1.0)]
            * batch
        ),
        jnp.stack(
            [jax.random.uniform(keys[4], (nb, 1), minval=0.0, maxval=1.0)]
            * batch
        ),
        jnp.stack(
            [jax.random.uniform(keys[5], (nb, 1), minval=0.0, maxval=1.0)]
            * batch
        ),
        jnp.stack(
            [jax.random.uniform(keys[6], (nb, 1), minval=0.0, maxval=1.0)]
            * batch
        ),
    ]
    xb = [
        jnp.array([[[-1.0]] * nb] * batch),
        jnp.array([[[1.0]] * nb] * batch),
        jnp.stack(
            [jax.random.uniform(keys[7], (nb, 1), minval=-1.0, maxval=1.0)]
            * batch
        ),
        jnp.stack(
            [jax.random.uniform(keys[8], (nb, 1), minval=-1.0, maxval=1.0)]
            * batch
        ),
    ]
    yb = [
        jnp.stack(
            [jax.random.uniform(keys[9], (nb, 1), minval=-1.0, maxval=1.0)]
            * batch
        ),
        jnp.stack(
            [jax.random.uniform(keys[10], (nb, 1), minval=-1.0, maxval=1.0)]
            * batch
        ),
        jnp.array([[[-1.0]] * nb] * batch),
        jnp.array([[[1.0]] * nb] * batch),
    ]
    ti = jnp.array([[[0.0]] * ni] * batch)
    xi = jnp.stack(
        [jax.random.uniform(keys[11], (ni, 1), minval=-1.0, maxval=1.0)] * batch
    )
    yi = jnp.stack(
        [jax.random.uniform(keys[12], (ni, 1), minval=-1.0, maxval=1.0)] * batch
    )
    return tc, xc, yc, tb, xb, yb, ti, xi, yi, fc


@partial(jax.jit, static_argnums=(1, 2))
def SepONet_train_generator_nonlinear_diffusion(fs, batch, nc, key):
    key, subkey = jax.random.split(key)
    idx = jax.random.choice(key, fs.shape[0], (batch,), replace=False)
    fc = fs[idx, :]
    keys = jax.random.split(subkey, 2)
    tc = jax.random.uniform(keys[0], (nc, 1), minval=0.0, maxval=1.0)
    xc = jax.random.uniform(keys[1], (nc, 1), minval=-1.0, maxval=1.0)
    yc = jax.random.uniform(keys[2], (nc, 1), minval=-1.0, maxval=1.0)
    tb = [tc, tc, tc, tc]
    xb = [jnp.array([[-1.0]]), jnp.array([[1.0]]), xc, xc]
    yb = [yc, yc, jnp.array([[-1.0]]), jnp.array([[1.0]])]
    ti = jnp.array([[0.0]])
    xi = xc
    yi = yc
    return tc, xc, yc, tb, xb, yb, ti, xi, yi, fc


#########################################################################
# Test generators (random functions and collocation ppints) for DeepONet and SepONet
#########################################################################
@jax.jit
def DeepONet_test_generator_nonlinear_diffusion(fs, u):
    nf = fs.shape[0]
    t = jnp.stack([jnp.linspace(0, 1, 101).reshape(-1, 1)] * nf)
    x = jnp.stack([jnp.linspace(-1, 1, 101).reshape(-1, 1)] * nf)
    y = jnp.stack([jnp.linspace(-1, 1, 101).reshape(-1, 1)] * nf)
    t_mesh, x_mesh, y_mesh = vmap(create_mesh, in_axes=(0, 0, 0))(t, x, y)
    t = t_mesh.reshape(nf, 101**3, 1)
    x = x_mesh.reshape(nf, 101**3, 1)
    y = y_mesh.reshape(nf, 101**3, 1)
    u = u.reshape(nf, 101**3, 1)
    return t, x, y, fs, u


@jax.jit
def SepONet_test_generator_nonlinear_diffusion(fs, u):
    t = jnp.linspace(0, 1, 101).reshape(-1, 1)
    x = jnp.linspace(-1, 1, 101).reshape(-1, 1)
    y = jnp.linspace(-1, 1, 101).reshape(-1, 1)
    return t, x, y, fs, u
