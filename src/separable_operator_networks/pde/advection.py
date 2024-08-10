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

from ..utils import create_mesh


# boundary and initial condition of advection equation
@jax.jit
def _adv_boundary_u(t):
    return jnp.sin((jnp.pi / 2) * t)


@jax.jit
def _adv_init_u(x):
    return jnp.sin(jnp.pi * x)


#########################################################################
# Loss functions for DeepONet/SepONet Models
#########################################################################


# DeepONet PI Loss
@eqx.filter_jit
def apply_model_DeepONet(
    model, tc, xc, tb, xb, ub, ti, xi, ui, fc, lam_b=100.0, lam_i=100.0
):
    def residual_loss(model, t, x, f):
        # compute u
        u = model(((t, x), f))
        # num of funcs
        nf = f.shape[0]
        # tangent vector du/du
        v = jnp.ones(u.shape)
        # 1st derivatives of u
        ut = vjp(lambda t: model(((t, x), f)), t)[1](v)[0]
        ux = vjp(lambda x: model(((t, x), f)), x)[1](v)[0]
        # interpolation
        f_fn = lambda x, f: jnp.interp(x.flatten(), jnp.linspace(0, 1, 128), f)  # noqa: E731
        f_x = vmap(f_fn, in_axes=(0, 0))(x, f)  # shape [nf, nx*nt]
        return jnp.mean((ut + f_x.reshape(nf, -1, 1) * ux) ** 2)

    def boundary_loss(model, t, x, u, f):
        return jnp.mean((model(((t, x), f)) - u) ** 2)

    def initial_loss(model, t, x, u, f):
        return jnp.mean((model(((t, x), f)) - u) ** 2)

    loss_fn = (  # noqa: E731
        lambda model: residual_loss(model, tc, xc, fc)
        + lam_b * boundary_loss(model, tb, xb, ub, fc)
        + lam_i * initial_loss(model, ti, xi, ui, fc)
    )

    loss, gradient = eqx.filter_value_and_grad(loss_fn)(model)

    return loss, gradient


# SepONet PI Loss
@eqx.filter_jit
def apply_model_SepONet(
    model, tc, xc, tb, xb, ub, ti, xi, ui, fc, lam_b=100.0, lam_i=100.0
):
    def residual_loss(model, t, x, f):
        # num of funcs and nx
        nf = f.shape[0]
        nx = x.shape[0]  # nt = nx here
        # tangent vector dx/dx
        v_t = jnp.ones(t.shape)
        v_x = jnp.ones(x.shape)
        # 1st derivatives of u
        ut = jvp(lambda t: model(((t, x), f)), (t,), (v_t,))[1]
        ux = jvp(lambda x: model(((t, x), f)), (x,), (v_x,))[1]
        # interpolation
        f_fn = lambda x, f: jnp.interp(x.flatten(), jnp.linspace(0, 1, 128), f)  # noqa: E731
        f_x = vmap(f_fn, in_axes=(None, 0))(x, f)  # shape [batch, nx]

        return jnp.mean((ut + f_x.reshape(nf, 1, nx, 1) * ux) ** 2)

    def boundary_loss(model, t, x, u, f):
        return jnp.mean((model(((t, x), f)) - u) ** 2)

    def initial_loss(model, t, x, u, f):
        return jnp.mean((model(((t, x), f)) - u) ** 2)

    ## debug and check magnitudes of different loss terms
    # jax.debug.print("residual_loss: {}",residual_loss(model, tc, xc, fc))
    # jax.debug.print("boundary_loss: {}",boundary_loss(model, tb, xb, fc))
    # jax.debug.print("initial_loss: {}",initial_loss(model, ti, xi, ui, fc))

    loss_fn = (  # noqa: E731
        lambda model: residual_loss(model, tc, xc, fc)
        + lam_b * boundary_loss(model, tb, xb, ub, fc)
        + lam_i * initial_loss(model, ti, xi, ui, fc)
    )

    loss, gradient = eqx.filter_value_and_grad(loss_fn)(model)

    return loss, gradient


#########################################################################
# GRF generator and ADR equation solver
#########################################################################
# RBF kernel
@jax.jit
def RBF(x1, x2, params):
    output_scale, lengthscales = params
    diffs = jnp.expand_dims(x1 / lengthscales, 1) - jnp.expand_dims(
        x2 / lengthscales, 0
    )
    r2 = jnp.sum(diffs**2, axis=2)
    return output_scale * jnp.exp(-0.5 * r2)


# randomly sample a Gaussian random field (128 sensors)
@jax.jit
def GP_sample(key):
    xmin, xmax = 0, 1
    gp_params = (1.0, 0.2)
    N = 512
    Nx = 128
    X = jnp.linspace(xmin, xmax, N)[:, None]
    x = jnp.linspace(xmin, xmax, Nx)  # Nx sensors
    jitter = 1e-10
    K = RBF(X, X, gp_params)
    L = jnp.linalg.cholesky(K + jitter * jnp.eye(N))
    gp_sample = jnp.dot(L, jax.random.normal(key, (N,)))
    # Create a callable interpolation function
    f_fn = lambda x: jnp.interp(x, X.flatten(), gp_sample)  # noqa: E731
    f_fn_pos = lambda x: f_fn(x) - f_fn(x).min() + 1  # noqa: E731
    f = f_fn_pos(x)
    return f


# A Advection numerical solver
def solve_CVC(f, Nt):
    """Solve 1D
    u_t + a(x) * u_x = 0
    with IBC u(x, 0)= sin(pi*x), u(0, t) = sin(pi*t/2).
    """
    xmin, xmax = 0, 1
    tmin, tmax = 0, 1

    Nx = len(f)

    # Create grid
    x = jnp.linspace(xmin, xmax, Nx)
    t = jnp.linspace(tmin, tmax, Nt)
    h = x[1] - x[0]
    dt = t[1] - t[0]
    lam = dt / h

    # advection velocity
    v = f

    # Initialize solution and apply initial & boundary conditions
    u = jnp.zeros([Nx, Nt])
    u = u.at[0, :].set(_adv_boundary_u(t))
    u = u.at[:, 0].set(_adv_init_u(x))

    # Compute finite difference operators
    a = (v[:-1] + v[1:]) / 2
    k = (1 - a * lam) / (1 + a * lam)
    K = jnp.eye(Nx - 1, k=0)
    K_temp = jnp.eye(Nx - 1, k=0)
    Trans = jnp.eye(Nx - 1, k=-1)

    def body_fn_x(i, carry):
        K, K_temp = carry
        K_temp = (-k[:, None]) * (Trans @ K_temp)
        K += K_temp
        return K, K_temp

    K, _ = jax.lax.fori_loop(0, Nx - 2, body_fn_x, (K, K_temp))
    D = jnp.diag(k) + jnp.eye(Nx - 1, k=-1)

    def body_fn_t(i, u):
        b = jnp.zeros(Nx - 1)
        b = b.at[0].set(
            _adv_boundary_u(i * dt) - k[0] * _adv_boundary_u((i + 1) * dt)
        )
        u = u.at[1:, i + 1].set(K @ (D @ u[1:, i] + b))
        return u

    UU = jax.lax.fori_loop(0, Nt - 1, body_fn_t, u)

    return UU


#########################################################################
# Train generators for DeepONet and SepONet
#########################################################################
# create GRFs for the training
def generate_training_data_ADV(Nf, key):
    jax.config.update("jax_enable_x64", True)
    keys = jax.random.split(key, Nf)
    fs = vmap(GP_sample)(keys)
    jax.config.update("jax_enable_x64", False)
    return fs


@partial(jax.jit, static_argnums=(1, 2))
def DeepONet_train_generator_ADV(fs, batch, nc, key):
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
    tb = jnp.stack(
        [jax.random.uniform(keys[2], (nb, 1), minval=0.0, maxval=1.0)] * batch
    )
    xb = jnp.array([[[0.0]] * nb] * batch)
    ub = _adv_boundary_u(tb)
    ti = jnp.array([[[0.0]] * ni] * batch)
    xi = jnp.stack(
        [jax.random.uniform(keys[3], (ni, 1), minval=0.0, maxval=1.0)] * batch
    )
    ui = _adv_init_u(xi)
    return tc, xc, tb, xb, ub, ti, xi, ui, fc


@partial(jax.jit, static_argnums=(1, 2))
def SepONet_train_generator_ADV(fs, batch, nc, key):
    key, subkey = jax.random.split(key)
    idx = jax.random.choice(key, fs.shape[0], (batch,), replace=False)
    fc = fs[idx, :]
    keys = jax.random.split(subkey, 2)
    tc = jax.random.uniform(keys[0], (nc, 1), minval=0.0, maxval=1.0)
    xc = jax.random.uniform(keys[1], (nc, 1), minval=0.0, maxval=1.0)
    tb = tc
    xb = jnp.array([[0.0]])
    ub = jnp.stack([_adv_boundary_u(tb.reshape(-1, 1, 1))] * batch)
    ti = jnp.array([[0.0]])
    xi = xc
    ui = jnp.stack([_adv_init_u(xi.reshape(1, -1, 1))] * batch)
    return tc, xc, tb, xb, ub, ti, xi, ui, fc


#########################################################################
# Test generators (random functions and collocation ppints) for DeepONet and SepONet
#########################################################################
def generate_test_data_ADV(Nf, key):
    jax.config.update("jax_enable_x64", True)
    keys = jax.random.split(key, Nf)
    fs = vmap(GP_sample)(keys)  # [Nf, 128]
    UU = vmap(solve_CVC, in_axes=(0, None))(fs, 128)
    u = jnp.transpose(UU, (0, 2, 1))  # [Nf, 128, 128]
    jax.config.update("jax_enable_x64", False)
    return fs, u


@jax.jit
def DeepONet_test_generator_ADV(fs, u):
    nf = fs.shape[0]
    t = jnp.stack([jnp.linspace(0, 1, 128).reshape(-1, 1)] * nf)
    x = jnp.stack([jnp.linspace(0, 1, 128).reshape(-1, 1)] * nf)
    t_mesh, x_mesh = vmap(create_mesh, in_axes=(0, 0))(t, x)
    t = t_mesh.reshape(nf, 128**2, 1)
    x = x_mesh.reshape(nf, 128**2, 1)
    u = u.reshape(nf, 128**2, 1)
    return t, x, fs, u


@jax.jit
def SepONet_test_generator_ADV(fs, u):
    t = jnp.linspace(0, 1, 128).reshape(-1, 1)
    x = jnp.linspace(0, 1, 128).reshape(-1, 1)
    return t, x, fs, u
