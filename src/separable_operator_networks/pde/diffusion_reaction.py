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
    model, tc, xc, tb, xb, ti, xi, fc, lam_b=1.0, lam_i=1.0
):
    def residual_loss(model, t, x, f):
        # compute u
        u = model(((t, x), f))
        # num of funcs
        nf = f.shape[0]
        # tangent vector du/du
        v = jnp.ones(u.shape)
        # 1st, 2nd derivatives of u
        ut = vjp(lambda t: model(((t, x), f)), t)[1](v)[0]
        _, uxx = hvp_fwdrev(lambda x: model(((t, x), f)), (x,), (v,), True)
        # interpolation
        f_fn = lambda x, f: jnp.interp(x.flatten(), jnp.linspace(0, 1, 128), f)  # noqa: E731
        f_x = vmap(f_fn, in_axes=(0, 0))(x, f)  # shape [nf, nx*nt]
        return jnp.mean(
            (ut - 0.01 * uxx - 0.01 * (u**2) - f_x.reshape(nf, -1, 1)) ** 2
        )

    def boundary_loss(model, t, x, f):
        loss = 0
        for i in range(2):
            loss += jnp.mean(model(((t[i], x[i]), f)) ** 2)
        return loss

    def initial_loss(model, t, x, f):
        return jnp.mean(model(((t, x), f)) ** 2)

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
    model, tc, xc, tb, xb, ti, xi, fc, lam_b=1.0, lam_i=1.0
):
    def residual_loss(model, t, x, f):
        # compute u
        u = model(((t, x), f))
        # num of funcs and nx
        nf = f.shape[0]
        nx = x.shape[0]  # nt = nx here
        # tangent vector dx/dx
        v_t = jnp.ones(t.shape)
        v_x = jnp.ones(x.shape)
        # 1st, 2nd derivatives of u
        ut = jvp(lambda t: model(((t, x), f)), (t,), (v_t,))[1]
        _, uxx = hvp_fwdfwd(lambda x: model(((t, x), f)), (x,), (v_x,), True)
        # interpolation
        f_fn = lambda x, f: jnp.interp(x.flatten(), jnp.linspace(0, 1, 128), f)  # noqa: E731
        f_x = vmap(f_fn, in_axes=(None, 0))(x, f)  # shape [batch, nx]

        return jnp.mean(
            (ut - 0.01 * uxx - 0.01 * (u**2) - f_x.reshape(nf, 1, nx, 1)) ** 2
        )

    def boundary_loss(model, t, x, f):
        loss = 0
        for i in range(2):
            loss += jnp.mean(model(((t[i], x[i]), f)) ** 2)
        return loss

    def initial_loss(model, t, x, f):
        return jnp.mean(model(((t, x), f)) ** 2)

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
    f = f_fn(x)
    return f


# A diffusion-reaction numerical solver
def solve_ADR(f, Nt):
    """Solve 1D
    u_t = (k(x) u_x)_x - v(x) u_x + g(u) + f(x)
    with zero initial and boundary conditions.
    """
    xmin, xmax = 0, 1
    tmin, tmax = 0, 1
    k = lambda x: 0.01 * jnp.ones_like(x)  # noqa: E731
    v = lambda x: jnp.zeros_like(x)  # noqa: E731
    g = lambda u: 0.01 * u**2  # noqa: E731
    dg = lambda u: 0.02 * u  # noqa: E731
    u0 = lambda x: jnp.zeros_like(x)  # noqa: E731
    Nx = len(f)

    # Create grid
    x = jnp.linspace(xmin, xmax, Nx)
    t = jnp.linspace(tmin, tmax, Nt)
    h = x[1] - x[0]
    dt = t[1] - t[0]
    h2 = h**2

    # Compute coefficients and forcing
    k = k(x)
    v = v(x)

    # Compute finite difference operators
    D1 = jnp.eye(Nx, k=1) - jnp.eye(Nx, k=-1)
    D2 = -2 * jnp.eye(Nx) + jnp.eye(Nx, k=-1) + jnp.eye(Nx, k=1)
    D3 = jnp.eye(Nx - 2)
    M = -jnp.diag(D1 @ k) @ D1 - 4 * jnp.diag(k) @ D2
    m_bond = 8 * h2 / dt * D3 + M[1:-1, 1:-1]
    v_bond = 2 * h * jnp.diag(v[1:-1]) @ D1[1:-1, 1:-1] + 2 * h * jnp.diag(
        v[2:] - v[: Nx - 2]
    )
    mv_bond = m_bond + v_bond
    c = 8 * h2 / dt * D3 - M[1:-1, 1:-1] - v_bond

    # Initialize solution and apply initial condition
    u = jnp.zeros((Nx, Nt))
    u = u.at[:, 0].set(u0(x))

    # Time-stepping update
    def body_fn(i, u):
        gi = g(u[1:-1, i])
        dgi = dg(u[1:-1, i])
        h2dgi = jnp.diag(4 * h2 * dgi)
        A = mv_bond - h2dgi
        b1 = 8 * h2 * (0.5 * f[1:-1] + 0.5 * f[1:-1] + gi)
        b2 = (c - h2dgi) @ u[1:-1, i].T
        # u = index_update(u, index[1:-1, i + 1], np.linalg.solve(A, b1 + b2))
        u = u.at[1:-1, i + 1].set(jnp.linalg.solve(A, b1 + b2))
        return u

    # Run loop
    UU = jax.lax.fori_loop(0, Nt - 1, body_fn, u)
    return UU


#########################################################################
# Train generators for DeepONet and SepONet
#########################################################################
# create GRFs for the training
def generate_training_data_DR(Nf, key):
    jax.config.update("jax_enable_x64", True)
    keys = jax.random.split(key, Nf)
    fs = vmap(GP_sample)(keys)
    jax.config.update("jax_enable_x64", False)
    return fs


@partial(jax.jit, static_argnums=(1, 2))
def DeepONet_train_generator_DR(fs, batch, nc, key):
    nb = nc
    ni = nc
    key, subkey = jax.random.split(key)
    idx = jax.random.choice(key, fs.shape[0], (batch,), replace=False)
    fc = fs[idx, :]
    keys = jax.random.split(subkey, 5)
    tc = jnp.stack(
        [jax.random.uniform(keys[0], (nc**2, 1), minval=0.0, maxval=1.0)]
        * batch
    )
    xc = jnp.stack(
        [jax.random.uniform(keys[1], (nc**2, 1), minval=0.0, maxval=1.0)]
        * batch
    )
    tb = [
        jnp.stack(
            [jax.random.uniform(keys[2], (nb, 1), minval=0.0, maxval=1.0)]
            * batch
        ),
        jnp.stack(
            [jax.random.uniform(keys[3], (nb, 1), minval=0.0, maxval=1.0)]
            * batch
        ),
    ]
    xb = [jnp.array([[[0.0]] * nb] * batch), jnp.array([[[1.0]] * nb] * batch)]
    ti = jnp.array([[[0.0]] * ni] * batch)
    xi = jnp.stack(
        [jax.random.uniform(keys[4], (ni, 1), minval=0.0, maxval=1.0)] * batch
    )
    return tc, xc, tb, xb, ti, xi, fc


@partial(jax.jit, static_argnums=(1, 2))
def SepONet_train_generator_DR(fs, batch, nc, key):
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
def generate_test_data_DR(Nf, key):
    jax.config.update("jax_enable_x64", True)
    keys = jax.random.split(key, Nf)
    fs = vmap(GP_sample)(keys)  # [Nf, 128]
    UU = vmap(solve_ADR, in_axes=(0, None))(fs, 128)
    u = jnp.transpose(UU, (0, 2, 1))  # [Nf, 128, 128]
    jax.config.update("jax_enable_x64", False)
    return fs, u


@jax.jit
def DeepONet_test_generator_DR(fs, u):
    nf = fs.shape[0]
    t = jnp.stack([jnp.linspace(0, 1, 128).reshape(-1, 1)] * nf)
    x = jnp.stack([jnp.linspace(0, 1, 128).reshape(-1, 1)] * nf)
    t_mesh, x_mesh = vmap(create_mesh, in_axes=(0, 0))(t, x)
    t = t_mesh.reshape(nf, 128**2, 1)
    x = x_mesh.reshape(nf, 128**2, 1)
    u = u.reshape(nf, 128**2, 1)
    return t, x, fs, u


@jax.jit
def SepONet_test_generator_DR(fs, u):
    t = jnp.linspace(0, 1, 128).reshape(-1, 1)
    x = jnp.linspace(0, 1, 128).reshape(-1, 1)
    return t, x, fs, u
