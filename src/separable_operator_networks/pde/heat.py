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

from diffrax import diffeqsolve, ODETerm, SaveAt, PIDController, Tsit5
from jaxtyping import Float, Array

#########################################################################
# Generate Ground Truth Data
#########################################################################

class SpatialDiscretisation(eqx.Module):
    x0: float = eqx.field(static=True)
    x_final: float = eqx.field(static=True)
    vals: Float[Array, str("n")]

    @property
    def δx(self):
        return (self.x_final - self.x0) / (len(self.vals) - 1)

    def binop(self, other, fn):
        if isinstance(other, SpatialDiscretisation):
            if self.x0 != other.x0 or self.x_final != other.x_final:
                raise ValueError("Mismatched spatial discretisations")
            other = other.vals
        return SpatialDiscretisation(self.x0, self.x_final, fn(self.vals, other))

    def __add__(self, other):
        return self.binop(other, lambda x, y: x + y)

    def __mul__(self, other):
        return self.binop(other, lambda x, y: x * y)

    def __radd__(self, other):
        return self.binop(other, lambda x, y: y + x)

    def __rmul__(self, other):
        return self.binop(other, lambda x, y: y * x)

    def __sub__(self, other):
        return self.binop(other, lambda x, y: x - y)

    def __rsub__(self, other):
        return self.binop(other, lambda x, y: y - x)

def laplacian(y: SpatialDiscretisation) -> SpatialDiscretisation:
    y_next = jnp.roll(y.vals, shift=1)
    y_prev = jnp.roll(y.vals, shift=-1)
    Δy = (y_next - 2 * y.vals + y_prev) / ((jnp.pi*y.δx)**2)
    # Dirichlet boundary condition
    Δy = Δy.at[0].set(0)
    Δy = Δy.at[-1].set(0)
    return SpatialDiscretisation(y.x0, y.x_final, Δy)

def solve_heat(y0):
    xmin, xmax = 0., 1.
    tmin, tmax = 0., 1.
    Nt = 128

    def vector_field(t, y, args):
        return laplacian(y)

    teval = jnp.linspace(tmin, tmax, Nt)
    δt = 0.0001

    y0 = y0.at[0].set(0)
    y0 = y0.at[-1].set(0)

    y0 = SpatialDiscretisation(xmin, xmax, y0)

    term = ODETerm(vector_field)
    saveat = SaveAt(ts=teval)
    rtol = 1e-10
    atol = 1e-10
    solver = Tsit5()
    stepsize_controller = PIDController(
        pcoeff=0.3, icoeff=0.4, rtol=rtol, atol=atol, dtmax=0.001,
    )

    sol = diffeqsolve(term, solver, t0=tmin, t1=tmax, dt0=δt, y0=y0, max_steps=16**5, saveat=saveat,
                      stepsize_controller=stepsize_controller)

    return sol.ys.vals

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

    # boundary conditions
    f = f * 4 * (1 - x) * x
    f = f.at[0].set(0)
    f = f.at[-1].set(0)
    return f

def generate_training_data(Nf=10000, key=None):
    keys = jax.random.split(key, Nf)
    y0 = jax.vmap(lambda key: GP_sample(key))(keys)
    return y0

def generate_test_data(Nf=100, key=None):
    y0 = generate_training_data(Nf=Nf, key=key)
    y = jax.vmap(solve_heat)(y0)
    return y0, y


#########################################################################
# Loss functions for DeepONet/SepONet Models
#########################################################################


# Deep Operator PI Loss
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
        _, uxx = hvp_fwdrev(lambda x: model(((t, x), f)), (x,), (v,), True)

        return jnp.mean((ut - uxx / (jnp.pi**2)) ** 2)

    def boundary_loss(model, t, x, f):
        loss = 0
        for i in range(2):
            loss += jnp.mean(model(((t[i], x[i]), f)) ** 2)
        return loss

    def initial_loss(model, t, x, f):
        # interpolation
        nf = f.shape[0]
        f_fn = lambda x, f: jnp.interp(x.flatten(), jnp.linspace(0, 1, 128), f)  # noqa: E731
        f_x = vmap(f_fn, in_axes=(0, 0))(x, f)

        return jnp.mean((model(((t, x), f)) - f_x.reshape(nf, -1, 1)) ** 2)

    loss_fn = (  # noqa: E731
        lambda model: residual_loss(model, tc, xc, fc)
        + lam_b * boundary_loss(model, tb, xb, fc)
        + lam_i * initial_loss(model, ti, xi, fc)
    )

    loss, gradient = eqx.filter_value_and_grad(loss_fn)(model)

    return loss, gradient


# Separable Deep Operator PI Loss
@eqx.filter_jit
def apply_model_SepONet(
    model, tc, xc, tb, xb, ti, xi, fc, lam_b=1.0, lam_i=20.0
):
    def residual_loss(model, t, x, f):
        # tangent vector dx/dx
        v_t = jnp.ones(t.shape)
        v_x = jnp.ones(x.shape)
        # 1st, 2nd derivatives of u
        ut = jvp(lambda t: model(((t, x), f)), (t,), (v_t,))[1]
        _, uxx = hvp_fwdfwd(lambda x: model(((t, x), f)), (x,), (v_x,), True)

        return jnp.mean((ut - uxx / (jnp.pi**2)) ** 2)

    def boundary_loss(model, t, x, f):
        loss = 0
        for i in range(2):
            loss += jnp.mean(model(((t[i], x[i]), f)) ** 2)
        return loss

    def initial_loss(model, t, x, f):
        nf = f.shape[0]
        # interpolation
        f_fn = lambda x, f: jnp.interp(x.flatten(), jnp.linspace(0, 1, 128), f)  # noqa: E731
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
def DeepONet_train_generator_heat(fs, batch, nc, key):
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
def SepONet_train_generator_heat(fs, batch, nc, key):
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
def DeepONet_test_generator_heat(fs, u):
    nf = fs.shape[0]
    t = jnp.stack([jnp.linspace(0, 1, 128).reshape(-1, 1)] * nf)
    x = jnp.stack([jnp.linspace(0, 1, 128).reshape(-1, 1)] * nf)
    t_mesh, x_mesh = vmap(create_mesh, in_axes=(0, 0))(t, x)
    t = t_mesh.reshape(nf, 128**2, 1)
    x = x_mesh.reshape(nf, 128**2, 1)
    u = u.reshape(nf, 128**2, 1)
    return t, x, fs, u


@jax.jit
def SepONet_test_generator_heat(fs, u):
    t = jnp.linspace(0, 1, 128).reshape(-1, 1)
    x = jnp.linspace(0, 1, 128).reshape(-1, 1)
    return t, x, fs, u
