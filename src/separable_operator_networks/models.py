"""
This module provides the implementation of the PINN, SPINN, DeepONet,
and SepONet models.
"""

__author__ = "Sean Hooten, Xinling Yu"
__email__ = "sean.hooten@hpe.com, xyu644@ucsb.edu"
__copyright__ = "Copyright 2024 Hewlett Packard Enterprise Development LP."
__license__ = "MIT"
__version__ = "0.0.1"

import jax
import jax.numpy as jnp
import equinox as eqx
import equinox.nn as nn

from . import utils

# einsum keys
# current support up to 10 dimensions (just add more characters if needed).

c2 = ["i", "j", "k", "l", "m", "n", "o", "p", "q", "r"]
c1 = [c + "yz" for c in c2]
c3 = [c + "xyz" for c in c2]

#########################################################################
# PINN and SPINN models
#########################################################################


class PINN(eqx.Module):
    """Physics Informed Neural Network (PINN) model class.

    Maps spatiotemporal points to field predictions along those points.

    Attributes:
        dim: int
            The number of spatiotemporal dimensions.
        field_dim: int
            The number of output fields.
        rank: int
            The rank (trunk output size) of the PINN model.
        trunk: eqx.Module
            The neural network of the PINN model.
    """

    dim: int
    field_dim: int
    rank: int
    trunk: eqx.Module

    def __init__(
        self,
        dim: int,
        field_dim: int = 1,
        depth: int = 3,
        hidden: int = 64,
        rank: int = 64,
        activation: callable = jax.nn.gelu,
        final_activation: callable = jax.nn.tanh,
        key: jax.random.key = None,
    ) -> eqx.Module:
        """Initialize the PINN model.

        Args:
            dim: int
                The number of spatiotemporal dimensions.
            field_dim: int
                The number of output fields.
            depth: int
                The number of hidden layers in the neural network.
            hidden: int
                The number of hidden units in each hidden layer.
            rank: int
                The rank (trunk output size) of the PINN model.
            activation: callable
                The activation function for the hidden layers.
            final_activation: callable
                The activation function for the final layer.
            key: jax.random.key
                The random key for initializing the neural network.

        Returns:
            eqx.Module
                The initialized PINN model.

        Note:
            Potentially will replace all kwargs with a config dictionary.
        """

        super().__init__()

        self.trunk = eqx.filter_vmap(
            nn.MLP(
                dim,
                rank * field_dim,
                hidden,
                depth,
                activation=activation,
                final_activation=final_activation,
                key=key,
            )
        )  # vmap over number of collocation points
        self.dim = dim
        self.field_dim = field_dim
        self.rank = rank

    def __call__(self, x: tuple[jnp.ndarray, ...]) -> jnp.ndarray:
        """Forward pass of the PINN model.

        Args:
            x: tuple
                The input spatiotemporal points. Each element of the length
                self.dim tuple is a jnp.ndarray with shape [N, 1] (before vmap).
                where N is the total number of collocation points.

        Returns:
            jnp.ndarray
                The field predictions along the spatiotemporal points.
                Will be shape [N, self.field_dim] (before vmap).
        """
        x = jnp.concatenate(x, axis=-1)  # [N, dim]
        t = self.trunk(x).reshape(
            -1, self.field_dim, self.rank
        )  # [N, field_dim, rank]
        return jnp.sum(t, axis=-1)  # [N, field_dim]


class SPINN(eqx.Module):
    """Separable Physics Informed Neural Network (SPINN) model class.

    Maps spatiotemporal points to field predictions along those points.
    The SPINN model uses a separable neural network architecture.

    Attributes:
        dim: int
            The number of spatiotemporal dimensions.
        field_dim: int
            The number of output fields.
        rank: int
            The rank (trunk output size) of the PINN model.
        trunk: list[eqx.Module]
            The neural networks of the SPINN model.
            There will be self.dim independent neural networks.
        outer_product_string: str
            The einsum string for the outer product of the SPINN model.
    """

    dim: int
    field_dim: int
    rank: int
    trunk: list[eqx.Module]
    outer_product_string: str

    def __init__(
        self,
        dim: int,
        field_dim: int = 1,
        depth: int = 3,
        hidden: int = 64,
        rank: int = 64,
        activation: callable = jax.nn.gelu,
        final_activation: callable = jax.nn.tanh,
        key: jax.random.key = None,
    ) -> eqx.Module:
        """Initialize the SPINN model.

        Args:
            dim: int
                The number of spatiotemporal dimensions.
            field_dim: int
                The number of output fields.
            depth: int
                The number of hidden layers in the neural network.
            hidden: int
                The number of hidden units in the neural network.
            rank: int
                The rank (trunk output size) of the SPINN model.
            activation: callable
                The activation function for the hidden layers.
            final_activation: callable
                The activation function for the final layer.
            key: jax.random.key
                The random key for initializing the neural network.

        Returns:
            eqx.Module
                The initialized SPINN model.

        Note:
            Potentially will replace all kwargs with a config dictionary.
        """
        super().__init__()

        def make_ensemble(keys: tuple[jax.random.key, ...]) -> list[eqx.Module]:
            """Create an ensemble of neural networks.

            Args:
                keys: tuple[jax.random.key, ...]
                    The random keys for initializing the neural networks.

            Returns:
                list[eqx.Module]
                    The initialized ensemble of neural networks.

            Note:
                Probably should be able to make this more efficient with vmap,
                but the issue is that each mlp might see different sized data
            """
            mlps = []
            for key in keys:
                mlp = eqx.filter_vmap(
                    nn.MLP(
                        1,
                        rank * field_dim,
                        hidden,
                        depth,
                        activation=activation,
                        final_activation=final_activation,
                        key=key,
                    )
                )  # vmap over number of points per dim
                mlps.append(mlp)
            return mlps

        subkeys = jax.random.split(key, num=dim)  # need dim separate mlps
        trunk = make_ensemble(subkeys)  # would be nice if we could vmap

        self.trunk = trunk
        self.dim = dim
        self.field_dim = field_dim
        self.rank = rank

        s1 = ""
        s2 = ""
        for i in range(dim):
            s1 = s1 + c1[i] + ","
            s2 = s2 + c2[i]
        self.outer_product_string = (
            s1[:-1] + "->" + s2 + "y"
        )  # e.g. 'iyz,jyz->ijy'

    def __call__(self, x: tuple[jnp.ndarray, ...]) -> jnp.ndarray:
        """Forward pass of the SPINN model.

        Args:
            x: tuple
                The input spatiotemporal points. Each element of the length
                self.dim tuple is a jnp.ndarray with shape [N_i, 1] (before vmap).
                where N_i is the the number of points sampled along dimension i.

        Returns:
            jnp.ndarray
                The field predictions along the spatiotemporal points.
                Will be shape [N_1, N_2, ..., N_dim, self.field_dim] (before vmap).
        """
        ts = []
        for i in range(len(x)):
            # can probably improve using jax.lax.scan
            ts.append(
                self.trunk[i](x[i]).reshape(
                    -1, self.field_dim, self.rank
                )  # [N_i, field_dim, rank]
            )
        return jnp.einsum(
            self.outer_product_string, *ts, optimize="optimal"
        )  # shape [N_1, ..., N_d, field_dim]


#########################################################################
# DEEPONET and SDEEPONET models
#########################################################################


class DeepONet(eqx.Module):
    """Deep operator network model class for operator learning and parametric PDEs.

    Maps input configuration function and spatiotemporal points to
    output function predictions along those points.

    Attributes:
        dim: int
            The number of spatiotemporal dimensions.
        branch_dim: int
            The number of measurements in input function space.
        field_dim: int
            The number of output fields.
        rank: int
            The rank (branch and trunk output size) of the branch and trunk nets.
        trunk: eqx.Module
            Decoder model of the DeepONet
        branch: eqx.Module
            Approximator model of the DeepONet
    """

    dim: int
    branch_dim: int
    field_dim: int
    rank: int
    trunk: eqx.Module
    branch: eqx.Module

    def __init__(
        self,
        dim: int,
        branch_dim: int,
        field_dim: int = 1,
        depth: int = 3,
        hidden: int = 64,
        rank: int = 64,
        branch_activation: callable = jax.nn.tanh,
        branch_final_activation: callable = utils.identity,
        trunk_activation: callable = utils.sine,
        trunk_final_activation: callable = utils.sine,
        key: jax.random.key = None,
    ) -> eqx.Module:
        """Initialize the DeepONet model.

        Args:
            dim: int
                The number of spatiotemporal dimensions.
            branch_dim: int
                The number of measurements in input function space.
            field_dim: int
                The number of output fields.
            depth: int
                The number of hidden layers in the branch and trunk nets.
            hidden: int
                The number of hidden units in the branch and trunk nets.
            rank: int
                The rank (branch and trunk output size) of the branch and trunk nets.
            branch_activation: callable
                The activation function for the branch hidden layers.
            branch_final_activation: callable
                The activation function for the final layer of the branch.
            trunk_activation: callable
                The activation function for the trunk hidden layers.
            trunk_final_activation: callable
                The activation function for the final layer of the trunk.
            key: jax.random.key
                The random key for initializing the branch and trunk nets.

        Returns:
            eqx.Module
                The initialized DeepONet model.

        Note:
            Potentially will replace NN kwargs with config dictionaries.
        """
        super().__init__()

        subkey1, subkey2 = jax.random.split(key)

        self.trunk = eqx.filter_vmap(
            nn.MLP(
                dim,
                rank * field_dim,
                hidden,
                depth,
                activation=trunk_activation,
                final_activation=trunk_final_activation,
                key=subkey1,
            )
        )  # vmap over number of points

        self.branch = nn.MLP(
            branch_dim,
            rank * field_dim,
            hidden,
            depth,
            activation=branch_activation,
            final_activation=branch_final_activation,
            key=subkey2,
        )

        self.dim = dim
        self.branch_dim = branch_dim
        self.field_dim = field_dim
        self.rank = rank

    def __call__(
        self, x__f: tuple[tuple[jnp.ndarray, ...], jnp.ndarray]
    ) -> jnp.ndarray:
        """Forward pass of the DeepONet model.

        Args:
            x__f: tuple
                The input spatiotemporal points and function measurements.
                x, f = x__f
                x: tuple
                    The input spatiotemporal points. Each element of the length
                    self.dim tuple is a jnp.ndarray with shape [N, 1] (before vmap)
                    where N is the total number of collocation points.
                f: jnp.array
                    The function measurements in input function space.
                    Should be shape [branch_dim] (before vmap).

        Returns:
            jnp.ndarray
                The field predictions along the spatiotemporal points.
                Will be shape [N, self.field_dim] (before vmap).
        """
        x, f = x__f
        x = jnp.concatenate(x, axis=-1)
        t = self.trunk(x).reshape(
            -1, self.field_dim, self.rank
        )  # [N, field_dim, rank]
        b = self.branch(f).reshape(
            self.field_dim, self.rank
        )  # [field_dim, rank]
        return jnp.einsum(
            "ijk,jk->ij", t, b, optimize="optimal"
        )  # [N, field_dim]


class SepONet(eqx.Module):
    """Separable operator network model class for operator learning and parametric PDEs.

    Maps input configuration function and spatiotemporal points to
    output function predictions along those points.

    SepONet uses a separable neural network architecture for each trunk network.
    This allows for the model to scale to higher dimensions and more collocation points
    by leveraging forward AD in the evaluation of high order derivatives.

    Attributes:
        dim: int
            The number of spatiotemporal dimensions.
        branch_dim: int
            The number of measurements in input function space.
        field_dim: int
            The number of output fields.
        rank: int
            The rank (branch and trunk output size) of the branch and trunk nets.
        trunk: list[eqx.Module]
            Decoder model of the SepONet. There will be self.dim independent neural
            networks in the trunk net ensemble.
        branch: eqx.Module
            Approximator model of the SepONet
    """

    dim: int
    branch_dim: int
    field_dim: int
    rank: int
    trunk: list[eqx.Module]
    branch: eqx.Module
    outer_product_string: str

    def __init__(
        self,
        dim: int,
        branch_dim: int,
        field_dim: int = 1,
        depth: int = 3,
        hidden: int = 64,
        rank: int = 64,
        branch_activation: callable = jax.nn.tanh,
        branch_final_activation: callable = utils.identity,
        trunk_activation: callable = utils.sine,
        trunk_final_activation: callable = utils.sine,
        key: jax.random.key = None,
    ) -> eqx.Module:
        """Initialize the SepONet model.

        Args:
            dim: int
                The number of spatiotemporal dimensions.
            branch_dim: int
                The number of measurements in input function space.
            field_dim: int
                The number of output fields.
            depth: int
                The number of hidden layers in the branch and trunk nets.
            hidden: int
                The number of hidden units in the branch and trunk nets.
            rank: int
                The rank (branch and trunk output size) of the branch and trunk nets.
            branch_activation: callable
                The activation function for the branch hidden layers.
            branch_final_activation: callable
                The activation function for the final layer of the branch.
            trunk_activation: callable
                The activation function for the trunk hidden layers.
            trunk_final_activation: callable
                The activation function for the final layer of the trunks.
            key: jax.random.key
                The random key for initializing the branch and trunk nets.

        Returns:
            eqx.Module
                The initialized SepONet model.

        Note:
            Potentially will replace NN kwargs with config dictionaries.
        """
        super().__init__()

        def make_ensemble(keys: tuple[jax.random.key, ...]) -> list[eqx.Module]:
            """Create an ensemble of neural networks.

            Args:
                keys: tuple[jax.random.key, ...]
                    The random keys for initializing the neural networks.

            Returns:
                list[eqx.Module]
                    The initialized ensemble of neural networks.

            Note:
                Probably should be able to make this more efficient with vmap,
                but the issue is that each mlp might see different sized data
            """
            mlps = []
            for i in range(len(keys)):
                mlp = eqx.filter_vmap(
                    nn.MLP(
                        1,
                        rank * field_dim,
                        hidden,
                        depth,
                        activation=trunk_activation,
                        final_activation=trunk_final_activation,
                        key=keys[i],
                    )
                )  # vmap over number of points per dim
                mlps.append(mlp)
            return mlps

        subkeys = jax.random.split(key, num=dim + 1)  # need dim separate mlps
        trunk = make_ensemble(subkeys[:-1])

        branch = eqx.filter_vmap(
            nn.MLP(
                branch_dim,
                rank * field_dim,
                hidden,
                depth,
                activation=branch_activation,
                final_activation=branch_final_activation,
                key=subkeys[-1],
            )
        )  # vmap over number of functions

        self.dim = dim
        self.field_dim = field_dim
        self.branch_dim = branch_dim
        self.trunk = trunk
        self.branch = branch
        self.rank = rank

        s1 = ""
        s2 = ""
        for i in range(dim):
            s1 = s1 + c1[i] + ","
            s2 = s2 + c2[i]
        self.outer_product_string = (
            s1 + "byz" + "->" + "b" + s2 + "y"
        )  # e.g. 'iyz,jyz,byz->bijy'
        print(self.outer_product_string)

    def __call__(
        self,
        x__f: tuple[tuple[jnp.ndarray, ...], jnp.ndarray],
        return_basis: bool = False,
    ) -> jnp.ndarray:
        """Forward pass of the SepONet model.

        Args:
            x__f: tuple
                The input spatiotemporal points and function measurements.
                x, f = x__f
                x: tuple
                    The input spatiotemporal points. Each element of the length
                    self.dim tuple is a jnp.ndarray with shape [N_i, 1] (before vmap)
                    where N_i is the the number of points sampled along dimension i.
                f: jnp.array
                    The function measurements in input function space.
                    Should be shape [N_f, branch_dim] (before vmap).
                    N_f is the number of functions.
            return_basis: bool
                If True, will return the basis tensors used in the outer product

        Returns:
            if not return_basis:
            jnp.ndarray
                The field predictions along the spatiotemporal points.
                Will be shape [N_f, N_1, ..., N_dim, self.field_dim] (before vmap).
        """
        x, f = x__f
        ts = []
        for i in range(len(x)):
            ts.append(
                self.trunk[i](x[i]).reshape(
                    -1, self.field_dim, self.rank
                )  # [Nx, field_dim, rank]
            )

        b = self.branch(f).reshape(
            -1, self.field_dim, self.rank
        )  # [Nf, field_dim, rank]

        if return_basis:
            return (
                ts,
                b,
                jnp.einsum(
                    self.outer_product_string, *ts, b, optimize="optimal"
                ),  # [N_f, N_1, ..., N_dim, field_dim]
            )
        else:
            return jnp.einsum(
                self.outer_product_string, *ts, b, optimize="optimal"
            )  # [N_f, N_1, ..., N_dim, field_dim]
