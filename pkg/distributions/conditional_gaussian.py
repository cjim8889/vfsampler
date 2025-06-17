from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from pkg.nn.gaussian_mlp import MeanStdNet

from .base import Distribution


class ConditionalDiagGaussian(Distribution):
    """
    p(r | x) = 𝒩(r; μ(x), diag[σ(x)²])

    Parameters
    ----------
    dim_r     : int
        Dimension of r.
    mean_fn   : Callable[[..., dim_x], [..., dim_r]]
        Maps x ↦ μ(x).
    std_fn    : Callable[[..., dim_x], [..., dim_r]]
        Maps x ↦ σ(x) (element-wise std-devs, strictly > 0).
    """

    def __init__(
        self,
        dim_r: int,
        mean_std_fn: Callable[[Float[Array, "... dim_x"]], tuple[Float[Array, "... dim_r"], Float[Array, "... dim_r"]]],
    ):
        self.dim_r = dim_r
        self.mean_std_fn = mean_std_fn
        self.batched_mean_std_fn = eqx.filter_vmap(mean_std_fn)
        self._log2pi = jnp.log(2.0 * jnp.pi)

    # ------------------------------------------------------------------ #
    #  Log-prob  (broadcasts over leading batch dims of r and x)          #
    # ------------------------------------------------------------------ #
    def log_prob(
        self,
        r: Float[Array, "... dim_r"],
        x: Float[Array, "... dim_x"],
    ) -> Float[Array, "..."]:
        mu, sigma = self.mean_std_fn(x)                       # (..., dim_r)
        z    = (r - mu) / sigma
        return -0.5 * (
            self.dim_r * self._log2pi
            + 2.0 * jnp.sum(jnp.log(sigma), axis=-1)
            + jnp.sum(z ** 2, axis=-1)
        )

    # ------------------------------------------------------------------ #
    #  Sampling                                                          #
    # ------------------------------------------------------------------ #
    def sample(
        self,
        key: PRNGKeyArray,
        x: Float[Array, "... dim_x"],
        sample_shape=(),
    ) -> Float[Array, "... sample_shape dim_r"]:
        if not sample_shape:
            eps = jax.random.normal(
                key,
                shape=x.shape,
            )

        else:
            eps  = jax.random.normal(
                key,
                shape=sample_shape + x.shape[:-1] + (self.dim_r,),
            )
        
        mu, sigma = self.batched_mean_std_fn(x)
        return mu + sigma * eps


class LearnableDiagGaussian(eqx.Module):
    """
    Same public API as before, but mean/std come from a Flax Module.
    """
    net: MeanStdNet
    conditional_gaussian: ConditionalDiagGaussian = eqx.field(static=True)
    dim_r: int = eqx.field(static=True)

    def __init__(self, dim_r: int, dim_x: int, key: PRNGKeyArray):
        self.net = MeanStdNet(dim_r, dim_x, key)
        self.conditional_gaussian = ConditionalDiagGaussian(
            dim_r,
            mean_std_fn=lambda x: self.net(x),
        )

        self.dim_r = dim_r

    def log_prob(self, r: Float[Array, "... dim_r"], x: Float[Array, "... dim_x"]) -> Float[Array, "..."]:
        return self.conditional_gaussian.log_prob(r, x)

    def sample(self, key: PRNGKeyArray, x: Float[Array, "... dim_x"], sample_shape=()) -> Float[Array, "... sample_shape dim_r"]:
        return self.conditional_gaussian.sample(key, x, sample_shape)
