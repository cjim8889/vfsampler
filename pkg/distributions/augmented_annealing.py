from typing import Any

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from .base import Distribution


class AugmentedAnnealedDistribution(eqx.Module, Distribution):
    # Static / differentiable field specifications for Equinox
    initial_density: Distribution = eqx.field(static=True)
    target_density: Distribution = eqx.field(static=True)
    augmented_density: Any  # learnable eqx.Module
    dim: int = eqx.field(static=True)
    n_samples_eval: int = eqx.field(static=True)
    augmented_dim: int = eqx.field(static=True)
    is_conditional: bool = eqx.field(static=True)

    def __init__(
        self,
        initial_density: Distribution,
        target_density: Distribution,
        augmented_density: Distribution,
        augmented_dim: int,
        is_conditional: bool = False,
    ):
        # Assign fields (runtime). Static/differentiable nature is declared at class level below.
        self.initial_density = initial_density
        self.target_density = target_density
        self.augmented_density = augmented_density  # learnable component (eqx.Module)
        self.dim = initial_density.dim + augmented_dim
        self.n_samples_eval = initial_density.n_samples_eval
        self.augmented_dim = augmented_dim
        self.is_conditional = is_conditional

    def log_prob(self, x: Float[Array, "n_samples dim"]) -> Float[Array, "n_samples"]:
        raise NotImplementedError("log_prob is not implemented for AugmentedAnnealedDistribution")

    def time_dependent_log_prob_without_augmentation(self, x: Float[Array, "dim"], t: Float[Array, ""]) -> Float[Array, ""]:
        chex.assert_shape(x, (self.dim - self.augmented_dim,))
        chex.assert_shape(t, ())

        initial_prob = (1 - t) * self.initial_density.log_prob(x)
        target_prob = t * self.target_density.log_prob(x)

        return initial_prob + target_prob
    
    def time_dependent_log_prob(self, xr: Float[Array, "dim"], t: Float[Array, ""]) -> Float[Array, ""]:
        x, r = xr[:self.dim - self.augmented_dim], xr[self.dim - self.augmented_dim:]

        chex.assert_shape(x, (self.dim - self.augmented_dim,))
        chex.assert_shape(r, (self.augmented_dim,))

        initial_prob = (1 - t) * self.initial_density.log_prob(x)
        target_prob = t * self.target_density.log_prob(x)
        augmented_prob = self.augmented_density.log_prob(r, x) if self.is_conditional else self.augmented_density.log_prob(r)

        return initial_prob + target_prob + augmented_prob

    def unnormalised_time_derivative(self, xr: Float[Array, "dim"], t: float) -> Float[Array, ""]:
        return jax.grad(lambda t: self.time_dependent_log_prob(xr, t))(t)

    def score_fn(self, xr: Float[Array, "dim"], t: float) -> Float[Array, "dim"]:
        return jax.grad(lambda x: self.time_dependent_log_prob(x, t))(xr)

    def sample_initial(self, key: PRNGKeyArray, shape: Array) -> Float[Array, "n_samples dim"]:
        return self.initial_density.sample(key, shape)
    
    def sample_augmented(self, key: PRNGKeyArray, shape: Array) -> Float[Array, "n_samples dim"]:
        return self.augmented_density.sample(key, shape)
    