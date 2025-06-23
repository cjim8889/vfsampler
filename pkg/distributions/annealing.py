import chex
import equinox as eqx
import jax
from jaxtyping import Array, Float, PRNGKeyArray

from .base import Distribution


class AnnealedDistribution(eqx.Module, Distribution):
    # Static / differentiable field specifications for Equinox
    initial_density: Distribution = eqx.field(static=True)
    target_density: Distribution = eqx.field(static=True)
    dim: int = eqx.field(static=True)
    n_samples_eval: int = eqx.field(static=True)

    def __init__(
        self,
        initial_density: Distribution,
        target_density: Distribution,
    ):
        # Assign fields (runtime). Static/differentiable nature is declared at class level below.
        self.initial_density = initial_density
        self.target_density = target_density
        self.dim = initial_density.dim
        self.n_samples_eval = initial_density.n_samples_eval

    def log_prob(self, x: Float[Array, "n_samples dim"]) -> Float[Array, "n_samples"]:
        raise NotImplementedError("log_prob is not implemented for AugmentedAnnealedDistribution")

    def time_dependent_unnormalised_log_prob(self, x: Float[Array, "dim"], t: Float[Array, ""]) -> Float[Array, ""]:
        chex.assert_shape(x, (self.dim,))
        chex.assert_shape(t, ())

        initial_prob = (1 - t) * self.initial_density.log_prob(x)
        target_prob = t * self.target_density.log_prob(x)

        return initial_prob + target_prob

    def unnormalised_time_derivative(self, xr: Float[Array, "dim"], t: float) -> Float[Array, ""]:
        return jax.grad(lambda t: self.time_dependent_unnormalised_log_prob(xr, t))(t)

    def score_fn(self, xr: Float[Array, "dim"], t: float) -> Float[Array, "dim"]:
        return jax.grad(lambda x: self.time_dependent_unnormalised_log_prob(x, t))(xr)

    def sample_initial(self, key: PRNGKeyArray, shape: Array) -> Float[Array, "n_samples dim"]:
        return self.initial_density.sample(key, shape)

    