from typing import Callable, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .utils import axis_aligned_fourier_modes
from pkg.nn.mlp import MLPVelocityField
import chex

class Particle(eqx.Module):
    x: Float[Array, "dim"]     
    t: float
    dt_logZt: Float[Array, "1"]


@eqx.filter_jit
def epsilon(
    v_theta: MLPVelocityField,
    particle: Particle,
    score_fn: Callable[[Float[Array, "dim"], float], Float[Array, "dim"]],
    time_derivative_log_density: Callable[[Float[Array, "dim"], float], float],
    num_frequencies: int,
) -> Float[Array, "num_frequencies"]:
    """Computes the local error using a Particle instance."""
    x, t, dt_logZt = particle.x, particle.t, particle.dt_logZt

    score = score_fn(x, t)
    v = v_theta(x, t)

    dt_log_density_unormalised = time_derivative_log_density(x, t)
    # dt_log_density = dt_log_density_unormalised - dt_logZt


    phi, grad_phi = axis_aligned_fourier_modes(x, num_frequencies, domain_range=(-50., 50.0))

    first_term = phi * dt_log_density_unormalised # (n_frequencies, )
    second_term = phi * (jnp.sum(score * v)) # (n_frequencies, )
    third_term = - jnp.sum(grad_phi * v) # (1, )

    residual = first_term + second_term + third_term

    chex.assert_shape(residual, (num_frequencies * 2 * x.shape[0],))
    return residual

batched_epsilon = jax.vmap(epsilon, in_axes=(None, 0, None, None, None))

def loss_fn(
    v_theta: MLPVelocityField,  
    particles: Particle,
    time_derivative_log_density: Callable[[Float[Array, "dim"], float], float],
    score_fn: Callable[[Float[Array, "dim"], float], Float[Array, "dim"]],
    num_frequencies: int,
) -> Tuple[float, Float[Array, "batch"]]:

    raw_epsilons = batched_epsilon(
        v_theta,
        particles,
        score_fn,
        time_derivative_log_density,    
        num_frequencies,
    )

    _loss = jnp.mean(raw_epsilons**2)
    return _loss, raw_epsilons