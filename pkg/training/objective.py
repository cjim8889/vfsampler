from typing import Callable, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .utils import divergence_wrt_R
from pkg.nn.mlp import AugmentedResidualField


class Particle(eqx.Module):
    xr: Float[Array, "dim"]     
    t: float
    dt_logZt: Float[Array, "1"]


@eqx.filter_jit
def epsilon(
    v_theta: AugmentedResidualField,
    particle: Particle,
    score_fn: Callable[[Float[Array, "dim"], float], Float[Array, "dim"]],
    time_derivative_log_density: Callable[[Float[Array, "dim"], float], float],
    r_dim: int,
) -> Float[Array, "1"]:
    """Computes the local error using a Particle instance."""
    xr, t, dt_logZt = particle.xr, particle.t, particle.dt_logZt

    score = score_fn(xr, t)
    score_x = score[:-r_dim]
    score_r = score[-r_dim:]

    div_v_r = divergence_wrt_R(v_theta.residual_v, xr, t, r_dim)
    v = v_theta(xr, t)
    v_x = v[:-r_dim]
    v_r = v[-r_dim:]

    dt_log_density_unormalised = time_derivative_log_density(xr, t)
    dt_log_density = dt_log_density_unormalised - dt_logZt

    lhs = div_v_r + jnp.sum(v_r * score_r)
    G = -jnp.sum(v_x * score_x) - dt_log_density

    result = lhs - G

    return result, G

batched_epsilon = jax.vmap(epsilon, in_axes=(None, 0, None, None, None))

def loss_fn(
    v_theta: AugmentedResidualField,
    particles: Particle,
    time_derivative_log_density: Callable[[Float[Array, "dim"], float], float],
    score_fn: Callable[[Float[Array, "dim"], float], Float[Array, "dim"]],
    r_dim: int,
    soft_constraint: bool = False,
) -> Tuple[float, Float[Array, "batch"]]:

    raw_epsilons, Gs = batched_epsilon(
        v_theta,
        particles,
        score_fn,
        time_derivative_log_density,
        r_dim,
    )

    _loss = jnp.mean(raw_epsilons**2)

    if soft_constraint:
        _loss = _loss + jnp.mean(Gs**2)

    return _loss, raw_epsilons