from typing import Callable, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from pkg.nn.mlp import MLPVelocityField


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
    test_fn: Callable[[Float[Array, "dim"], float], Tuple[float, Float[Array, "dim"]]],
) -> Tuple[float, float]:
    """Computes the local error using a Particle instance.
    
    Args:
        v_theta: Velocity field model
        particle: Particle containing x, t, dt_logZt
        score_fn: Score function
        time_derivative_log_density: Time derivative of log density
        test_fn: Test function that takes x, t and returns scalar (phi, grad_phi)
        
    Returns:
        residual: The computed residual value
        phi: The test function value at (x, t)
    """
    x, t, dt_logZt = particle.x, particle.t, particle.dt_logZt

    score = score_fn(x, t)
    v = v_theta(x, t)

    dt_log_density_unormalised = time_derivative_log_density(x, t)
    dt_log_density = dt_log_density_unormalised - dt_logZt

    # Use the provided test function instead of fixed fourier modes
    phi, grad_phi = test_fn(x, t)
    grad_phi = grad_phi.reshape(-1, x.shape[0])


    first_term = phi * dt_log_density  # scalar
    second_term = phi * (jnp.sum(score * v))  # scalar
    third_term = - jnp.dot(grad_phi, v)  # scalar
    residual = first_term + second_term + third_term

    print(f"Shape of grad_phi: {grad_phi.shape}")
    print(f"Shape of phi: {phi.shape}")

    return residual, jnp.mean(grad_phi ** 2, axis=1)

batched_epsilon = jax.vmap(epsilon, in_axes=(None, 0, None, None, None))

def loss_fn(
    v_theta: MLPVelocityField,  
    particles: Particle,
    time_derivative_log_density: Callable[[Float[Array, "dim"], float], float],
    score_fn: Callable[[Float[Array, "dim"], float], Float[Array, "dim"]],
    test_fn: Callable[[Float[Array, "dim"], float], Tuple[float, Float[Array, "dim"]]],
) -> Tuple[float, Float[Array, "batch"], Float[Array, "batch"]]:
    """Compute loss using the provided test function.
    
    Args:
        v_theta: Velocity field model
        particles: Batch of particles
        time_derivative_log_density: Time derivative of log density
        score_fn: Score function
        test_fn: Test function that takes x, t and returns scalar (phi, grad_phi)
        
    Returns:
        loss: Mean squared residual
        raw_epsilons: Raw residual values per particle
        phi_values: Test function output values per particle
    """

    raw_epsilons, phi_values = batched_epsilon(
        v_theta,
        particles,
        score_fn,
        time_derivative_log_density,    
        test_fn,
    )

    _loss = jnp.mean(raw_epsilons**2)
    return _loss, raw_epsilons, phi_values