from typing import Callable, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from pkg.nn.mlp import MLPVelocityField


class Particle(eqx.Module):
    x: Float[Array, "dim"]   
    x_0: Float[Array, "dim"]
    t: float
    # dt_logZt: Float[Array, "1"]


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

@eqx.filter_jit
def LHS_expectation_form(
    v_theta: MLPVelocityField,
    x: Float[Array, "dim"],
    t: float,
    test_fn: Callable[[Float[Array, "dim"], float], float],
) -> Tuple[float, float]:
    """Computes the local error using a Particle instance."""

    dt_test_fn = jax.grad(lambda t: test_fn(x, t))(t)
    dx_test_fn = jax.grad(lambda x: test_fn(x, t))(x)

    v_x = v_theta(x, t)

    grad_norm_xt = jnp.sqrt(jnp.sum(jnp.square(dx_test_fn)) + jnp.square(dt_test_fn))

    return jnp.sum(v_x * dx_test_fn) + dt_test_fn, grad_norm_xt

batched_LHS_expectation_form = jax.vmap(LHS_expectation_form, in_axes=(None, 0, 0, None))

@eqx.filter_jit
def RHS_expectation_form(
    x_0: Float[Array, "batch_dim dim"],
    x_1: Float[Array, "batch_dim dim"],
    test_fn: Callable[[Float[Array, "dim"], float], float],
) -> Tuple[float, float]:
    """Computes the RHS expectation form of the test function."""
    batch_test_fn_0 = jax.vmap(lambda x: test_fn(x, jnp.array([0.0])))(x_0)
    batch_test_fn_1 = jax.vmap(lambda x: test_fn(x, jnp.array([1.0])))(x_1)
    return jnp.mean(batch_test_fn_0), jnp.mean(batch_test_fn_1)
    
def loss_fn_expectation_form(
    v_theta: MLPVelocityField,
    particles: Particle,
    test_fn: Callable[[Float[Array, "dim"], float], float],
) -> Tuple[float, float]:
    """Computes the loss using the expectation form of the test function."""

    LHS, grad_norm_xt  = batched_LHS_expectation_form(v_theta, particles.x, particles.t, test_fn)
    fn_0, fn_1 = RHS_expectation_form(particles.x_0, particles.x_1, test_fn)
    return jnp.mean(LHS) + fn_0 - fn_1, grad_norm_xt

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