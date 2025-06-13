from typing import Callable, Optional

import blackjax
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

@eqx.filter_jit
def sample_hamiltonian_monte_carlo(
    key: PRNGKeyArray,
    time_dependent_log_density: Callable[[Float[Array, "dim"], float], float],
    x: Float[Array, "dim"],
    t: float,
    step_size: Float[Array, ""],
    inverse_mass_matrix: Optional[Float[Array, "dim dim"]],
    num_integration_steps: int,
    num_hmc_steps: int = 1,
) -> Float[Array, "dim"]:
    """
    Hamiltonian Monte Carlo using BlackJAX.
    
    Args:
        key: Random key
        time_dependent_log_density: Log density function
        x: Initial position
        t: Time parameter
        num_steps: Number of HMC steps
        integration_steps: Number of integration steps per HMC step
        step_size: Step size
        inverse_mass_matrix: Optional covariance matrix/diagonal
        
    Returns:
        Final position after HMC
    """
    dim = x.shape[-1]
    _inverse_mass_matrix = jnp.eye(dim) if inverse_mass_matrix is None else inverse_mass_matrix

    # Initialize Blackjax HMC kernel with passed parameters
    hmc = blackjax.hmc(
        logdensity_fn=lambda state: time_dependent_log_density(state, t),
        step_size=step_size,
        inverse_mass_matrix=_inverse_mass_matrix,
        num_integration_steps=num_integration_steps,
    )
    hmc_kernel = jax.jit(hmc.step)
    initial_state = hmc.init(x)

    # Define the loop body for sequential HMC steps
    @jax.jit
    def one_step(state, rng_key):
        state, _ = hmc_kernel(rng_key, state)
        return state, state # Carry the state, output the state

    # Perform HMC steps using lax.scan
    keys = jax.random.split(key, num_hmc_steps) # Use num_hmc_steps
    final_state, _ = jax.lax.scan(one_step, initial_state, keys)

    return final_state.position # Return only the final position
