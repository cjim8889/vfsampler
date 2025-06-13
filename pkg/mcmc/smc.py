from typing import Callable, Dict, Optional, Tuple, Union

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from blackjax.smc.resampling import systematic
from jaxtyping import Array, Float, Int, PRNGKeyArray

from .hmc import sample_hamiltonian_monte_carlo


@jax.jit
def log_weights_to_weights(log_weights: Float[Array, "n_samples"]) -> Float[Array, "n_samples"]:
    log_sum_w = jax.scipy.special.logsumexp(log_weights)
    log_normalized_weights = log_weights - log_sum_w
    weights = jnp.exp(log_normalized_weights)

    return weights

@jax.jit
def ess_from_logweights(log_w: jax.Array) -> jax.Array:
    log_w = log_w - jax.scipy.special.logsumexp(log_w)
    w      = jnp.exp(log_w)
    return 1.0 / jnp.sum(w * w)

@eqx.filter_jit
def generate_samples_with_smc(
    key: PRNGKeyArray,
    initial_samples: Float[Array, "n_samples dim"],
    time_dependent_log_density: Callable[[Float[Array, "dim"], float], float],
    ts: Float[Array, "n_timesteps"],
    num_hmc_steps: int = 10,
    num_integration_steps: int = 3,
    step_size: float = 0.1,
    ess_threshold: float = 0.6,
    resampling_fn: Callable[
        [PRNGKeyArray, Float[Array, "n_samples"], int], Int[Array, "n_samples"]
    ] = systematic,
    initial_log_weights: Optional[Float[Array, "n_samples"]] = None,
    hmc_parameters: Optional[Dict] = None,
) -> Dict[str, Union[Float[Array, "n_timesteps n_samples dim"],
                       Float[Array, "n_timesteps n_samples"],
                       Float[Array, "n_timesteps"]]]:

    num_samples = initial_samples.shape[0]
    chex.assert_rank(initial_samples, 2)
    if initial_log_weights is None:
        initial_log_weights = jnp.full((num_samples,), -jnp.log(num_samples), dtype=jnp.float32)

    chex.assert_rank(initial_log_weights, 1)
    
    num_scan_steps = ts.shape[0] - 1
    sample_keys = jax.random.split(key, num_samples * num_scan_steps).reshape(
        num_scan_steps, num_samples, -1
    )

    particles = {
        "positions": initial_samples,
        "log_weights": initial_log_weights,
        "ess": jnp.array(1.0),
    }

    def _delta(positions, t, t_prev):
        return time_dependent_log_density(
            positions, t
        ) - time_dependent_log_density(positions, t_prev)

    batched_delta = jax.vmap(_delta, in_axes=(0, None, None))
    
    batched_hmc_step = jax.jit(jax.vmap(
        lambda key, pos, t, step_size, inv_mass_matrix, num_integration_steps: sample_hamiltonian_monte_carlo(
            key=key,
            time_dependent_log_density=time_dependent_log_density,
            x=pos,
            t=t,
            step_size=step_size,
            inverse_mass_matrix=inv_mass_matrix,
            num_integration_steps=num_integration_steps,
            num_hmc_steps=num_hmc_steps,
        ),
        in_axes=(0, 0, None, None, None, None)
    ))

    @jax.jit
    def _resample(key: PRNGKeyArray, 
                  positions: Float[Array, "n_samples dim"], 
                  log_weights: Float[Array, "n_samples"]
                 ) -> Tuple[Float[Array, "n_samples dim"], Float[Array, "n_samples"]]:

        weights = log_weights_to_weights(log_weights)
        indices = resampling_fn(key, weights, num_samples)

        new_positions = jnp.take(positions, indices, axis=0)

        new_log_weights = jnp.full((num_samples,), -jnp.log(num_samples), dtype=jnp.float32)

        return new_positions, new_log_weights

    def step(carry, inputs):
        particles_prev, t_idx = carry
        keys, hmc_params_t = inputs
        
        t_prev = ts[t_idx]
        t = ts[t_idx + 1]
        
        ess_val = ess_from_logweights(particles_prev["log_weights"])
        ess_percentage = ess_val / num_samples

        def do_resample():
            resample_key, _ = jax.random.split(keys[0])
            new_positions, new_log_weights = _resample(
                resample_key, particles_prev["positions"], particles_prev["log_weights"]
            )

            return {"positions": new_positions, "log_weights": new_log_weights}

        def do_nothing():
            log_weights_normalized = particles_prev["log_weights"] - jax.scipy.special.logsumexp(
                particles_prev["log_weights"]
            )

            return {
                "positions": particles_prev["positions"],
                "log_weights": log_weights_normalized,
            }

        # Conditionally resample based on ESS percentage
        particles_new = jax.lax.cond(
            ess_percentage < ess_threshold,
            do_resample,
            do_nothing,
        )
        particles_new["ess"] = ess_percentage

        propagated_positions = particles_new["positions"]

        if hmc_params_t is not None:
            hmc_step_size = hmc_params_t["step_size"]
            hmc_inv_mass_matrix = hmc_params_t["inverse_mass_matrix"]
            hmc_num_integration_steps = hmc_params_t["num_integration_steps"]
        else:
            hmc_step_size = step_size
            hmc_inv_mass_matrix = None
            hmc_num_integration_steps = num_integration_steps

        propagated_positions = batched_hmc_step(
            keys,
            propagated_positions,
            t,
            hmc_step_size,
            hmc_inv_mass_matrix,
            hmc_num_integration_steps,
        )

        w_delta = batched_delta(propagated_positions, t, t_prev)
        next_log_weights = particles_new["log_weights"] + w_delta
        next_log_weights = next_log_weights - jax.scipy.special.logsumexp(
            next_log_weights
        )

        new_carry = (
            {
                "positions": propagated_positions,
                "log_weights": next_log_weights,
                "ess": particles_new["ess"],
            },
            t_idx + 1,
        )

        return new_carry, {
            "positions": propagated_positions,
            "log_weights": next_log_weights,
            "ess": particles_new["ess"],
        }

    if hmc_parameters is not None:
        chex.assert_tree_shape_prefix(hmc_parameters, (ts.shape[0],)) # Sanity check
        scan_hmc_params = jax.tree.map(lambda x: x[1:], hmc_parameters)
    else:   
        scan_hmc_params = None

    _, scan_particles = jax.lax.scan(
        step,
        (particles, 0),  # Initial carry: particles at ts[0] and time index 0
        (sample_keys, scan_hmc_params),  # Inputs for each time step
    )
    
    all_positions = jnp.concatenate([
        jnp.expand_dims(particles["positions"], axis=0),
        scan_particles["positions"]
    ], axis=0)
    
    all_log_weights = jnp.concatenate([
        jnp.expand_dims(particles["log_weights"], axis=0),
        scan_particles["log_weights"]
    ], axis=0)
    
    all_ess = jnp.concatenate([
        jnp.expand_dims(particles["ess"], axis=0),
        scan_particles["ess"]
    ], axis=0)
    
    all_weights = jax.vmap(log_weights_to_weights)(all_log_weights)

    return {
        "positions": all_positions,
        "weights": all_weights,
        "ess": all_ess,
    }