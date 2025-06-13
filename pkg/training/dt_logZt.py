from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


@eqx.filter_jit
def estimate_dt_logZt(
    xs: Float[Array, "n_timesteps n_samples dim"],
    weights: Float[Array, "n_timesteps n_samples"],
    ts: Float[Array, "n_timesteps"],
    time_derivative_log_density: Callable[[Float[Array, "dim"], float], float],
) -> Float[Array, "n_timesteps"]:
    def scan_body(_, scan_slice):
        x_t, w_t, t = scan_slice
        dt_log_density_t = jax.vmap(lambda x: time_derivative_log_density(x, t))(x_t)

        combined_term_t = dt_log_density_t
        step_contribution = jnp.sum(combined_term_t * w_t)

        return None, step_contribution

    scan_inputs = (xs, weights, ts)

    _, final_dt_logZt = jax.lax.scan(scan_body, None, scan_inputs)

    return final_dt_logZt.flatten()
