from typing import Callable, Dict, Tuple, Union

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from .euler import solve_neural_ode_euler


@eqx.filter_jit
def generate_samples(
    key: PRNGKeyArray,
    v_theta: Callable[[Float[Array, "dim"], float], Float[Array, "dim"]],
    num_samples: int,
    ts: Float[Array, "n_timesteps"],
    sample_fn: Callable[
        [PRNGKeyArray, Tuple[int, ...]], Float[Array, "n_samples dim"]
    ],
    save_trajectory: bool = False,
    estimate_dt_logZ: bool = False,
) -> Dict[
    str,
    Union[
        Float[Array, "n_timesteps n_samples dim"],
        Float[Array, "n_timesteps n_samples"],
    ],
]:
    initial_samples = sample_fn(key, (num_samples,))
    final_samples, _ = solve_neural_ode_euler(
        v_theta=v_theta if not estimate_dt_logZ else lambda *args: v_theta(*args)[0],
        y0=initial_samples,
        ts=ts,
        use_shortcut=False,
        save_trajectory=save_trajectory,
    )
    return {
        "positions": final_samples,
        "weights": jnp.ones((ts.shape[0], num_samples)) / num_samples,
    }