from typing import Callable

from jaxtyping import Array, Float
import equinox as eqx
import jax
import jax.numpy as jnp

@eqx.filter_jit
def divergence(
    v_theta: Callable[[Float[Array, "dim"], float], Float[Array, "dim"]],
    x: Float[Array, "dim"],
    t: float,
) -> float:
    def v_x(x):
        return v_theta(x, t)

    return jnp.trace(jax.jacfwd(v_x)(x))


@eqx.filter_jit
def divergence_wrt_R(
    v_theta: Callable[[Float[Array, "dim"], float], Float[Array, "dim"]],
    xr: Float[Array, "dim"],
    t: float,
    r_dim: int,
) -> float:
    x, r = xr[:-r_dim], xr[-r_dim:]

    def v_r(r):
        return v_theta(r, t)

    return jnp.trace(jax.jacfwd(v_r)(r))