import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from .utils import init_linear_weights, zero_init


class MeanStdNet(eqx.Module):
    """θ(x) ↦ (μ(x), σ(x)) with a tiny MLP."""
    mlp: eqx.nn.MLP                      # all parameters live here
    dim_r: int = eqx.field(static=True)

    def __init__(self, dim_r: int, dim_x: int, key: PRNGKeyArray, hidden: int = 64, depth: int = 2):
        self.dim_r = dim_r
        self.mlp = eqx.nn.MLP(
            in_size=dim_x,
            out_size=2 * dim_r,
            width_size=hidden,
            depth=depth,
            activation=jax.nn.silu,
            key=key,
        )

        self.mlp = init_linear_weights(self.mlp, zero_init, key)

    def __call__(self, x: Float[Array, "... dim_x"]
                 ) -> tuple[Float[Array, "... dim_r"],
                            Float[Array, "... dim_r"]]:
        h = self.mlp(x)                  # (..., 2·dim_r)
        μ, logσ = jnp.split(h, 2, axis=-1)
        σ = 0.01 + jax.nn.softplus(logσ)  # keep strictly > 0
        return μ, σ

