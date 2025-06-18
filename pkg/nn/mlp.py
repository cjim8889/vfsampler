from typing import Callable, List

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from .utils import init_linear_weights, xavier_init


class AdaptiveFeatureProjection(eqx.Module):
    """Conditioning module for time"""

    time_mlp: eqx.nn.MLP
    transform: eqx.nn.Linear
    activation: eqx.nn.Lambda

    def __init__(
        self, 
        dim: int, 
        key: PRNGKeyArray, 
        activation: Callable[[Float[Array, "..."]], Float[Array, "..."]] = jax.nn.silu
    ):
        t_key, proj_key = jax.random.split(key, 2)
        self.time_mlp = eqx.nn.MLP(1, dim, dim, 2, key=t_key)
        self.transform = eqx.nn.Linear(dim, dim, key=proj_key)
        self.activation = eqx.nn.Lambda(activation)

    def __call__(self, t: Float[Array, "1"]) -> Float[Array, "dim"]:
        t_feat = self.activation(self.time_mlp(t))
        return self.transform(t_feat)

class MLPVelocityField(eqx.Module):
    input_proj: eqx.nn.Linear
    blocks: List[eqx.nn.Sequential]
    norm: eqx.nn.LayerNorm
    output_proj: eqx.nn.Linear
    conditioning: AdaptiveFeatureProjection
    dt: float

    def __init__(
        self, 
        key: PRNGKeyArray,
        in_dim: int,
        out_dim: int, 
        hidden_dim: int, 
        depth: int = 4, 
        dt: float = 0.01
    ):
        keys = jax.random.split(key, 6)
        self.dt = dt
        # Input processing
        in_dim = in_dim + 1
        self.input_proj = eqx.nn.Linear(in_dim, hidden_dim, key=keys[0])

        # Residual blocks
        self.blocks = [
            eqx.nn.Sequential(
                [
                    eqx.nn.Linear(hidden_dim, hidden_dim, key=k),
                    eqx.nn.LayerNorm(hidden_dim),
                    eqx.nn.Lambda(jax.nn.gelu),
                ]
            )
            for k in jax.random.split(keys[1], depth)
        ]

        # Conditioning system
        self.conditioning = AdaptiveFeatureProjection(hidden_dim, keys[2])

        # Output projection
        self.norm = eqx.nn.LayerNorm(hidden_dim)
        self.output_proj = eqx.nn.Linear(hidden_dim, out_dim, key=keys[3])
        self._init_weights(keys[4])

    def _init_weights(self, key: PRNGKeyArray) -> None:
        """Ensure stable initialization with dt scaling"""

        key1, key2, key3 = jax.random.split(key, 3)
        self.input_proj = init_linear_weights(
            self.input_proj, xavier_init, key1, scale=self.dt
        )

        self.output_proj = init_linear_weights(
            self.output_proj, xavier_init, key2, scale=self.dt
        )

        self.blocks = init_linear_weights(
            self.blocks, xavier_init, key3, scale=self.dt
        )

    def __call__(self, x: Float[Array, "in_dim"], t: float) -> Float[Array, "out_dim"]:
        if isinstance(t, float):
            t = jnp.array([t])

        t = t.reshape(1)
        inputs = jnp.concatenate([x, t], axis=-1)

        h = self.input_proj(inputs)

        cond = self.conditioning(t)

        for block in self.blocks:
            h = block(h + cond)  # Additive conditioning

        return self.output_proj(self.norm(h))


class AugmentedResidualField(eqx.Module):
    f_residual: MLPVelocityField
    v_residual: MLPVelocityField
    f_natural: Callable[[Float[Array, "augmented_dim"]], Float[Array, "x_dim"]] = eqx.field(static=True)
    v_natural: Callable[[Float[Array, "x_dim"], float], Float[Array, "augmented_dim"]] = eqx.field(static=True)
    x_dim: int = eqx.field(static=True)
    augmented_dim: int = eqx.field(static=True)

    def __init__(self, 
        key: PRNGKeyArray, 
        x_dim: int, 
        hidden_dim: int, 
        augmented_dim: int,
        f_natural: Callable[[Float[Array, "augmented_dim"]], Float[Array, "x_dim"]],
        v_natural: Callable[[Float[Array, "x_dim"], float], Float[Array, "augmented_dim"]],
        depth: int = 4, 
        dt: float = 0.01,
    ):
        self.x_dim = x_dim
        self.augmented_dim = augmented_dim

        self.f_natural = f_natural
        self.v_natural = v_natural

        keys = jax.random.split(key, 2)
        self.f_residual = MLPVelocityField(keys[0], augmented_dim, x_dim, hidden_dim, depth, dt)
        self.v_residual = MLPVelocityField(keys[1], x_dim + augmented_dim, augmented_dim, hidden_dim, depth, dt)

    def __call__(self, xr: Float[Array, "x_dim + augmented_dim"], t: float) -> Float[Array, "x_dim"]:
        x, r = xr[:self.x_dim], xr[self.x_dim:]
        f_residual = self.f_residual(r, t) + self.f_natural(r)
        v_residual = self.v_residual(xr, t) + self.v_natural(x, t)
        return jnp.concatenate([f_residual, v_residual], axis=0)
    
    def residual_f(self, r: Float[Array, "augmented_dim"], t: float) -> Float[Array, "x_dim"]:
        return self.f_residual(r, t)
    
    def residual_v(self, xr: Float[Array, "x_dim + augmented_dim"], t: float) -> Float[Array, "augmented_dim"]:
        return self.v_residual(xr, t)
    