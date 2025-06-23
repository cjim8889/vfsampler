from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


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
    print(x.shape, r.shape)
    def v_r(r):
        out = v_theta(jnp.concatenate([x, r]), t)
        return out
    

    return jnp.trace(jax.jacfwd(v_r)(r))


def axis_aligned_fourier_modes(
    x: Float[Array, "... dim"],
    max_frequency: int,
    *,
    domain_range: tuple[float, float] = (0.0, 1.0),
    use_2pi: bool = True,
) -> tuple[
    Float[Array, "... n_basis"],           # φ(x)
    Float[Array, "... n_basis dim"],       # ∇φ(x)
]:
    """
    Axis-aligned Fourier basis (sine & cosine) and their gradients.

    For every spatial dimension j = 0,…,d-1 and every integer
    n = 1,…,max_frequency this routine builds the one-dimensional modes

        sin( k_n · x ) ,   cos( k_n · x ) ,
        k_n = n * 2π/L  (or π/L if use_2pi=False)

    where L = domain_range[1]-domain_range[0].   Only one coordinate is
    active at a time, hence *axis-aligned*.

    Parameters
    ----------
    x
        Array of shape (..., dim) — works on a single point or a batch.
    max_frequency
        Highest integer frequency on each axis.
    domain_range
        Tuple (xmin, xmax); the same interval is assumed in every axis.
    use_2pi
        True  ➜ periodic modes  2πn / L
        False ➜ half-range modes πn / L  (Dirichlet / Neumann)

    Returns
    -------
    phi
        (..., n_basis)          with  n_basis = 2 * dim * max_frequency
    grad_phi
        (..., n_basis, dim)
    """
    xmin, xmax = domain_range
    L          = xmax - xmin
    factor     = (2 * jnp.pi if use_2pi else jnp.pi) / L

    # ------------------------------------------------------------------
    # Build the list of axis-aligned wave-vectors  w_k  (M = dim*max_freq)
    # ------------------------------------------------------------------
    dim = x.shape[-1]
    ws  = []
    for j in range(dim):
        for n in range(1, max_frequency + 1):
            w           = jnp.zeros(dim)
            w           = w.at[j].set(n * factor)
            ws.append(w)
    ws         = jnp.stack(ws)                     # (M, dim)
    M          = ws.shape[0]

    # ------------------------------------------------------------------
    # Evaluate  φ(x)  and  ∇φ(x)                works on arbitrary batch
    # ------------------------------------------------------------------
    x_shifted  = x - xmin                         # align phase at xmin
    w_dot_x    = jnp.einsum("...d,md->...m", x_shifted, ws)   # (..., M)

    sin_vals   = jnp.sin(w_dot_x)                 # (..., M)
    cos_vals   = jnp.cos(w_dot_x)

    phi        = jnp.concatenate([sin_vals, cos_vals], axis=-1)        # (..., 2M)

    grad_sin   =  ws * cos_vals[..., None]        # (..., M, dim)
    grad_cos   = -ws * sin_vals[..., None]
    grad_phi   = jnp.concatenate([grad_sin, grad_cos], axis=-2)        # (..., 2M, dim)

    return phi, grad_phi
