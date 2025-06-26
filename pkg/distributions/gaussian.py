import distrax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxtyping import Array, Float, PRNGKeyArray

from .base import Distribution


class MultivariateGaussian(Distribution):

    def __init__(
        self,
        dim: int = 2,
        mean: float = 0.0,
        sigma: float = 1.0,
        n_samples_eval: int = 1024,
    ):
        self.dim = dim
        self.n_samples_eval = n_samples_eval

        self.sigma = jnp.asarray(sigma)
        if self.sigma.ndim == 0:
            scale_diag = jnp.full((dim,), self.sigma)
        else:
            if self.sigma.shape[0] != dim:
                raise ValueError(
                    f"Sigma shape {self.sigma.shape} does not match dimension {dim}."
                )
            scale_diag = self.sigma

        self.mean = jnp.ones((dim,)) * mean

        self.distribution = distrax.MultivariateNormalDiag(
            loc=self.mean, scale_diag=scale_diag
        )

    def log_prob(self, x: Float[Array, "n_samples dim"]) -> Float[Array, "n_samples"]:
        return self.distribution.log_prob(x)

    def sample(self, key: PRNGKeyArray, shape: Array) -> Float[Array, "n_samples dim"]:
        return self.distribution.sample(seed=key, sample_shape=shape)

    def visualise(self, samples: Float[Array, "n_samples dim"]) -> plt.Figure:
        fig, ax = plt.subplots(1, figsize=(6, 6))
        if self.dim == 2:
            # Plot contour lines for the distribution
            # Create a grid
            grid_size = 100
            x_lin = jnp.linspace(
                self.mean[0] - 3 * self.sigma[0],
                self.mean[0] + 3 * self.sigma[0],
                grid_size,
            )
            y_lin = jnp.linspace(
                self.mean[1] - 3 * self.sigma[1],
                self.mean[1] + 3 * self.sigma[1],
                grid_size,
            )
            X, Y = jnp.meshgrid(x_lin, y_lin)
            grid = jnp.stack([X, Y], axis=-1).reshape(-1, 2)  # Shape: (grid_size**2, 2)

            # Compute log_prob for each grid point
            log_probs = self.log_prob(grid).reshape(grid_size, grid_size)

            # Plot contours
            ax.contour(X, Y, log_probs, levels=20, cmap="viridis")
            ax.set_xlim(
                self.mean[0] - 3 * self.sigma[0], self.mean[0] + 3 * self.sigma[0]
            )
            ax.set_ylim(
                self.mean[1] - 3 * self.sigma[1], self.mean[1] + 3 * self.sigma[1]
            )

            # Overlay scatter plot of samples
            ax.scatter(
                samples[:, 0],
                samples[:, 1],
                alpha=0.5,
                s=10,
                color="red",
                label="Samples",
            )

            ax.set_title("Multivariate Gaussian (2D)")
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")
            ax.legend()
            ax.grid(True)

        return fig

class ZeroMeanMultivariateGaussian(Distribution):
    """Translation–invariant multivariate Gaussian.

    This distribution is defined over a collection of ``n_particles`` points that
    live in a *d*-dimensional Euclidean space (``space_dim``).  The support is
    the sub-space where the centre-of-mass of the configuration is exactly
    zero

    Sampling: we first draw from an unconstrained Gaussian and then subtract the
    sample centre-of-mass so that the final configuration is zero-mean.

    Log-probability: for an input configuration *x* we remove its mean and
    evaluate the underlying Gaussian density at the centred configuration.  The
    resulting density is constant along global translations, making it
    invariant under the Euclidean group E(N).
    """

    def __init__(
        self,
        n_particles: int,
        space_dim: int = 3,
        sigma: float = 1.0,
        n_samples_eval: int = 1024,
    ) -> None:
        if space_dim not in (2, 3):
            raise ValueError(
                "`space_dim` is expected to be 2 or 3 for most applications; got "
                f"{space_dim}."
            )

        self.n_particles = n_particles
        self.space_dim = space_dim
        self.dim = n_particles * space_dim  # flattened dimension
        self.n_samples_eval = n_samples_eval

        self.sigma = jnp.asarray(sigma)
        if self.sigma.ndim == 0:
            scale_diag = jnp.full((self.dim,), self.sigma)
        else:
            if self.sigma.shape[0] != self.dim:
                raise ValueError(
                    f"Sigma shape {self.sigma.shape} does not match dimension {self.dim}."
                )
            scale_diag = self.sigma

        # Underlying isotropic / diagonal Gaussian (mean is zero by definition)
        self._gaussian = distrax.MultivariateNormalDiag(loc=jnp.zeros((self.dim,)), scale_diag=scale_diag)

    # ---------------------------------------------------------------------
    # Helper utilities
    # ---------------------------------------------------------------------
    def _flatten(self, x: Array) -> Array:
        """Reshape *x* to shape (batch, dim)."""
        if x.ndim == 3:  # (batch, n_particles, space_dim)
            return x.reshape(x.shape[0], self.dim)
        elif x.ndim == 2 and x.shape[-1] == self.dim:
            return x
        else:
            raise ValueError(
                "Input must have shape (batch, n_particles, space_dim) or (batch, dim)."
            )

    def _unflatten(self, x: Array) -> Array:
        """Reshape *x* to shape (batch, n_particles, space_dim)."""
        if x.ndim == 2 and x.shape[-1] == self.dim:
            return x.reshape(x.shape[0], self.n_particles, self.space_dim)
        elif x.ndim == 3:
            return x
        else:
            raise ValueError(
                "Input must have shape (batch, dim) or (batch, n_particles, space_dim)."
            )

    # ---------------------------------------------------------------------
    # Public API required by the `Distribution` base class
    # ---------------------------------------------------------------------
    def log_prob(self, x: Float[Array, "n_samples dim"],) -> Float[Array, "n_samples"]:
        """Log-probability of *x* (translation-invariant).

        Acceptable input shapes:
        1. `(dim,)` – a single flattened configuration.
        2. `(n_particles, space_dim)` – a single unflattened configuration.
        3. `(batch, dim)` – batched flattened configurations.
        4. `(batch, n_particles, space_dim)` – batched unflattened configurations.
        """

        # ------------------------------------------------------------------
        # Promote single samples to a batch of size 1 so that downstream code
        # can assume the leading dimension is "batch".
        # ------------------------------------------------------------------
        x = jnp.asarray(x)
        # Detect whether the caller provided a single configuration
        _single_input = False
        if x.ndim == 1:  # (dim,)
            _single_input = True
            x = x[None, :]
        elif x.ndim == 2 and x.shape == (self.n_particles, self.space_dim):
            _single_input = True
            x = x[None, :, :]

        # Reshape → (batch, n_particles, space_dim)
        x_unf = self._unflatten(x)

        # Remove centre-of-mass
        com = jnp.mean(x_unf, axis=1, keepdims=True)  # (batch, 1, space_dim)
        x_centered = x_unf - com

        # Flatten and evaluate density
        x_flat = x_centered.reshape(x_centered.shape[0], self.dim)
        log_p = self._gaussian.log_prob(x_flat)

        # Return scalar if only a single configuration was provided
        if _single_input:
            return log_p.squeeze(axis=0)
        return log_p

    def sample(
        self, key: PRNGKeyArray, shape: Array
    ) -> Float[Array, "n_samples dim"]:
        """Draw samples with zero centre-of-mass."""
        # Sample from underlying Gaussian, then center
        raw = self._gaussian.sample(seed=key, sample_shape=shape)  # (shape, dim)
        raw_unf = raw.reshape(raw.shape[0], self.n_particles, self.space_dim)
        com = jnp.mean(raw_unf, axis=1, keepdims=True)
        centered = raw_unf - com  # zero COM
        return centered.reshape(raw.shape[0], self.dim)

    def visualise(self, samples: Float[Array, "n_samples dim"]) -> plt.Figure:
        # Fall back to the parent class visualisation for 2D only.
        if self.space_dim != 2:
            raise NotImplementedError(
                "Visualisation is currently implemented only for 2D configurations."
            )

        # Convert to unflattened representation for plotting
        samples_unf = self._unflatten(samples)
        centered_samples = samples_unf - jnp.mean(samples_unf, axis=1, keepdims=True)
        flat_centered = centered_samples.reshape(centered_samples.shape[0], self.dim)

        # Re-use visualisation from `MultivariateGaussian` by delegating
        temp_gaussian = MultivariateGaussian(dim=self.dim, sigma=self.sigma, mean=0.0)
        return temp_gaussian.visualise(flat_centered)
