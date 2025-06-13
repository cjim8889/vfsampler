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
