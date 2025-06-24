import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxtyping import Array, Float, PRNGKeyArray

from pkg.distributions.base import Distribution


class MultiDoubleWellEnergy(Distribution):
    def __init__(
        self,
        dim: int,
        n_particles: int,
        a: float = 0.9,
        b: float = -4.0,
        c: float = 0.0,
        offset: float = 4.0,
        n_samples_eval: int = 2048,
    ):
        self.n_samples_eval = n_samples_eval
        self.dim = dim
        self.n_particles = n_particles
        self.n_spatial_dim = dim // n_particles

        self.a = a
        self.b = b
        self.c = c
        self.offset = offset

    def compute_distances(self, x: Float[Array, "dim"], epsilon: float = 1e-8) -> Float[Array, "n_pairs"]:
        x = x.reshape(self.n_particles, self.n_spatial_dim)

        # Get indices of upper triangular pairs
        i, j = jnp.triu_indices(self.n_particles, k=1)

        # Calculate displacements between pairs
        dx = x[i] - x[j]

        # Compute distances
        distances = jnp.sqrt(jnp.sum(dx**2, axis=-1) + epsilon)

        return distances

    def batched_remove_mean(self, x: Float[Array, "n_samples dim"]) -> Float[Array, "n_samples dim"]:
        return x - jnp.mean(x, axis=1, keepdims=True)

    def multi_double_well_energy(self, x: Float[Array, "dim"]) -> Float[Array, ""]:
        dists = self.compute_distances(x)
        dists = dists - self.offset

        energies = self.a * dists**4 + self.b * dists**2 + self.c

        total_energy = jnp.sum(energies)
        return total_energy

    def log_prob(self, x: Float[Array, "dim"]) -> Float[Array, ""]:
        p_t = -self.multi_double_well_energy(x)
        return p_t

    def score(self, x: Float[Array, "dim"]) -> Float[Array, "dim"]:
        return jax.grad(self.log_prob)(x)

    def sample(
        self, key: PRNGKeyArray, sample_shape: Array = ()
    ) -> Float[Array, "..."]:
        raise NotImplementedError(
            "Sampling is not implemented for MultiDoubleWellEnergy"
        )

    def interatomic_dist(self, x: Float[Array, "n_samples dim"]) -> Float[Array, "n_samples n_pairs"]:
        x = x.reshape(-1, self.n_particles, self.n_spatial_dim)
        distances = jax.vmap(lambda x: self.compute_distances(x))(x)

        return distances

    def batched_log_prob(self, xs: Float[Array, "n_samples dim"]) -> Float[Array, "n_samples"]:
        return jax.vmap(self.log_prob)(xs)

    def visualise(self, samples: Float[Array, "n_samples dim"]) -> plt.Figure:
        samples = jnp.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        dist_samples = self.interatomic_dist(samples)

        axs[0].hist(
            dist_samples.flatten(),
            bins=100,
            alpha=0.5,
            density=True,
            histtype="step",
            linewidth=4,
        )

        axs[0].set_xlabel("Interatomic distance")
        axs[0].legend(["Generated data"])

        energy_samples = -self.batched_log_prob(samples)

        min_energy = -26
        max_energy = 0

        axs[1].hist(
            energy_samples,
            bins=100,
            density=True,
            alpha=0.4,
            range=(min_energy, max_energy),
            color="r",
            histtype="step",
            linewidth=4,
            label="Generated data",
        )
        axs[1].set_xlabel("Energy")
        axs[1].legend()

        fig.canvas.draw()
        return fig