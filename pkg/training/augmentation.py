import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


@eqx.filter_jit
def random_rotation_3d(key: PRNGKeyArray) -> Float[Array, "3 3"]:
    """Generates a random 3D rotation matrix using Rodrigues' rotation formula."""
    key_angle, key_axis = jax.random.split(key)
    angle = jax.random.uniform(key_angle, shape=(), minval=0.0, maxval=2 * jnp.pi)
    # Sample a random axis uniformly from the sphere
    axis = jax.random.normal(key_axis, shape=(3,))
    axis = axis / jnp.linalg.norm(axis)
    # Construct the skew-symmetric matrix for the axis
    K = jnp.array([[0, -axis[2], axis[1]],
                   [axis[2], 0, -axis[0]],
                   [-axis[1], axis[0], 0]])
    I = jnp.eye(3)
    # Rodrigues rotation formula
    R = I + jnp.sin(angle) * K + (1 - jnp.cos(angle)) * (K @ K)
    return R


@eqx.filter_jit
def random_rotation(
    x: Float[Array, "dim"],
    key: PRNGKeyArray,
    num_particles: int,
    spatial_dim: int = 2
) -> Float[Array, "dim"]:
    """Apply random rotation to particle coordinates.
    
    Args:
        x: Flattened particle coordinates of shape (dim,)
        key: Random key
        num_particles: Number of particles
        spatial_dim: Spatial dimension (2 or 3)
    
    Returns:
        Rotated coordinates of same shape as input
    """
    if spatial_dim not in [2, 3]:
        raise ValueError("spatial_dim must be 2 or 3")
    
    # Reshape to (num_particles, spatial_dim)
    coords_per_particle = x.shape[0] // num_particles
    if coords_per_particle != spatial_dim:
        # If dimensions don't match expected structure, return unchanged
        return x
    
    x_reshaped = x.reshape(num_particles, spatial_dim)
    
    if spatial_dim == 2:
        # 2D rotation
        angle = jax.random.uniform(key, shape=(), minval=0.0, maxval=2 * jnp.pi)
        cos_a, sin_a = jnp.cos(angle), jnp.sin(angle)
        R = jnp.array([[cos_a, -sin_a],
                       [sin_a, cos_a]])
    else:
        # 3D rotation
        R = random_rotation_3d(key)
    
    # Apply rotation
    x_rotated = jnp.dot(x_reshaped, R.T)
    return x_rotated.reshape(-1)


@eqx.filter_jit
def random_translation(
    x: Float[Array, "dim"],
    key: PRNGKeyArray,
    translation_scale: float,
    num_particles: int,
    spatial_dim: int = 2
) -> Float[Array, "dim"]:
    """Apply random translation to particle coordinates.
    
    Args:
        x: Flattened particle coordinates of shape (dim,)
        key: Random key
        translation_scale: Scale of translation (uniform in [-scale, scale])
        num_particles: Number of particles
        spatial_dim: Spatial dimension (2 or 3)
    
    Returns:
        Translated coordinates of same shape as input
    """
    if spatial_dim not in [2, 3]:
        raise ValueError("spatial_dim must be 2 or 3")
    
    # Reshape to (num_particles, spatial_dim)
    coords_per_particle = x.shape[0] // num_particles
    if coords_per_particle != spatial_dim:
        # If dimensions don't match expected structure, return unchanged
        return x
    
    x_reshaped = x.reshape(num_particles, spatial_dim)
    
    # Sample random translation vector
    translation = jax.random.uniform(
        key,
        shape=(1, spatial_dim),
        minval=-translation_scale,
        maxval=translation_scale
    )
    
    # Apply translation (broadcasts over all particles)
    x_translated = x_reshaped + translation
    return x_translated.reshape(-1)


@eqx.filter_jit
def random_normal_noise(
    x: Float[Array, "dim"],
    key: PRNGKeyArray,
    noise_scale: float
) -> Float[Array, "dim"]:
    """Add random normal noise to coordinates.
    
    Args:
        x: Input coordinates of shape (dim,)
        key: Random key
        noise_scale: Standard deviation of noise
    
    Returns:
        Noisy coordinates of same shape as input
    """
    noise = jax.random.normal(key, shape=x.shape) * noise_scale
    return x + noise


@eqx.filter_jit
def augment_particle_coordinates(
    x: Float[Array, "dim"],
    key: PRNGKeyArray,
    num_particles: int,
    spatial_dim: int = 2,
    enable_rotation: bool = False,
    enable_translation: bool = False,
    enable_noise: bool = False,
    translation_scale: float = 1.0,
    noise_scale: float = 0.1
) -> Float[Array, "dim"]:
    """Apply multiple augmentations to particle coordinates.
    
    Args:
        x: Flattened particle coordinates
        key: Random key
        num_particles: Number of particles
        spatial_dim: Spatial dimension (2 or 3)
        enable_rotation: Whether to apply rotation
        enable_translation: Whether to apply translation
        enable_noise: Whether to add noise
        translation_scale: Scale for translation
        noise_scale: Scale for noise
    
    Returns:
        Augmented coordinates
    """
    keys = jax.random.split(key, 3)
    
    # Apply augmentations sequentially if enabled
    result = x
    
    if enable_rotation:
        result = random_rotation(result, keys[0], num_particles, spatial_dim)
    
    if enable_translation:
        result = random_translation(result, keys[1], translation_scale, num_particles, spatial_dim)
    
    if enable_noise:
        result = random_normal_noise(result, keys[2], noise_scale)
    
    return result


# Vectorized versions for batch processing
batch_random_rotation = jax.vmap(random_rotation, in_axes=(0, 0, None, None))
batch_random_translation = jax.vmap(random_translation, in_axes=(0, 0, None, None, None))
batch_random_normal_noise = jax.vmap(random_normal_noise, in_axes=(0, 0, None))
batch_augment_particle_coordinates = jax.vmap(
    augment_particle_coordinates, 
    in_axes=(0, 0, None, None, None, None, None, None, None)
) 