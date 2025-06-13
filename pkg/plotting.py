import itertools
from typing import Callable, Optional, Tuple

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def plot_contours(log_prob_func: Callable,
                  ax: Optional[plt.Axes] = None,
                  bounds: Tuple[float, float] = (-5.0, 5.0),
                  grid_width_n_points: int = 200,
                  n_contour_levels: Optional[int] = None,
                  log_prob_min: float = -1000.0):
    """Plot contours of a log_prob_func that is defined on 2D"""
    if ax is None:
        fig, ax = plt.subplots(1)

    x_points_dim1_np = np.linspace(bounds[0], bounds[1], grid_width_n_points)
    x_points_dim2_np = np.linspace(bounds[0], bounds[1], grid_width_n_points)
    
    x_points_for_func = np.array(list(itertools.product(x_points_dim1_np, x_points_dim2_np)))

    log_p_x = log_prob_func(x_points_for_func)
    
    log_p_x_jnp = jnp.asarray(log_p_x)
    log_p_x_clipped_jnp = jnp.clip(log_p_x_jnp, a_min=log_prob_min)
    log_p_x_clipped_np = np.asarray(log_p_x_clipped_jnp)
    
    log_p_x_reshaped_np = log_p_x_clipped_np.reshape((grid_width_n_points, grid_width_n_points))

    x_coords_reshaped_np = x_points_for_func[:, 0].reshape((grid_width_n_points, grid_width_n_points))
    y_coords_reshaped_np = x_points_for_func[:, 1].reshape((grid_width_n_points, grid_width_n_points))

    if n_contour_levels is not None:
        ax.contour(x_coords_reshaped_np, y_coords_reshaped_np, log_p_x_reshaped_np, levels=n_contour_levels)
    else:
        ax.contour(x_coords_reshaped_np, y_coords_reshaped_np, log_p_x_reshaped_np)


def plot_marginal_pair(
    samples: jnp.ndarray,
    ax: Optional[plt.Axes] = None,
    marginal_dims: Tuple[int, int] = (0, 1),
    bounds: Tuple[float, float] = (-5.0, 5.0),
    alpha: float = 0.5,
    markersize: float = 1.5,
):
    """Plot samples from marginal of distribution for a given pair of dimensions."""
    if not ax:
        fig, ax = plt.subplots(1)
    
    samples_clipped = jnp.clip(samples, bounds[0], bounds[1])
    
    samples_np = np.asarray(samples_clipped)
    
    ax.plot(
        samples_np[:, marginal_dims[0]], 
        samples_np[:, marginal_dims[1]], 
        "o", 
        alpha=alpha,
        markersize=markersize
    )
