from typing import Tuple, Callable, List

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .mlp import MLPVelocityField


class TrainableTestFunction(eqx.Module):
    """A trainable test function that outputs a single phi and grad_phi for given x."""
    
    phi_network: MLPVelocityField
    
    def __init__(self, key, in_dim: int, hidden_dim: int = 64, depth: int = 3):
        """Initialize the trainable test function.
        
        Args:
            key: Random key for initialization
            in_dim: Input dimension (dimension of x)
            hidden_dim: Hidden dimension of the MLP
            depth: Depth of the MLP
        """
        self.phi_network = MLPVelocityField(
            key=key,
            in_dim=in_dim,
            out_dim=1,  # Single output
            hidden_dim=hidden_dim,
            depth=depth,
            dt=0.01,  # Not used since we don't use time
        )
    
    def __call__(self, x: Float[Array, "dim"], t: float) -> Tuple[float, Float[Array, "dim"]]:
        """Compute phi and grad_phi for given x and t.
        
        Args:
            x: Input coordinates
            t: Time parameter
            
        Returns:
            phi: Scalar test function value
            grad_phi: Gradient of test function of shape (dim,)
        """
        # Compute phi value (scalar)
        phi = self.phi_network(x, t)[0]  # Extract scalar from shape (1,)
        
        # Compute gradient with respect to x
        def phi_fn(x_):
            return self.phi_network(x_, t)[0]
        
        grad_phi = jax.grad(phi_fn)(x)
        
        return phi, grad_phi


class FixedTestFunction(eqx.Module):
    """A fixed (non-trainable) test function that applies multiple temperature scalings to a log probability function.
    
    This class uses a provided log probability function and applies multiple temperature scalings:
    phi_i(x, t) = log_prob(x, t) / temperature_i for each temperature_i
    
    When using this test function, the training loop will automatically detect
    that it's not trainable and skip the test function optimization phase.
    """
    
    # All fields are static (no trainable parameters)
    log_prob_fn: Callable[[Float[Array, "dim"], float], float] = eqx.field(static=True)
    temperatures: List[float] = eqx.field(static=True)
    
    def __init__(self, log_prob_fn: Callable[[Float[Array, "dim"], float], float], temperatures: List[float]):
        """Initialize the fixed test function with a log probability function and list of temperatures.
        
        Args:
            log_prob_fn: Function that computes log probability given (x, t). 
                        May or may not be normalized.
            temperatures: List of temperature parameters for scaling the log probability.
                         Higher temperature makes the function smoother.
        """
        self.log_prob_fn = log_prob_fn
        self.temperatures = temperatures
    
    def __call__(self, x: Float[Array, "dim"], t: float) -> Tuple[Float[Array, "n_temps"], Float[Array, "n_temps dim"]]:
        """Compute phi and grad_phi for given x and t using temperature-scaled log probability for each temperature.
        
        Args:
            x: Input coordinates 
            t: Time parameter
            
        Returns:
            phi: Array of test function values (log_prob(x, t) / temperature_i) of shape (n_temps,)
            grad_phi: Array of gradients of test functions of shape (n_temps, dim)
        """
        # Compute base log probability once
        log_prob = self.log_prob_fn(x, t)
        
        # Convert temperatures to JAX array and compute phi values using vectorization
        temperatures_array = jnp.array(self.temperatures)
        phi_values = log_prob / temperatures_array
        
        # Compute gradients for each temperature using vmap
        def compute_grad_for_temp(temp):
            def phi_fn(x_):
                return self.log_prob_fn(x_, t) / temp
            return jax.grad(phi_fn)(x)
        
        grad_phi_values = jax.vmap(compute_grad_for_temp)(temperatures_array)
        
        return phi_values, grad_phi_values