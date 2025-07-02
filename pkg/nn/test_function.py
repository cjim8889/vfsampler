from typing import Tuple

import equinox as eqx
import jax
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