import torch

class BaseManifold:
    """Base class for all manifolds.

    The implementation should provide:
    1. Basic operations: projection to tangent space, exponential and logarithmic maps
    2. Geodesic operations: geodesic paths and velocities
    3. Sampling: uniform sampling on the manifold
    4. Wrapping/unwrapping: maps between R^n and the manifold
    """
    
    def __init__(self, eps=1e-7):
        self.eps = eps
    
    def wrap(self, samples):
        """Map points from ambient space to manifold."""
        raise NotImplementedError
        
    def unwrap(self, samples):
        """Map points from manifold to ambient space."""
        raise NotImplementedError
        
    def project_to_tangent(self, x, v):
        """Project vector v onto tangent space at point x."""
        raise NotImplementedError
        
    def geodesic(self, x0, x1, t):
        """Compute geodesic between x0 and x1 at time t."""
        raise NotImplementedError
        
    def geodesic_velocity(self, x0, x1, t):
        """Compute velocity of geodesic between x0 and x1 at time t."""
        raise NotImplementedError
        
    def log_map(self, x0, x1):
        """Compute logarithmic map from x0 to x1."""
        raise NotImplementedError
        
    def exp_map(self, x0, v):
        """Compute exponential map of v at x0."""
        raise NotImplementedError
        
    def sample(self, batch_size, device="cpu"):
        """Sample points uniformly from manifold."""
        raise NotImplementedError 
    
    def distance(self, x, y):
        """Compute geodesicdistance between x and y."""
        raise NotImplementedError