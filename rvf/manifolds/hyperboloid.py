import torch
from rvf.manifolds.base import BaseManifold


class HyperboloidManifold(BaseManifold):
    """
    Class for operations on the hyperboloid model of hyperbolic space H^n.

    The hyperboloid model H^n is embedded in R^(n+1) with the Minkowski metric:
    H^n = { x in R^(n+1): -x_0^2 + x_1^2 + ... + x_n^2 = -1, x_0 > 0 }
    """
    def __init__(self, eps=1e-8):
        """Initialize the hyperboloid manifold."""
        super().__init__(eps)
        self.origin = None  # Base point at (1, 0, ..., 0)

    def _minkowski_dot(self, x, y):
        """Compute the Minkowski inner product: <x, y>_M = -x0*y0 + sum_{i=1..n} x_i * y_i."""
        return -x[..., 0] * y[..., 0] + (x[..., 1:] * y[..., 1:]).sum(dim=-1)

    def _get_origin(self, batch_size, dim, device):
        """Get the origin (1,0,...,0) on the hyperboloid for a given batch and dimension."""
        first = torch.ones(batch_size, 1, device=device)
        zeros = torch.zeros(batch_size, dim, device=device)
        self.origin = torch.cat([first, zeros], dim=-1)
        return self.origin

    def project_to_tangent(self, x, v):
        """Project vector v onto the tangent space at x (ensure <x, v>_M = 0)."""
        assert x.shape == v.shape, f"x.shape: {x.shape} != v.shape: {v.shape}"
        dot = self._minkowski_dot(x, v).unsqueeze(-1)
        return v + dot * x

    def exp_map(self, x0, v):
        """
        Exponential map at x0: move along the geodesic by tangent vector v.
        Formula: cosh(norm_v) * x0 + sinh(norm_v) * (v / norm_v)
        """
        assert x0.shape == v.shape, f"x0.shape: {x0.shape} != v.shape: {v.shape}"
        # Minkowski norm of v: sqrt(<v, v>_M)
        vv = self._minkowski_dot(v, v)
        norm_v = torch.sqrt(torch.clamp(vv, min=self.eps)).unsqueeze(-1)
        direction = v / norm_v
        x = torch.cosh(norm_v) * x0 + torch.sinh(norm_v) * direction

        # Find which samples are off-manifold and correct them
        norm_xx = self._minkowski_dot(x, x)
        mask = (norm_xx + 1.0).abs() > 1e-8

        if mask.any():
            # Split out the spatial part x_1...x_n
            spatial = x[..., 1:]
            # Recompute x_0 = sqrt(1 + ||spatial||^2)
            x0_new = torch.sqrt(1.0 + (spatial**2).sum(dim=-1))
            x0_new = x0_new.unsqueeze(-1)
            # Rebuild x: corrected time-component for masked samples
            x = x.clone()
            x[mask, 0:1] = x0_new[mask]

        return x

    def log_map(self, x0, x1):
        """Compute the logarithmic map from x0 to x1.

        Formula:
        c = - minkowski_dot(x0, x1)
        dist_hyperbolic = arccosh(c) = ln(c + sqrt(c^2 - 1))
        log_map(x0, x1) = dist_hyperbolic * (x1 - c * x0) / sqrt(c^2 - 1)

        Args:
            x0 (torch.Tensor): Base point on the hyperboloid
            x1 (torch.Tensor): Target point on the hyperboloid

        Returns:
            torch.Tensor: Tangent vector v at x0 such that exp_{x0}(v) = x1
        """
        dot = self._minkowski_dot(x0, x1)
        # Numerical stability: ensure dot <= -1 since we use arccosh
        dot_clamped = torch.clamp(dot, max=-1.0 - self.eps)
        # Hyperbolic distance
        c = -dot_clamped
        dist = torch.log(c + torch.sqrt(c * c - 1.0 + self.eps))
        # Tangent component
        u = x1 - c.unsqueeze(-1) * x0
        u_norm = torch.sqrt(torch.clamp(c * c - 1.0, min=self.eps)).unsqueeze(-1)
        return dist.unsqueeze(-1) * u / u_norm

    def geodesic(self, x0, x1, t):
        """Compute the geodesic path from x0 to x1 parametrized by t in [0,1]."""
        v = self.log_map(x0, x1)
        # Broadcast t to last dimension if needed
        if t.dim() < v.dim():
            t = t.unsqueeze(-1)
        return self.exp_map(x0, v * t)

    def geodesic_velocity(self, x0, x1, t):
        """Compute the velocity of the geodesic at parameter t."""
        v = self.log_map(x0, x1)
        vv = self._minkowski_dot(v, v)
        alpha = torch.sqrt(torch.clamp(vv, min=0.0) + self.eps).unsqueeze(-1)
        direction = v / (alpha + self.eps)
        if t.dim() < v.dim():
            t = t.unsqueeze(-1)
        t_alpha = alpha * t
        return alpha * (torch.sinh(t_alpha) * x0 + torch.cosh(t_alpha) * direction)

    def wrap(self, samples):
        """Map Euclidean samples in R^n to the hyperboloid H^n via exp map at origin."""
        batch_size, dim = samples.shape
        device = samples.device
        origin = self._get_origin(batch_size, dim, device)
        zeros = torch.zeros(batch_size, 1, device=device)
        v = torch.cat([zeros, samples], dim=-1)
        return self.exp_map(origin, v)

    def unwrap(self, samples):
        """Map points on the hyperboloid back to R^n via log map at origin."""
        batch_size, dim1 = samples.shape
        dim = dim1 - 1
        device = samples.device
        origin = self._get_origin(batch_size, dim, device)
        v = self.log_map(origin, samples)
        return v[..., 1:]

    def distance(self, x, y, eps=1e-7):
        """Compute the hyperbolic distance between x and y.

        Formula: arccosh(-<x,y>_M)
        where arccosh(u) = ln(u + sqrt(u^2 - 1))
        """
        dot = self._minkowski_dot(x, y)
        c = torch.clamp(-dot, min=1.0 + eps)
        sq = torch.sqrt(c * c - 1.0)
        return torch.log(c + sq)

    def sample(self, *args, **kwargs):
        """Uniform sampling wrapped onto the hyperboloid."""
        return self.wrap(torch.randn(*args, **kwargs))

    def project_to_hyperboloid(self, x):
        """Project points in R^(d+1) to the hyperboloid H^d."""
        batch_size, dim = x.shape
        origin = self._get_origin(batch_size, dim-1, x.device)
        inner = self._minkowski_dot(x, origin).unsqueeze(-1)
        x_tangent = x + inner * origin
        x_proj = self.exp_map(origin, x_tangent)
        return x_proj
