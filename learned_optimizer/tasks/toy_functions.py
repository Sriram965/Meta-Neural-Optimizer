import tensorflow as tf
from .base import BaseTask


class RosenbrockTask(BaseTask):
    """
    The Rosenbrock function — a classic hard benchmark.
    
    Global minimum is at theta = (1, 1, ..., 1) where f = 0.
    The challenge is a narrow curved valley that most first-order
    optimizers struggle to navigate without oscillating.
    
    dim must be even and >= 2. We split theta into pairs (x, y)
    and sum the Rosenbrock contribution from each pair.
    """

    def __init__(self, dim: int = 2):
        assert dim >= 2 and dim % 2 == 0, "dim must be even and >= 2"
        self._dim = dim

    def sample_theta(self) -> tf.Tensor:
        # Start randomly in [-2, 2] — far from the minimum at (1,...,1)
        return tf.random.uniform([self._dim], minval=-2.0, maxval=2.0)

    def loss(self, theta: tf.Tensor) -> tf.Tensor:
        # Split theta into x (even indices) and y (odd indices)
        x = theta[0::2]  
        y = theta[1::2]   

        # Classic Rosenbrock formula applied to each (x, y) pair
        return tf.reduce_sum(
            (1.0 - x) ** 2 + 100.0 * (y - x ** 2) ** 2
        )

    @property
    def dim(self) -> int:
        return self._dim
    
    @property
    def name(self) -> str:
        return f"Rosenbrock(dim={self._dim})"