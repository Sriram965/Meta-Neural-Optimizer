import tensorflow as tf
import numpy as np
from .base import BaseTask


class QuadraticTask(BaseTask):
   

    def __init__(self, dim: int = 10, condition_number: float = 1.0, seed: int = None):
        self._dim = dim
        self.condition_number = condition_number
        rng = np.random.default_rng(seed)

        R = rng.standard_normal((dim, dim))
        R = R / (np.linalg.norm(R, 'fro') + 1e-8)  # normalizing  to prevent overflow
        A = R.T @ R * dim + 0.1 * np.eye(dim, dtype=np.float64)

        if condition_number > 1.0:
            eigvals, eigvecs = np.linalg.eigh(A)
            new_eigvals = np.linspace(1.0, condition_number, dim, dtype=np.float64)
            A = (eigvecs * new_eigvals) @ eigvecs.T

        # Single cast at the very end
        self.A = tf.constant(A.astype(np.float32), dtype=tf.float32)

        
    def sample_theta(self) -> tf.Tensor:

        return tf.random.normal([self._dim], stddev=1.0)

    def loss(self, theta: tf.Tensor) -> tf.Tensor:

        Atheta = tf.linalg.matvec(self.A, theta)  
        return tf.reduce_sum(theta * Atheta)      

    @property
    def dim(self) -> int:
        return self._dim
    
    @property
    def name(self) -> str:
        return f"Quadratic(dim={self._dim}, cond={self.condition_number})"
    

class NoisyQuadraticTask(QuadraticTask):
    """
    Same as QuadraticTask but with Gaussian noise added to the loss.
    
    """

    def __init__(self, dim: int = 10, condition_number: float = 1.0,
                 noise_std: float = 0.1, seed: int = None):
        
        super().__init__(dim=dim, condition_number=condition_number, seed=seed)
        self.noise_std = noise_std

    def loss(self, theta: tf.Tensor) -> tf.Tensor:
        base_loss = super().loss(theta)

        # Adding  Gaussian noise
        noise = tf.random.normal([], stddev=self.noise_std)

        return base_loss + noise
    
    @property
    def name(self) -> str:
        return f"NoisyQuadratic(dim={self._dim}, std={self.noise_std})"