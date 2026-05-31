import tensorflow as tf
import numpy as np
from .base import BaseTask


class QuadraticTask(BaseTask):
   

    def __init__(self, dim: int = 10, condition_number: float = 1.0, seed: int = None):
        self._dim = dim

        # Use a seeded random generator so tasks are reproducible
        rng = np.random.default_rng(seed)

        # building a positive definite matrix via R^T R
        R = rng.standard_normal((dim, dim)).astype(np.float32)
        A = R.T @ R + 0.1 * np.eye(dim, dtype=np.float32)

        # rescaling  eigenvalues to control difficulty
        if condition_number > 1.0:
            eigvals, eigvecs = np.linalg.eigh(A)
            new_eigvals = np.linspace(1.0, condition_number, dim).astype(np.float32)
            A = (eigvecs * new_eigvals) @ eigvecs.T

        self.A = tf.constant(A, dtype=tf.float32)

    def sample_theta(self) -> tf.Tensor:

        return tf.random.normal([self._dim], stddev=1.0)

    def loss(self, theta: tf.Tensor) -> tf.Tensor:

        Atheta = tf.linalg.matvec(self.A, theta)  
        return tf.reduce_sum(theta * Atheta)      

    @property
    def dim(self) -> int:
        return self._dim
    

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

        # Adding  Gaussian noise — different every call, simulating mini-batch variance
        noise = tf.random.normal([], stddev=self.noise_std)

        return base_loss + noise