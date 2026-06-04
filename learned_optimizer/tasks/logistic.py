import tensorflow as tf
import numpy as np
from .base import BaseTask


class LogisticRegressionTask(BaseTask):
    """
    Binary classification via cross-entropy loss.
    """

    def __init__(self, n_samples: int = 200, input_dim: int = 20,
             batch_size: int = 32, seed: int = None):
        self._dim = input_dim
        self.batch_size = batch_size
        self.n_samples = n_samples

        rng = np.random.default_rng(seed)

        # All computation in float64 to avoid overflow
        w_true = rng.standard_normal(input_dim)          
        X = rng.standard_normal((n_samples, input_dim))   
        y = (X @ w_true > 0).astype(np.float32)         

        # Cast to float32 only when creating TF constants
        self.X = tf.constant(X.astype(np.float32))
        self.y = tf.constant(y)

    @property
    def name(self) -> str:
        return f"LogisticRegression(n={self.n_samples}, dim={self._dim})"

    def sample_theta(self) -> tf.Tensor:
        
        return tf.random.normal([self._dim], stddev=0.01)

    def loss(self, theta: tf.Tensor) -> tf.Tensor:
        # Draw a random mini-batch of indices
        indices = tf.random.uniform(
            [self.batch_size], minval=0, maxval=self.n_samples, dtype=tf.int32
        )

        # Gather the mini-batch features and labels
        X_batch = tf.gather(self.X, indices)   # [batch_size, input_dim]
        y_batch = tf.gather(self.y, indices)   # [batch_size]

        # Compute predictions as raw logits (no sigmoid yet)
        logits = tf.linalg.matvec(X_batch, theta)   # [batch_size]

        # Cross-entropy loss — TF handles the sigmoid internally for stability
        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(y_batch, logits)
        )
        return loss

    @property
    def dim(self) -> int:
        return self._dim