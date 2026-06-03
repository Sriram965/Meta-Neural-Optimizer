import tensorflow as tf
from .features import N_FEATURES


class MLPOptimizer(tf.keras.Model):
    """
    The learned optimizer — a small MLP applied coordinatewise.

    Architecture:
        features (8-dim) → Dense(32, tanh) → Dense(32, tanh) → Dense(1, tanh) * scale
    """

    def __init__(self, hidden_sizes=(32, 32), output_scale=0.1):
        super().__init__()

        self.output_scale = output_scale

        # Build hidden layers with tanh activations.
        # tanh is more stable than ReLU here because it bounds activations
        # to (-1, 1), preventing explosive outputs during early meta-training.
        self.hidden_layers = [
            tf.keras.layers.Dense(
                units,
                activation="tanh",
                kernel_initializer="glorot_uniform",
                name=f"hidden_{i}",
            )
            for i, units in enumerate(hidden_sizes)
        ]

        # Output layer: one scalar update per parameter.
        # tanh bounds the output, then we scale it down to keep
        # initial updates small while meta-training stabilises.
        self.output_layer = tf.keras.layers.Dense(
            1,
            activation="tanh",
            kernel_initializer="glorot_uniform",
            name="output",
        )

    def call(self, features: tf.Tensor) -> tf.Tensor:
        """
        Forward pass through the optimizer network.
    
        """
        x = features              

        for layer in self.hidden_layers:
            x = layer(x)                   

        x = self.output_layer(x)           

        updates = tf.squeeze(x, axis=-1) * self.output_scale 

        return updates