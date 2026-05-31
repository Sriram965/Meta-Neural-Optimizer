import tensorflow as tf


def compute_meta_loss(losses, eps: float = 1e-8) -> tf.Tensor:
    """
    Compute the meta-loss from an inner-loop loss trajectory.

    We use the average log-loss formulation from Wichrowska et al. (2017):

        L_meta = (1/K) * sum_{t=1}^{K} log( f_t + eps )

    Args:
        losses : list of K scalar tensors from the inner loop
        eps    : small floor to prevent log(0) when loss is exactly zero

    Returns:
        meta_loss : a single scalar tensor
    """
    # Stack the list of K scalars into one tensor of shape [K]
    stacked = tf.stack(losses)

    log_losses = tf.math.log(stacked + eps)

    # Average across all K steps — this is the single number the
    # outer loop will differentiate through to update phi.
    return tf.reduce_mean(log_losses)