import tensorflow as tf
from ..optimizer.features import (
    init_states, update_momentum, update_rms, compute_features
)


def run_inner_loop(task, opt_net, K: int, theta_init=None):
    """
    Run K steps of the learned optimizer on a task.

    This function must be called inside an outer GradientTape that is
    watching opt_net.trainable_variables. The outer tape traces through
    the MLP forward passes (opt_net(features)) at each step, building
    the computation graph that connects phi to the meta-loss.

    Args:
        task       : any BaseTask instance
        opt_net    : MLPOptimizer instance
        K          : number of inner optimisation steps
        theta_init : optional starting theta; sampled from task if None

    Returns:
        losses : list of K scalar tensors, one loss value per inner step
        theta  : the final parameter tensor after K steps
    """

    # Initialise theta 
    if theta_init is None:
        theta = task.sample_theta()
    else:
        theta = tf.identity(theta_init)

    n_params = theta.shape[0]

    # Initialise running states
    momentum_states, rms_states = init_states(n_params)

    losses = []

    for step in range(K):

        # compute the task gradient 
        with tf.GradientTape() as inner_tape:
            inner_tape.watch(theta)
            task_loss = task.loss(theta)

        grad = inner_tape.gradient(task_loss, theta)  

        # stop gradient on the inner gradient
        grad = tf.stop_gradient(grad)
        

        # record the loss BEFORE this step's update 
        losses.append(task_loss)

        # update running states and build features
        momentum_states = update_momentum(grad, momentum_states)
        rms_states      = update_rms(grad, rms_states)
        features        = compute_features(
            grad, momentum_states, rms_states, step, K
        ) 

        # optimizer network forward pass
        # THIS is the operation the outer tape is tracking.
        # update depends on opt_net's weights phi, so the outer tape
        # can differentiate meta_loss all the way back to phi through here.
        update = opt_net(features)   

        # apply the update 
        theta = theta + update

    return losses, theta