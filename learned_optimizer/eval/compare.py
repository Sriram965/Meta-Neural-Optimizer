"""
Evaluate the learned optimizer against tuned standard baselines.
"""

import tensorflow as tf
import numpy as np
from ..meta_train.inner_loop import run_inner_loop



# This covers everything from very conservative (1e-4) to
# quite aggressive (0.1) learning rates.
LR_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1]

# The baselines we compare against.
# Each maps a name to a function that builds the optimizer given a lr.
BASELINES = {
    "sgd": lambda lr: tf.keras.optimizers.SGD(
        learning_rate=lr
    ),
    "sgd_momentum": lambda lr: tf.keras.optimizers.SGD(
        learning_rate=lr, momentum=0.9
    ),
    "rmsprop": lambda lr: tf.keras.optimizers.RMSprop(
        learning_rate=lr
    ),
    "adam": lambda lr: tf.keras.optimizers.Adam(
        learning_rate=lr
    ),
}


def _run_one_baseline(optimizer, task, n_steps: int, theta_init) -> list:
    """
    Run a single standard optimizer for n_steps on a task and return the loss trajectory.
    """

    theta = tf.Variable(tf.identity(theta_init))
    losses = []

    for _ in range(n_steps):
        with tf.GradientTape() as tape:
            loss = task.loss(theta)
        losses.append(float(loss))
        grad = tape.gradient(loss, theta)
        optimizer.apply_gradients([(grad, theta)])

    return losses


def _tune_and_run(baseline_name: str, task, n_steps: int, theta_init) -> list:
    """
    Grid search over learning rates and return the best loss trajectory.
    'Best' here is defined as the lowest final loss after n_steps, since that's what we ultimately care about.
    """
    best_final_loss = float("inf")
    best_trajectory = None

    for lr in LR_GRID:
        try:
            opt = BASELINES[baseline_name](lr)
            trajectory = _run_one_baseline(opt, task, n_steps, theta_init)

            if trajectory[-1] < best_final_loss:
                best_final_loss = trajectory[-1]
                best_trajectory = trajectory
        except Exception:
            pass

    # If every lr failed, return a trajectory of NaNs
    if best_trajectory is None:
        return [float("nan")] * n_steps

    return best_trajectory


def evaluate_all(opt_net, tasks, n_steps: int = 200, n_seeds: int = 5) -> dict:
    """
    Full evaluation: learned optimizer vs all baselines on all tasks.

    We run multiple seeds per (task, optimizer) pair so we can report
    mean and standard deviation ,so averaging over seeds gives a
    more reliable picture of true performance.

    Args:
        opt_net  : trained MLPOptimizer
        tasks    : list of BaseTask instances (fixed, seeded eval tasks)
        n_steps  : number of inner optimization steps per evaluation run
        n_seeds  : number of random starting points to average over

    Returns:
        results  : nested dict  task_name → optimizer_name → list of trajectories
                   each trajectory is a list of n_steps float loss values
                   shape conceptually: [n_seeds, n_steps]
    """
    results = {}

    for task in tasks:
        task_name = task.name
        print(f"\nEvaluating: {task_name}")
        results[task_name] = {}

        learned_trajectories = []
        for seed in range(n_seeds):
            tf.random.set_seed(seed)
            theta_init = task.sample_theta()
            losses, _ = run_inner_loop(
                task, opt_net, K=n_steps, theta_init=theta_init
            )
            learned_trajectories.append([float(l) for l in losses])
        results[task_name]["learned"] = learned_trajectories
        avg = np.mean([t[-1] for t in learned_trajectories])
        print(f"  learned        final loss (avg): {avg:.4f}")

        # Baselines with learning rate tuning
        for baseline_name in BASELINES:
            trajectories = []
            for seed in range(n_seeds):
                tf.random.set_seed(seed)
                theta_init = task.sample_theta()
                traj = _tune_and_run(baseline_name, task, n_steps, theta_init)
                trajectories.append(traj)
            results[task_name][baseline_name] = trajectories
            avg = np.mean([t[-1] for t in trajectories])
            print(f"  {baseline_name:<15} final loss (avg): {avg:.4f}")

    return results