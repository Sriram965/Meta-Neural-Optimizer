"""
evaluate.py — run full comparison after training is complete.

Loads the trained optimizer weights and compares against
tuned SGD, SGD+Momentum, RMSProp, and Adam on all five
evaluation tasks. Produces loss curves and a rank table.
"""

import os
import numpy as np
import tensorflow as tf

from learned_optimizer.tasks.quadratic     import QuadraticTask, NoisyQuadraticTask
from learned_optimizer.tasks.logistic      import LogisticRegressionTask
from learned_optimizer.tasks.toy_functions import RosenbrockTask
from learned_optimizer.optimizer.mlp_opt   import MLPOptimizer
from learned_optimizer.meta_train.inner_loop import run_inner_loop
from learned_optimizer.eval.compare        import evaluate_all
from learned_optimizer.eval.plot           import plot_loss_curves, print_rank_table


def build_eval_tasks():
    """
    Fixed seeded evaluation tasks — identical to those used in train.py
    so results are directly comparable across runs.
    """
    return [
        QuadraticTask(dim=10, condition_number=1.0,  seed=0),
        QuadraticTask(dim=10, condition_number=50.0, seed=1),
        NoisyQuadraticTask(dim=10, noise_std=0.2,    seed=2),
        LogisticRegressionTask(n_samples=200, input_dim=20, seed=3),
        RosenbrockTask(dim=2),
    ]


def main():
    os.makedirs("plots", exist_ok=True)

    print("=" * 55)
    print("  Meta-Learned Optimizer — Full Evaluation")
    print("=" * 55)

    # Build and warm-start the optimizer network
    opt_net = MLPOptimizer(hidden_sizes=(32, 32), output_scale=0.1)
    dummy   = QuadraticTask(dim=5, seed=99)
    _       = run_inner_loop(dummy, opt_net, K=2)

    # Load the trained weights
    weights_path = "checkpoints/opt_net_final.weights.h5"
    if not os.path.exists(weights_path):
        print(f"ERROR: weights not found at {weights_path}")
        print("Make sure training has completed first.")
        return

    opt_net.load_weights(weights_path)
    print(f"Loaded weights from {weights_path}")

    # Count parameters
    n_params = sum(np.prod(v.shape) for v in opt_net.trainable_variables)
    print(f"Optimizer network: {n_params} trainable parameters")

    # Build evaluation tasks
    eval_tasks = build_eval_tasks()
    print(f"\nEvaluating on {len(eval_tasks)} tasks with 10 seeds each...")
    print("(This may take a few minutes due to baseline grid search)\n")

    # Run full evaluation — 10 seeds, 200 inner steps
    results = evaluate_all(
        opt_net  = opt_net,
        tasks    = eval_tasks,
        n_steps  = 200,
        n_seeds  = 10,
    )

    # Plot loss curves
    plot_loss_curves(results, "plots/final_loss_curves.png")

    # Print rank table
    print_rank_table(results)

    print("Evaluation complete.")
    print("Loss curves saved → plots/final_loss_curves.png")


if __name__ == "__main__":
    main()