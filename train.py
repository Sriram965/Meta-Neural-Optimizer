"""
train.py — main entry point for the meta-learned optimizer project.

Usage:
    python train.py           # full training run (50,000 meta-steps)
    python train.py --quick   # smoke test (200 meta-steps only)

Outputs:
    checkpoints/opt_net_final     — saved optimizer weights
    plots/meta_training.png       — meta-loss and grad-norm curves
"""
import os
import argparse
import numpy as np
import tensorflow as tf

from learned_optimizer.tasks.quadratic     import QuadraticTask, NoisyQuadraticTask
from learned_optimizer.tasks.logistic      import LogisticRegressionTask
from learned_optimizer.tasks.toy_functions import RosenbrockTask
from learned_optimizer.optimizer.mlp_opt   import MLPOptimizer
from learned_optimizer.meta_train.inner_loop import run_inner_loop
from learned_optimizer.meta_train.outer_loop import MetaTrainer



# Hyperparameters — change these to experiment
N_META_STEPS  = 50_000   
INNER_STEPS   = 50    
OUTER_LR      = 1e-4    
GRAD_CLIP     = 1.0      
HIDDEN_SIZES  = (32, 32)
WEIGHT_DECAY  = 1e-4    
WARMUP_STEPS  = 1_000   
SEED          = 42


def set_seed(seed: int):
    """Fixing  all random seeds for reproducibility."""
    tf.random.set_seed(seed)
    np.random.seed(seed)


def build_task_sampler():
    """
    Returns a callable that samples a random task from the Tier-1 suite.
    The sampling weights are designed so the optimizer sees mostly
    quadratic tasks (which are fundamental) but also gets meaningful
    exposure to the stochastic and nonconvex cases.
    """
    rng = np.random.default_rng()  

    def sampler():
        choice = rng.choice(5, p=[0.25, 0.25, 0.20, 0.20, 0.10])

        if choice == 0:
            # Well-conditioned quadratic — basic convergence test
            return QuadraticTask(
                dim=int(rng.integers(5, 30)),
                condition_number=1.0,
            )
        elif choice == 1:
            # Ill-conditioned quadratic — tests curvature adaptation
            return QuadraticTask(
                dim=int(rng.integers(5, 20)),
                condition_number=float(rng.uniform(10, 100)),
            )
        elif choice == 2:
            # Noisy quadratic — tests robustness to gradient noise
            return NoisyQuadraticTask(
                dim=int(rng.integers(5, 20)),
                noise_std=float(rng.uniform(0.01, 0.5)),
            )
        elif choice == 3:
            # Logistic regression — real ML-flavoured stochastic task
            return LogisticRegressionTask(
                n_samples=int(rng.integers(100, 400)),
                input_dim=int(rng.integers(10, 50)),
                batch_size=32,
            )
        else:
            # Rosenbrock — hard nonconvex benchmark (10% of the time)
            return RosenbrockTask(dim=2)

    return sampler


def build_eval_tasks():
    """
    Fixed, seeded evaluation tasks — always the same so results are comparable.

    These are the tasks we will compare the learned optimizer against
    Adam, SGD, and RMSProp on.
    """
    return [
        QuadraticTask(dim=10, condition_number=1.0,  seed=0),
        QuadraticTask(dim=10, condition_number=50.0, seed=1),
        NoisyQuadraticTask(dim=10, noise_std=0.2,    seed=2),
        LogisticRegressionTask(n_samples=200, input_dim=20, seed=3),
        RosenbrockTask(dim=2),
    ]


def plot_meta_training(history: dict):
    """Plot meta-loss and gradient norm curves from training history."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs("plots", exist_ok=True)
        steps = history["step"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.semilogy(steps, history["meta_loss"], color="#e41a1c", linewidth=1.5)
        ax1.set_title("Meta-Loss over Training", fontweight="bold")
        ax1.set_xlabel("Meta-step")
        ax1.set_ylabel("Meta-loss (log scale)")
        ax1.grid(True, which="both", alpha=0.3)

        ax2.semilogy(steps, history["grad_norm"], color="#377eb8", linewidth=1.5)
        ax2.set_title("Gradient Norm over Training", fontweight="bold")
        ax2.set_xlabel("Meta-step")
        ax2.set_ylabel("Gradient norm (log scale)")
        ax2.grid(True, which="both", alpha=0.3)

        plt.suptitle("Meta-Training Curves", fontsize=13)
        plt.tight_layout()
        plt.savefig("plots/meta_training.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Training curves saved → plots/meta_training.png")

    except Exception as e:
        print(f"Plotting skipped: {e}")


def evaluate(opt_net, eval_tasks, n_steps: int = 200):
    """
    Quick evaluation: run the learned optimizer on each eval task
    and print the final loss. Full comparison against baselines
    will be added in the eval/ module next.
    """
    print("\n--- Evaluation ---")
    for task in eval_tasks:
        task_name = type(task).__name__

        # Run the learned optimizer
        losses, _ = run_inner_loop(task, opt_net, K=n_steps)
        final_loss = float(losses[-1])

        print(f"{task_name:<35} final loss: {final_loss:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick", action="store_true",
        help="Run 200 meta-steps instead of 50,000 (for testing)"
    )
    args = parser.parse_args()

    n_steps = 200 if args.quick else N_META_STEPS

    set_seed(SEED)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    print("=" * 55)
    print("  Meta-Learned Optimizer — Tier 1 (TensorFlow)")
    print("=" * 55)
    print(f"  Meta-steps   : {n_steps}")
    print(f"  Inner steps  : {INNER_STEPS}")
    print(f"  Hidden sizes : {HIDDEN_SIZES}")
    print(f"  Outer LR     : {OUTER_LR}")
    print(f"  Grad clip    : {GRAD_CLIP}")
    print("=" * 55)

    # Build the optimizer network
    opt_net = MLPOptimizer(hidden_sizes=HIDDEN_SIZES, output_scale=0.1)

    # Warm-start the Keras layers with one dummy forward passfor building weights
    dummy_task = QuadraticTask(dim=5, seed=99)
    _ = run_inner_loop(dummy_task, opt_net, K=2)
    n_params = sum(np.prod(v.shape) for v in opt_net.trainable_variables)
    print(f"\n  Optimizer network: {n_params} trainable parameters\n")

    # Build the task sampler and trainer
    task_sampler = build_task_sampler()
    trainer = MetaTrainer(
        opt_net        = opt_net,
        task_sampler   = task_sampler,
        outer_lr       = OUTER_LR,
        meta_grad_clip = GRAD_CLIP,
        inner_steps    = INNER_STEPS,
        weight_decay   = WEIGHT_DECAY,
        warmup_steps   = WARMUP_STEPS,
    )

    # Run meta-training
    print("[Meta-training...]\n")
    history = trainer.train(
        n_meta_steps     = n_steps,
        log_every        = max(1, n_steps // 200),
        checkpoint_every = max(1, n_steps // 10),
        checkpoint_dir   = "checkpoints",
    )

    # Saving final weights
    opt_net.save_weights("checkpoints/opt_net_final.weights.h5")
    print("\n  Final weights saved → checkpoints/opt_net_final")

    # Plot training curves
    plot_meta_training(history)

    # Quick evaluation on fixed test tasks
    eval_tasks = build_eval_tasks()
    evaluate(opt_net, eval_tasks, n_steps=200 if not args.quick else 50)
    


if __name__ == "__main__":
    main()