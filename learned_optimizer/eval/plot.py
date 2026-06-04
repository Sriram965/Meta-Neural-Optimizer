"""
Turn raw evaluation results into loss curves and a rank table.

"""
import numpy as np
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt


STYLE = {
    "learned":      {"color": "#e41a1c", "lw": 2.5, "zorder": 5},
    "adam":         {"color": "#377eb8", "lw": 1.8, "zorder": 4},
    "sgd_momentum": {"color": "#4daf4a", "lw": 1.8, "zorder": 3},
    "rmsprop":      {"color": "#984ea3", "lw": 1.8, "zorder": 2},
    "sgd":          {"color": "#999999", "lw": 1.2, "zorder": 1},
}

LABELS = {
    "learned":      "Learned (ours)",
    "adam":         "Adam (tuned)",
    "sgd_momentum": "SGD + Momentum",
    "rmsprop":      "RMSProp (tuned)",
    "sgd":          "SGD (tuned)",
}


def plot_loss_curves(results: dict, output_path: str = "plots/loss_curves.png"):
    """
    One subplot per task, log-scale y-axis.
    Shows mean ± 1 std deviation across seeds.
    The shaded band gives a visual sense of how consistent
    each optimizer is across different random starting points.
    """
    task_names = list(results.keys())
    n_tasks = len(task_names)

    fig, axes = plt.subplots(1, n_tasks, figsize=(5 * n_tasks, 4.5), squeeze=False)

    for col, task_name in enumerate(task_names):
        ax = axes[0][col]
        task_results = results[task_name]

        for opt_name, trajectories in task_results.items():
            arr  = np.array(trajectories, dtype=float)
            mean = np.mean(arr, axis=0)
            std  = np.std(arr,  axis=0)
            steps = np.arange(len(mean))
            s = STYLE.get(opt_name, {"color": "black", "lw": 1.5, "zorder": 0})

            ax.semilogy(
                steps, mean,
                label=LABELS.get(opt_name, opt_name),
                color=s["color"], linewidth=s["lw"], zorder=s["zorder"]
            )
            # Shaded uncertainty band
            ax.fill_between(
                steps,
                np.maximum(mean - std, 1e-12),
                mean + std,
                alpha=0.15, color=s["color"]
            )

        ax.set_title(task_name, fontsize=10, fontweight="bold")
        ax.set_xlabel("Optimization step")
        ax.set_ylabel("Loss (log scale)")
        ax.legend(fontsize=7, framealpha=0.8)
        ax.grid(True, which="both", alpha=0.25)

    plt.suptitle("Learned Optimizer vs Tuned Baselines", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Loss curves saved → {output_path}")


def print_rank_table(results: dict):
    """
    Print final loss for every (task, optimizer) pair,
    then compute and print the average rank across all tasks.
    The average rank gives a single-number summary of overall performance,
    """
    if not results:
        print("No results to display.")
        return

    opt_names  = list(next(iter(results.values())).keys())
    task_names = list(results.keys())
    col_w = 20

    # Build header
    header = f"{'Task':<30}" + "".join(
        f"{LABELS.get(o, o):>{col_w}}" for o in opt_names
    )
    divider = "-" * len(header)

    print("\n" + "=" * len(header))
    print("Final Loss — mean across seeds (lower is better)")
    print("=" * len(header))
    print(header)
    print(divider)

    all_ranks = {o: [] for o in opt_names}

    for task_name in task_names:
        row = f"{task_name:<30}"
        final_losses = {}

        for opt in opt_names:
            arr = np.array(results[task_name][opt])
            final_losses[opt] = float(np.mean(arr[:, -1]))
            row += f"{final_losses[opt]:>{col_w}.4f}"

        # Rank optimizers on this task (1 = lowest final loss = best)
        ranked = sorted(final_losses.keys(), key=lambda o: final_losses[o])
        for rank, opt in enumerate(ranked, 1):
            all_ranks[opt].append(rank)

        print(row)

    # Print average rank row
    print(divider)
    avg_row = f"{'Average Rank':<30}"
    for opt in opt_names:
        avg = float(np.mean(all_ranks[opt]))
        avg_row += f"{avg:>{col_w}.2f}"
    print(avg_row)
    print("=" * len(header))
    print("Rank 1 = best on that task. Lower average rank = better overall.\n")