import os
import tensorflow as tf
import numpy as np
from .inner_loop import run_inner_loop
from .meta_loss import compute_meta_loss


class MetaTrainer:
    """
    Orchestrates the full meta-training loop.

    Each meta-step:
      1. Sample a random task from the task suite
      2. Run the inner loop for K steps → loss trajectory
      3. Compute the meta-loss from the trajectory
      4. Add L2 weight decay to the meta-loss
      5. Backpropagate and clip gradients
      6. Apply the outer Adam update to phi
    """

    def __init__(
        self,
        opt_net,
        task_sampler,
        outer_lr: float = 1e-4,
        meta_grad_clip: float = 1.0,
        inner_steps: int = 50,
        weight_decay: float = 1e-4,
        warmup_steps: int = 1000,
    ):
        self.opt_net        = opt_net
        self.task_sampler   = task_sampler
        self.inner_steps    = inner_steps
        self.meta_grad_clip = meta_grad_clip
        self.weight_decay   = weight_decay
        self.warmup_steps   = warmup_steps
        self.base_lr        = outer_lr

        self.outer_optimizer = tf.keras.optimizers.Adam(
            learning_rate=outer_lr, beta_1=0.9, beta_2=0.999
        )

    def _current_lr(self, step: int) -> float:
        """
        This prevents large destructive updates during the unstable
        early phase of meta-training when phi is still essentially random.
        """
        if step < self.warmup_steps:
            fraction = step / self.warmup_steps
            return self.base_lr * (0.1 + 0.9 * fraction)
        return self.base_lr

    def _single_meta_step(self, task):
        """
        Run one complete meta-training step.
        Returns (meta_loss, gradient_norm) for logging.
        """
        with tf.GradientTape() as outer_tape:

            # Run the inner loop — the outer tape watches every
            # opt_net(features) call inside here and builds the
            # computation graph connecting phi to the meta-loss.
            losses, _ = run_inner_loop(
                task, self.opt_net, K=self.inner_steps
            )

            meta_loss = compute_meta_loss(losses)

            # L2 weight decay — gently penalise large weights in phi.
            l2_penalty = tf.add_n([
                tf.nn.l2_loss(v)
                for v in self.opt_net.trainable_variables
            ]) * self.weight_decay

            total_loss = meta_loss + l2_penalty

        # Compute gradients of total_loss with respect to phi
        meta_grads = outer_tape.gradient(
            total_loss, self.opt_net.trainable_variables
        )

        # Clipping the  gradients — the single most important stability intervention.
        clipped_grads, grad_norm = tf.clip_by_global_norm(
            meta_grads, self.meta_grad_clip
        )

        # Applying  the clipped gradients to phi
        self.outer_optimizer.apply_gradients(
            zip(clipped_grads, self.opt_net.trainable_variables)
        )

        return meta_loss, grad_norm

    def train(
        self,
        n_meta_steps: int,
        log_every: int = 100,
        checkpoint_every: int = 5000,
        checkpoint_dir: str = "checkpoints",
    ) -> dict:
        """
        Run the full meta-training loop for n_meta_steps iterations.
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        history = {"step": [], "meta_loss": [], "grad_norm": []}

        for step in range(n_meta_steps):

            # Updating the learning rate according to the warm-up schedule
            self.outer_optimizer.learning_rate.assign(
                self._current_lr(step)
            )

            # Sampling a fresh task — new random parameters every step
            task = self.task_sampler()

            meta_loss, grad_norm = self._single_meta_step(task)
            
            if tf.math.is_nan(meta_loss) or tf.math.is_inf(meta_loss):
                print(f"  [step {step}] NaN detected — skipping")
                continue

            # Logging
            if step % log_every == 0:
                ml = float(meta_loss)
                gn = float(grad_norm)
                lr = self._current_lr(step)
                history["step"].append(step)
                history["meta_loss"].append(ml)
                history["grad_norm"].append(gn)
                print(
                    f"step {step:6d} | "
                    f"meta_loss {ml:+8.4f} | "
                    f"grad_norm {gn:.4f} | "
                    f"lr {lr:.2e}"
                )

            # Save a checkpoint periodically so we don't lose progress if training is interrupted
            if step > 0 and step % checkpoint_every == 0:
                path = os.path.join(checkpoint_dir, f"opt_net_step_{step}.weights.h5")
                self.opt_net.save_weights(path)
                print(f"  checkpoint saved → {path}")

        return history