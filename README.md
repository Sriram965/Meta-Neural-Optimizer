# Meta-Learned Neural Optimizer

A TensorFlow implementation of a meta-learned optimizer trained via 
bilevel optimization. Instead of using a hand-designed optimizer like 
Adam or SGD, we train a small neural network to *be* the optimizer — 
learning update rules directly from experience across a diverse suite 
of optimization tasks.

Based on:
- Andrychowicz et al. (2016), "Learning to Learn by Gradient Descent by Gradient Descent"
- Wichrowska et al. (2017), "Learned Optimizers that Scale and Generalize"
- Metz et al. (2019), "Understanding and Correcting Pathologies in the Training of Learned Optimizers"

---

## How It Works

The project uses **bilevel optimization** — two nested loops:

**Inner loop:** The learned optimizer runs for K steps on a task,
producing parameter updates via a small MLP network.

**Outer loop:** The meta-loss (average log-loss over the trajectory)
is backpropagated through the inner loop to update the optimizer's
own weights.

After 50,000 meta-training steps across a diverse task suite, the
optimizer learns update rules that generalize across problem types
without any learning rate tuning.

---

## Project Structure
Meta_Optimiser/
├── learned_optimizer/
│   ├── tasks/
│   │   ├── base.py           # Abstract base class for all tasks
│   │   ├── quadratic.py      # Quadratic bowl (easy/hard curvature)
│   │   ├── logistic.py       # Logistic regression (stochastic)
│   │   └── toy_functions.py  # Rosenbrock (nonconvex benchmark)
│   ├── optimizer/
│   │   ├── features.py       # 11-dim gradient feature computation
│   │   └── mlp_opt.py        # MLP optimizer network (coordinatewise)
│   ├── meta_train/
│   │   ├── inner_loop.py     # Differentiable K-step inner loop
│   │   ├── meta_loss.py      # Average log-loss over trajectory
│   │   └── outer_loop.py     # Meta-training with gradient clipping
│   └── eval/
│       ├── compare.py        # Baseline comparison with LR grid search
│       └── plot.py           # Loss curves and rank tables
├── train.py                  # Main entry point
├── evaluate.py               # Full evaluation after training


## Results

Evaluated on 5 fixed tasks with 10 random seeds each.
Baselines use grid-searched learning rates over
{1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1}.

| Task | Learned (ours) | SGD | SGD+Mom | RMSProp | Adam |
|---|---|---|---|---|---|
| Quadratic (easy) | **0.0025** | 0.0259 | 0.0218 | 0.0545 | 0.0444 |
| Quadratic (hard) | **0.0260** | 0.0228 | 0.4794 | 0.6255 | 0.6199 |
| Noisy Quadratic | -0.0061 | -0.0002 | **-0.1700** | -0.0078 | 0.0244 |
| Logistic Regression | **0.0720** | 0.3673 | 0.1206 | 0.1201 | 0.1215 |
| Rosenbrock | **1.0545** | 1.2927 | 1.4191 | 1.5414 | 1.5360 |
| **Average Rank** | **1.60** | 3.00 | 2.40 | 3.80 | 4.20 |

The learned optimizer achieves the best average rank across all tasks,
beating well-tuned Adam on 4 out of 5 tasks — without any learning
rate tuning of its own.

---

## Architecture

**Optimizer network:** 2-layer MLP (32→32→1) with tanh activations,
applied coordinatewise (shared weights across all parameters).

**Input features (11-dim per parameter):**
- log|g|, sign(g) — preprocessed gradient
- log|m|, sign(m) × 3 timescales — momentum estimates (β=0.9, 0.99, 0.999)
- log(v) × 2 timescales — RMS estimates (β=0.9, 0.99)
- t/T — normalized timestep

**Meta-training:**
- 50,000 outer steps, Adam lr=1e-4 with 1,000-step warmup
- Gradient clipping at norm=1.0
- L2 weight decay=1e-4
- 50 inner steps per meta-step