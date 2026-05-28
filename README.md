# Meta-Learned Neural Optimizer

A TensorFlow implementation of a meta-learned MLP optimizer,
trained via bilevel optimization on a suite of convex and
near-convex tasks.

Based on:
- Andrychowicz et al. (2016), "Learning to Learn by Gradient Descent by Gradient Descent"
- Wichrowska et al. (2017), "Learned Optimizers that Scale and Generalize"
- Metz et al. (2019), "Understanding and Correcting Pathologies in the Training of Learned Optimizers"

## Project Structure

    learned_optimizer/
    ├── tasks/          # Optimization task definitions
    ├── optimizer/      # MLP optimizer network
    ├── meta_train/     # Inner and outer training loops
    └── eval/           # Evaluation and plotting

## Setup

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt