# 1. Central Claim

The paper asserts that a relatively simple learning procedure—**back-propagation of errors**—is powerful enough to construct useful internal, distributed representations in hidden layers of neural networks.

By adjusting connection weights via gradient descent, the network can automatically extract regularities and domain features without manual feature engineering, bypassing the strict structural limitations that plagued earlier architectures like the single-layer perceptron.

# 2. Core Architecture & Algorithms

## Network Forward Pass

The simplest implementation uses a strictly feed-forward, layered architecture consisting of:

- An input layer
- An arbitrary number of hidden layers
- An output layer

Constraints:

- Connections within a layer are forbidden.
- Feedback loops are forbidden.
- Connections may skip intermediate layers.

# 3. Evaluation Datasets & Verification Baseline

To verify your implementation, choose one of the benchmark tasks detailed in the paper.

## Task Option A: The Mirror Symmetry Detector

### Dataset

- 64 possible binary input vectors of length 6.
- Target output is `1` if the vector is symmetrical about its center (e.g., `101101`).
- Target output is `0` otherwise.

### Network Layout

```text
6 Input Units → 2 Hidden Units → 1 Output Unit
```

### Initialization

- Uniform random weights in the range `[-0.3, 0.3]`.

### Hyperparameters

- Learning rate (ε) = `0.1`
- Momentum (α) = `0.9`

### Expected Behavior / Metric

- Complete convergence should occur within approximately 1,425 epochs.
- Upon convergence, the weights on each side of a hidden unit's midpoint should:
  - Mirror each other in magnitude.
  - Have opposite signs.
  - Form ratios close to `1 : 2 : 4`.
