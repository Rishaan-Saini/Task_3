# 1. Central Claim

The paper shows that a simple method called **back-propagation** allows artificial neural networks to automatically figure out the hidden patterns and rules of a task on their own. By calculating how mistakes trace back through the network, it changes its internal connections to create custom features. This proved that multi-layer networks could solve complex, non-linear problems that older, simpler systems (like early perceptrons) couldn't handle.

# 2. Core Architecture & Algorithm

## The Forward Pass (Guessing)

The network passes information from the bottom (inputs) to the top (outputs). Each neuron:

1. Adds up the incoming signals from the layer below it.
2. Multiplies them by their connection weights (strengths).
3. Runs the total through an S-shaped curve filter (the sigmoid function) to determine its output.

To give neurons a flexible baseline to trigger, an extra constant input of `1` is attached to act as a trainable threshold (**bias**).

## The Backward Pass (Learning from Mistakes)

After the network makes a guess, the code measures how far off the final output is from the correct answer.

The algorithm then works backward from the output layer to the input layer. Using the **chain rule** from calculus, it calculates exactly how much each individual connection weight contributed to the final mistake.

## The Update (Fixing Weights)

Instead of making immediate adjustments after every example, the network aggregates its corrections over the entire dataset (**batch learning**).

It then uses **momentum**, which updates the weights like a heavy ball rolling down a hill:

- Builds up speed in useful directions.
- Smooths out sudden changes.
- Helps roll through flat regions of the error landscape.

# 3. Dataset, Metrics, and Verification Baseline

## The Task: Mirror Symmetry Detector

### Dataset

- All 64 possible combinations of a 6-digit binary code.
- Example: `101101`
- Target output = `1` if the pattern is perfectly symmetrical around the center.
- Target output = `0` otherwise.

### Network Layout

```text
6 Inputs → 2 Hidden Neurons → 1 Output Neuron
```

### Initialization

- All starting weights are randomized between `-0.3` and `0.3`.

### Success Criteria

Early in training, the network often gets stuck on a flat plateau:

- Total Error = `4.0`
- Accuracy = `87.5%`

At this stage, it simply predicts `0` for every input.

A successful implementation should:

- Escape this plateau.
- Reach `100%` classification accuracy.
- Typically converge after roughly `1,400–3,000` training cycles.

### Verification of Learned Representation

When training succeeds, the two hidden neurons should automatically organize their weights into mirror-image patterns:

- Opposite positive/negative signs.
- Symmetric around the center.
- Magnitudes approximately following a `1 : 2 : 4` ratio from the edges toward the center.

These hidden units effectively become learned **symmetry detectors**.
