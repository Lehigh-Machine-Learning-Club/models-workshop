
# Mechanistic Interpretability Dashboard

An interactive, educational dashboard built for the Lehigh Machine Learning Club to demystify machine learning models. Built completely in Python using [Streamlit](https://streamlit.io/) and [NumPy](https://numpy.org/).

Our philosophical approach is simple: **if you can't see it, you can't understand it**. We don't just show you how to train a model; we open the black box to show you its hidden mechanics, layer by layer, pixel by pixel, epoch by epoch.

## What We've Built

This application structurally unfolds the theory behind neural networks into two highly accessible, interactive phases:

### Phase 1: Neural Networks Toy MLP

A **2-3-1 Multi-Layer Perceptron** built completely from scratch using NumPy. It tackles a synthetic 2D binary classification problem ("Poisonous Fruit Detector").

-**The Engine:** All forward and backward propagation code is fully exposed (with pseudocode provided in-app).

-**The Playground:** Users can tweak the 13 network parameters natively or watch them shift through pre-calculated training phases to witness the decision boundary adapt to the data in real-time.

-**Interpretability:** Look inside the network at the individual "Feature Detectors" to see what specific logic each hidden neuron is looking for.

### Phase 2: Neural Networks MNIST Scale-Up

Once the principles from Phase 1 are understood, we scale the exact same architectural concepts to 784 pixels and 109,000 parameters classifying the real-world **MNIST** dataset.

-**Architectural Analysis:** Comparing deep vs shallow networks head-to-head.

-**Deconstruction:** 3Blue1Brown-inspired exploration into the hidden features and abstract representations neural networks create.

-**Interactive Sandbox:** Draw your own digit and watch layer activations propagate dynamically, including an uncertainty/confidence threshold check for random scribbles.

## Model Checkpoints & Pre-Computation Architecture

The dashboard relies on pre-computed model checkpoints to ensure smooth animation and reliable scrubbing. By default, Streamlit’s execution model makes high-frequency state updates (like playing thousands of training epochs live) prone to stuttering. To resolve this, **we pre-compute the entire training runs** locally.

*(Note: the pre-calculated `models/` checkpoints are tracked directly in this repository, so no initial script-running is required for the application to function!)*

If you wish to re-train the models or generate new checkpoints from scratch, you can run the provided scripts:

1.**Pre-computing the Toy MLP (Phase 1):**

```bash

python scripts/precompute_toy_training.py --max-epochs 10000 --checkpoint-every 5

```

*This trains the NumPy network simultaneously across 5 distinct activation functions, serializing the weights and decision boundaries chunk-by-chunk to `.npz` files.*

2.**Training the Multi-Layer MNIST PyTorch Models (Phase 2):**

```bash

python scripts/train_mnist.py

```

*This downloads the MNIST dataset, trains both a shallow and deep architecture, compares them, saves the master `.pt` model state, and generates sample inference dictionaries for the visualizations.*

## Repository Structure

A concise breakdown of how this codebase is organized:

```text

mechanistic-interpretability-dashboard/

│

├── app.py                     # Main Streamlit landing page

├── requirements.txt           # Project dependencies

├── .gitignore                 # Git ignore config (models preserved)

│

├── pages/                     # Streamlit App Navigation Pages

│   ├── 1_Linear_Regression.py

│   ├── 2_Classification.py

│   ├── 3_Neural_Networks_Toy.py

│   └── 4_Neural_Networks_MNIST.py

│

├── src/                       # Core Application Modules

│   ├── mlp.py                 # Pure NumPy fully-connected network architecture

│   ├── mnist_mlp.py           # PyTorch equivalent MNIST model architecture

│   ├── precompute_toy.py      # Trajectory serialization tooling for Phase 1

│   ├── data.py                # Dataset generators for the toy problems

│   ├── visualizations.py      # Plotly functions and graphing for Phase 1

│   ├── mnist_visualizations.py# Plotly functions and graphing for Phase 2

│   └── ui_components.py       # Custom Streamlit helper widgets (CSS, tooltips)

│

├── scripts/                   # Model Generation & Training CLI Tools

│   ├── precompute_toy_training.py

│   └── train_mnist.py

│

├── models/                    # Saved Checkpoint Data (Tracked loosely via Git)

│   ├── mnist_mlp.pt           # Compiled PyTorch master weights

│   ├── toy_checkpoints_*.npz  # Frozen numpy trajectories of the toy model

│   └── *.json                 # History and architecture breakdown metrics

│

├── assets/                    # Educational images and diagrams

└── documentations/            # Implementation plans, drafts, and notes 

```

## Getting Started

1. Set up a local Python virtual environment:

   ```bash

   ```

python3 -m venv .venv

source .venv/bin/activate

```

2. Install dependencies:

   ```bash

pip install -r requirements.txt

```

3. Run the dashboard:

   ```bash

   ```

streamlit run app.py

```

```
