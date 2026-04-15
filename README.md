# Creating Your First Models: Interactive Workshop Dashboard

An interactive educational dashboard built by the [Lehigh Machine Learning Club](https://github.com/Lehigh-Machine-Learning-Club) for our **Creating Your First Models** introductory workshop. Built entirely in Python.

**Our philosophy: if you can't see it, you can't understand it.** Every weight, gradient, and activation is visible - no black boxes.

---

## Quick Start (Workshop Attendees)

You need **Python 3.9+** installed. Open a terminal and run:

```bash
git clone https://github.com/Lehigh-Machine-Learning-Club/models-workshop.git
cd models-workshop
pip install -r requirements.txt
streamlit run app.py
```

The dashboard will open in your browser automatically. Use the **sidebar** to navigate between sections.

> **Trouble with the install?** If `streamlit-drawable-canvas` fails to install on your machine, everything else will still work - you just won't be able to use the "Draw Your Own Digit" canvas on the MNIST page. A file upload fallback is provided. You can skip it by installing everything else manually:
> ```bash
> pip install streamlit numpy pandas plotly scikit-learn torch torchvision matplotlib Pillow
> ```

---

## Code Setup

You can set up the project in two ways:

### Option A: Manual Setup

```bash
git clone https://github.com/Lehigh-Machine-Learning-Club/models-workshop.git
cd models-workshop
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### Option B: Workshop Shell Scripts (Recommended)

After cloning the repo, run one script from the project root.
By default, scripts use committed model artifacts and only retrain/recompute if files are missing.

- **macOS / Linux**
  ```bash
  bash shell_scripts/setup_workshop_unix.sh
  ```

- **Windows (cmd/.bat)**
  ```bat
  shell_scripts\setup_workshop_windows.bat
  ```

- **Windows (PowerShell fallback)**
  ```powershell
  powershell -ExecutionPolicy Bypass -File .\shell_scripts\setup_workshop_windows.ps1
  ```

Optional script flags for fully fresh artifacts:
- Unix: `--recompute-toy --recompute-mnist`
- Windows (.bat): `--recompute-toy --recompute-mnist`
- Windows (PowerShell): `-RecomputeToy -RecomputeMnist`

---

## What's Inside

The dashboard walks through four sections, building from simple to complex:

| Section | What You'll Learn |
|---|---|
| **1. Linear Regression** | Fitting lines and curves to data. How models minimize error. |
| **2. Classification** | KNN and Logistic Regression. Visualizing decision boundaries. |
| **3. Neural Networks: Toy MLP** | A tiny 2-3-1 network built from scratch in NumPy. Compare activations + learning rates, then scrub training checkpoints. |
| **4. Neural Networks: MNIST** | Scale to 784-pixel inputs with a compact `784→16→16→10` MLP. Inspect learned features, layer activations, and drawing-time uncertainty. |

No prior ML experience is assumed. Each section includes hover tooltips on technical terms, plain-English explanations alongside the math, and interactive controls so you can experiment yourself.

---

## Event Details

- **Event:** Creating Your First Model - An Introductory Workshop
- **Hosted by:** Lehigh Machine Learning Club
- **Date:** April 15, 2025
- **Location:** Packard Lab 416
- **Focus areas:** Regression, Classification, and Neural Networks

---

## Technical Details (For Contributors)

### Architecture

- **Framework:** [Streamlit](https://streamlit.io/) 1.54+
- **Visualization:** [Plotly](https://plotly.com/python/) for all interactive charts
- **ML Backend:** NumPy (from-scratch MLP in Section 3), PyTorch (pretrained MNIST model in Section 4)
- **UI:** Custom CSS hover tooltips, `st.popover()` glossary entries, sidebar mode controls, and cached on-demand training for toy checkpoints

### How the Toy MLP Playground Works

The toy page supports three modes:
- **Manual Tweaking**: edit all 13 parameters directly
- **Pre-computed Checkpoints**: load saved runs from `models/toy_checkpoints_*.npz`
- **On-demand Training (Cached)**: generate checkpoints in-app for custom activation/LR/epoch settings, then scrub through them

This keeps playback smooth while still allowing quick experimentation.

To regenerate artifacts from scratch:

```bash
# Toy MLP (Phase 1): trains across 5 activation functions, saves checkpoints
python scripts/precompute_toy_training.py --max-epochs 5500 --checkpoint-every 2

# MNIST (Phase 2): trains baseline + compact models, saves model + analytics artifacts
python scripts/train_mnist.py
```

### MNIST Notes (Current Setup)

- Primary demo model: **`784→16→16→10`** (`src/mnist_mlp.py`)
- Baseline comparator: **`784→64→10`**
- Draw-canvas inference includes:
  - MNIST-style preprocessing (crop, center, scale to 28x28)
  - confidence + entropy uncertainty checks
  - temperature-controlled softmax calibration for display-time probabilities

### Repository Structure

```
models-workshop/
├── app.py                        # Landing page
├── requirements.txt              # Dependencies
│
├── pages/                        # Dashboard sections (Streamlit multipage)
│   ├── 1_Linear_Regression.py
│   ├── 2_Classification.py
│   ├── 3_Neural_Networks_Toy.py
│   └── 4_Neural_Networks_MNIST.py
│
├── src/                          # Core modules
│   ├── mlp.py                    # NumPy MLP (forward + backprop from scratch)
│   ├── mnist_mlp.py              # PyTorch MNIST model
│   ├── data.py                   # Synthetic dataset generators
│   ├── precompute_toy.py         # Checkpoint serialization for toy MLP
│   ├── visualizations.py         # Plotly charts (Sections 1-3)
│   ├── mnist_visualizations.py   # Plotly charts (Section 4)
│   └── ui_components.py          # Shared tooltips, metrics, dividers
│
├── scripts/                      # Offline training scripts
│   ├── precompute_toy_training.py
│   └── train_mnist.py
│
├── shell_scripts/                # Cross-platform workshop setup scripts
│   ├── setup_workshop_unix.sh
│   ├── setup_workshop_windows.bat
│   └── setup_workshop_windows.ps1
│
├── models/                       # Pre-computed checkpoints (committed to repo)
│   ├── mnist_mlp.pt
│   ├── toy_checkpoints_*.npz
│   └── *.json
│
├── assets/                       # Educational images and diagrams
└── documentations/               # Design notes and planning docs
```


---

## Offline Training Scripts (Concise)

- `scripts/precompute_toy_training.py`
  - Runs full-batch NumPy training on the synthetic fruit dataset.
  - Saves checkpoint trajectories (`epochs`, params, loss/acc, boundary grid) to `models/toy_checkpoints_*.npz`.
  - These files power scrub-based playback in the toy page.

- `scripts/train_mnist.py`
  - Trains both MNIST architectures (baseline + compact), compares metrics, and saves artifacts.
  - Writes:
    - `models/mnist_mlp.pt` (primary compact model weights)
    - `models/training_history.json`
    - `models/architecture_comparison.json`
    - `models/test_predictions.npz`, `models/sample_digits.npz`, `models/misclassified.npz`
  - Optional regularization knobs are available (`--l1-lambda`, `--l1-target`) for interpretability experiments.

---

## License

Built by the Lehigh Machine Learning Club. For educational use.
