# Creating Your First Models: Interactive Workshop Dashboard

An interactive educational dashboard built by the [Lehigh Machine Learning Club](https://github.com/Lehigh-Machine-Learning-Club) for our **Creating Your First Model** introductory workshop. Built entirely in Python.

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

## What's Inside

The dashboard walks through four sections, building from simple to complex:

| Section | What You'll Learn |
|---|---|
| **1. Linear Regression** | Fitting lines and curves to data. How models minimize error. |
| **2. Classification** | KNN and Logistic Regression. Visualizing decision boundaries. |
| **3. Neural Networks: Toy MLP** | A tiny 2-3-1 network built from scratch in NumPy. Watch it learn to classify "poisonous fruits" in real-time. |
| **4. Neural Networks: MNIST** | Scale up to 784 pixels and 109K parameters. Explore how a trained network reads handwritten digits, then draw your own. |

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
- **UI:** Custom CSS hover tooltips, `st.popover()` glossary entries, `@st.fragment` for animation isolation

### How the Toy MLP Playground Works

The dashboard does **not** train models live in the browser. Instead, training runs are pre-computed offline and saved as checkpoint files. This gives smooth scrubbing and animation (the Streamlit execution model would stutter otherwise). The checkpoints are committed directly to this repo, so cloning is all you need.

To regenerate checkpoints from scratch:

```bash
# Toy MLP (Phase 1) - trains across 5 activation functions, saves weight snapshots
python scripts/precompute_toy_training.py --max-epochs 10000 --checkpoint-every 5

# MNIST (Phase 2) - trains shallow + deep architectures, saves model + inference artifacts
python scripts/train_mnist.py
```

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
├── models/                       # Pre-computed checkpoints (committed to repo)
│   ├── mnist_mlp.pt
│   ├── toy_checkpoints_*.npz
│   └── *.json
│
├── assets/                       # Educational images and diagrams
└── documentations/               # Design notes and planning docs
```

### Known Limitations

- Streamlit caps at roughly 10 FPS for complex Plotly charts during `@st.fragment` animation.
- `streamlit-drawable-canvas` can be finicky on some platforms (Windows ARM, some Linux distros). File upload fallback is provided.
- The `use_container_width` parameter is deprecated in Streamlit 1.54+; migration to `width='stretch'` is pending.

---

## License

Built by the Lehigh Machine Learning Club. For educational use.
