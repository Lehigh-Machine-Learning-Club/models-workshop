"""
Phase 2: MNIST Scale-Up (Multi-Layer Perceptron)
Scaling Phase 1 principles from 2D toy data to 784-dimensional handwritten digits.

Sections:
  A. "From Fruits to Pixels" — Brief narrative bridge (no architecture comparison here)
  B. "Exploring MNIST" — Data explorer + 3B1B images (pixel grid, layer deconstruction)
  C. "How The Model Was Trained" — Training methodology, optimizer, loss, parameter breakdown
  D. "Architecture Comparison" — Standalone section comparing Small vs Large
  E. "Inside the Trained Network" — Feature detectors, 3B1B hidden features image, parameter elaboration
  F. "Watch a Digit Flow Through" — Layer activation inspector
  G. "Draw Your Own Digit" — Interactive canvas with confidence thresholding
  H. "Where the Model Struggles" — Error analysis
"""
import streamlit as st
import numpy as np
import torch
import os
import json

from src.mnist_mlp import MNIST_MLP
from src.mnist_visualizations import (
    plot_digit_grid, plot_pixel_heatmap, plot_feature_detector_grid,
    plot_layer_activations, plot_activation_flow,
    plot_confusion_matrix, plot_confidence_distribution,
    plot_output_probabilities, plot_architecture_comparison,
    plot_training_curves,
)
from src.ui_components import (
    inject_tooltip_css, tip, glossary_popover,
    metric_row, section_header, section_divider, math_block,
)

st.set_page_config(page_title="Neural Networks: MNIST", layout="wide")

# ────────────────────────────────────────────────────────
# CSS + Setup
# ────────────────────────────────────────────────────────
inject_tooltip_css()

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
ASSETS_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets')

# Plotly config to reduce flickering
PLOTLY_CONFIG = {
    'displayModeBar': False,
    'staticPlot': False,
}


@st.cache_resource
def load_model():
    """Load the pretrained MNIST MLP model."""
    model_path = os.path.join(MODELS_DIR, 'mnist_mlp.pt')
    if os.path.exists(model_path):
        try:
            model = MNIST_MLP.load_pretrained(model_path)
            return model
        except RuntimeError:
            st.error("Model checkpoint architecture mismatch. Re-train with `python scripts/train_mnist.py` for the 784→16→16→10 setup.")
            return None
    else:
        st.error(f"Model not found at {model_path}. Run `python scripts/train_mnist.py` first.")
        return None


@st.cache_data
def load_training_data():
    """Load saved training history, comparison, predictions, and samples."""
    data = {}

    history_path = os.path.join(MODELS_DIR, 'training_history.json')
    if os.path.exists(history_path):
        with open(history_path) as f:
            data['history'] = json.load(f)

    comparison_path = os.path.join(MODELS_DIR, 'architecture_comparison.json')
    if os.path.exists(comparison_path):
        with open(comparison_path) as f:
            data['comparison'] = json.load(f)

    predictions_path = os.path.join(MODELS_DIR, 'test_predictions.npz')
    if os.path.exists(predictions_path):
        npz = np.load(predictions_path)
        data['predictions'] = npz['predictions']
        data['labels'] = npz['labels']
        data['probabilities'] = npz['probabilities']

    samples_path = os.path.join(MODELS_DIR, 'sample_digits.npz')
    if os.path.exists(samples_path):
        npz = np.load(samples_path)
        data['sample_images'] = npz['images']
        data['sample_labels'] = npz['labels']

    misclassified_path = os.path.join(MODELS_DIR, 'misclassified.npz')
    if os.path.exists(misclassified_path):
        npz = np.load(misclassified_path)
        data['misc_images'] = npz['images']
        data['misc_true'] = npz['true_labels']
        data['misc_pred'] = npz['pred_labels']
        data['misc_probs'] = npz['probabilities']

    return data


def get_param_breakdown_rows(model):
    """Return per-layer parameter metadata for the currently loaded model."""
    h1 = model.hidden1
    h2 = model.hidden2
    layer_1_params = 784 * h1 + h1
    layer_2_params = h1 * h2 + h2
    layer_3_params = h2 * 10 + 10
    total = layer_1_params + layer_2_params + layer_3_params
    first_layer_share = 100.0 * layer_1_params / max(total, 1)
    return {
        'h1': h1,
        'h2': h2,
        'layer_1_params': layer_1_params,
        'layer_2_params': layer_2_params,
        'layer_3_params': layer_3_params,
        'total': total,
        'first_layer_share': first_layer_share,
    }


def preprocess_canvas_digit(image, threshold_frac=0.20):
    """
    Make freehand input more MNIST-like:
    1) threshold + crop to ink bbox
    2) resize glyph to ~20x20
    3) center in a 28x28 canvas
    """
    from PIL import Image

    img = np.asarray(image, dtype=np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    img = np.clip(img, 0.0, 1.0)

    max_val = float(img.max())
    if max_val <= 1e-6:
        return np.zeros((28, 28), dtype=np.float32), 0.0

    thr = threshold_frac * max_val
    mask = img > thr
    ink_ratio = float(mask.mean())
    if not np.any(mask):
        return np.zeros((28, 28), dtype=np.float32), 0.0

    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop = img[y0:y1, x0:x1]

    h, w = crop.shape
    side = max(h, w)
    pad_y = (side - h) // 2
    pad_x = (side - w) // 2
    square = np.pad(crop, ((pad_y, side - h - pad_y), (pad_x, side - w - pad_x)), mode='constant')

    pil = Image.fromarray((square * 255).astype(np.uint8), mode='L')
    glyph_20 = np.array(pil.resize((20, 20), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0

    out = np.zeros((28, 28), dtype=np.float32)
    out[4:24, 4:24] = glyph_20
    return out, ink_ratio


# ────────────────────────────────────────────────────────
# Load everything
# ────────────────────────────────────────────────────────
model = load_model()
training_data = load_training_data()

if model is None:
    st.stop()

# ════════════════════════════════════════════════════════
# PAGE HEADER
# ════════════════════════════════════════════════════════
st.title("Mechanistic Interpretability: Neural Networks")
st.markdown("### Phase 2: Scaling to MNIST — From 2 Features to 784 Pixels")
st.caption(f"Architecture: {model.architecture} • {model.count_parameters():,} parameters • Trained on 60,000 handwritten digits")

section_divider()

# ════════════════════════════════════════════════════════
# SECTION A: "From Fruits to Pixels" — Brief Bridge
# ════════════════════════════════════════════════════════
section_header("From Fruits to Pixels", "The same mechanics, massively scaled up", "")

st.markdown(f"""
Now that we understand how a tiny 2-3-1 network learns curved decision boundaries from just 13 parameters, 
let's apply the **exact same principles** to a real-world problem: recognizing handwritten digits.

The key change isn't conceptual — it's **scale**:

| Aspect | Phase 1 (Toy MLP) | Phase 2 (MNIST) |
|--------|-------------------|-----------------|
| Input features | 2 (spikiness, spottiness) | 784 (28×28 pixels) |
| Hidden neurons | 3 | {model.hidden1} → {model.hidden2} (two layers) |
| Output | 1 (binary: safe/poison) | 10 (digits 0-9) |
| Parameters | 13 | {model.count_parameters():,} |
| Dataset size | 200 samples | 70,000 images |
| Training | NumPy (from scratch) | {tip("PyTorch", "A deep learning framework by Meta that handles automatic differentiation (backprop), GPU acceleration, and efficient tensor operations. It does the same thing our NumPy MLP does — just much faster and more scalable.")} |

Every neuron still acts as a {tip("feature detector", "A trained neuron that fires strongly for specific input patterns. In Phase 1, neurons detected combinations of spikiness/spottiness. Here, they detect specific stroke patterns in digit images.")} — 
just detecting pixel patterns instead of fruit traits. The forward pass, backward pass, and gradient descent 
all work identically. The GPU just lets us do it orders of magnitude faster through 
{tip("matrix multiplication", "The workhorse operation of neural networks. Instead of computing each neuron one at a time, we multiply entire matrices of inputs × weights simultaneously. This operation parallelizes perfectly on GPUs.")} 
at scale.
""", unsafe_allow_html=True)

section_divider()

# ════════════════════════════════════════════════════════
# SECTION B: "Exploring MNIST" — Enhanced with 3B1B images
# ════════════════════════════════════════════════════════
section_header("Exploring MNIST", "What does the data look like?", "")

# 3B1B Style: Pixel grid image
st.markdown("#### How a Computer Sees a Digit")

col_3b1b_img, col_3b1b_text = st.columns([1.3, 1])

with col_3b1b_img:
    pixel_grid_path = os.path.join(ASSETS_DIR, '1. 24x24 Image Viz.png')
    if os.path.exists(pixel_grid_path):
        st.image(pixel_grid_path, caption="Each pixel is a number between 0.0 (black) and 1.0 (white) — an individual 'activation' that feeds into the network", width=600)

with col_3b1b_text:
    st.markdown(f"""
    To a neural network, an image isn't a picture — it's a **list of numbers**.
    
    Each handwritten digit in MNIST is a **28 × 28 pixel** grayscale image. 
    Every pixel has a brightness value between 0.0 (black) and 1.0 (white). 
    
    When we feed this image to our network, we 
    {tip("flatten", "Reshaping a 2D grid (28×28) into a 1D vector (784 elements) by reading pixels left-to-right, top-to-bottom. The spatial arrangement is lost, but the pixel values are preserved.")} 
    the 28×28 grid into a single vector of **784 numbers** — and each number becomes one input neuron.
    """, unsafe_allow_html=True)
    
    st.latex(r"28 \times 28 = 784 \text{ input neurons}")
    st.caption("↳ Each pixel = one input feature = one number the network can read")

# Layer deconstruction image
st.markdown("---")
st.markdown("#### How the Network Decomposes a Digit")

col_decomp_img, col_decomp_text = st.columns([1.3, 1])

with col_decomp_img:
    decomp_path = os.path.join(ASSETS_DIR, '3. Layer Deconstruction.png')
    if os.path.exists(decomp_path):
        st.image(decomp_path, caption="A digit '9' decomposed: the original image (left), sub-features detected by hidden layers (center images), and the full network (right)", width=600)

with col_decomp_text:
    st.markdown(f"""
    The network doesn't see the digit as a whole — it **decomposes** it layer by layer:
    
    1. **First hidden layer** neurons detect simple patterns: edges, strokes, curves
    2. **Second hidden layer** neurons combine those into higher-level features: loops, intersections, specific digit components
    3. **Output layer** combines all features to vote on which digit (0-9) it is
    
    Think of it like a team of specialists: one neuron might detect "upper loop" 
    (important for 9, 8, 6), another detects "vertical stroke" 
    (important for 1, 7, 4), and the output layer weighs all their opinions.
    """)

# Sample digit grid
st.markdown("---")
if 'sample_images' in training_data:
    col_grid, col_detail = st.columns([1.5, 1])

    with col_grid:
        st.markdown("#### Sample Digits (5 per class)")
        fig_grid = plot_digit_grid(
            training_data['sample_images'],
            training_data['sample_labels']
        )
        st.plotly_chart(fig_grid, key="digit_grid", config=PLOTLY_CONFIG)

    with col_detail:
        st.markdown("#### Inspect a Digit")
        digit_choice = st.selectbox("Select digit class", list(range(10)), key="digit_explore")

        mask = training_data['sample_labels'] == digit_choice
        available = training_data['sample_images'][mask]

        if len(available) > 0:
            idx_in_class = st.slider("Sample index", 0, min(len(available)-1, 4), 0, key="sample_in_class")
            img = available[idx_in_class]

            fig_px = plot_pixel_heatmap(img, title=f"Digit {digit_choice} — Pixel Intensities")
            st.plotly_chart(fig_px, key="digit_heatmap", config=PLOTLY_CONFIG)

            st.caption(f"This 28×28 grid gets **flattened** into a vector of {28*28} numbers, which becomes the input to our network.")
else:
    st.info("Sample digits not found. Run `python scripts/train_mnist.py` to generate them.")

section_divider()

# ════════════════════════════════════════════════════════
# SECTION C: "How The Model Was Trained"
# ════════════════════════════════════════════════════════
section_header("How The Model Was Trained", "Training methodology, optimizer, and loss function", "")

st.markdown(f"""
Unlike our Phase 1 toy network (which we trained live in the browser with raw NumPy), 
the MNIST model was pre-trained using {tip("PyTorch", "A deep learning framework that automates gradient computation via 'autograd' and enables GPU-accelerated matrix operations. Conceptually identical to our NumPy implementation, but orders of magnitude faster.")} — 
the industry-standard framework for deep learning research.
""", unsafe_allow_html=True)

col_train_method, col_train_details = st.columns([1, 1])

with col_train_method:
    st.markdown("#### Training Configuration")
    st.markdown("""
    | Setting | Value | Why |
    |---------|-------|-----|
    | **Optimizer** | Adam | Adaptive learning rate — faster convergence than plain SGD |
    | **Learning Rate** | 0.001 | Standard default for Adam optimizer |
    | **Loss Function** | Cross-Entropy | The multi-class generalization of our binary BCE from Phase 1 |
    | **Batch Size** | 128 | Process 128 images before each weight update |
    | **Epochs** | 15 | Full passes through all 60,000 training images |
    | **Data Split** | 60K train / 10K test | Standard MNIST benchmark split |
    """)

with col_train_details:
    st.markdown("#### The Training Loop (same as Phase 1!)")
    st.code("""
# PyTorch version — same logic as our NumPy pseudocode
for epoch in range(15):
    for batch_X, batch_y in train_loader:
        # Forward pass
        predictions = model(batch_X)
        loss = cross_entropy(predictions, batch_y)
        
        # Backward pass (PyTorch autograd)
        loss.backward()   # Computes ALL gradients automatically
        
        # Update weights
        optimizer.step()  # Adam optimizer updates all params
        optimizer.zero_grad()
    """, language="python")
    
    st.caption("↳ PyTorch's `loss.backward()` does the same chain-rule gradient computation we did by hand in Phase 1 — just automatically for any network architecture.")

# Parameter breakdown
st.markdown("---")
st.markdown("#### Parameter Breakdown by Layer")

param_info = get_param_breakdown_rows(model)
h1 = param_info['h1']
h2 = param_info['h2']

st.markdown(f"""
Our model has **{model.count_parameters():,} total learnable parameters**. Here's exactly where they live:

| Layer | Input → Output | Weight Shape | Bias Shape | Param Count | Role |
|-------|---------------|-------------|------------|-------------|------|
| **Layer 1** (Input → Hidden 1) | 784 → {h1} | 784 × {h1} | {h1} | **{param_info['layer_1_params']:,}** | First-pass stroke template detectors (edges, curves, local motifs) |
| **Layer 2** (Hidden 1 → Hidden 2) | {h1} → {h2} | {h1} × {h2} | {h2} | **{param_info['layer_2_params']:,}** | Composes simple stroke cues into higher-level digit parts |
| **Layer 3** (Hidden 2 → Output) | {h2} → 10 | {h2} × 10 | 10 | **{param_info['layer_3_params']:,}** | Converts abstract evidence into 10 digit logits |
| **Total** | | | | **{model.count_parameters():,}** | |

> **Notice the distribution**: ~{param_info['first_layer_share']:.1f}% of all parameters are in Layer 1 (the "feature detector" layer). 
> This is because the input space is enormous (784 pixels) and each neuron needs a weight for every pixel.
> The deeper layers are much smaller because they operate on compressed, abstract representations.
""", unsafe_allow_html=True)

# Training curves
if 'history' in training_data:
    st.markdown("---")
    st.markdown("#### Training Curves")
    st.caption("Curves compare the baseline MLP and the compact 16×16 model. The compact model is easier to interpret but may trade off some peak accuracy.")
    fig_curves = plot_training_curves(training_data['history'])
    st.plotly_chart(fig_curves, key="training_curves", config=PLOTLY_CONFIG)

section_divider()

# ════════════════════════════════════════════════════════
# SECTION D: "Architecture Comparison" — Standalone
# ════════════════════════════════════════════════════════
section_header("Architecture Comparison", "Does a deeper network perform better?", "")

st.markdown(f"""
We trained **two architectures** on the same data: a wider single-hidden-layer baseline and a compact 16×16 two-layer model. 
This is a common practice in ML — {tip("ablation study", "A systematic experiment where you vary one aspect of a model (e.g., depth, width) while keeping everything else constant, to understand the effect of that specific change.")} — 
to understand trade-offs among accuracy, parameter count, and interpretability.
""", unsafe_allow_html=True)

if 'comparison' in training_data:
    comp = training_data['comparison']
    comp_keys = list(comp.keys())

    if len(comp_keys) < 2:
        st.warning("Comparison data is incomplete. Re-run `python scripts/train_mnist.py` to regenerate metrics.")
    else:
        key_a, key_b = comp_keys[0], comp_keys[1]
        model_a = comp[key_a]
        model_b = comp[key_b]

        col_a, col_vs, col_b = st.columns([1, 0.3, 1])

        with col_a:
            st.markdown(f"#### Model A: {key_a.title()}")
            st.markdown(f"**`{model_a['architecture']}`**")
            metric_row({
                "Test Accuracy": f"{model_a['final_test_acc']*100:.2f}%",
                "Parameters": f"{model_a['parameters']:,}",
                "Avg Epoch": f"{model_a['avg_epoch_time']:.2f}s",
            })

        with col_vs:
            st.markdown("")
            st.markdown("")
            st.markdown("###")
            st.markdown("### vs")

        with col_b:
            st.markdown(f"#### Model B: {key_b.title()} ✓")
            st.markdown(f"**`{model_b['architecture']}`**")
            metric_row({
                "Test Accuracy": f"{model_b['final_test_acc']*100:.2f}%",
                "Parameters": f"{model_b['parameters']:,}",
                "Avg Epoch": f"{model_b['avg_epoch_time']:.2f}s",
            })

        # Architecture comparison chart
        st.markdown("---")
        fig_comp = plot_architecture_comparison(comp)
        st.plotly_chart(fig_comp, key="arch_comparison", config=PLOTLY_CONFIG)

        acc_diff = (model_b['final_test_acc'] - model_a['final_test_acc']) * 100
        param_ratio = model_b['parameters'] / max(model_a['parameters'], 1)

        st.markdown(f"""
        **Key Takeaway:** Model B differs by **{acc_diff:+.2f}%** test accuracy with **{param_ratio:.2f}×** the parameter count.
        For this walkthrough, we prioritize **clear neuron-level interpretability** (compact hidden layers) while still keeping strong MNIST performance.
        """)
else:
    st.info("Architecture comparison data not found. Run `python scripts/train_mnist.py`.")

section_divider()

# ════════════════════════════════════════════════════════
# SECTION E: "Inside the Trained Network" — Enhanced
# ════════════════════════════════════════════════════════
section_header("Inside the Trained Network", "What do the neurons look for?", "")

# 3B1B Hidden Features Image
col_3b1b_hidden, col_3b1b_hidden_text = st.columns([1.3, 1])

with col_3b1b_hidden:
    hidden_features_path = os.path.join(ASSETS_DIR, '2. MLP Architecture - Hidden features.png')
    if os.path.exists(hidden_features_path):
        st.image(hidden_features_path, caption="3Blue1Brown-style visualization: each hidden neuron learns to detect a specific visual pattern — like 'upper loop' detectors", width=600)

with col_3b1b_hidden_text:
    st.markdown(f"""
    Each neuron in the first hidden layer has **784 weights** — one for each pixel. 
    If we reshape these weights back into a 28×28 grid, we can visualize what pixel pattern 
    each neuron is looking for.
    
    These are called {tip("feature maps", "A visualization of a neuron's weights reshaped to match the input dimensions. Bright areas indicate pixels that strongly activate the neuron; dark areas indicate pixels that suppress it.")} 
    or **receptive fields**.
    
    **Think of each neuron as a stencil:** when the input image's bright pixels 
    overlap with the neuron's positive weights, the neuron fires strongly. 
    When they overlap with negative weights, the neuron is suppressed.
    
    The output layer then combines all second-layer activations to 
    decide which digit it's looking at — essentially voting based on 
    which feature detectors fired.
    """, unsafe_allow_html=True)

st.markdown("---")

col_fmaps, col_inspect = st.columns([1.5, 1])

with col_fmaps:
    st.markdown(f"#### Feature Detectors (Top {min(16, model.hidden1)} neurons by weight magnitude)")
    feature_maps = model.get_feature_maps(layer_idx=0)  # (hidden1, 28, 28)
    fig_features = plot_feature_detector_grid(feature_maps, n_neurons=16)
    st.plotly_chart(fig_features, key="feature_maps", config=PLOTLY_CONFIG)

with col_inspect:
    st.markdown("#### Inspect Hidden Layers")
    inspect_layer = st.radio(
        "Layer to inspect",
        ["Hidden Layer 1 (pixel-space detectors)", "Hidden Layer 2 (compositions of Layer-1 detectors)"],
        key="mnist_inspect_layer",
    )

    if inspect_layer.startswith("Hidden Layer 1"):
        max_neuron_idx = max(0, len(feature_maps) - 1)
        neuron_idx = st.number_input(
            f"Neuron index (0-{max_neuron_idx})",
            min_value=0,
            max_value=max_neuron_idx,
            value=0,
            key="neuron_inspect_l1"
        )

        fig_single = plot_pixel_heatmap(
            feature_maps[neuron_idx],
            title=f"Layer 1 Neuron {neuron_idx} Receptive Field"
        )
        st.plotly_chart(fig_single, key="neuron_single_l1", config=PLOTLY_CONFIG)

        st.markdown(f"""
        **Layer 1 interpretation:** Red/positive regions are pixels that excite this neuron.  
        Blue/negative regions inhibit it.  
        These are the closest thing to direct "stroke templates" in this MLP.
        """)
    else:
        layer2_weights = model.get_feature_maps(layer_idx=1)  # (hidden2, hidden1)
        max_l2_idx = max(0, layer2_weights.shape[0] - 1)
        l2_idx = st.number_input(
            f"Layer-2 neuron index (0-{max_l2_idx})",
            min_value=0,
            max_value=max_l2_idx,
            value=0,
            key="neuron_inspect_l2"
        )
        comp_weights = layer2_weights[l2_idx]
        fig_l2 = plot_layer_activations(
            comp_weights,
            layer_name=f"Layer 2 Neuron {l2_idx} Input Weights",
            top_k=None
        )
        st.plotly_chart(fig_l2, key="neuron_single_l2", config=PLOTLY_CONFIG)

        top_contributors = np.argsort(np.abs(comp_weights))[-5:][::-1]
        contributor_text = ", ".join([f"H1[{idx}] ({comp_weights[idx]:+.3f})" for idx in top_contributors])
        st.markdown(f"""
        **Layer 2 interpretation:** This neuron does not look at pixels directly.
        It mixes outputs from Layer 1 detectors.

        Top contributors: {contributor_text}
        """)

section_divider()

# ════════════════════════════════════════════════════════
# SECTION F: "Watch a Digit Flow Through"
# ════════════════════════════════════════════════════════
section_header("Watch a Digit Flow Through", "Layer-by-layer activation inspection", "")

st.markdown(f"""
Select a digit below to watch its pixel values propagate through the network:  
**Input (784) → Hidden Layer 1 ({model.hidden1}) → Hidden Layer 2 ({model.hidden2}) → Output (10)**

At each layer, only a subset of neurons will "fire" (have high activation). 
The pattern of which neurons fire is the network's internal {tip("representation", "The network's internal encoding of the input at a given layer. Earlier layers detect low-level features (edges, strokes); later layers combine these into higher-level concepts (loops, curves).")} of the digit.
""", unsafe_allow_html=True)

if 'sample_images' in training_data:
    col_select, col_flow = st.columns([0.8, 1.5])

    with col_select:
        flow_digit = st.selectbox("Digit class", list(range(10)), key="flow_digit")
        flow_mask = training_data['sample_labels'] == flow_digit
        flow_available = training_data['sample_images'][flow_mask]

        if len(flow_available) > 0:
            flow_sample_idx = st.slider("Sample", 0, min(len(flow_available)-1, 4), 0, key="flow_sample")
            flow_img = flow_available[flow_sample_idx]

            fig_input = plot_pixel_heatmap(flow_img, title=f"Input: Digit {flow_digit}")
            st.plotly_chart(fig_input, key="flow_input", config=PLOTLY_CONFIG)

    with col_flow:
        if len(flow_available) > 0:
            # Get activations
            x_tensor = torch.FloatTensor(flow_img).unsqueeze(0)
            activations = model.get_layer_activations(x_tensor)

            # Output probabilities
            predicted = int(np.argmax(activations['output_probs']))
            fig_probs = plot_output_probabilities(activations['output_probs'], predicted)
            st.plotly_chart(fig_probs, key="flow_probs", config=PLOTLY_CONFIG)

            # Layer activations flow
            fig_flow = plot_activation_flow(
                {
                    f'Hidden 1 ({model.hidden1})': activations['hidden1'],
                    f'Hidden 2 ({model.hidden2})': activations['hidden2'],
                    'Output (10)': activations['output_probs'],
                },
                layer_names=[f'Hidden 1 ({model.hidden1})', f'Hidden 2 ({model.hidden2})', 'Output (10)']
            )
            st.plotly_chart(fig_flow, key="flow_activations", config=PLOTLY_CONFIG)

            st.caption(f"Predicted: **{predicted}** | Actual: **{flow_digit}** | "
                       f"{'Correct' if predicted == flow_digit else 'Wrong'}")

section_divider()

# ════════════════════════════════════════════════════════
# SECTION G: "Draw Your Own Digit" — With Confidence Thresholding
# ════════════════════════════════════════════════════════
section_header("Draw Your Own Digit", "Test the network with your handwriting", "")

col_canvas, col_result = st.columns([1, 1.2])

with col_canvas:
    with st.expander("Uncertainty & OOD controls", expanded=False):
        confidence_threshold = st.slider(
            "Minimum confidence threshold",
            min_value=0.50,
            max_value=0.99,
            value=0.80,
            step=0.01,
            key="mnist_confidence_threshold",
            help="Below this, we flag the prediction as uncertain."
        )
        uncertainty_threshold = st.slider(
            "Maximum entropy uncertainty",
            min_value=0.20,
            max_value=0.95,
            value=0.55,
            step=0.01,
            key="mnist_uncertainty_threshold",
            help="Entropy normalized to [0,1]. Higher means flatter probability distribution."
        )
        temperature = st.slider(
            "Softmax temperature",
            min_value=0.5,
            max_value=3.0,
            value=1.4,
            step=0.1,
            key="mnist_temperature",
            help="Higher temperature flattens overconfident logits for display-time calibration."
        )

    try:
        from streamlit_drawable_canvas import st_canvas

        canvas_result = st_canvas(
            fill_color="black",
            background_color="black",
            stroke_width=20,
            stroke_color="white",
            update_streamlit=True,
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas_draw",
        )
        st.caption("Draw a digit (0-9) in white on the black canvas.")

        has_drawing = (canvas_result.image_data is not None and
                       np.max(canvas_result.image_data[:, :, :3]) > 0)
    except ImportError:
        st.warning("Canvas not available. Install `streamlit-drawable-canvas` or use the file uploader.")
        has_drawing = False
        canvas_result = None

    # Fallback: file upload
    uploaded = st.file_uploader("Or upload a digit image", type=['png', 'jpg', 'jpeg'], key="digit_upload")

with col_result:
    input_image = None

    if has_drawing and canvas_result is not None:
        # Process canvas
        img_array = canvas_result.image_data[:, :, :3]  # Drop alpha
        grayscale = np.mean(img_array, axis=2) / 255.0
        input_image = grayscale
    elif uploaded is not None:
        from PIL import Image
        pil_img = Image.open(uploaded).convert('L').resize((28, 28))
        input_image_raw = np.array(pil_img) / 255.0
        input_image = input_image_raw

    if input_image is not None and np.max(input_image) > 0:
        # Resize to 28×28 before MNIST-like preprocessing
        from PIL import Image
        if input_image.shape[0] != 28:
            pil = Image.fromarray((input_image * 255).astype(np.uint8), mode='L')
            pil = pil.resize((28, 28), Image.Resampling.LANCZOS)
            img_28 = np.array(pil) / 255.0
        else:
            img_28 = input_image

        # OOD-robust preprocessing: crop, center, and normalize digit layout
        img_processed, ink_ratio = preprocess_canvas_digit(img_28)

        # Show what the model sees
        col_sees, col_pred = st.columns(2)

        with col_sees:
            st.markdown("#### What the model sees")
            fig_sees = plot_pixel_heatmap(img_processed, title="Preprocessed 28×28")
            st.plotly_chart(fig_sees, key="canvas_preprocessed", config=PLOTLY_CONFIG)

        with col_pred:
            # Run prediction
            # Normalize like MNIST training
            img_normalized = (img_processed - 0.1307) / 0.3081
            x_tensor = torch.FloatTensor(img_normalized).unsqueeze(0).unsqueeze(0)

            activations = model.get_layer_activations(x_tensor)
            logits = activations['output_logits'] / max(temperature, 1e-6)
            exp_logits = np.exp(logits - np.max(logits))
            calibrated_probs = exp_logits / np.sum(exp_logits)
            predicted = int(np.argmax(calibrated_probs))
            confidence = float(calibrated_probs[predicted])

            # Entropy-based uncertainty
            entropy = -np.sum(calibrated_probs * np.log(calibrated_probs + 1e-10))
            max_entropy = -np.log(1.0 / 10)  # Uniform distribution entropy
            uncertainty = entropy / max_entropy  # 0 = certain, 1 = completely uncertain

            st.markdown(f"#### Prediction: **{predicted}**")
            st.markdown(f"Confidence: **{confidence:.1%}**")
            st.caption(f"Ink coverage after preprocessing: {ink_ratio:.1%}")

            # Confidence + entropy + low-ink OOD warnings
            if (confidence < confidence_threshold) or (uncertainty > uncertainty_threshold) or (ink_ratio < 0.01):
                st.warning(f"""
                **The model is uncertain / potentially out-of-distribution.**

                - Confidence: {confidence:.1%} (threshold {confidence_threshold:.0%})
                - Entropy uncertainty: {uncertainty:.1%} (threshold {uncertainty_threshold:.0%})
                - Ink coverage: {ink_ratio:.1%}

                Try writing a single centered digit with continuous strokes.
                
                *Note: Neural networks always produce a prediction (softmax forces probabilities to sum to 1), 
                even for random scribbles. We therefore combine confidence + entropy + input quality checks.*
                """)

            st.caption(f"Entropy: {entropy:.2f} / {max_entropy:.2f} = {uncertainty:.1%} uncertainty")

            fig_probs = plot_output_probabilities(calibrated_probs, predicted)
            st.plotly_chart(fig_probs, key="canvas_probs", config=PLOTLY_CONFIG)
    else:
        st.info("Draw a digit on the canvas (or upload an image) to see the network classify it in real-time.")

section_divider()

# ════════════════════════════════════════════════════════
# SECTION H: "Where the Model Struggles"
# ════════════════════════════════════════════════════════
section_header("Where the Model Struggles", "Error analysis and failure modes", "")

if 'predictions' in training_data:
    preds = training_data['predictions']
    labels = training_data['labels']
    probs = training_data['probabilities']

    total = len(labels)
    correct = np.sum(preds == labels)
    acc = correct / total

    metric_row({
        "Total Test Samples": f"{total:,}",
        "Correct": f"{correct:,}",
        "Wrong": f"{total - correct:,}",
        "Accuracy": f"{acc*100:.2f}%",
    })

    st.markdown("")

    col_cm, col_conf = st.columns([1, 1])

    with col_cm:
        fig_cm = plot_confusion_matrix(labels, preds)
        st.plotly_chart(fig_cm, key="confusion_matrix", config=PLOTLY_CONFIG)
        st.caption("Darker cells on the diagonal = good. Off-diagonal = confusion between digits.")

    with col_conf:
        fig_conf = plot_confidence_distribution(probs, labels, preds)
        st.plotly_chart(fig_conf, key="confidence_dist", config=PLOTLY_CONFIG)
        st.caption("Correct predictions tend to be high-confidence. Wrong predictions are often low-confidence — the model 'knows' when it's unsure.")

    # Misclassified examples
    if 'misc_images' in training_data and len(training_data['misc_images']) > 0:
        st.markdown("#### Misclassified Examples")
        st.caption("Here are some digits the model got wrong. Can you see why?")

        misc_imgs = training_data['misc_images']
        misc_true = training_data['misc_true']
        misc_pred = training_data['misc_pred']
        misc_probs = training_data['misc_probs']

        n_show = min(10, len(misc_imgs))
        cols = st.columns(min(5, n_show))

        for i in range(n_show):
            with cols[i % 5]:
                img = misc_imgs[i]
                if img.ndim == 3:
                    img = img.squeeze(0)

                fig_m = plot_pixel_heatmap(img, title=f"True:{misc_true[i]} Pred:{misc_pred[i]}")
                fig_m.update_layout(height=200, width=200, margin=dict(l=5, r=5, t=30, b=5))
                st.plotly_chart(fig_m, key=f"misclassified_{i}", config=PLOTLY_CONFIG)
else:
    st.info("Test predictions not found. Run `python scripts/train_mnist.py` to generate them.")

section_divider()

# ════════════════════════════════════════════════════════
# Bridge to Advanced Topics
# ════════════════════════════════════════════════════════
st.info("""
**Where This Leads**  
This MLP is the foundation. The same principles scale to:
- **Convolutional Neural Networks (CNNs)** — feature detectors that slide across the image (preserving spatial structure)
- **Transformers & Language Models** — attention mechanisms as feature detectors for text (GPT, BERT)
- **Advanced Interpretability** — mechanistic analysis of what larger models encode internally

The mechanics you've seen today — weights, activations, feature detection, gradient descent — 
are the **exact same building blocks** used in systems like GPT-4, DALL-E, and image recognition at scale.
""")
