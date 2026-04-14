"""
Phase 1: Poisonous Fruit Detector (2-3-1 MLP)
A guided educational tour through the mechanics of a simple neural network.

Sections:
  A. "The Dataset & The Problem" — merged narrative intro + data + why linear fails
  B. "So What Are Neural Networks?" — perceptron concept, MLP stacking, forward/backward,
     pseudocode, features vs hidden features, parameter breakdown
  C. "The Playground" — checkpoint-based slider playback (no live training)
  D. "Training Progress" — loss/accuracy curves with annotations
  E. "What Did the Neurons Learn?" — interpretability & latent features
"""
import streamlit as st
import numpy as np
import os

from src.data import get_poisonous_fruit_data, get_linear_baseline_accuracy
from src.mlp import MLP, ACTIVATIONS, ACTIVATION_DESCRIPTIONS
from src.precompute_toy import load_checkpoints, restore_mlp_from_checkpoint
from src.visualizations import (
    plot_decision_boundary_from_grid, plot_network_graph,
    plot_activation_gallery, plot_activation_curve,
    plot_loss_curve, plot_neuron_boundaries,
    plot_linear_failure, plot_sample_flow,
)
from src.ui_components import (
    inject_tooltip_css, tip, glossary_popover,
    metric_row, section_header, section_divider, math_block,
)

st.set_page_config(page_title="Neural Networks: Toy MLP", layout="wide")

# ────────────────────────────────────────────────────────
# Inject custom CSS
# ────────────────────────────────────────────────────────
inject_tooltip_css()

ASSETS_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

# ────────────────────────────────────────────────────────
# Load pre-computed checkpoint data
# ────────────────────────────────────────────────────────
CHECKPOINT_FILES = {
    'Sigmoid': 'toy_checkpoints_sigmoid.npz',
    'ReLU': 'toy_checkpoints_relu.npz',
    'Tanh': 'toy_checkpoints_tanh.npz',
    'None (Linear)': 'toy_checkpoints_none_linear.npz',
    'Step Function': 'toy_checkpoints_step_function.npz',
}


@st.cache_data
def load_checkpoint_data(activation_name):
    """Load pre-computed checkpoints from disk."""
    filename = CHECKPOINT_FILES.get(activation_name)
    if not filename:
        return None
    filepath = os.path.join(MODELS_DIR, filename)
    if os.path.exists(filepath):
        return load_checkpoints(filepath)
    return None


# ────────────────────────────────────────────────────────
# Session State Initialization
# ────────────────────────────────────────────────────────
if 'X' not in st.session_state:
    X, y = get_poisonous_fruit_data()
    st.session_state.X = X
    st.session_state.y = y
if 'is_playing' not in st.session_state:
    st.session_state.is_playing = False
if 'training_done' not in st.session_state:
    st.session_state.training_done = False

X = st.session_state.X
y = st.session_state.y

# ════════════════════════════════════════════════════════
# PAGE HEADER
# ════════════════════════════════════════════════════════
st.title("Mechanistic Interpretability: Neural Networks")
st.markdown("### Phase 1: Poisonous Fruit Detector — a 2-3-1 MLP from Scratch")
st.caption("Built entirely with NumPy • No frameworks, no black boxes • Every weight and gradient visible")

section_divider()

# ════════════════════════════════════════════════════════
# SECTION A: "The Dataset & The Problem" (MERGED)
# ════════════════════════════════════════════════════════
section_header("The Dataset & The Problem", "Exploring the data before we build a solution", "")

st.markdown(f"""
Before we start dissecting the components and functionalities of each component in a neural network, 
let's first talk a little bit about this hypothetical 2D dataset. 

This is a dataset about **alien fruits** discovered on a distant planet. Each fruit has two measurable traits: 
{tip("Spikiness", "How many spikes protrude from the fruit's surface — normalized to [0, 1] range.")} (x₁) and 
{tip("Spottiness", "How many spots cover the fruit's skin — normalized to [0, 1] range.")} (x₂). 
Some fruits are **Safe** to eat, while others are **Poisonous**.

The problem we face here is a classic {tip("binary classification", "A machine learning task where each input must be assigned to one of exactly two classes. Here: Safe (0) or Poisonous (1).")} task — 
detecting if a given fruit is poisonous or not based on its two features. 
As Matt and Amandeep already showed you in the previous section about Regression and Classification, 
we can use simple algorithms like "logistic regression", "KNN", etc. to approximate the necessary 
{tip("decision function", "A function f(x) that maps input features to a class label. The 'boundary' where f(x) changes its prediction is the decision boundary.")} to solve this problem.

While these classic algorithms are lightweight and easy to implement, they're not **"universal"**. 
This is where neural networks come into play. They are extremely versatile and can be thought of as 
{tip("universal function approximators", "A mathematical result (Universal Approximation Theorem) stating that a neural network with a single hidden layer and non-linear activation can approximate any continuous function to arbitrary precision, given enough neurons.")} 
— and they scale very well with the complexity of the problem and the scale of data. 

For this small dataset, we could obviously adapt linear models by adding polynomial terms, etc. — but now 
let's skip the logistic model and look at how we can solve this problem from a **neural network perspective**.
""", unsafe_allow_html=True)

# ── Two-column: Fruit sketch + Scatter plot with linear failure ──
col_img, col_plot = st.columns([1, 1.2])

with col_img:
    fruit_img_path = os.path.join(ASSETS_DIR, 'fruitdata.png')
    if os.path.exists(fruit_img_path):
        st.image(fruit_img_path, caption="Artist's rendering of the alien fruits — note the varying spikes and spots!", width=450)
    else:
        st.info("Fruit sketch not found — place your image at `assets/fruit_sketch.png`")

    st.markdown(f"""
    This is a {tip("synthetic dataset", "Data generated by a mathematical formula rather than collected from the real world. Synthetic data lets us control the exact complexity.")}, 
    created using `sklearn.datasets.make_moons(n=200, noise=0.15)` — 
    two interlocking crescent-moon shapes that are **impossible to separate with a straight line**.
    
    | Feature | Range | Interpretation |
    |---------|-------|----------------|
    | x₁ (Spikiness) | 0–1 | How spiky the fruit is (normalized) |
    | x₂ (Spottiness) | 0–1 | How spotty the fruit is (normalized) |
    | Label y | 0 or 1 | 0 = Safe, 1 = Poisonous |
    """, unsafe_allow_html=True)

with col_plot:
    lin_acc, lin_model = get_linear_baseline_accuracy(X, y)
    fig_linear = plot_linear_failure(X, y, lin_model)
    st.plotly_chart(fig_linear, key="linear_fail", config={'displayModeBar': False})
    st.caption(f"The best a linear model can do: **{lin_acc*100:.1f}%** accuracy. Yellow ✗ marks = misclassified points.")

    with st.popover("Why does the linear model fail?"):
        st.markdown(f"""
        **Logistic Regression Accuracy: {lin_acc*100:.1f}%**
        
        A logistic regression model computes:
        """)
        st.latex(r"\hat{y} = \sigma(W_1 x_1 + W_2 x_2 + b)")
        st.markdown("""
        The decision boundary is where the probability = 0.5, which is always a **straight line**:
        """)
        st.latex(r"W_1 x_1 + W_2 x_2 + b = 0")
        st.markdown("No matter how you tilt or shift this line, it can't capture the moon-shaped clusters.")

section_divider()

# ════════════════════════════════════════════════════════
# SECTION B: "So What Are Neural Networks?"
# ════════════════════════════════════════════════════════
section_header("So What Are Neural Networks?", "Building intuition from neurons to networks", "")

# ── B.1: The Perceptron ──
col_perc_text, col_perc_img = st.columns([1.3, 1])

with col_perc_text:
    st.markdown(f"""
    #### The Core Building Block: The Perceptron
    
    At its heart, a neural network is built from a simple computational unit called a 
    {tip("perceptron", "The simplest form of a neural network — a single neuron that takes weighted inputs, sums them with a bias, and passes the result through an activation function.")}. 
    A perceptron takes some inputs (x₁, x₂, ... xₙ), multiplies each by a learnable **weight**, 
    adds a **bias** term, and passes the result through an 
    {tip("activation function", "A non-linear function applied to the weighted sum. Without it, stacking layers has no benefit — the entire network would collapse to a single linear transformation.")}:
    """, unsafe_allow_html=True)
    
    st.latex(r"\hat{y} = f\left(\sum_{i=1}^{n} w_i x_i + b\right)")
    st.caption("↳ Same as logistic regression! The perceptron IS a generalized linear model with a flexible activation.")
    
    st.markdown(f"""
    **The key insight**: a single perceptron can only learn a linear boundary — just like logistic regression. 
    But when we **stack** multiple perceptrons into layers, the network can learn 
    {tip("non-linear boundaries", "Decision boundaries that curve, bend, and wrap around data clusters — impossible for any single linear model to achieve.")} 
    that are far more powerful.
    
    This is called a **Multi-Layer Perceptron (MLP)** — and it's the foundation of modern deep learning.
    """, unsafe_allow_html=True)

with col_perc_img:
    perc_img_path = os.path.join(ASSETS_DIR, 'perceptron.webp')
    if os.path.exists(perc_img_path):
        st.image(perc_img_path, caption="A single perceptron: weighted inputs → sum → activation → output", width=400)
    else:
        st.info("Place your perceptron diagram at `assets/perceptron.webp`")

# ── B.2: Types of NNs and scope ──
st.markdown("---")
st.markdown("""
#### Types of Neural Networks

Neural networks come in many architectures, each designed for specific types of data and tasks:

| Architecture | Best For | Example Applications |
|---|---|---|
| **MLP** (Multi-Layer Perceptron) | Tabular data, simple classification/regression | Fraud detection, price prediction |
| **CNN** (Convolutional Neural Network) | Images, spatial data | Image recognition, medical imaging |
| **RNN / LSTM** | Sequential data, time series | Speech recognition, stock prediction |
| **Transformer** | Language, attention-based tasks | GPT, BERT, machine translation |

> **For this event**, we'll be exploring **MLPs for classification tasks** — first this tiny 2-3-1 network (Phase 1), 
> then a larger 784→128→64→10 MNIST digit classifier (Phase 2).
""")

# ── B.3: Our Toy Network Architecture ──
st.markdown("---")
st.markdown("#### Our Toy Neural Network: 2-3-1 Architecture")

col_arch_text, col_arch_viz = st.columns([1, 1.2])

with col_arch_text:
    st.markdown(f"""
    Our solution to the fruit classification problem: a tiny neural network with **13 learnable parameters**.
    
    The network has three layers:
    - **Input Layer** (2 nodes): receives Spikiness and Spottiness
    - **Hidden Layer** (3 nodes): the "brain" — learns abstract combinations of input features  
    - **Output Layer** (1 node): outputs P(Poisonous)
    
    Each hidden node acts as a {tip("feature detector", "A hidden neuron that learns to respond to a specific pattern or combination of input features. Think of it as a tiny specialist — one might detect 'high spikes + low spots' while another detects the opposite.")} — 
    it learns to fire when it sees a specific pattern.
    """, unsafe_allow_html=True)
    
    st.markdown("**Parameter Breakdown:**")
    st.markdown("""
    | Parameter | Shape | Count | Role |
    |-----------|-------|-------|------|
    | **W₁** (Input→Hidden weights) | 2 × 3 | 6 | How much each input affects each hidden neuron |
    | **b₁** (Hidden biases) | 1 × 3 | 3 | Baseline activation threshold for each hidden neuron |
    | **W₂** (Hidden→Output weights) | 3 × 1 | 3 | How much each hidden neuron contributes to the final answer |
    | **b₂** (Output bias) | 1 × 1 | 1 | Baseline probability offset |
    | **Total** | | **13** | Every single one is exposed as a slider below! |
    """)

with col_arch_viz:
    # Show the initial network graph (untrained)
    init_mlp = MLP(seed=42)
    x_sample = X[0:1]
    init_mlp.forward(X, 'Sigmoid')
    fig_init_net = plot_network_graph(init_mlp, x_sample, 'Sigmoid')
    st.plotly_chart(fig_init_net, key="init_network", config={'displayModeBar': False})
    st.caption("The 2-3-1 network with random initial weights (before training). Edge thickness = weight magnitude, color = sign.")

# ── B.4: Forward + Backward Pass ──
st.markdown("---")

col_forward, col_backward = st.columns([1, 1])

with col_forward:
    st.markdown("#### The Forward Pass")
    st.markdown("Data flows left → right through the network:")

    math_block(r"Z_1 = X \cdot W_1 + b_1", "Multiply inputs by weights, add bias → 'pre-activation'")
    math_block(r"A_1 = f(Z_1)", "Apply the activation function (the non-linear magic)")
    math_block(r"Z_2 = A_1 \cdot W_2 + b_2", "Hidden activations → output pre-activation")
    math_block(r"\hat{y} = \sigma(Z_2)", "Sigmoid squashes output to a probability [0, 1]")
    
    st.markdown(f"""
    The {tip("learning rate", "Controls how big each weight update step is. Too small = slow learning (takes forever to converge). Too large = overshooting (bounces past the optimum and may diverge).")} (α) 
    determines how aggressively we update. Once training is done (all iterations complete), 
    we use the **same forward pass** for {tip("inference", "Using the trained model to make predictions on new, unseen data. No backward pass needed — just forward propagation through the frozen weights.")}: 
    feed new data through the frozen weights to get predictions.
    """, unsafe_allow_html=True)

with col_backward:
    st.markdown("#### The Backward Pass (Backpropagation)")
    st.markdown("After computing the prediction, we measure the error and update weights:")

    math_block(
        r"\mathcal{L} = -\frac{1}{N}\sum\left[y\log(\hat{y}) + (1-y)\log(1-\hat{y})\right]",
        "Binary Cross-Entropy Loss — penalizes confident wrong predictions heavily"
    )
    
    st.markdown("Gradients flow backward through the **chain rule**:")
    st.latex(r"\frac{\partial \mathcal{L}}{\partial W} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial Z} \cdot \frac{\partial Z}{\partial W}")
    
    st.markdown("Then we nudge each weight in the direction that reduces loss:")
    st.latex(r"W \leftarrow W - \alpha \cdot \frac{\partial \mathcal{L}}{\partial W}")
    st.caption("↳ α is the learning rate — how big each gradient descent step is")
    
    st.markdown("""
    This process repeats for many **epochs** (passes through the full dataset). 
    After hundreds or thousands of epochs, the weights converge to values that produce 
    an accurate curved decision boundary.
    """)

# ── B.5: General NN Pseudocode ──
st.markdown("---")
st.markdown("#### General Neural Network Pseudocode")
st.caption("This is the complete training loop for any feedforward network — our 2-3-1 MLP follows exactly this template.")

st.code("""
def train_neural_network(X, y, epochs, learning_rate):
    # 1. Initialize random weights and biases
    W1, b1, W2, b2 = initialize_random_params()
    
    for epoch in range(epochs):
        # === FORWARD PASS ===
        Z1 = X @ W1 + b1              # Pre-activation (hidden layer)
        A1 = activation(Z1)            # Post-activation (hidden layer)
        Z2 = A1 @ W2 + b2             # Pre-activation (output layer)
        y_hat = sigmoid(Z2)            # Predicted probabilities
        
        # === COMPUTE LOSS ===
        loss = cross_entropy(y, y_hat)
        
        # === BACKWARD PASS (Backpropagation) ===
        dZ2 = y_hat - y               # Output error gradient
        dW2 = A1.T @ dZ2 / N          # Weight gradient (hidden → output)
        db2 = sum(dZ2) / N            # Bias gradient (output)
        dA1 = dZ2 @ W2.T              # Error flowing back into hidden layer
        dZ1 = dA1 * d_activation(Z1)  # Chain rule through activation
        dW1 = X.T @ dZ1 / N           # Weight gradient (input → hidden)
        db1 = sum(dZ1) / N            # Bias gradient (hidden)
        
        # === UPDATE PARAMETERS (Gradient Descent) ===
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
    
    return W1, b1, W2, b2   # Trained model!

# === INFERENCE (after training) ===
def predict(x_new, W1, b1, W2, b2):
    Z1 = x_new @ W1 + b1
    A1 = activation(Z1)
    Z2 = A1 @ W2 + b2
    return sigmoid(Z2)       # Probability of being poisonous
""", language="python")

# ── B.6: Features vs Hidden Features ──
st.markdown("---")
st.markdown("#### Features vs. Hidden Features")

st.markdown(f"""
Our input features (Spikiness, Spottiness) are **explicit** — we chose them, we measured them, we know what they mean. 
But each hidden neuron learns its own **implicit feature** — a combination of inputs that it specializes in detecting.

After training, each of our 3 hidden neurons becomes a specialized 
{tip("feature detector", "A trained neuron that fires strongly for specific input patterns and weakly for others. In our network, one neuron might detect 'high spikiness AND low spottiness' — an implicit feature that wasn't in the original data.")}:

- **Neuron H₀** might learn to fire when **spikiness is high and spottiness is low**
- **Neuron H₁** might fire for **both features being moderate**  
- **Neuron H₂** might fire for **spottiness dominating spikiness**

The final output neuron combines these three implicit features (via W₂) to make the classification decision. 
**This is the core insight of neural networks**: they automatically discover useful intermediate representations 
that the programmer never had to manually engineer.
""", unsafe_allow_html=True)

# ── B.7: Activation Function Gallery ──
st.markdown("---")
st.markdown("#### Activation Function Gallery")
st.markdown(f"""
The {tip("activation function", "The mathematical function applied to each neuron's output. Without it, stacking layers has no benefit — the entire network collapses to a single linear transformation.")} 
is what gives the network its power to learn curves. Here are all 5 options and their derivatives:
""", unsafe_allow_html=True)

fig_gallery = plot_activation_gallery()
st.plotly_chart(fig_gallery, key="act_gallery", config={'displayModeBar': False})

with st.expander("Activation function details", expanded=False):
    for name, desc in ACTIVATION_DESCRIPTIONS.items():
        st.markdown(f"**{name}:** {desc}")

section_divider()

# ════════════════════════════════════════════════════════
# SECTION C: "The Playground" — Checkpoint-Based Playback
# ════════════════════════════════════════════════════════
section_header("The Playground", "Train the network and watch it learn", "")

st.markdown("""
**Try it yourself!** First, try adjusting the 13 parameter sliders manually (below) to see 
if you can find a good fit. Then switch to "Auto-Play" mode to watch gradient descent 
do it automatically — smoothly scrubbing through 10,000 pre-computed training epochs.
""")

# ── Sidebar Controls ──
st.sidebar.header("Playground Controls")

activation_name = st.sidebar.selectbox(
    "Hidden Layer Activation",
    list(ACTIVATIONS.keys()),
    index=2,  # Default: Sigmoid
    help="Changes the non-linear function applied in the hidden layer. Each activation produces different decision boundary shapes."
)

# Load checkpoints for selected activation
checkpoint_data = load_checkpoint_data(activation_name)

if checkpoint_data is None:
    st.error(f"Pre-computed checkpoints not found for '{activation_name}'. Run `python scripts/precompute_toy_training.py` first.")
    st.stop()

n_checkpoints = len(checkpoint_data['epochs'])
stored_lr = float(checkpoint_data['lr'])

st.sidebar.caption(f"Pre-computed: {n_checkpoints} checkpoints, lr={stored_lr}, max epoch={int(checkpoint_data['epochs'][-1])}")

# ── Playback Mode Toggle ──
st.sidebar.markdown("---")
st.sidebar.markdown("#### Playback Mode")

mode = st.sidebar.radio(
    "Mode",
    ["Manual Tweaking", "Pre-computed Playback"],
    index=1,
    help="Manual: adjust weights by hand. Playback: scrub through pre-computed training epochs."
)

# Plotly config to reduce flickering
PLOTLY_CONFIG = {
    'displayModeBar': False,
    'staticPlot': False,
    'scrollZoom': False,
}

if mode == "Pre-computed Playback":
    # ── Checkpoint slider ──
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Training Playback")
    
    # Speed control
    playback_speed = st.sidebar.select_slider(
        "Playback Speed",
        options=[1, 2, 5, 10, 20, 50],
        value=5,
        help="How many checkpoints to advance per animation tick."
    )

    # Play/Pause/Reset
    col_play, col_reset = st.sidebar.columns(2)
    
    def toggle_play():
        st.session_state.is_playing = not st.session_state.is_playing
        if st.session_state.is_playing:
            st.session_state.training_done = False

    def do_reset():
        st.session_state.is_playing = False
        st.session_state.training_done = False
        st.session_state.ckpt_idx = 0
    
    play_label = "Pause" if st.session_state.is_playing else "Play"
    if st.session_state.training_done:
        play_label = "Done"
    
    col_play.button(play_label, use_container_width=True, on_click=toggle_play,
                    disabled=st.session_state.training_done)
    col_reset.button("Reset", use_container_width=True, on_click=do_reset)
    
    if 'ckpt_idx' not in st.session_state:
        st.session_state.ckpt_idx = 0
    
    # Slider for manual scrubbing
    ckpt_idx = st.sidebar.slider(
        "Training Step", 0, n_checkpoints - 1,
        st.session_state.ckpt_idx,
        key="ckpt_slider",
        help="Scrub through the pre-computed training trajectory."
    )
    st.session_state.ckpt_idx = ckpt_idx
    
    # Auto-advance via fragment
    @st.fragment(run_every="80ms")
    def auto_advance():
        if st.session_state.is_playing and not st.session_state.training_done:
            new_idx = st.session_state.ckpt_idx + playback_speed
            if new_idx >= n_checkpoints:
                new_idx = n_checkpoints - 1
                st.session_state.is_playing = False
                st.session_state.training_done = True
            st.session_state.ckpt_idx = new_idx
    
    auto_advance()
    
    # Sample selector
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Sample Inspector")
    sample_idx = st.sidebar.slider("Select sample index", 0, len(X)-1, 0,
                                    help="Choose a data point to trace through the network.")
    
    # ── Load checkpoint and render ──
    idx = st.session_state.ckpt_idx
    epoch = int(checkpoint_data['epochs'][idx])
    loss = float(checkpoint_data['losses'][idx])
    acc = float(checkpoint_data['accuracies'][idx])
    boundary_grid = checkpoint_data['boundaries'][idx]
    grid_x = checkpoint_data['grid_x']
    grid_y = checkpoint_data['grid_y']
    
    # Restore MLP for network graph
    mlp = restore_mlp_from_checkpoint(checkpoint_data, idx)
    
    # Status
    status_icon = "" if st.session_state.training_done else ("" if st.session_state.is_playing else "")
    metric_row({
        "Epoch": f"{epoch} / {int(checkpoint_data['epochs'][-1])}",
        "Loss": f"{loss:.4f}",
        "Accuracy": f"{acc*100:.1f}%",
        "Parameters": "13",
        "Status": f"{status_icon} {'Playing' if st.session_state.is_playing else ('Done' if st.session_state.training_done else 'Paused')}",
    })
    
    st.markdown("")
    
    # ── Visualization ──
    col_boundary, col_network = st.columns([1.2, 1])
    
    with col_boundary:
        show_neurons = st.checkbox("Show hidden neuron boundaries", value=False, key="show_neuron_lines_cb_play")
        fig_db = plot_decision_boundary_from_grid(
            boundary_grid, grid_x, grid_y, X, y,
            epoch, loss, acc,
            show_neuron_lines=show_neurons,
            mlp_model=mlp if show_neurons else None,
            activation_name=activation_name,
        )
        st.plotly_chart(fig_db, key="db_chart_play", config=PLOTLY_CONFIG)
    
    with col_network:
        fig_act = plot_activation_curve(activation_name)
        st.plotly_chart(fig_act, key="act_curve_play", config=PLOTLY_CONFIG)
        
        sample_X = X[sample_idx:sample_idx+1]
        fig_net = plot_network_graph(mlp, sample_X, activation_name)
        st.plotly_chart(fig_net, key="net_graph_play", config=PLOTLY_CONFIG)

    # ── Training Curves ──
    st.markdown("#### Training Progress")
    
    with st.expander("What are these curves showing?", expanded=False):
        st.markdown(f"""
        **Loss Curve (red, left axis):**  
        The {tip("Binary Cross-Entropy (BCE) Loss", "Measures how different the model's predicted probabilities are from the true labels. Lower = better.")} 
        quantifies how wrong the model's predictions are.
        
        **Accuracy Curve (green, right axis):**  
        The fraction of fruits classified correctly. 
        
        **Each point is one {tip("epoch", "One complete pass through the entire training dataset. In each epoch, the model sees all 200 fruits, computes gradients, and updates weights.")}.**
        """, unsafe_allow_html=True)
    
    # Show all losses/accuracies up to current checkpoint
    loss_slice = checkpoint_data['losses'][:idx+1].tolist()
    acc_slice = checkpoint_data['accuracies'][:idx+1].tolist()
    epoch_slice = checkpoint_data['epochs'][:idx+1].tolist()

    if len(loss_slice) > 1:
        col_curve, col_stats = st.columns([2, 1])
        
        with col_curve:
            fig_loss = plot_loss_curve(loss_slice, acc_slice, epoch_labels=epoch_slice)
            st.plotly_chart(fig_loss, key="loss_curve_play", config=PLOTLY_CONFIG)
        
        with col_stats:
            st.markdown("**Training Summary**")
            st.markdown(f"""
            | Metric | Start | Current | Best |
            |--------|-------|---------|------|
            | Loss | {loss_slice[0]:.4f} | {loss_slice[-1]:.4f} | {min(loss_slice):.4f} |
            | Accuracy | {acc_slice[0]*100:.1f}% | {acc_slice[-1]*100:.1f}% | {max(acc_slice)*100:.1f}% |
            | Epoch | 0 | {epoch} | — |
            """)
            
            loss_reduction = ((loss_slice[0] - loss_slice[-1]) / max(loss_slice[0], 1e-10)) * 100
            st.metric("Loss Reduction", f"{loss_reduction:.1f}%", delta=f"{loss_slice[-1] - loss_slice[0]:.4f}")
    else:
        fig_loss = plot_loss_curve([], [])
        st.plotly_chart(fig_loss, key="loss_curve_empty", config=PLOTLY_CONFIG)
        st.caption("Press Play or drag the slider to see the training curves develop.")
    
    # ── Sample Flow Inspector ──
    st.markdown("#### Sample Flow Inspector")
    st.caption("Trace how a single data point's values propagate through every layer of the network.")
    
    x_sample = X[sample_idx:sample_idx+1]
    y_label = y[sample_idx, 0]
    fig_flow = plot_sample_flow(mlp, x_sample.flatten(), activation_name, sample_idx, y_label)
    st.plotly_chart(fig_flow, key="flow_play", config=PLOTLY_CONFIG)

else:
    # ══ MANUAL TWEAKING MODE ══
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Manual Parameter Tweaking")
    st.sidebar.caption("Adjust weights and biases to see the decision boundary shift in real-time.")
    
    if 'manual_mlp' not in st.session_state:
        st.session_state.manual_mlp = MLP(seed=42)
    
    mlp = st.session_state.manual_mlp
    
    with st.sidebar.expander("Weights W1 (Input → Hidden)", expanded=False):
        for i in range(2):
            for j in range(3):
                val = st.slider(
                    f"W1[{i},{j}]", -5.0, 5.0,
                    float(mlp.W1[i, j]), 0.01,
                    key=f"W1_{i}_{j}_manual",
                    label_visibility="visible"
                )
                mlp.W1[i, j] = val

    with st.sidebar.expander("Biases b1 (Hidden Layer)", expanded=False):
        for j in range(3):
            val = st.slider(
                f"b1[0,{j}]", -5.0, 5.0,
                float(mlp.b1[0, j]), 0.01,
                key=f"b1_{j}_manual"
            )
            mlp.b1[0, j] = val

    with st.sidebar.expander("Weights W2 (Hidden → Output)", expanded=False):
        for i in range(3):
            val = st.slider(
                f"W2[{i},0]", -5.0, 5.0,
                float(mlp.W2[i, 0]), 0.01,
                key=f"W2_{i}_manual"
            )
            mlp.W2[i, 0] = val

    with st.sidebar.expander("Bias b2 (Output Layer)", expanded=False):
        val = st.slider(
            "b2[0,0]", -5.0, 5.0,
            float(mlp.b2[0, 0]), 0.01,
            key="b2_0_manual"
        )
        mlp.b2[0, 0] = val
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Sample Inspector")
    sample_idx = st.sidebar.slider("Select sample index", 0, len(X)-1, 0, key="sample_idx_manual",
                                    help="Choose a data point to trace through the network.")
    
    # Compute current state
    mlp.forward(X, activation_name)
    current_loss, current_acc = mlp.compute_loss_accuracy(y)
    
    metric_row({
        "Loss": f"{current_loss:.4f}",
        "Accuracy": f"{current_acc*100:.1f}%",
        "Parameters": "13",
        "Mode": "Manual",
    })

    st.markdown("")
    
    col_boundary, col_network = st.columns([1.2, 1])
    
    with col_boundary:
        show_neurons = st.checkbox("Show hidden neuron boundaries", value=False, key="show_neuron_lines_cb_manual")
        from src.visualizations import plot_decision_boundary
        fig_db = plot_decision_boundary(
            mlp, X, y, activation_name,
            0, current_loss, current_acc,
            grid_resolution=50,
            show_neuron_lines=show_neurons
        )
        st.plotly_chart(fig_db, key="db_chart_manual", config=PLOTLY_CONFIG)
    
    with col_network:
        fig_act = plot_activation_curve(activation_name)
        st.plotly_chart(fig_act, key="act_curve_manual", config=PLOTLY_CONFIG)
        
        sample_X = X[sample_idx:sample_idx+1]
        fig_net = plot_network_graph(mlp, sample_X, activation_name)
        st.plotly_chart(fig_net, key="net_graph_manual", config=PLOTLY_CONFIG)
    
    # Sample Flow Inspector
    st.markdown("#### Sample Flow Inspector")
    x_sample = X[sample_idx:sample_idx+1]
    y_label = y[sample_idx, 0]
    fig_flow = plot_sample_flow(mlp, x_sample.flatten(), activation_name, sample_idx, y_label)
    st.plotly_chart(fig_flow, key="flow_manual", config=PLOTLY_CONFIG)

section_divider()

# ════════════════════════════════════════════════════════
# SECTION E: "What Did the Neurons Learn?"
# ════════════════════════════════════════════════════════
section_header("What Did the Neurons Learn?", "Interpretability — peeking inside the hidden layer", "")

# Use the final checkpoint's MLP for this section
if mode == "Pre-computed Playback":
    final_mlp = restore_mlp_from_checkpoint(checkpoint_data, st.session_state.ckpt_idx)
else:
    final_mlp = st.session_state.manual_mlp

st.markdown(f"""
After training, each {tip("hidden neuron", "A node in the hidden layer that computes a weighted sum of inputs, adds a bias, and applies an activation function. Each neuron learns to detect a specific feature combination.")} 
in our network has specialized. By examining their weights, we can understand **what pattern each neuron has learned to detect**.
""", unsafe_allow_html=True)

col_labels, col_boundaries = st.columns([1, 1.5])

with col_labels:
    st.markdown("#### Neuron Identities")
    nicknames, details = final_mlp.get_neuron_labels()

    for j in range(3):
        st.markdown(f"""
        **Hidden Neuron {j}: {nicknames[j]}**  
        {details[j]}  
        Weight on output: W2[{j},0] = `{final_mlp.W2[j, 0]:.3f}`
        """)

with col_boundaries:
    st.markdown("#### Individual Neuron Activation Maps")
    st.caption("Each neuron fires in a different region of the input space. The final decision boundary is a combination of all three.")
    fig_neuron = plot_neuron_boundaries(final_mlp, X, y, activation_name)
    st.plotly_chart(fig_neuron, key="neuron_maps", config=PLOTLY_CONFIG)

# Bridge to Phase 2
st.markdown("---")
st.info("""
**What's Next?** This tiny 2-3-1 network has just **13 parameters** and works on a 2D dataset. 
In **Phase 2**, we scale this exact same architecture to **784 input pixels → 128 neurons → 64 neurons → 10 output classes** 
for handwritten digit recognition (MNIST). The mechanics are identical — just more neurons, more dimensions, more power.
""")
