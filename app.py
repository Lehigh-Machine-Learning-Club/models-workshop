import streamlit as st
import time

from src.data import get_poisonous_fruit_data
from src.mlp import MLP, ACTIVATIONS
from src.visualizations import plot_decision_boundary, plot_network_graph

st.set_page_config(page_title="Mechanistic Interpretability Dashboard", layout="wide")

st.title("Mechanistic Interpretability: Neural Networks")
st.markdown("### Phase 1: Poisonous Fruit Detector (2-3-1 MLP)")

# Initialize session state variables
if 'mlp' not in st.session_state:
    st.session_state.mlp = MLP()
if 'X' not in st.session_state:
    X, y = get_poisonous_fruit_data()
    st.session_state.X = X
    st.session_state.y = y
if 'epoch' not in st.session_state:
    st.session_state.epoch = 0
if 'is_playing' not in st.session_state:
    st.session_state.is_playing = False
if 'loss' not in st.session_state:
    st.session_state.loss = 1.0
if 'accuracy' not in st.session_state:
    st.session_state.accuracy = 0.5
if 'init_seed' not in st.session_state:
    st.session_state.init_seed = 42

# -----------------
# Right Column Controls (Sidebar)
# -----------------
st.sidebar.header("Global Controls")

activation_name = st.sidebar.selectbox("Hidden Layer Activation", list(ACTIVATIONS.keys()), index=2)
lr = st.sidebar.slider("Learning Rate", 0.01, 2.0, 0.1)
fps = st.sidebar.slider("Animation Speed (FPS)", 1, 60, 30)
epochs_per_frame = st.sidebar.slider("Epochs per Frame", 1, 50, 5)

col_auto = st.sidebar.columns(3)

# Button state handlers
if col_auto[0].button("Play/Pause ⏯️"):
    st.session_state.is_playing = not st.session_state.is_playing
if col_auto[1].button("Step ⏭️"):
    st.session_state.is_playing = False
    for _ in range(epochs_per_frame):
        st.session_state.mlp.forward(st.session_state.X, activation_name)
        st.session_state.mlp.backward(st.session_state.y, activation_name, lr)
        st.session_state.epoch += 1
if col_auto[2].button("Reset 🔄"):
    st.session_state.init_seed += 1
    st.session_state.mlp = MLP(seed=st.session_state.init_seed)
    st.session_state.epoch = 0
    st.session_state.is_playing = False

st.sidebar.markdown("---")
st.sidebar.subheader("Manual Parameter Tweaking")
st.sidebar.caption("Pause training to tweak weights and observe the decision boundary shift in real-time.")

# Mapping sliders directly to mlp attributes
disabled = st.session_state.is_playing

st.sidebar.markdown("**Weights (W1) - Input to Hidden**")
for i in range(2):
    for j in range(3):
        val = st.sidebar.slider(f"W1[{i}, {j}]", -5.0, 5.0, float(st.session_state.mlp.W1[i, j]), 0.01, key=f"W1_{i}_{j}", disabled=disabled)
        if not disabled: st.session_state.mlp.W1[i, j] = val

st.sidebar.markdown("**Biases (b1) - Hidden Layer**")
for j in range(3):
    val = st.sidebar.slider(f"b1[0, {j}]", -5.0, 5.0, float(st.session_state.mlp.b1[0, j]), 0.01, key=f"b1_{j}", disabled=disabled)
    if not disabled: st.session_state.mlp.b1[0, j] = val

st.sidebar.markdown("**Weights (W2) - Hidden to Output**")
for i in range(3):
    val = st.sidebar.slider(f"W2[{i}, 0]", -5.0, 5.0, float(st.session_state.mlp.W2[i, 0]), 0.01, key=f"W2_{i}", disabled=disabled)
    if not disabled: st.session_state.mlp.W2[i, 0] = val

st.sidebar.markdown("**Bias (b2) - Output Layer**")
val = st.sidebar.slider("b2[0, 0]", -5.0, 5.0, float(st.session_state.mlp.b2[0, 0]), 0.01, key="b2_0", disabled=disabled)
if not disabled: st.session_state.mlp.b2[0, 0] = val

# -----------------
# Ensure Metrics Are Fresh (captures manual slider changes)
# -----------------
st.session_state.mlp.forward(st.session_state.X, activation_name)
st.session_state.loss, st.session_state.accuracy = st.session_state.mlp.compute_loss_accuracy(st.session_state.y)

# -----------------
# Main View Rendering (Left & Middle Columns)
# -----------------
plot_container = st.empty()

def render_ui():
    with plot_container.container():
        col1, col2 = st.columns([1.2, 1])
        
        with col1:
            st.markdown("### The Output (Decision Boundary)")
            fig_db = plot_decision_boundary(
                st.session_state.mlp, 
                st.session_state.X, 
                st.session_state.y, 
                activation_name,
                st.session_state.epoch, 
                st.session_state.loss, 
                st.session_state.accuracy
            )
            st.plotly_chart(fig_db, use_container_width=True)
            
        with col2:
            st.markdown("### The Mechanism (Network State)")
            # Use the first sample's data for the forward pass visual demonstration
            sample_X = st.session_state.X[0:1]
            fig_net = plot_network_graph(st.session_state.mlp, sample_X, activation_name)
            st.plotly_chart(fig_net, use_container_width=True)

# Call UI function to explicitly render the placeholder content
render_ui()

# -----------------
# Animation Loop Logic
# -----------------
if st.session_state.is_playing:
    # Perform math updates
    for _ in range(epochs_per_frame):
        st.session_state.mlp.forward(st.session_state.X, activation_name)
        st.session_state.mlp.backward(st.session_state.y, activation_name, lr)
        st.session_state.epoch += 1
    
    # Calculate new metrics for the upcoming render
    st.session_state.loss, st.session_state.accuracy = st.session_state.mlp.compute_loss_accuracy(st.session_state.y)
    
    # Precise sleep mapping to simulated FPS
    time.sleep(1.0 / fps)
    
    # Trigger Streamlit top-to-bottom re-execution
    st.rerun()
