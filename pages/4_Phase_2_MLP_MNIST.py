import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import numpy as np
import plotly.graph_objects as go
from src.mnist_mlp import MNIST_MLP

st.set_page_config(page_title="Phase 2: MNIST MLP", layout="wide")
st.title("Mechanistic Interpretability: Phase 2 (MNIST)")
st.markdown("Scaling up our understanding of neural networks from a 2D toy dataset to 784 pixels.")

# Initialize model
if 'mnist_model' not in st.session_state:
    # Initialize a dummy/untrained model for the skeleton
    # In a full implementation, we load pretrained weights here
    model = MNIST_MLP(hidden_size=64) 
    model.eval()
    st.session_state.mnist_model = model

col1, col2, col3 = st.columns([1, 1.5, 1])

with col1:
    st.subheader("Draw a Digit")
    # Interactive canvas
    canvas_result = st_canvas(
        fill_color="black",
        background_color="black",
        stroke_width=20,
        stroke_color="white",
        update_streamlit=True,
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    st.caption("Draw a number (0-9) here in white on the black canvas.")
    
    if st.button("Clear Canvas"):
        # The key="canvas" handles internal state, but typically you need logic
        # or another widget to truly force a canvas reset natively in Streamlit
        st.rerun()

with col2:
    st.subheader("Feature Detectors (Hidden Layer)")
    st.markdown("Each hidden neuron looks for a specific pattern. Here are 4 random feature maps extracted from the initialized weights:")
    
    # Extract weights from the first hidden layer (shape: 64, 784)
    weights = st.session_state.mnist_model.fc1.weight.detach().numpy()
    
    # Plot 4 random feature maps as 28x28 heatmaps
    num_to_display = 4
    fig = go.Figure()
    
    # This is currently a dummy multi-heatmap layout
    # In pure Plotly it requires subplots for a grid layout. 
    # For skeleton purposes, we'll render just the 1st neuron's feature detector
    z = weights[0].reshape((28, 28))
    fig.add_trace(go.Heatmap(
        z=z,
        colorscale='RdBu',
        showscale=False
    ))
    fig.update_layout(
        title="Neuron 0 Feature Map (28x28)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, autorange='reversed'),
        height=300,
        width=300,
        margin=dict(l=10, r=10, t=30, b=10)
    )
    st.plotly_chart(fig, use_container_width=False)
    
with col3:
    st.subheader("Output Probabilities")
    st.markdown("Classification Confidence:")
    
    # Process canvas image if it exists
    probs = np.zeros(10)
    if canvas_result.image_data is not None:
        img_array = canvas_result.image_data
        
        # Convert RGBA to grayscale, scale to [0, 1]
        grayscale = np.mean(img_array[:, :, :3], axis=2) / 255.0
        
        # Small workaround: skip computation if canvas is purely black
        if np.max(grayscale) > 0:
            tensor_x = torch.FloatTensor(grayscale).unsqueeze(0)
            with torch.no_grad():
                logits = st.session_state.mnist_model(tensor_x)
                probs = torch.softmax(logits, dim=1).numpy()[0]
                
    fig_bar = go.Figure(go.Bar(
        x=list(range(10)),
        y=probs,
        marker_color='lightblue'
    ))
    fig_bar.update_layout(
        xaxis=dict(tickvals=list(range(10)), title="Digit Class"),
        yaxis=dict(range=[0, 1], title="Probability"),
        height=300,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig_bar, use_container_width=True)
