"""
MNIST-specific visualization functions for Phase 2.
All functions return Plotly figures for use with st.plotly_chart().
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


# Shared colors
DIGIT_COLORS = [
    '#FF6584', '#6C63FF', '#00C9A7', '#FFD93D', '#4D96FF',
    '#FF9671', '#845EC2', '#D65DB1', '#F9F871', '#00D2FC'
]


def plot_digit_grid(images, labels, n_per_class=5):
    """
    Renders a grid of sample digits organized by class (0-9).
    
    Args:
        images: numpy array of shape (N, 1, 28, 28) or (N, 28, 28)
        labels: numpy array of shape (N,)
        n_per_class: number of samples per digit class
    """
    fig = make_subplots(
        rows=2, cols=5,
        subplot_titles=[f"Digit {d}" for d in range(10)],
        horizontal_spacing=0.03,
        vertical_spacing=0.08,
    )
    
    for digit in range(10):
        mask = labels == digit
        digit_imgs = images[mask][:n_per_class]
        
        if len(digit_imgs) == 0:
            continue
        
        # Take first sample for display
        img = digit_imgs[0]
        if img.ndim == 3:
            img = img.squeeze(0)  # Remove channel dim
        
        row = digit // 5 + 1
        col = digit % 5 + 1
        
        fig.add_trace(go.Heatmap(
            z=np.flipud(img),
            colorscale='Gray',
            showscale=False,
            hovertemplate=f"Digit {digit}<br>Row: %{{y}}<br>Col: %{{x}}<br>Pixel: %{{z:.2f}}<extra></extra>"
        ), row=row, col=col)
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='#0E1117',
        paper_bgcolor='white',
        font=dict(color='#333'),
    )
    
    # Remove axis ticks from all subplots
    for i in range(1, 11):
        r, c = (i-1)//5+1, (i-1)%5+1
        fig.update_xaxes(showticklabels=False, showgrid=False, row=r, col=c)
        fig.update_yaxes(showticklabels=False, showgrid=False, row=r, col=c)
    
    return fig


def plot_pixel_heatmap(image_28x28, title="Pixel Values"):
    """
    Renders a single 28×28 digit as an intensity heatmap with value annotations.
    """
    if image_28x28.ndim == 3:
        image_28x28 = image_28x28.squeeze(0)
    
    img = np.flipud(image_28x28)
    
    fig = go.Figure(go.Heatmap(
        z=img,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title="Intensity", thickness=12),
        hovertemplate="Row: %{y}<br>Col: %{x}<br>Value: %{z:.3f}<extra></extra>"
    ))
    
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=13, color='#333')),
        xaxis=dict(showgrid=False, showticklabels=False, title=""),
        yaxis=dict(showgrid=False, showticklabels=False, title=""),
        height=350,
        width=350,
        margin=dict(l=20, r=40, t=40, b=20),
        paper_bgcolor='white',
        font=dict(color='#333'),
    )
    return fig


def plot_feature_detector_grid(weights, n_neurons=16):
    """
    Renders first-layer weight maps as a grid of 28×28 heatmaps.
    
    Args:
        weights: numpy array of shape (n_total_neurons, 28, 28)
        n_neurons: how many to display (sorted by weight magnitude)
    """
    n_neurons = min(n_neurons, len(weights))
    
    # Sort by weight magnitude (most "interesting" neurons first)
    magnitudes = np.sum(np.abs(weights.reshape(len(weights), -1)), axis=1)
    top_indices = np.argsort(magnitudes)[-n_neurons:][::-1]
    
    cols = 4
    rows = (n_neurons + cols - 1) // cols
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"Neuron {idx}" for idx in top_indices],
        horizontal_spacing=0.04,
        vertical_spacing=0.08,
    )
    
    for i, idx in enumerate(top_indices):
        r = i // cols + 1
        c = i % cols + 1
        
        w = weights[idx]
        
        fig.add_trace(go.Heatmap(
            z=np.flipud(w),
            colorscale='RdBu',
            zmid=0,
            showscale=False,
            hovertemplate=f"Neuron {idx}<br>Row: %{{y}}<br>Col: %{{x}}<br>Weight: %{{z:.4f}}<extra></extra>"
        ), row=r, col=c)
    
    fig.update_layout(
        height=200 * rows,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='white',
        font=dict(color='#333'),
    )
    
    for i in range(1, rows * cols + 1):
        r, c = (i-1)//cols+1, (i-1)%cols+1
        fig.update_xaxes(showticklabels=False, showgrid=False, row=r, col=c)
        fig.update_yaxes(showticklabels=False, showgrid=False, row=r, col=c)
    
    return fig


def plot_layer_activations(activations, layer_name, top_k=None):
    """
    Bar chart of neuron activations for a single layer.
    
    Args:
        activations: 1D numpy array of activation values
        layer_name: display name for the layer
        top_k: if set, only show top k neurons by activation value
    """
    n = len(activations)
    indices = np.arange(n)
    values = activations.copy()
    
    if top_k and top_k < n:
        top_indices = np.argsort(np.abs(values))[-top_k:]
        top_indices = np.sort(top_indices)
        indices = top_indices
        values = values[top_indices]
    
    colors = ['#6C63FF' if v > 0 else '#FF6584' for v in values]
    
    fig = go.Figure(go.Bar(
        x=indices, y=values,
        marker_color=colors,
        hovertemplate="Neuron %{x}<br>Activation: %{y:.4f}<extra></extra>"
    ))
    
    fig.update_layout(
        title=dict(text=f"<b>{layer_name}</b> ({n} neurons)", font=dict(size=13, color='#333')),
        xaxis_title="Neuron Index",
        yaxis_title="Activation Value",
        height=250,
        margin=dict(l=50, r=20, t=40, b=40),
        plot_bgcolor='#fafafa',
        paper_bgcolor='white',
        font=dict(color='#333'),
    )
    return fig


def plot_activation_flow(layer_activations, layer_names=None):
    """
    Multi-panel flow diagram showing activations through all layers.
    
    Args:
        layer_activations: dict of {layer_name: activations_array}
        layer_names: optional list of names in order
    """
    if layer_names is None:
        layer_names = list(layer_activations.keys())
    
    n_layers = len(layer_names)
    fig = make_subplots(
        rows=1, cols=n_layers,
        subplot_titles=layer_names,
        horizontal_spacing=0.06,
    )
    
    colors_list = ['#a8a4e6', '#6C63FF', '#00C9A7', '#FFD93D', '#FF6584']
    
    for i, name in enumerate(layer_names):
        acts = layer_activations[name]
        n = len(acts)
        
        # For large layers, show top 20
        if n > 30:
            top_idx = np.argsort(np.abs(acts))[-20:]
            top_idx = np.sort(top_idx)
            display_vals = acts[top_idx]
            display_idx = top_idx
            subtitle = f"(showing top 20 of {n})"
        else:
            display_vals = acts
            display_idx = np.arange(n)
            subtitle = ""
        
        fig.add_trace(go.Bar(
            x=display_idx, y=display_vals,
            marker_color=colors_list[i % len(colors_list)],
            showlegend=False,
            hovertemplate=f"{name}<br>Neuron %{{x}}<br>Value: %{{y:.4f}}<extra></extra>"
        ), row=1, col=i+1)
    
    fig.update_layout(
        height=280,
        margin=dict(l=40, r=20, t=50, b=30),
        plot_bgcolor='#fafafa',
        paper_bgcolor='white',
        font=dict(color='#333'),
    )
    return fig


def plot_confusion_matrix(y_true, y_pred):
    """
    Annotated confusion matrix heatmap.
    """
    n_classes = 10
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    
    # Normalize by row for color intensity
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    
    # Create text annotations
    text_matrix = [[str(cm[i, j]) for j in range(n_classes)] for i in range(n_classes)]
    
    fig = go.Figure(go.Heatmap(
        z=cm_normalized,
        x=list(range(n_classes)),
        y=list(range(n_classes)),
        colorscale='Blues',
        showscale=True,
        colorbar=dict(title="Rate", thickness=12),
        text=text_matrix,
        texttemplate="%{text}",
        textfont=dict(size=11),
        hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{text}<br>Rate: %{z:.3f}<extra></extra>"
    ))
    
    fig.update_layout(
        title=dict(text="<b>Confusion Matrix</b>", font=dict(size=14, color='#333')),
        xaxis=dict(title="Predicted Label", tickvals=list(range(10)), dtick=1),
        yaxis=dict(title="True Label", tickvals=list(range(10)), dtick=1, autorange='reversed'),
        height=450,
        width=500,
        margin=dict(l=60, r=40, t=50, b=50),
        paper_bgcolor='white',
        font=dict(color='#333'),
    )
    return fig


def plot_confidence_distribution(probs, y_true, y_pred):
    """
    Histogram of prediction confidence for correct vs incorrect predictions.
    """
    max_probs = np.max(probs, axis=1)
    correct_mask = y_true == y_pred
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=max_probs[correct_mask],
        name='Correct', opacity=0.7,
        marker_color='#00C9A7',
        nbinsx=30,
    ))
    
    fig.add_trace(go.Histogram(
        x=max_probs[~correct_mask],
        name='Incorrect', opacity=0.7,
        marker_color='#FF6584',
        nbinsx=30,
    ))
    
    fig.update_layout(
        title=dict(text="<b>Prediction Confidence Distribution</b>", font=dict(size=14, color='#333')),
        xaxis_title="Max Probability (Confidence)",
        yaxis_title="Count",
        barmode='overlay',
        height=300,
        margin=dict(l=50, r=20, t=50, b=40),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(color='#333')),
        plot_bgcolor='#fafafa',
        paper_bgcolor='white',
        font=dict(color='#333'),
    )
    return fig


def plot_output_probabilities(probs, predicted_label=None):
    """
    Bar chart of output probabilities for a single prediction.
    """
    colors = ['#e0e0e0'] * 10
    if predicted_label is not None:
        colors[predicted_label] = '#6C63FF'
    
    fig = go.Figure(go.Bar(
        x=list(range(10)),
        y=probs,
        marker_color=colors,
        text=[f"{p:.1%}" for p in probs],
        textposition='outside',
        hovertemplate="Digit %{x}<br>Probability: %{y:.4f}<extra></extra>"
    ))
    
    fig.update_layout(
        title=dict(text="<b>Classification Confidence</b>", font=dict(size=13, color='#333')),
        xaxis=dict(tickvals=list(range(10)), title="Digit Class", dtick=1),
        yaxis=dict(range=[0, 1.1], title="Probability"),
        height=300,
        margin=dict(l=50, r=20, t=40, b=40),
        plot_bgcolor='#fafafa',
        paper_bgcolor='white',
        font=dict(color='#333'),
    )
    return fig


def plot_architecture_comparison(comparison_data):
    """
    Side-by-side comparison chart of two architectures.
    """
    small = comparison_data['small']
    large = comparison_data['large']
    
    metrics = ['Parameters', 'Test Accuracy (%)', 'Test Loss', 'Avg Epoch Time (s)']
    small_vals = [
        small['parameters'],
        small['final_test_acc'] * 100,
        small['final_test_loss'],
        small['avg_epoch_time'],
    ]
    large_vals = [
        large['parameters'],
        large['final_test_acc'] * 100,
        large['final_test_loss'],
        large['avg_epoch_time'],
    ]
    
    fig = make_subplots(rows=1, cols=4, subplot_titles=metrics)
    
    for i, (metric, sv, lv) in enumerate(zip(metrics, small_vals, large_vals)):
        fig.add_trace(go.Bar(
            x=['Small', 'Large'], y=[sv, lv],
            marker_color=['#FFD93D', '#6C63FF'],
            text=[f"{sv:.2f}", f"{lv:.2f}"],
            textposition='outside',
            showlegend=False,
        ), row=1, col=i+1)
    
    fig.update_layout(
        height=300,
        margin=dict(l=40, r=20, t=50, b=30),
        plot_bgcolor='#fafafa',
        paper_bgcolor='white',
        font=dict(color='#333'),
    )
    return fig


def plot_training_curves(history_data):
    """
    Training loss/accuracy curves for comparison.
    
    Args:
        history_data: dict with 'small' and 'large' keys, each containing
                      'train_loss', 'train_acc', 'test_loss', 'test_acc' lists
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Loss', 'Accuracy'],
    )
    
    epochs_small = list(range(1, len(history_data['small']['test_loss']) + 1))
    epochs_large = list(range(1, len(history_data['large']['test_loss']) + 1))
    
    # Loss curves
    fig.add_trace(go.Scatter(
        x=epochs_small, y=history_data['small']['test_loss'],
        mode='lines', name='Small (test)',
        line=dict(color='#FFD93D', width=2),
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=epochs_large, y=history_data['large']['test_loss'],
        mode='lines', name='Large (test)',
        line=dict(color='#6C63FF', width=2),
    ), row=1, col=1)
    
    # Accuracy curves
    fig.add_trace(go.Scatter(
        x=epochs_small, y=[a * 100 for a in history_data['small']['test_acc']],
        mode='lines', name='Small (test)',
        line=dict(color='#FFD93D', width=2, dash='dash'),
        showlegend=False,
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=epochs_large, y=[a * 100 for a in history_data['large']['test_acc']],
        mode='lines', name='Large (test)',
        line=dict(color='#6C63FF', width=2, dash='dash'),
        showlegend=False,
    ), row=1, col=2)
    
    fig.update_layout(
        height=300,
        margin=dict(l=50, r=20, t=50, b=40),
        plot_bgcolor='#fafafa',
        paper_bgcolor='white',
        font=dict(color='#333'),
        legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='right', x=1, font=dict(color='#333')),
    )
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy (%)", row=1, col=2)
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    
    return fig
