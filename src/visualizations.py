"""
Visualization functions for the Mechanistic Interpretability Dashboard.
All functions return Plotly figures for use with st.plotly_chart().
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# ---- Shared Color Palette ----
COLORS = {
    'primary': '#6C63FF',
    'secondary': '#FF6584',
    'safe': '#00C9A7',        # Safe / edible fruit
    'poison': '#FF6584',       # Poisonous fruit
    'positive_weight': 'rgba(0, 201, 167, 0.7)',   # Green for positive weights
    'negative_weight': 'rgba(255, 101, 132, 0.7)', # Red for negative weights
    'bg_dark': '#0E1117',
    'bg_card': '#1a1a2e',
    'text': '#e0e0e0',
    'grid': 'rgba(255,255,255,0.05)',
    'node_input': '#a8a4e6',
    'node_output': '#64b5f6',
}

# Diverging colorscale: safe (blue/teal) → boundary → poison (red/pink)
DECISION_COLORSCALE = [
    [0.0, '#00C9A7'],
    [0.3, '#a8e6cf'],
    [0.5, '#f5f5dc'],
    [0.7, '#ffaaa5'],
    [1.0, '#FF6584'],
]


def plot_decision_boundary(mlp_model, X, y, activation_name, epoch, loss, accuracy,
                           grid_resolution=50, show_neuron_lines=False):
    """
    Renders the decision boundary contour + training data scatter.
    Enhanced with custom colorscale and optional neuron boundary lines.
    """
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_resolution),
                         np.linspace(y_min, y_max, grid_resolution))
    
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = mlp_model.forward(grid, activation_name)
    Z = Z.reshape(xx.shape)
    
    fig = go.Figure()
    
    # Decision surface contour
    fig.add_trace(go.Contour(
        x=np.linspace(x_min, x_max, grid_resolution),
        y=np.linspace(y_min, y_max, grid_resolution),
        z=Z,
        colorscale=DECISION_COLORSCALE,
        opacity=0.6,
        showscale=True,
        colorbar=dict(title="P(Poison)", thickness=12, len=0.6, x=1.02),
        contours=dict(showlines=True, coloring='heatmap'),
        hoverinfo='skip'
    ))
    
    # Decision boundary contour line at 0.5
    fig.add_trace(go.Contour(
        x=np.linspace(x_min, x_max, grid_resolution),
        y=np.linspace(y_min, y_max, grid_resolution),
        z=Z,
        showscale=False,
        contours=dict(
            start=0.5, end=0.5, size=0,
            coloring='lines',
        ),
        line=dict(color='white', width=3, dash='dot'),
        hoverinfo='skip'
    ))
    
    y_flat = y.flatten()
    
    # Safe fruits (class 0)
    mask_safe = (y_flat == 0)
    fig.add_trace(go.Scatter(
        x=X[mask_safe, 0], y=X[mask_safe, 1],
        mode='markers',
        name='Safe',
        marker=dict(
            color=COLORS['safe'], size=9,
            line=dict(color='white', width=1.5),
            symbol='circle'
        ),
    ))
    
    # Poisonous fruits (class 1)
    mask_poison = (y_flat == 1)
    fig.add_trace(go.Scatter(
        x=X[mask_poison, 0], y=X[mask_poison, 1],
        mode='markers',
        name='Poison',
        marker=dict(
            color=COLORS['secondary'], size=9,
            line=dict(color='white', width=1.5),
            symbol='diamond'
        ),
    ))
    
    # Optional: neuron boundary lines
    if show_neuron_lines:
        boundaries = mlp_model.get_neuron_boundaries(activation_name)
        neuron_colors = ['#FFD93D', '#6BCB77', '#4D96FF']
        for bnd in boundaries:
            j = bnd['neuron_idx']
            if bnd['type'] == 'line':
                x_line = np.linspace(x_min, x_max, 100)
                y_line = bnd['slope'] * x_line + bnd['intercept']
                mask = (y_line >= y_min) & (y_line <= y_max)
                fig.add_trace(go.Scatter(
                    x=x_line[mask], y=y_line[mask],
                    mode='lines', name=f'H{j} boundary',
                    line=dict(color=neuron_colors[j % 3], width=2, dash='dash'),
                ))
            else:
                fig.add_vline(
                    x=bnd['x_intercept'], 
                    line=dict(color=neuron_colors[j % 3], width=2, dash='dash'),
                    annotation_text=f"H{j}"
                )
    
    fig.update_layout(
        title=dict(
            text=f"<b>Decision Boundary</b>  •  Epoch {epoch}  •  Loss {loss:.4f}  •  Acc {accuracy*100:.1f}%",
            font=dict(size=14, color='#333')
        ),
        xaxis_title="Spikiness (Feature 1)",
        yaxis_title="Spottiness (Feature 2)",
        height=480,
        margin=dict(l=50, r=60, t=50, b=50),
        plot_bgcolor='#fafafa',
        paper_bgcolor='white',
        font=dict(color='#333'),
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02,
            xanchor='right', x=1, font=dict(size=11, color='#333')
        ),
        xaxis=dict(gridcolor=COLORS['grid']),
        yaxis=dict(gridcolor=COLORS['grid']),
    )
    return fig


def plot_decision_boundary_from_grid(boundary_Z, grid_x, grid_y, X, y,
                                      epoch, loss, accuracy,
                                      show_neuron_lines=False, mlp_model=None,
                                      activation_name='Sigmoid'):
    """
    Renders a decision boundary from a pre-computed grid (no MLP forward pass needed).
    Used for checkpoint-based playback.
    """
    x_min, x_max = grid_x[0], grid_x[-1]
    y_min, y_max = grid_y[0], grid_y[-1]

    fig = go.Figure()

    # Decision surface contour
    fig.add_trace(go.Contour(
        x=grid_x, y=grid_y, z=boundary_Z,
        colorscale=DECISION_COLORSCALE,
        opacity=0.6,
        showscale=True,
        colorbar=dict(title="P(Poison)", thickness=12, len=0.6, x=1.02,
                      tickfont=dict(color='#333')),
        contours=dict(showlines=True, coloring='heatmap'),
        hoverinfo='skip'
    ))

    # Decision boundary line at 0.5
    fig.add_trace(go.Contour(
        x=grid_x, y=grid_y, z=boundary_Z,
        showscale=False,
        contours=dict(start=0.5, end=0.5, size=0, coloring='lines'),
        line=dict(color='white', width=3, dash='dot'),
        hoverinfo='skip'
    ))

    y_flat = y.flatten()

    # Safe fruits
    mask_safe = (y_flat == 0)
    fig.add_trace(go.Scatter(
        x=X[mask_safe, 0], y=X[mask_safe, 1],
        mode='markers', name='\U0001f34f Safe',
        marker=dict(color=COLORS['safe'], size=9, line=dict(color='white', width=1.5), symbol='circle'),
    ))

    # Poisonous fruits
    mask_poison = (y_flat == 1)
    fig.add_trace(go.Scatter(
        x=X[mask_poison, 0], y=X[mask_poison, 1],
        mode='markers', name='\u2620\ufe0f Poison',
        marker=dict(color=COLORS['secondary'], size=9, line=dict(color='white', width=1.5), symbol='diamond'),
    ))

    # Optional: neuron boundary lines
    if show_neuron_lines and mlp_model is not None:
        boundaries = mlp_model.get_neuron_boundaries(activation_name)
        neuron_colors = ['#FFD93D', '#6BCB77', '#4D96FF']
        for bnd in boundaries:
            j = bnd['neuron_idx']
            if bnd['type'] == 'line':
                x_line = np.linspace(x_min, x_max, 100)
                y_line = bnd['slope'] * x_line + bnd['intercept']
                mask = (y_line >= y_min) & (y_line <= y_max)
                fig.add_trace(go.Scatter(
                    x=x_line[mask], y=y_line[mask],
                    mode='lines', name=f'H{j} boundary',
                    line=dict(color=neuron_colors[j % 3], width=2, dash='dash'),
                ))
            else:
                fig.add_vline(
                    x=bnd['x_intercept'],
                    line=dict(color=neuron_colors[j % 3], width=2, dash='dash'),
                    annotation_text=f"H{j}"
                )

    fig.update_layout(
        title=dict(
            text=f"<b>Decision Boundary</b>  \u2022  Epoch {epoch}  \u2022  Loss {loss:.4f}  \u2022  Acc {accuracy*100:.1f}%",
            font=dict(size=14, color='#333')
        ),
        xaxis_title="Spikiness (Feature 1)",
        yaxis_title="Spottiness (Feature 2)",
        height=480,
        margin=dict(l=50, r=60, t=50, b=50),
        plot_bgcolor='#fafafa',
        paper_bgcolor='white',
        font=dict(color='#333'),
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02,
            xanchor='right', x=1, font=dict(size=11, color='#333')
        ),
        xaxis=dict(gridcolor=COLORS['grid']),
        yaxis=dict(gridcolor=COLORS['grid']),
    )
    return fig


def plot_network_graph(mlp_model, X_sample, activation_name):
    """
    Renders the 2-3-1 network topology with live activations, heatmap node coloring,
    and proportional edge weights.
    """
    vals = mlp_model.predict_single(X_sample.flatten(), activation_name)
    
    layer_sizes = mlp_model.layer_sizes
    layer_labels = ['Input', 'Hidden', 'Output']
    
    # Node positions
    x_positions = [0.0, 1.0, 2.0]
    all_nodes = []  # list of (x, y, value, layer_idx, node_idx)
    
    for l_idx, size in enumerate(layer_sizes):
        y_positions = np.linspace(0.85, 0.15, size)
        for i in range(size):
            if l_idx == 0:
                val = vals['input'][i]
            elif l_idx == 1:
                val = vals['A1'][i]
            else:
                val = vals['A2'][0]
            all_nodes.append((x_positions[l_idx], y_positions[i], val, l_idx, i))
    
    fig = go.Figure()
    
    # Build node position lookup for edges
    positions = {}
    for node in all_nodes:
        x, y, val, l_idx, n_idx = node
        positions[(l_idx, n_idx)] = (x, y)
    
    # Draw edges (W1: Input → Hidden)
    for i in range(layer_sizes[0]):
        for j in range(layer_sizes[1]):
            w = mlp_model.W1[i, j]
            color = COLORS['positive_weight'] if w > 0 else COLORS['negative_weight']
            width = max(0.5, min(abs(w) * 3, 10))
            p1 = positions[(0, i)]
            p2 = positions[(1, j)]
            fig.add_trace(go.Scatter(
                x=[p1[0], p2[0]], y=[p1[1], p2[1]],
                mode='lines',
                line=dict(color=color, width=width),
                hovertemplate=f"<b>W1[{i},{j}]</b> = {w:.3f}<extra></extra>",
                showlegend=False
            ))
    
    # Draw edges (W2: Hidden → Output)
    for i in range(layer_sizes[1]):
        for j in range(layer_sizes[2]):
            w = mlp_model.W2[i, j]
            color = COLORS['positive_weight'] if w > 0 else COLORS['negative_weight']
            width = max(0.5, min(abs(w) * 3, 10))
            p1 = positions[(1, i)]
            p2 = positions[(2, j)]
            fig.add_trace(go.Scatter(
                x=[p1[0], p2[0]], y=[p1[1], p2[1]],
                mode='lines',
                line=dict(color=color, width=width),
                hovertemplate=f"<b>W2[{i},{j}]</b> = {w:.3f}<extra></extra>",
                showlegend=False
            ))
    
    # Draw nodes
    for node in all_nodes:
        x, y, val, l_idx, n_idx = node
        
        # Color mapping
        if l_idx == 0:
            node_color = COLORS['node_input']
            label = f"In{n_idx}"
            border_color = '#7c78d4'
        elif l_idx == 1:
            # Heatmap: pale yellow (low) → deep red (high)
            intensity = min(1.0, max(0.0, abs(val)))
            r = 255
            g = int(255 * (1 - intensity * 0.85))
            b = int(255 * (1 - intensity * 0.9))
            node_color = f'rgb({r},{g},{b})'
            label = f"H{n_idx}"
            border_color = '#cc4444' if intensity > 0.5 else '#886644'
        else:
            node_color = COLORS['node_output']
            label = "Out"
            border_color = '#4488cc'
        
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            marker=dict(
                size=55,
                color=node_color,
                line=dict(color=border_color, width=2.5),
            ),
            text=f"<b>{label}</b><br>{val:.3f}",
            textfont=dict(size=11, color='#222'),
            textposition='middle center',
            hovertemplate=f"<b>{label}</b><br>Value: {val:.4f}<extra></extra>",
            showlegend=False
        ))
    
    # Layer labels
    for l_idx, label in enumerate(layer_labels):
        fig.add_annotation(
            x=x_positions[l_idx], y=-0.02,
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(size=12, color='#666'),
        )
    
    # Bias annotations
    for j in range(layer_sizes[1]):
        b_val = mlp_model.b1[0, j]
        pos = positions[(1, j)]
        fig.add_annotation(
            x=pos[0] - 0.15, y=pos[1] + 0.06,
            text=f"b={b_val:.2f}",
            showarrow=False,
            font=dict(size=9, color='#999'),
        )
    
    b2_val = mlp_model.b2[0, 0]
    out_pos = positions[(2, 0)]
    fig.add_annotation(
        x=out_pos[0] - 0.15, y=out_pos[1] + 0.06,
        text=f"b={b2_val:.2f}",
        showarrow=False,
        font=dict(size=9, color='#999'),
    )
    
    fig.update_layout(
        title=dict(text="<b>Network State</b>  •  Live Activations", font=dict(size=14, color='#333')),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.3, 2.3]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.12, 1.0]),
        height=480,
        margin=dict(l=10, r=10, t=50, b=30),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#333'),
    )
    return fig


def plot_activation_gallery():
    """
    Renders all 5 activation functions and their derivatives side-by-side.
    Returns a Plotly figure with subplots.
    """
    from src.mlp import ACTIVATIONS, ACTIVATION_DESCRIPTIONS
    
    names = list(ACTIVATIONS.keys())
    fig = make_subplots(
        rows=2, cols=5,
        subplot_titles=[n for n in names] + [f"d/dx {n.split('(')[0].strip()}" for n in names],
        vertical_spacing=0.15,
        horizontal_spacing=0.05,
    )
    
    x = np.linspace(-5, 5, 200)
    colors = ['#6C63FF', '#FF6584', '#00C9A7', '#FFD93D', '#64b5f6']
    
    for i, name in enumerate(names):
        act_fn, d_act_fn = ACTIVATIONS[name]
        y_act = act_fn(x)
        y_deriv = d_act_fn(x)
        
        # Activation function
        fig.add_trace(go.Scatter(
            x=x, y=y_act, mode='lines',
            line=dict(color=colors[i], width=2.5),
            name=name, showlegend=False,
            hovertemplate=f"x=%{{x:.2f}}<br>f(x)=%{{y:.3f}}<extra>{name}</extra>"
        ), row=1, col=i+1)
        
        # Zero line
        fig.add_trace(go.Scatter(
            x=[-5, 5], y=[0, 0], mode='lines',
            line=dict(color='rgba(150,150,150,0.3)', width=1, dash='dot'),
            showlegend=False, hoverinfo='skip'
        ), row=1, col=i+1)
        
        # Derivative
        fig.add_trace(go.Scatter(
            x=x, y=y_deriv, mode='lines',
            line=dict(color=colors[i], width=2, dash='dash'),
            name=f"d_{name}", showlegend=False,
            hovertemplate=f"x=%{{x:.2f}}<br>f'(x)=%{{y:.3f}}<extra>d/dx {name}</extra>"
        ), row=2, col=i+1)
        
        fig.add_trace(go.Scatter(
            x=[-5, 5], y=[0, 0], mode='lines',
            line=dict(color='rgba(150,150,150,0.3)', width=1, dash='dot'),
            showlegend=False, hoverinfo='skip'
        ), row=2, col=i+1)
    
    fig.update_layout(
        height=380,
        margin=dict(l=30, r=20, t=40, b=20),
        plot_bgcolor='#fafafa',
        paper_bgcolor='white',
        font=dict(size=10, color='#333'),
    )
    
    # Update all axes
    for i in range(1, 11):
        fig.update_xaxes(range=[-5, 5], showgrid=True, gridcolor='rgba(0,0,0,0.06)', row=(i-1)//5+1, col=(i-1)%5+1)
        fig.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.06)', row=(i-1)//5+1, col=(i-1)%5+1)
    
    return fig


def plot_activation_curve(activation_name):
    """
    Renders a single activation function as a compact mini-plot.
    Used as the "activation visualizer thumbnail" next to the network graph.
    """
    from src.mlp import ACTIVATIONS
    
    act_fn, _ = ACTIVATIONS[activation_name]
    x = np.linspace(-5, 5, 200)
    y = act_fn(x)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='lines',
        line=dict(color=COLORS['primary'], width=3),
        fill='tozeroy',
        fillcolor='rgba(108,99,255,0.08)',
        hovertemplate="x=%{x:.2f}<br>f(x)=%{y:.3f}<extra></extra>"
    ))
    
    # Zero lines
    fig.add_hline(y=0, line=dict(color='rgba(0,0,0,0.15)', width=1))
    fig.add_vline(x=0, line=dict(color='rgba(0,0,0,0.15)', width=1))
    
    short_name = activation_name.split('(')[0].strip()
    fig.update_layout(
        title=dict(text=f"<b>ƒ(x) = {short_name}</b>", font=dict(size=12, color='#333')),
        height=180,
        margin=dict(l=30, r=10, t=35, b=20),
        xaxis=dict(range=[-5, 5], showgrid=False, title=''),
        yaxis=dict(showgrid=False, title=''),
        plot_bgcolor='#fafafa',
        paper_bgcolor='white',
        font=dict(color='#333'),
    )
    return fig


def plot_loss_curve(loss_history, accuracy_history, epoch_labels=None):
    """
    Dual-axis running chart: loss (left axis) + accuracy (right axis).
    Optionally uses epoch_labels for x-axis instead of sequential indices.
    """
    if not loss_history:
        # Empty placeholder
        fig = go.Figure()
        fig.update_layout(
            title="Training Curve (waiting for data...)",
            height=250, margin=dict(l=40, r=40, t=40, b=30),
            font=dict(color='#333'),
            paper_bgcolor='white',
        )
        return fig
    
    if epoch_labels is not None:
        epochs = list(epoch_labels)
    else:
        epochs = list(range(1, len(loss_history) + 1))
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(go.Scatter(
        x=epochs, y=loss_history,
        mode='lines', name='Loss',
        line=dict(color=COLORS['secondary'], width=2),
    ), secondary_y=False)
    
    fig.add_trace(go.Scatter(
        x=epochs, y=accuracy_history,
        mode='lines', name='Accuracy',
        line=dict(color=COLORS['safe'], width=2),
    ), secondary_y=True)
    
    fig.update_layout(
        title=dict(text="<b>Training Progress</b>", font=dict(size=13, color='#333')),
        height=250,
        margin=dict(l=50, r=50, t=40, b=30),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                    font=dict(color='#333')),
        plot_bgcolor='#fafafa',
        paper_bgcolor='white',
        font=dict(color='#333'),
    )
    fig.update_yaxes(title_text="Loss", secondary_y=False, gridcolor='rgba(0,0,0,0.06)')
    fig.update_yaxes(title_text="Accuracy", secondary_y=True, range=[0, 1.05], gridcolor='rgba(0,0,0,0.06)')
    fig.update_xaxes(title_text="Epoch", gridcolor='rgba(0,0,0,0.06)')
    
    return fig


def plot_neuron_boundaries(mlp_model, X, y, activation_name):
    """
    Shows each hidden neuron's individual decision boundary as a subplot.
    For each neuron, shows the region it "fires" (activation > 0.5 for sigmoid, > 0 for ReLU/etc).
    """
    from src.mlp import ACTIVATIONS
    
    n_hidden = mlp_model.layer_sizes[1]
    act_fn, _ = ACTIVATIONS[activation_name]
    
    fig = make_subplots(
        rows=1, cols=n_hidden,
        subplot_titles=[f"Hidden Neuron {j}" for j in range(n_hidden)],
        horizontal_spacing=0.08,
    )
    
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    res = 40
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, res), np.linspace(y_min, y_max, res))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Compute Z1 for all grid points
    Z1_grid = np.dot(grid, mlp_model.W1) + mlp_model.b1  # (res*res, 3)
    A1_grid = act_fn(Z1_grid)
    
    neuron_colors = ['#FFD93D', '#6BCB77', '#4D96FF']
    
    for j in range(n_hidden):
        activation_map = A1_grid[:, j].reshape(xx.shape)
        
        fig.add_trace(go.Contour(
            x=np.linspace(x_min, x_max, res),
            y=np.linspace(y_min, y_max, res),
            z=activation_map,
            colorscale=[[0, '#ffffff'], [1, neuron_colors[j]]],
            showscale=False,
            opacity=0.7,
            hoverinfo='skip',
        ), row=1, col=j+1)
        
        # Data points
        y_flat = y.flatten()
        fig.add_trace(go.Scatter(
            x=X[:, 0], y=X[:, 1],
            mode='markers',
            marker=dict(
                color=['#00C9A7' if v == 0 else '#FF6584' for v in y_flat],
                size=5, line=dict(color='white', width=0.5),
            ),
            showlegend=False,
        ), row=1, col=j+1)
    
    fig.update_layout(
        height=300,
        margin=dict(l=30, r=20, t=40, b=30),
        plot_bgcolor='#fafafa',
        paper_bgcolor='white',
        showlegend=False,
        font=dict(color='#333'),
    )
    
    return fig


def plot_linear_failure(X, y, linear_model=None):
    """
    Shows a logistic regression's linear boundary failing on the moons data.
    Highlights the misclassified points.
    """
    from sklearn.linear_model import LogisticRegression
    
    y_flat = y.flatten()
    
    if linear_model is None:
        linear_model = LogisticRegression(max_iter=200)
        linear_model.fit(X, y_flat)
    
    accuracy = linear_model.score(X, y_flat)
    preds = linear_model.predict(X)
    
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = linear_model.predict_proba(grid)[:, 1].reshape(xx.shape)
    
    fig = go.Figure()
    
    fig.add_trace(go.Contour(
        x=np.linspace(x_min, x_max, 50),
        y=np.linspace(y_min, y_max, 50),
        z=Z,
        colorscale='RdBu',
        opacity=0.4,
        showscale=False,
        hoverinfo='skip'
    ))
    
    # Correct predictions
    correct_mask = (preds == y_flat)
    fig.add_trace(go.Scatter(
        x=X[correct_mask, 0], y=X[correct_mask, 1],
        mode='markers', name='Correct',
        marker=dict(
            color=['#00C9A7' if v == 0 else '#FF6584' for v in y_flat[correct_mask]],
            size=7, line=dict(color='white', width=1),
        ),
    ))
    
    # Misclassified
    wrong_mask = ~correct_mask
    if np.any(wrong_mask):
        fig.add_trace(go.Scatter(
            x=X[wrong_mask, 0], y=X[wrong_mask, 1],
            mode='markers', name='Misclassified ✗',
            marker=dict(
                color='yellow', size=10, symbol='x',
                line=dict(color='black', width=2),
            ),
        ))
    
    fig.update_layout(
        title=dict(
            text=f"<b>Linear Model (Logistic Regression)</b>  •  Acc: {accuracy*100:.1f}%  •  The straight line can't bend!",
            font=dict(size=13, color='#333')
        ),
        xaxis_title="Spikiness", yaxis_title="Spottiness",
        height=400,
        margin=dict(l=50, r=20, t=50, b=50),
        plot_bgcolor='#fafafa',
        paper_bgcolor='white',
        font=dict(color='#333'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(color='#333')),
    )
    return fig


def plot_sample_flow(mlp_model, x_single, activation_name, sample_idx=0, y_true=None):
    """
    Renders an annotated flow diagram showing a single sample's values 
    propagating through the network. Shows input → Z1 → A1 → Z2 → A2.
    """
    vals = mlp_model.predict_single(x_single, activation_name)
    
    fig = go.Figure()
    
    # Stages along x-axis
    stages = ['Input', 'Pre-Act (Z1)', 'Hidden (A1)', 'Pre-Out (Z2)', 'Output (A2)']
    stage_x = [0, 1, 2, 3, 4]
    
    data_map = {
        'Input': vals['input'],
        'Pre-Act (Z1)': vals['Z1'],
        'Hidden (A1)': vals['A1'],
        'Pre-Out (Z2)': vals['Z2'],
        'Output (A2)': vals['A2'],
    }
    
    colors_per_stage = ['#a8a4e6', '#FFD93D', '#6BCB77', '#64b5f6', '#FF6584']
    
    for s_idx, stage in enumerate(stages):
        values = data_map[stage]
        n = len(values)
        y_positions = np.linspace(0.8, 0.2, n) if n > 1 else [0.5]
        
        for v_idx, (yp, val) in enumerate(zip(y_positions, values)):
            fig.add_trace(go.Scatter(
                x=[stage_x[s_idx]], y=[yp],
                mode='markers+text',
                marker=dict(size=40, color=colors_per_stage[s_idx], 
                           line=dict(color='#333', width=1.5)),
                text=f"{val:.3f}",
                textfont=dict(size=10, color='#222'),
                textposition='middle center',
                showlegend=False,
                hovertemplate=f"<b>{stage}</b>[{v_idx}] = {val:.4f}<extra></extra>"
            ))
    
    # Draw arrows between stages
    for s_idx in range(len(stages) - 1):
        n_from = len(data_map[stages[s_idx]])
        n_to = len(data_map[stages[s_idx + 1]])
        y_from = np.linspace(0.8, 0.2, n_from) if n_from > 1 else [0.5]
        y_to = np.linspace(0.8, 0.2, n_to) if n_to > 1 else [0.5]
        
        for yf in y_from:
            for yt in y_to:
                fig.add_trace(go.Scatter(
                    x=[stage_x[s_idx] + 0.15, stage_x[s_idx + 1] - 0.15],
                    y=[yf, yt],
                    mode='lines',
                    line=dict(color='rgba(150,150,150,0.2)', width=1),
                    showlegend=False, hoverinfo='skip'
                ))
    
    # Stage labels
    for s_idx, stage in enumerate(stages):
        fig.add_annotation(
            x=stage_x[s_idx], y=-0.05,
            text=f"<b>{stage}</b>",
            showarrow=False,
            font=dict(size=10, color='#666'),
        )
    
    title_extra = ""
    if y_true is not None:
        pred = "Poison" if vals['A2'][0] > 0.5 else "Safe"
        actual = "Poison" if y_true > 0.5 else "Safe"
        correct = "✓" if (vals['A2'][0] > 0.5) == (y_true > 0.5) else "✗"
        title_extra = f"  •  Predicted: {pred}  •  Actual: {actual}  {correct}"
    
    fig.update_layout(
        title=dict(text=f"<b>Sample #{sample_idx} Flow</b>{title_extra}", font=dict(size=13, color='#333')),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 4.5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.15, 0.95]),
        height=320,
        margin=dict(l=10, r=10, t=45, b=30),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#333'),
    )
    return fig
