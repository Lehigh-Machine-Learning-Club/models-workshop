import plotly.graph_objects as go
import numpy as np

def plot_decision_boundary(mlp_model, X, y, activation_name, epoch, loss, accuracy):
    # Create meshgrid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                         np.linspace(y_min, y_max, 50))
    
    grid = np.c_[xx.ravel(), yy.ravel()]
    # forward pass on grid
    Z = mlp_model.forward(grid, activation_name)
    Z = Z.reshape(xx.shape)
    
    fig = go.Figure()
    
    # Contour for decision boundary
    fig.add_trace(go.Contour(
        x=np.linspace(x_min, x_max, 50),
        y=np.linspace(y_min, y_max, 50),
        z=Z,
        colorscale='RdBu',
        opacity=0.5,
        showscale=False,
        hoverinfo='skip'
    ))
    
    # Scatter for training data points
    fig.add_trace(go.Scatter(
        x=X[:, 0], y=X[:, 1],
        mode='markers',
        marker=dict(
            color=y.flatten(),
            colorscale='RdBu',
            line=dict(color='white', width=1),
            size=8
        ),
        showlegend=False
    ))
    
    fig.update_layout(
        title=f"Decision Boundary | Epoch: {epoch} | Loss: {loss:.4f} | Acc: {accuracy*100:.1f}%",
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Spikes (Feature 1)",
        yaxis_title="Spots (Feature 2)",
        height=450
    )
    return fig

def plot_network_graph(mlp_model, X_sample, activation_name):
    """
    Renders the topology and live weights/activations for the 2-3-1 network.
    """
    # Run a single forward pass for a sample to get node activations
    _ = mlp_model.forward(X_sample, activation_name)
    
    # Extract values
    x_vals = X_sample.flatten()
    z1_vals = mlp_model.Z1.flatten()
    a1_vals = mlp_model.A1.flatten()
    a2_vals = mlp_model.A2.flatten()
    
    layer_sizes = mlp_model.layer_sizes
    
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    
    # Horizontal spacing based on layers
    x_space = [0, 1, 2]
    
    # Build Nodes
    positions = []
    for l_idx, size in enumerate(layer_sizes):
        y_space = np.linspace(0.8, 0.2, size)
        layer_pos = []
        for i in range(size):
            node_x.append(x_space[l_idx])
            node_y.append(y_space[i])
            layer_pos.append((x_space[l_idx], y_space[i]))
            
            # Text & Color formatting based on activation
            if l_idx == 0:
                val = x_vals[i]
                node_text.append(f"<b>In_{i}</b><br>{val:.2f}")
                node_color.append('lightgray')
            elif l_idx == 1:
                val = a1_vals[i]
                # Scale color strictly based on magnitude of activation (0 to 1 mapping to yellow -> red)
                intensity = min(255, max(0, int(abs(val) * 255)))
                node_text.append(f"<b>H_{i}</b><br>{val:.2f}")
                node_color.append(f'rgb(255, {255-intensity}, {255-intensity})')
            else:
                val = a2_vals[0]
                node_text.append(f"<b>Out</b><br>{val:.2f}")
                node_color.append('lightblue')
                
        positions.append(layer_pos)
        
    fig = go.Figure()
    
    # Build Edges (W1)
    for i in range(layer_sizes[0]):
        for j in range(layer_sizes[1]):
            w = mlp_model.W1[i, j]
            color = 'rgba(0,128,0,0.6)' if w > 0 else 'rgba(255,0,0,0.6)'
            width = max(1, abs(w) * 3)
            p1 = positions[0][i]
            p2 = positions[1][j]
            fig.add_trace(go.Scatter(
                x=[p1[0], p2[0]], y=[p1[1], p2[1]],
                mode='lines',
                line=dict(color=color, width=width),
                hovertemplate=f"W1[{i},{j}] = {w:.2f}<extra></extra>",
                showlegend=False
            ))
            
    # Build Edges (W2)
    for i in range(layer_sizes[1]):
        for j in range(layer_sizes[2]):
            w = mlp_model.W2[i, j]
            color = 'rgba(0,128,0,0.6)' if w > 0 else 'rgba(255,0,0,0.6)'
            width = max(1, abs(w) * 3)
            p1 = positions[1][i]
            p2 = positions[2][j]
            fig.add_trace(go.Scatter(
                x=[p1[0], p2[0]], y=[p1[1], p2[1]],
                mode='lines',
                line=dict(color=color, width=width),
                hovertemplate=f"W2[{i},{j}] = {w:.2f}<extra></extra>",
                showlegend=False
            ))
            
    # Draw Nodes last so they sit on top of the edges
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(
            size=60,
            color=node_color,
            line=dict(color='black', width=2)
        ),
        text=node_text,
        textfont=dict(size=12, color='black'),
        hoverinfo='none',
        showlegend=False
    ))
    
    fig.update_layout(
        title="Network Graph (Sample Auto-Forward)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=40, b=20),
        height=450,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    return fig
