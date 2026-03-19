import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Linear Regression", layout="wide")
st.title("Section 1: Linear Regression & Polynomials")
st.markdown("Dissecting the core components, parameters, and biases of mathematical curves.")

@st.cache_data
def load_data():
    try:
        # Load Auto-MPG dataset
        data = fetch_openml(name='autoMpg', version=1, parser='auto')
        df = data.frame.dropna()
        # Ensure numerical targets
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'class' in numeric_cols: numeric_cols.remove('class')

        target_col = data.target_names[0] if data.target_names else 'mpg'
        if target_col not in df.columns and 'class' in df.columns:
            target_col = 'class'

        if df[target_col].dtype.name == 'category':
            df[target_col] = df[target_col].astype(float)
        
        features = [c for c in numeric_cols if c != target_col]
        return df, features, target_col
    except Exception as e:
        # Fallback to a synthetic dataset if openml fails
        np.random.seed(42)
        X1 = np.random.uniform(10, 50, 200)
        X2 = np.random.uniform(5, 25, 200)
        Y = 2.5 * X1 - 1.2 * X2 + 15 + np.random.normal(0, 10, 200)
        df = pd.DataFrame({'Feature_1': X1, 'Feature_2': X2, 'Target': Y})
        return df, ['Feature_1', 'Feature_2'], 'Target'

df, features_list, target_col = load_data()

# -----------------
# Sidebar Controls
# -----------------
st.sidebar.header("Model Configuration")
selected_features = st.sidebar.multiselect("Select Independent Variables (X)", features_list, default=[features_list[0]])

if len(selected_features) == 0:
    st.warning("Please select at least one feature.")
    st.stop()

poly_degree = st.sidebar.slider("Polynomial Degree (Non-linear)", 1, 3, 1)
test_size = st.sidebar.slider("Test Set Split Ratio (%)", 10, 50, 20) / 100.0

# -----------------
# Data Preparation
# -----------------
X = df[selected_features].values
y = df[target_col].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Polynomial mapping
poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Model Training
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Predictions
y_train_pred = model.predict(X_train_poly)
y_test_pred = model.predict(X_test_poly)

# Metrics
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

# -----------------
# Mathematical Rendering
# -----------------
col1, col2 = st.columns([1.5, 1])

with col2:
    st.subheader("Mechanistics & Math")
    st.markdown("### The Regression Equation")
    
    st.markdown("The model attempts to learn the parameters (Weights $\mathbf{W}$ and Bias $b$) that minimize the Residual Sum of Squares.")
    
    # Render the exact equation dynamically
    feature_names = poly.get_feature_names_out(selected_features)
    weights = model.coef_
    bias = model.intercept_
    
    eq_parts = [f" {w:.2f} \\times \\text{{{name.replace(' ', '_')}}}" for w, name in zip(weights, feature_names)]
    equation_str = f"\\hat{{y}} = {bias:.2f} + " + " + ".join(eq_parts)
    
    if len(weights) <= 6:
        st.latex(equation_str)
    else:
        st.latex(f"\\hat{{y}} = {bias:.2f} + \sum_{{i=1}}^{{{len(weights)}}} W_i x_i")
        st.caption(f"(Showing {len(weights)} polynomial complexity terms)")
        
    st.markdown("### Evaluation Metrics")
    m_c1, m_c2 = st.columns(2)
    m_c1.metric("Test MSE", f"{mse_test:.2f}")
    m_c2.metric("Test R² Score", f"{r2_test:.4f}")
    
    st.markdown("### Residual Analysis")
    residuals = y_test - y_test_pred
    fig_res = px.histogram(x=residuals, nbins=20, title="Residual Errors (Test Set)", labels={'x': 'Error', 'count': 'Frequency'})
    fig_res.add_vline(x=0, line_color="red", line_dash="dash")
    fig_res.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_res, use_container_width=True)

with col1:
    st.subheader("Visualization Space")
    
    if len(selected_features) == 1:
        # Simple Linear Regression -> 2D Scatter + Curve
        fig = go.Figure()
        
        # Scatter Train
        fig.add_trace(go.Scatter(x=X_train[:, 0], y=y_train, mode='markers', name='Train Data', marker=dict(color='blue', opacity=0.5)))
        # Scatter Test
        fig.add_trace(go.Scatter(x=X_test[:, 0], y=y_test, mode='markers', name='Test Data', marker=dict(color='orange', opacity=0.8)))
        
        # Regression Line/Curve mapping
        x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100).reshape(-1, 1)
        x_range_poly = poly.transform(x_range)
        y_range_pred = model.predict(x_range_poly)
        
        fig.add_trace(go.Scatter(x=x_range[:, 0], y=y_range_pred, mode='lines', name='Regression Fit', line=dict(color='red', width=3)))
        
        fig.update_layout(title="Simple Regression Fit", xaxis_title=selected_features[0], yaxis_title=target_col, height=500)
        st.plotly_chart(fig, use_container_width=True)
        
    elif len(selected_features) == 2 and poly_degree == 1:
        # Multiple Linear Regression (2 vars) -> 3D Scatter + Plane
        fig = go.Figure()
        
        fig.add_trace(go.Scatter3d(
            x=X_test[:, 0], y=X_test[:, 1], z=y_test,
            mode='markers', name='Test Data',
            marker=dict(size=4, color='orange', opacity=0.8)
        ))
        
        x_min, x_max = X[:, 0].min(), X[:, 0].max()
        y_min, y_max = X[:, 1].min(), X[:, 1].max()
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 20), np.linspace(y_min, y_max, 20))
        grid = np.c_[xx.ravel(), yy.ravel()]
        zz = model.predict(grid).reshape(xx.shape)
        
        fig.add_trace(go.Surface(x=xx, y=yy, z=zz, colorscale='Blues', opacity=0.5, name='Prediction Plane', showscale=False))
        
        fig.update_layout(title="Multiple Regression Plane (3D)", scene=dict(
            xaxis_title=selected_features[0],
            yaxis_title=selected_features[1],
            zaxis_title=target_col
        ), height=500)
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("Visualizing Actual vs Predicted for higher dimensions or multi-feature polynomials.")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test, y=y_test_pred, mode='markers', name='Predictions', marker=dict(color='purple')))
        
        min_val, max_val = min(y_test.min(), y_test_pred.min()), max(y_test.max(), y_test_pred.max())
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Perfect Predict Line', line=dict(color='red', dash='dash')))
        
        fig.update_layout(title="Actual vs Predicted (Test Set)", xaxis_title="True Values", yaxis_title="Predicted Values", height=500)
        st.plotly_chart(fig, use_container_width=True)
