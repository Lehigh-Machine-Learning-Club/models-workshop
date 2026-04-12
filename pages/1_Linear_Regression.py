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

st.set_page_config(page_title="Linear, Polynomial & Multiple Regression", layout="wide")

st.markdown("""
<style>
.tooltip-term {
  border-bottom: 1px dashed #64748b;
  cursor: help;
  position: relative;
  display: inline-block;
}
.tooltip-term .tooltip-text {
  visibility: hidden;
  background-color: #1e293b;
  color: #f1f5f9;
  text-align: left;
  border-radius: 6px;
  padding: 8px 12px;
  position: absolute;
  z-index: 100;
  bottom: 125%;
  left: 0;
  width: 260px;
  font-size: 0.78rem;
  line-height: 1.4;
  box-shadow: 0 4px 12px rgba(0,0,0,0.3);
  opacity: 0;
  transition: opacity 0.2s;
}
.tooltip-term:hover .tooltip-text {
  visibility: visible;
  opacity: 1;
}

.section-badge {
    display: inline-block;
    padding: 2px 8px;
    background-color: #1e293b;
    color: #38bdf8;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 10px;
}

.title-accent {
    width: 60px;
    height: 4px;
    background-color: #0ea5e9;
    border-radius: 2px;
    margin-bottom: 30px;
}

.insight-box {
    background-color: #1e293b;
    border-left: 4px solid #38bdf8;
    padding: 15px;
    border-radius: 4px;
    margin: 15px 0;
    color: #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

st.title("Linear, Polynomial & Multiple Regression")
st.markdown('<div class="title-accent"></div>', unsafe_allow_html=True)

def tooltip(term, text):
    return f'<span class="tooltip-term" style="font-weight:600;">{term}<span class="tooltip-text">{text}</span></span>'

@st.cache_data
def load_data():
    try:
        data = fetch_openml(name='autoMpg', version=1, parser='auto')
        df = data.frame.dropna()
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
        np.random.seed(42)
        X1 = np.random.uniform(10, 50, 200)
        X2 = np.random.uniform(5, 25, 200)
        Y = 2.5 * X1 - 1.2 * X2 + 15 + np.random.normal(0, 10, 200)
        df = pd.DataFrame({'Feature_1': X1, 'Feature_2': X2, 'Target': Y})
        return df, ['Feature_1', 'Feature_2'], 'Target'

df, features_list, target_col = load_data()

def gradient_descent_step(X, y, w, b, lr):
    predictions = w * X + b
    error = predictions - y
    N = len(y)
    dw = (2 / N) * np.sum(error * X)
    db = (2 / N) * np.sum(error)
    w_new = w - lr * dw
    b_new = b - lr * db
    loss = (1 / N) * np.sum(error ** 2)
    return w_new, b_new, loss

@st.cache_data
def compute_loss_surface(X_flat, y, opt_w, opt_b, spread_w, spread_b, grid_size=80):
    w_range = np.linspace(opt_w - spread_w, opt_w + spread_w, grid_size)
    b_range = np.linspace(opt_b - spread_b, opt_b + spread_b, grid_size)
    
    W_grid, B_grid = np.meshgrid(w_range, b_range)
    Loss_grid = np.zeros_like(W_grid)
    for i in range(len(b_range)):
        for j in range(len(w_range)):
            preds = W_grid[i, j] * X_flat + B_grid[i, j]
            Loss_grid[i, j] = np.mean((preds - y) ** 2)
            
    return w_range, b_range, Loss_grid

# -----------------
# Sidebar Controls
# -----------------
st.sidebar.header("Model Configuration")
selected_features = st.sidebar.multiselect("Select Independent Variables (X)", features_list, default=[features_list[0]])

if len(selected_features) == 1:
    fit_mode = st.sidebar.radio("Optimization Mode", ["Auto-Fit (Algorithm)", "Manual Fit (Human)"])
    if fit_mode == "Manual Fit (Human)":
        st.sidebar.caption("🎯 Drag w and b to fit the line manually. Can you beat the algorithm?")
        
    show_residuals = st.sidebar.checkbox("Show Residuals", value=False)
    
    if fit_mode == "Manual Fit (Human)":
        manual_w = st.sidebar.slider("Weight (w)", -5.0, 5.0, 0.0, step=0.001)
        manual_b = st.sidebar.slider("Bias (b)", -50.0, 100.0, 0.0, step=0.1)
    else:
        manual_w = None
        manual_b = None
else:
    fit_mode = "Auto-Fit (Algorithm)"
    show_residuals = False
    manual_w = None
    manual_b = None

poly_degree = st.sidebar.slider("Polynomial Degree (Non-linear)", 1, 3, 1)
test_size = st.sidebar.slider("Test Set Split Ratio (%)", 10, 50, 20) / 100.0



# -----------------
# Layout Tabs
# -----------------
tab_data, tab_foundations, tab_model, tab_hood = st.tabs(
    [" Dataset & Problem", " Foundations", " Model", " Under the Hood"]
)

with tab_data:
    desc_map = {
        'mpg': "Miles per gallon — the target variable we predict",
        'cylinders': "Number of engine cylinders (4, 6, or 8)",
        'cylinders': "Number of engine cylinders (4, 6, or 8)",
        'displacement': "Engine displacement in cubic inches",
        'horsepower': "Engine horsepower",
        'weight': "Vehicle weight in pounds",
        'acceleration': "Time to accelerate from 0 to 60 mph (seconds)",
        'model_year': "Model year of the car (e.g. 70 = 1970)",
        'origin': "Origin of the car (1: USA, 2: Europe, 3: Japan)"
    }

    # 1. Problem Statement
    st.subheader("Can we predict a car's fuel efficiency?")
    st.markdown("In the late 1970s and early 1980s, cars varied wildly in fuel efficiency. Some guzzled gas, others were surprisingly economical. The question is: given a car's physical and mechanical properties — its weight, engine size, horsepower — can we build a mathematical model that accurately predicts how many miles per gallon it will achieve?")
    st.markdown("This is a regression problem: we are predicting a continuous numeric value (mpg), not a category. The model we build will learn a mathematical relationship between the input properties and fuel efficiency.")
    st.divider()

    # 2. Data Overview
    st.subheader("Data Overview")
    st.markdown("This dataset was originally from the UCI Machine Learning Repository and contains data about cars from the late 1970s and early 1980s. The goal is to predict a car's fuel efficiency (miles per gallon) based on its physical and mechanical properties.")
    
    col1, col2 = st.columns(2)
    col1.metric("Total Rows", len(df))
    col2.metric("Total Columns", len(df.columns))
    st.divider()
    
    # 3. Column Details
    st.subheader("Column Details")
    
    col_details = []
    for col in df.columns:
        desc = desc_map.get(col, "Feature column")
        col_details.append({
            "Column Name": col,
            "Data Type": str(df[col].dtype),
            "Description": desc
        })
    df_desc = pd.DataFrame(col_details)
    st.dataframe(df_desc, use_container_width=True, hide_index=True)
    st.divider()
    
    # 4. Sample Data
    st.subheader("Sample Data (First 5 Rows)")
    st.dataframe(df.head(5), use_container_width=True)
    st.divider()

    # 5. Key Terminology
    st.subheader("Key Terminology")
    tc1, tc2 = st.columns(2)
    with tc1:
        st.markdown('<span style="color:#0ea5e9; font-size:1.2rem; font-weight:700;">Independent Variables (Features)</span>', unsafe_allow_html=True)
        st.markdown("These are the INPUT columns — the properties we use to make a prediction. Think of them as the 'questions' we ask about each car.")
        
        feat_html = ""
        for f in features_list:
            desc = desc_map.get(f, "Feature column")
            feat_html += f"<div style='background-color:#1e293b; padding:8px 12px; margin-bottom:6px; border-radius:6px; border-left:3px solid #64748b;'><strong>{f}</strong> <span style='color:#94a3b8; font-size:0.9em;'>— {desc}</span></div>"
        if feat_html:
            st.markdown(feat_html, unsafe_allow_html=True)
            
        st.caption("We choose which of these to feed into our model using the sidebar. More features can improve predictions — but also risk overfitting.")
        
    with tc2:
        st.markdown('<span style="color:#0ea5e9; font-size:1.2rem; font-weight:700;">Dependent Variable (Target)</span>', unsafe_allow_html=True)
        st.markdown("This is the OUTPUT column — the single value we are trying to predict. Everything the model learns is aimed at getting this value right.")
        
        t_desc = desc_map.get(target_col, "Target output column")
        st.markdown(f"<div style='background-color:#0f172a; padding:15px; margin: 15px 0; border-radius:8px; border:1px solid #1e293b; border-left:4px solid #0ea5e9; text-align:center;'><h3 style='margin:0; color:#0ea5e9; font-weight:700;'>{target_col}</h3><span style='color:#cbd5e1; font-style:italic; font-size:0.95rem;'>{t_desc}</span></div>", unsafe_allow_html=True)
        
        st.markdown(f"For every sample in the dataset, we know the actual **{target_col}**. The model's job is to learn a formula that gets as close to these real values as possible.")
        
    st.info("In simple terms: Features are what we KNOW about a data point. The target is what we want to PREDICT.")

with tab_foundations:
    # SECTION A
    st.markdown("### What is Regression?")
    st.markdown("Regression is a supervised learning technique that predicts a continuous numerical output from input features. Unlike classification (which predicts categories like 'spam' or 'not spam'), regression predicts values like price, temperature, or — in our case — miles per gallon.")

    # SECTION B
    st.markdown("### Types of Regression")
    col1, col2, col3 = st.columns(3)
    np.random.seed(42)

    with col1:
        x_lin = np.linspace(0, 10, 30)
        y_lin = 2 * x_lin + 3 + np.random.randn(30) * 2
        fig_lin = go.Figure()
        fig_lin.add_trace(go.Scatter(x=x_lin, y=y_lin, mode='markers', name='Data', marker=dict(color='blue', opacity=0.5)))
        fig_lin.add_trace(go.Scatter(x=x_lin, y=2*x_lin+3, mode='lines', name='Fit', line=dict(color='red', width=3)))
        fig_lin.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), showlegend=False, title="Simple Linear Regression")
        st.plotly_chart(fig_lin, use_container_width=True)
        st.latex(r"y = w \cdot x + b")
        st.caption("One feature, one straight line. The simplest form.")

    with col2:
        x_poly = np.linspace(0, 10, 30)
        y_poly = 0.5 * x_poly**2 - 3 * x_poly + 10 + np.random.randn(30) * 2
        fig_poly = go.Figure()
        fig_poly.add_trace(go.Scatter(x=x_poly, y=y_poly, mode='markers', name='Data', marker=dict(color='blue', opacity=0.5)))
        fig_poly.add_trace(go.Scatter(x=x_poly, y=0.5*x_poly**2 - 3*x_poly + 10, mode='lines', name='Fit', line=dict(color='red', width=3)))
        fig_poly.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), showlegend=False, title="Polynomial Regression")
        st.plotly_chart(fig_poly, use_container_width=True)
        st.latex(r"y = w_2 x^2 + w_1 x + b")
        st.caption("Still one feature, but the relationship is curved. We add polynomial terms (x², x³...) to capture this.")

    with col3:
        x1_mult = np.random.rand(30) * 10
        x2_mult = np.random.rand(30) * 10
        y_mult = 3 * x1_mult + 2 * x2_mult + 5 + np.random.randn(30) * 2
        fig_mult = go.Figure()
        fig_mult.add_trace(go.Scatter3d(x=x1_mult, y=x2_mult, z=y_mult, mode='markers', marker=dict(size=4, color='blue', opacity=0.5)))
        
        xx_mult, yy_mult = np.meshgrid(np.linspace(0, 10, 10), np.linspace(0, 10, 10))
        zz_mult = 3 * xx_mult + 2 * yy_mult + 5
        fig_mult.add_trace(go.Surface(x=xx_mult, y=yy_mult, z=zz_mult, colorscale='Reds', opacity=0.5, showscale=False))
        fig_mult.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), showlegend=False, title="Multiple Regression", scene=dict(xaxis_title="x1", yaxis_title="x2", zaxis_title="y"))
        st.plotly_chart(fig_mult, use_container_width=True)
        st.latex(r"y = w_1 x_1 + w_2 x_2 + b")
        st.caption("Two or more features. The line becomes a plane (or hyperplane in higher dimensions).")

    st.divider()

    # SECTION C
    st.markdown("### Understanding Residuals")
    np.random.seed(42)
    x_res = np.linspace(1, 9, 8)
    y_true_res = 2 * x_res + 3
    y_noisy_res = y_true_res + np.random.randn(8) * 3
    
    fig_res = go.Figure()
    fig_res.add_trace(go.Scatter(x=x_res, y=y_true_res, mode='lines', name='Regression Line', line=dict(color='red', width=3)))
    fig_res.add_trace(go.Scatter(x=x_res, y=y_noisy_res, mode='markers', name='Actual Data', marker=dict(color='blue', size=8)))
    
    # Draw residual lines
    for i in range(len(x_res)):
        fig_res.add_trace(go.Scatter(
            x=[x_res[i], x_res[i]], y=[y_noisy_res[i], y_true_res[i]],
            mode='lines', line=dict(color='red', dash='dash', width=1), showlegend=False
        ))
        
    # Annotate one residual
    idx = 4
    fig_res.add_annotation(
        x=x_res[idx], y=(y_noisy_res[idx] + y_true_res[idx])/2,
        text="Residual = Actual − Predicted", showarrow=True, arrowhead=2, ax=60, ay=0
    )
    fig_res.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_res, use_container_width=True)
    
    st.latex(r"e_i = y_i - \hat{y}_i")
    st.markdown("A residual is the vertical distance between an actual data point and the regression line. Positive means the model under-predicted; negative means it over-predicted. The model tries to make these as small as possible.")

    st.divider()

    # SECTION D
    st.markdown("### Model Complexity: Finding the Sweet Spot")
    np.random.seed(42)
    x_cmplx = np.linspace(0, 10, 20)
    y_cmplx = 0.4 * x_cmplx**2 - 2 * x_cmplx + 5 + np.random.randn(20) * 1.5
    
    col_u, col_g, col_o = st.columns(3)
    
    x_smooth = np.linspace(0, 10, 100)
    
    with col_u:
        p1 = np.polyfit(x_cmplx, y_cmplx, 1)
        y_u = np.polyval(p1, x_smooth)
        fig_u = go.Figure()
        fig_u.add_trace(go.Scatter(x=x_cmplx, y=y_cmplx, mode='markers', marker=dict(color='blue', size=6)))
        fig_u.add_trace(go.Scatter(x=x_smooth, y=y_u, mode='lines', line=dict(color='#f59e0b', width=3)))
        fig_u.update_layout(title="Underfit (Degree 1)", height=300, margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
        st.plotly_chart(fig_u, use_container_width=True)
        st.error(" Too simple — misses the pattern (high bias)")
        
    with col_g:
        p2 = np.polyfit(x_cmplx, y_cmplx, 2)
        y_g = np.polyval(p2, x_smooth)
        fig_g = go.Figure()
        fig_g.add_trace(go.Scatter(x=x_cmplx, y=y_cmplx, mode='markers', marker=dict(color='blue', size=6)))
        fig_g.add_trace(go.Scatter(x=x_smooth, y=y_g, mode='lines', line=dict(color='#22c55e', width=3)))
        fig_g.update_layout(title="Good Fit (Degree 2)", height=300, margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
        st.plotly_chart(fig_g, use_container_width=True)
        st.success(" Just right — captures the pattern without chasing noise")
        
    with col_o:
        p10 = np.polyfit(x_cmplx, y_cmplx, 10)
        y_o = np.polyval(p10, x_smooth)
        fig_o = go.Figure()
        fig_o.add_trace(go.Scatter(x=x_cmplx, y=y_cmplx, mode='markers', marker=dict(color='blue', size=6)))
        fig_o.add_trace(go.Scatter(x=x_smooth, y=y_o, mode='lines', line=dict(color='#ef4444', width=3)))
        fig_o.update_layout(title="Overfit (Degree 10)", height=300, margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
        st.plotly_chart(fig_o, use_container_width=True)
        st.error("❌ Too complex — memorizes noise, fails on new data (high variance)")

    st.markdown('<div class="insight-box">As you increase polynomial degree, the model becomes more flexible. But more flexibility isn\'t always better — it can start memorizing noise in the training data instead of learning the true pattern. This tradeoff between simplicity and complexity is called the bias-variance tradeoff.</div>', unsafe_allow_html=True)

    st.divider()

    # SECTION E
    st.markdown("### Why Split the Data?")
    st.markdown("We split our dataset into two parts:\n- **Training set (80%)**: Used to fit the model — find the best weights.\n- **Testing set (20%)**: Held back, never seen during training. Used to check if the model generalizes to new, unseen data.\n\nIf a model scores well on training data but poorly on test data, it's overfitting — it memorized the training examples instead of learning the underlying pattern.\n\nThe split is random — we shuffle the data before splitting so neither set is biased toward a particular range.")



with tab_model:
    if len(selected_features) == 0:
        st.warning("Please select at least one feature.")
    else:
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
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_r2 = r2_score(y_train, y_train_pred)
        
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(y_test, y_test_pred)
        
        sklearn_train_mse = train_mse
        sklearn_test_mse = test_mse
        sklearn_train_r2 = train_r2
        
        if fit_mode == "Manual Fit (Human)" and len(selected_features) == 1:
            y_train_pred = manual_w * X_train[:, 0] + manual_b
            y_test_pred  = manual_w * X_test[:, 0]  + manual_b
            train_mse = mean_squared_error(y_train, y_train_pred)
            train_rmse = np.sqrt(train_mse)
            train_r2 = r2_score(y_train, y_train_pred)
            
            test_mse  = mean_squared_error(y_test,  y_test_pred)
            test_rmse = np.sqrt(test_mse)
            test_r2   = r2_score(y_test, y_test_pred)

        # -----------------
        # Visual Rendering
        # -----------------
        col_viz, col_math = st.columns([1.5, 1])

        with col_math:
            with st.container(border=True):

                st.subheader("Mechanistics & Math")
                
                st.markdown("The model attempts to learn the parameters (Weights $\mathbf{W}$ and Bias $b$) that minimize the Residual Sum of Squares.")
                
                # Render the exact equation dynamically
                feature_names = poly.get_feature_names_out(selected_features)
                weights = model.coef_
                bias = model.intercept_
                
                eq_parts = [f" {w:.2f} \\times \\text{{{name.replace(' ', '_')}}}" for w, name in zip(weights, feature_names)]
                equation_str = f"\\hat{{y}} = {bias:.2f} + " + " + ".join(eq_parts)
                
                if len(selected_features) == 1 and poly_degree == 1:
                    generic_form = "ŷ = mx + b"
                elif len(selected_features) > 1 and poly_degree == 1:
                    generic_form = "ŷ = b + w₁x₁ + w₂x₂ + ..."
                elif len(selected_features) == 1 and poly_degree > 1:
                    generic_form = "ŷ = b + w₁x + w₂x² + ..."
                else:
                    generic_form = "ŷ = b + Σ wᵢxᵢ"
                
                with st.container(border=True):
                    st.markdown(f"<div style='text-align:center; color:#94a3b8; font-size:0.85rem; letter-spacing:1px; margin-bottom:10px;'><span style='text-transform:uppercase; font-weight:600;'>Syntax:</span> <span style='font-family: monospace; font-size: 0.95rem;'>{generic_form}</span></div>", unsafe_allow_html=True)
                    if len(weights) <= 6:
                        st.latex(equation_str)
                    else:
                        st.latex(f"\\hat{{y}} = {bias:.2f} + \sum_{{i=1}}^{{{len(weights)}}} W_i x_i")
                        st.caption(f"<div style='text-align:center;'>(Showing {len(weights)} polynomial complexity terms)</div>", unsafe_allow_html=True)
                
            st.divider()
                
            with st.container(border=True):

                st.markdown('### Evaluation Metrics')

                table_html = f"""
                <style>
                .metric-table {{ width: 100%; border-collapse: collapse; margin-top: 10px; margin-bottom: 20px; }}
                .metric-table th {{ text-align: left; padding: 10px; border-bottom: 2px solid #334155; color: #94a3b8; font-weight: 600; text-transform: uppercase; font-size: 0.85rem; letter-spacing: 1px; }}
                .metric-table td {{ padding: 10px; border-bottom: 1px solid #1e293b; color: #f1f5f9; }}
                .metric-table tr:hover {{ background-color: #1e293b; transition: background-color 0.2s; }}
                .metric-table tr:last-child td {{ border-bottom: none; }}
                </style>
                <table class="metric-table">
                  <tr>
                    <th>Metric</th>
                    <th>Train</th>
                    <th>Test</th>
                  </tr>
                  <tr>
                    <td>{tooltip('MSE', 'Mean Squared Error: Average of squared differences between predicted and actual values. Lower is better.')}</td>
                    <td>{train_mse:.2f}</td>
                    <td>{test_mse:.2f}</td>
                  </tr>
                  <tr>
                    <td>{tooltip('RMSE', 'Root Mean Squared Error: Square root of MSE. Same units as the target variable.')}</td>
                    <td>{train_rmse:.2f}</td>
                    <td>{test_rmse:.2f}</td>
                  </tr>
                  <tr>
                    <td>{tooltip('R² Score', 'Coefficient of Determination: Proportion of variance explained by the model. 1.0 = perfect, 0.0 = no better than mean.')}</td>
                    <td>{train_r2:.4f}</td>
                    <td>{test_r2:.4f}</td>
                  </tr>
                </table>
                """
                st.markdown(table_html, unsafe_allow_html=True)

                r2_gap = train_r2 - test_r2

                if r2_gap < 0.05:
                    st.success('🟢 Good Fit — Train and Test scores are close. Model generalizes well.')
                elif r2_gap < 0.15:
                    st.warning('🟡 Slight Overfit — Small gap between Train and Test. Monitor complexity.')
                else:
                    st.error('🔴 Overfitting Detected — Large gap between Train and Test. Reduce polynomial degree or features.')
            
            st.divider()

            with st.container(border=True):
                st.markdown(f"### {tooltip('Residual Analysis', 'The difference between actual and predicted values for each point.')}", unsafe_allow_html=True)
                residuals = y_test - y_test_pred
                fig_res = px.histogram(x=residuals, nbins=20, title="Residual Errors (Test Set)", labels={'x': 'Error', 'count': 'Frequency'})
                fig_res.add_vline(x=0, line_color="red", line_dash="dash")
                fig_res.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_res, use_container_width=True)

        with col_viz:
            with st.container(border=True):

                st.subheader("Visualization Space")
                
                if len(selected_features) == 1:
                    # Simple Linear Regression -> 2D Scatter + Curve
                    fig = go.Figure()
                    
                    # Scatter Train
                    fig.add_trace(go.Scatter(x=X_train[:, 0], y=y_train, mode='markers', name='Train Data', marker=dict(color='blue', opacity=0.5)))
                    # Scatter Test
                    fig.add_trace(go.Scatter(x=X_test[:, 0], y=y_test, mode='markers', name='Test Data', marker=dict(color='orange', opacity=0.8)))
                    
                    if show_residuals:
                        for xi, yi_true, yi_pred in zip(X_test[:, 0], y_test, y_test_pred):
                            fig.add_trace(go.Scatter(
                                x=[xi, xi],
                                y=[yi_true, yi_pred],
                                mode='lines',
                                line=dict(color='rgba(100,100,100,0.4)', width=1, dash='dash'),
                                showlegend=False,
                                hoverinfo='skip'
                            ))

                    # Regression Line/Curve mapping
                    x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 200).reshape(-1, 1)
                    
                    if fit_mode == "Manual Fit (Human)":
                        y_range = manual_w * x_range[:, 0] + manual_b
                    else:
                        y_range = model.predict(poly.transform(x_range))
                    
                    fig.add_trace(go.Scatter(x=x_range[:, 0], y=y_range, mode='lines', name='Regression Fit', line=dict(color='red', width=3)))
                    
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
                    
                    if show_residuals:
                        for i in range(len(X_test)):
                            z_pred_point = model.predict(X_test[i].reshape(1, -1))[0]
                            fig.add_trace(go.Scatter3d(
                                x=[X_test[i, 0], X_test[i, 0]],
                                y=[X_test[i, 1], X_test[i, 1]],
                                z=[y_test[i], z_pred_point],
                                mode='lines',
                                line=dict(color='rgba(255,255,255,0.4)', width=2, dash='dot'),
                                showlegend=False,
                                hoverinfo='skip'
                            ))
                    
                    fig.add_trace(go.Surface(
                        x=xx, y=yy, z=zz,
                        colorscale='Viridis',
                        opacity=0.7,
                        name='Prediction Plane',
                        showscale=False
                    ))
                    
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

        # -----------------
        # Live Code Viewer
        # -----------------
        st.divider()
        with st.expander("View Python Code", expanded=False):
            if fit_mode == "Manual Fit (Human)" and len(selected_features) == 1:
                code_str = f'''import pandas as pd
import numpy as np

# Load Data
df = pd.read_csv('auto-mpg.csv')  # example static file
X = df[['{selected_features[0]}']].values
y = df['{target_col}'].values

# Manual Parameters
w = {manual_w}
b = {manual_b}

# Predictions
predictions = w * X + b
'''
            else:
                code_str = f'''import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load Data
df = pd.read_csv('auto-mpg.csv')  # example static file
X = df[{selected_features}].values
y = df['{target_col}'].values

# Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_size}, random_state=42)

# Polynomial mapping
poly = PolynomialFeatures(degree={poly_degree}, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Model Training
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Predictions
predictions = model.predict(X_test_poly)
'''
            st.code(code_str, language="python")
            st.caption("This code updates live as you change the sidebar controls. Copy it to run independently.")

with tab_hood:
    if len(selected_features) == 0:
        st.warning("Please select at least one feature from the sidebar to begin.")
    elif len(selected_features) != 1:
        st.info("Gradient descent visualization is available for single-feature regression only. Please select exactly one feature.")
    else:
        st.markdown("### Method 1: OLS — The Exact Solution")
        st.info("Ordinary Least Squares (OLS) is how sklearn actually solves linear regression. Instead of iterating, it uses a direct formula from linear algebra called the Normal Equation to compute the exact optimal weights in a single step.")
        st.latex(r"\mathbf{w} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}")
        
        X_with_bias = np.column_stack([X_train[:, 0], np.ones(len(X_train))])
        XtX = X_with_bias.T @ X_with_bias
        XtX_inv = np.linalg.inv(XtX)
        Xty = X_with_bias.T @ y_train
        w_ols = XtX_inv @ Xty

        st.code(f"X^T X =\n{XtX}\n\n(X^T X)^-1 =\n{XtX_inv}\n\nX^T y =\n{Xty}", language="plaintext")
        
        st.markdown(f"**Result:** w = {w_ols[0]:.6f}, b = {w_ols[1]:.6f}")
        st.markdown("**Computed in:** 1 step, 0 iterations")
        st.caption(f"sklearn result: w = {model.coef_[0]:.6f}, b = {model.intercept_:.6f} — identical, because sklearn uses OLS internally.")
        
        with st.expander("Why not always use OLS?"):
            st.markdown("OLS requires computing $(\mathbf{X}^T \mathbf{X})^{-1}$, which is a matrix inversion. For small datasets like ours, this is instant. But with millions of rows or thousands of features, matrix inversion becomes very slow and memory-intensive. That is when gradient descent becomes useful — it trades exactness for scalability.")
            
        st.divider()
        
        st.markdown(f"### Method 2: {tooltip('Gradient Descent', 'An optimization algorithm used to minimize the loss function by iteratively moving in the direction of steepest descent.')} — The Iterative Alternative", unsafe_allow_html=True)
        st.info("Gradient descent is an alternative to OLS that finds the same answer through repeated approximation. Instead of solving a matrix equation, it starts with random weights and improves them step by step. It is slower for small problems like ours, but scales to datasets with millions of rows where OLS would run out of memory.")
        
        # Section A: Controls
        ctrl_container = st.container(border=True)
        ctrl_container.markdown('<div class="section-badge">OPTIMIZATION CONTROLS</div>', unsafe_allow_html=True)
        col1, col2, col3 = ctrl_container.columns(3)
        lr = col1.select_slider("Learning Rate", options=[0.0001, 0.001, 0.01, 0.05, 0.1, 0.5], value=0.01, help="The step size taken at each iteration. Too large and it diverges, too small and it takes too long to converge.")
        max_iters = col2.slider("Max Iterations", 10, 500, 100, step=10, help="The maximum number of steps the algorithm will take before stopping.")
        init_weights = col3.radio("Initial Weights", ["Random", "Zero"], index=0, help="The starting values for the parameters before the algorithm begins learning.")
        
        if 'gd_w' not in st.session_state:
            st.session_state.gd_w = np.random.randn() * 0.1 if init_weights == "Random" else 0.0
            st.session_state.gd_b = np.random.randn() * 0.1 if init_weights == "Random" else 0.0
            st.session_state.gd_i = 0
            # initial loss
            preds = st.session_state.gd_w * X_train[:, 0] + st.session_state.gd_b
            st.session_state.loss_history = [(1/len(y_train)) * np.sum((preds - y_train) ** 2)]
            st.session_state.w_history = [st.session_state.gd_w]
            st.session_state.b_history = [st.session_state.gd_b]
            st.session_state.dw = 0.0
            st.session_state.db = 0.0
            
        def reset_gd():
            st.session_state.gd_w = np.random.randn() * 0.1 if init_weights == "Random" else 0.0
            st.session_state.gd_b = np.random.randn() * 0.1 if init_weights == "Random" else 0.0
            st.session_state.gd_i = 0
            preds = st.session_state.gd_w * X_train[:, 0] + st.session_state.gd_b
            st.session_state.loss_history = [(1/len(y_train)) * np.sum((preds - y_train) ** 2)]
            st.session_state.w_history = [st.session_state.gd_w]
            st.session_state.b_history = [st.session_state.gd_b]
            st.session_state.dw = 0.0
            st.session_state.db = 0.0

        if 'last_init_weights' not in st.session_state:
            st.session_state.last_init_weights = init_weights

        if init_weights != st.session_state.last_init_weights:
            st.session_state.last_init_weights = init_weights
            reset_gd()

        c_btn1, c_btn2, c_btn3 = ctrl_container.columns([1, 1, 2])
        if c_btn1.button(" Run Full Training", help="Runs all iterations at once natively reproducing the complete gradient plunge."):
            for _ in range(max_iters):
                old_w, old_b = st.session_state.gd_w, st.session_state.gd_b
                new_w, new_b, loss = gradient_descent_step(X_train[:, 0], y_train, old_w, old_b, lr)
                st.session_state.gd_w, st.session_state.gd_b = new_w, new_b
                st.session_state.dw = (old_w - new_w) / lr
                st.session_state.db = (old_b - new_b) / lr
                st.session_state.loss_history.append(loss)
                st.session_state.w_history.append(new_w)
                st.session_state.b_history.append(new_b)
                st.session_state.gd_i += 1
                
        if c_btn2.button(" Step Once", help="Runs a single mathematical step calculation down the gradient surface."):
            old_w, old_b = st.session_state.gd_w, st.session_state.gd_b
            new_w, new_b, loss = gradient_descent_step(X_train[:, 0], y_train, old_w, old_b, lr)
            st.session_state.gd_w, st.session_state.gd_b = new_w, new_b
            st.session_state.dw = (old_w - new_w) / lr
            st.session_state.db = (old_b - new_b) / lr
            st.session_state.loss_history.append(loss)
            st.session_state.w_history.append(new_w)
            st.session_state.b_history.append(new_b)
            st.session_state.gd_i += 1
            
        if c_btn3.button(" Reset Weights", help="Reinitializes parameters to default."):
            reset_gd()

        # Section C: Live Parameter Display
        st.divider()
        with st.container(border=True):
            st.markdown('<div class="section-badge">LIVE METRICS</div>', unsafe_allow_html=True)
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Current w", f"{st.session_state.gd_w:.6f}", help="The parameter that determines the slope of the regression line.")
            mc2.metric("Current b", f"{st.session_state.gd_b:.6f}", help="The parameter that determines the y-intercept of the regression line.")
            mc3.metric("Current Loss", f"{st.session_state.loss_history[-1]:.4f}", help="Mean Squared Error, a measure of how far off the model's predictions are from the actual values.")
            mc4.metric("Iteration", f"{st.session_state.gd_i}/{max_iters}", help="The current step in the optimization sequence.")
            
            st.caption(f"sklearn optimal: w = {model.coef_[0]:.6f}, b = {model.intercept_:.6f}, MSE = {train_mse:.4f}")
            
            st.latex(rf"w \leftarrow w - \alpha \cdot \frac{{\partial L}}{{\partial w}} = {st.session_state.w_history[-2] if len(st.session_state.w_history)>1 else st.session_state.gd_w:.4f} - {lr} \times {st.session_state.dw:.4f} = {st.session_state.gd_w:.4f}")
            st.latex(rf"b \leftarrow b - \alpha \cdot \frac{{\partial L}}{{\partial b}} = {st.session_state.b_history[-2] if len(st.session_state.b_history)>1 else st.session_state.gd_b:.4f} - {lr} \times {st.session_state.db:.4f} = {st.session_state.gd_b:.4f}")

        # Section D: Animating Plots
        st.divider()
        anim_container = st.container(border=True)
        anim_container.markdown('<div class="section-badge">INTERACTIVE ANIMATOR</div>', unsafe_allow_html=True)
        col_plot1, col_plot2 = anim_container.columns(2)
        with col_plot1:
            st.markdown(f"**{tooltip('Regression Fit (Animated)', 'The regression line drawn using the current w and b values from gradient descent. Watch it rotate toward the optimal fit.')}**", unsafe_allow_html=True)
            
            if len(st.session_state.w_history) > 1:
                hist_idx = st.slider("Scrub Iteration History", 0, len(st.session_state.w_history)-1, len(st.session_state.w_history)-1)
                display_w = st.session_state.w_history[hist_idx]
                display_b = st.session_state.b_history[hist_idx]
            else:
                display_w = st.session_state.gd_w
                display_b = st.session_state.gd_b
            
            fig_anim = go.Figure()
            fig_anim.add_trace(go.Scatter(x=X_train[:, 0], y=y_train, mode='markers', name='Train Data', marker=dict(color='blue', opacity=0.5)))
            
            t_x = np.array([X_train[:, 0].min(), X_train[:, 0].max()])
            t_y_gd = display_w * t_x + display_b
            fig_anim.add_trace(go.Scatter(x=t_x, y=t_y_gd, mode='lines', name='GD Current', line=dict(color='red', width=3)))
            
            t_y_sk = model.coef_[0] * t_x + model.intercept_
            fig_anim.add_trace(go.Scatter(x=t_x, y=t_y_sk, mode='lines', name='Optimal (sklearn)', line=dict(color='green', width=1, dash='dash')))
            
            fig_anim.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_anim, use_container_width=True)

        with col_plot2:
            st.markdown(f"**{tooltip('Loss Curve', 'Shows how the error (MSE) decreases with each iteration of gradient descent. The dashed red line is the sklearn optimal.')}**", unsafe_allow_html=True)
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(y=st.session_state.loss_history, mode='lines', name='Loss', line=dict(color='blue')))
            fig_loss.add_hline(y=sklearn_train_mse, line_dash='dash', line_color='red', annotation_text='sklearn MSE')
            fig_loss.add_trace(go.Scatter(x=[len(st.session_state.loss_history)-1], y=[st.session_state.loss_history[-1]], mode='markers', marker=dict(color='red', size=8), name='Current'))
            fig_loss.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="Iteration", yaxis_title="MSE Loss")
            st.plotly_chart(fig_loss, use_container_width=True)

        # Section E: Insights
        if st.session_state.gd_i > 0:
            initial_loss = st.session_state.loss_history[0]
            current_loss = st.session_state.loss_history[-1]
            if current_loss > initial_loss * 10:
                st.error(" Diverged! The learning rate is too high. The gradients are overshooting the minimum. Try a smaller learning rate.")
            elif abs(current_loss - sklearn_train_mse) / sklearn_train_mse < 0.05:
                st.success(" Converged! Gradient descent found weights very close to the optimal solution. The remaining gap is due to stopping early.")
            elif current_loss < st.session_state.loss_history[-2] - 0.0001:
                st.info(" Still converging — try more iterations or a higher learning rate.")
            else:
                st.warning(" Stuck. The learning rate might be too small, or more iterations are needed.")

        # Section F / Loss Surface Visualization
        st.divider()
        st.markdown(f"### {tooltip('Loss Surface', 'A visual representation of the loss for every possible combination of weight and bias.')}: The Landscape Gradient Descent Navigates", unsafe_allow_html=True)
        st.info("Every possible combination of weight (w) and bias (b) produces a different MSE loss. OLS jumps directly to the bottom of this bowl. Gradient descent has to find it by rolling downhill. Both arrive at the same point — the global minimum — but through very different paths.")
        
        opt_w, opt_b = model.coef_[0], model.intercept_
        spread_w = max(abs(opt_w) * 3, 0.5)
        spread_b = max(abs(opt_b) * 0.8, 10.0)
        
        w_range, b_range, Loss_grid = compute_loss_surface(X_train[:, 0], y_train, opt_w, opt_b, spread_w, spread_b)
        
        surf_container = st.container(border=True)
        surf_container.markdown('<div class="section-badge">TOPOLOGY VIEWERS</div>', unsafe_allow_html=True)
        col_surf1, col_surf2 = surf_container.columns([1, 1])
        with col_surf1:
            st.markdown(f"**{tooltip('3D Loss Surface', 'The bowl-shaped surface shows MSE for every possible (w, b) combination. Gradient descent rolls downhill to find the lowest point.')}**", unsafe_allow_html=True)
            fig_surf = go.Figure(data=[go.Surface(x=w_range, y=b_range, z=Loss_grid, colorscale='Viridis', opacity=0.85)])
            fig_surf.add_trace(go.Scatter3d(x=[opt_w], y=[opt_b], z=[sklearn_train_mse], mode='markers', marker=dict(size=6, color='red', symbol='diamond'), name='Global Minimum'))
            if len(st.session_state.loss_history) > 1:
                fig_surf.add_trace(go.Scatter3d(x=st.session_state.w_history, y=st.session_state.b_history, z=st.session_state.loss_history, mode='lines+markers', line=dict(color='red', width=3), marker=dict(size=2, color='red'), name='GD Path'))
            fig_surf.update_layout(scene=dict(xaxis_title='Weight (w)', yaxis_title='Bias (b)', zaxis_title='MSE Loss', camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))), height=500, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_surf, use_container_width=True)
            
        with col_surf2:
            st.markdown(f"**{tooltip('Contour Map (Top-Down View)', 'A top-down view of the loss surface, like a topographic map. Each ring represents a constant error level. The red path shows the route gradient descent took.')}**", unsafe_allow_html=True)
            fig_cont = go.Figure(data=[go.Contour(x=w_range, y=b_range, z=Loss_grid, colorscale='Viridis', ncontours=25)])
            fig_cont.add_trace(go.Scatter(x=[opt_w], y=[opt_b], mode='markers', marker=dict(size=12, color='red', symbol='star'), name='Optimal'))
            if len(st.session_state.loss_history) > 1:
                fig_cont.add_trace(go.Scatter(x=st.session_state.w_history, y=st.session_state.b_history, mode='lines+markers', line=dict(color='red', width=2), marker=dict(size=6, symbol='arrow-right'), name='Path'))
                fig_cont.add_trace(go.Scatter(x=[st.session_state.w_history[0]], y=[st.session_state.b_history[0]], mode='markers', marker=dict(size=10, color='yellow'), name='Start'))
                fig_cont.add_trace(go.Scatter(x=[st.session_state.w_history[-1]], y=[st.session_state.b_history[-1]], mode='markers', marker=dict(size=10, color='green'), name='End'))
            fig_cont.update_layout(xaxis_title='Weight (w)', yaxis_title='Bias (b)', height=500, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_cont, use_container_width=True)
            
        st.markdown(f"The surface above shows MSE loss for {len(w_range) * len(b_range):,} different (w, b) combinations. The red diamond marks the minimum loss of **{sklearn_train_mse:.4f}** at `w={opt_w:.6f}`, `b={opt_b:.6f}`.")
        if len(st.session_state.loss_history) > 1:
            try: gap = abs(st.session_state.loss_history[-1] - sklearn_train_mse) / sklearn_train_mse 
            except: gap = 0
            st.success(f"**Path Tracking:** The red path shows how gradient descent navigated from its starting point to reach within **{gap:.2%}** of the optimal solution in **{st.session_state.gd_i}** steps.")
