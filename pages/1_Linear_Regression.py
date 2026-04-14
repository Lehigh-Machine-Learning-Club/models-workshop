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
import os

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

/* TABS HIGHLIGHT ACCENT OVERRIDES */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: transparent;
}
.stTabs [data-baseweb="tab"] {
    background-color: #1e293b;
    border-radius: 8px 8px 0px 0px !important;
    padding: 12px 24px !important;
    border: 1px solid #334155;
    border-bottom: none;
    transition: all 0.2s ease-in-out;
}
.stTabs [aria-selected="true"] {
    background-color: #38bdf8 !important;
    color: #0f172a !important;
    font-weight: 800 !important;
    border-color: #38bdf8 !important;
    text-shadow: none;
    box-shadow: 0 -4px 12px rgba(56, 189, 248, 0.2);
}
.stTabs [aria-selected="false"]:hover {
    background-color: #334155 !important;
    color: #f8fafc !important;
}

/* NATIVE HOVER DEFINITION STYLING */
div[data-testid="stTooltipContent"] {
    background-color: #0f172a !important;
    color: #e0f2fe !important;
    border: 1px solid #38bdf8 !important;
    border-radius: 6px !important;
    padding: 14px 16px !important;
    box-shadow: 0 4px 16px rgba(56, 189, 248, 0.25) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    line-height: 1.5 !important;
}
</style>
""", unsafe_allow_html=True)

st.title("Linear, Polynomial & Multiple Regression")
st.markdown('<div class="title-accent"></div>', unsafe_allow_html=True)

st.markdown("""
**In this workshop:**  
✦ Discover the pattern in data and formalize it into a prediction formula  
✦ Manually fit a regression line and try to beat the algorithm  
✦ Understand how the model finds the best weights (OLS vs gradient descent)  
✦ See overfitting happen in real time as model complexity increases  
✦ Walk away with sklearn code you can copy to your own projects  
""")
st.divider()

def tooltip(term, text):
    return f'<span class="tooltip-term" style="font-weight:600;">{term}<span class="tooltip-text">{text}</span></span>'

@st.cache_data
def load_data():
    local_path = "auto-mpg.csv"
    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
        # Normalize target column name
        if 'class' in df.columns and 'mpg' not in df.columns:
            df = df.rename(columns={'class': 'mpg'})
        
        target_col = 'mpg'
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        features = [c for c in numeric_cols if c != target_col]
        return df, features, target_col
    try:
        data = fetch_openml(name='autoMpg', version=1, parser='auto')
        df = data.frame.dropna()
        
        # Normalize target column name
        if 'class' in df.columns and 'mpg' not in df.columns:
            df = df.rename(columns={'class': 'mpg'})
            
        target_col = 'mpg'

        if df[target_col].dtype.name == 'category':
            df[target_col] = df[target_col].astype(float)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Save locally for future
        df.to_csv(local_path, index=False)
        
        features = [c for c in numeric_cols if c != target_col]
        return df, features, target_col
    except Exception as e:
        np.random.seed(42)
        X1 = np.random.uniform(10, 50, 200)
        X2 = np.random.uniform(5, 25, 200)
        Y = 2.5 * X1 - 1.2 * X2 + 15 + np.random.normal(0, 10, 200)
        df = pd.DataFrame({'Feature_1': X1, 'Feature_2': X2, 'mpg': Y})
        return df, ['Feature_1', 'Feature_2'], 'mpg'

df, features_list, target_col = load_data()

# Global Baseline
mean_mpg = df[target_col].mean()
baseline_mse = np.mean((df[target_col] - mean_mpg) ** 2)
baseline_rmse = np.sqrt(baseline_mse)

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
# Sidebar & Configuration
# -----------------
st.sidebar.header("Model Controls")

selected_features = st.sidebar.multiselect("Features (X):", features_list, default=[features_list[0]], help="The input variables the model will use to predict the target. Select multiple for Multiple Regression.")

with st.sidebar.expander("Advanced Settings"):
    poly_degree = st.slider("Polynomial Degree", 1, 10, 1, help="The highest power of the feature used. Degree 1 is a straight line, Degree 2 allows one curve, etc.")
    test_size = st.slider("Test Split %", 10, 50, 20, help="The percentage of data hidden from the model during training, used later to evaluate its accuracy safely.") / 100.0

if len(selected_features) == 0:
    st.warning("Please select at least one feature.")
    st.stop()

# -----------------
# Global Model Training (powers all tabs)
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



# -----------------
# Layout Tabs
# -----------------
tab_data, tab_foundations, tab_model, tab_hood = st.tabs(
    ["The Problem", "How Regression Works", "Build & Evaluate", "How the Algorithm Learns"]
)

with tab_data:
    # 1.1 The Challenge
    st.subheader("The Challenge")
    st.markdown("Here's a car from 1978. It weighs 3,504 lbs, has 130 horsepower, and a 307 cubic-inch engine.\n\n**How many miles per gallon does it get?**\n\nYou probably have an intuition — heavier car, worse mileage. But can we turn that intuition into a precise formula? One that works for ANY car in the dataset?\n\nThat's regression: turning patterns in data into a prediction machine.")
    
    with st.container(border=True):
        col_guess, col_reveal = st.columns(2)
        with col_guess:
            student_guess = st.number_input("Your guess (mpg):", min_value=5.0, max_value=60.0, value=20.0, step=0.5)
            reveal_btn = st.button("Reveal actual mpg", type="primary")
        with col_reveal:
            if reveal_btn:
                st.metric("Actual MPG", "18.0")
                st.metric("Your error", f"{abs(student_guess - 18.0):.1f} mpg off")
                st.markdown(f"A regression model trained on this data gets within **~{baseline_rmse:.1f} mpg** on average. Let's build one.")

    st.divider()

    # 1.2 Dataset at a glance
    st.subheader("The Dataset at a Glance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Cars", len(df), help="The total number of samples (rows) in our dataset.")
    c2.metric("Features Available", f"{len(features_list)} numeric", help="The columns (like weight, horsepower) we can use to make our prediction.")
    c3.metric("Target Variable", f"{target_col}", help="The variable we are trying to predict.")
    c4.metric("Target Range", f"{df[target_col].min()} - {df[target_col].max()}", help="The minimum and maximum values of our target variable.")

    with st.container(border=True):
        st.markdown("**Quick vocabulary check:**\nThe columns we use to make predictions (weight, horsepower, etc.) are called **features**.You'll also see them called independent variables, predictors, or just `X`.\nThe column we're trying to predict (mpg) is called the **target** — also known as the dependent variable, response, label, or `y`.\n\nSame concepts, different names depending on who's talking.")
        st.markdown("""
        | Context | Input Columns | Output Column |
        |---------|--------------|---------------|
        | Textbook | Independent variables | Dependent variable |
        | sklearn code | `X` | `y` |
        | Data team | Features | Target |
        | Research paper | Predictors / Covariates | Response |
        | ML Course | Features | Label |
        """)

    st.divider()

    # 1.3 See the Pattern Before the Math
    st.subheader("See the Pattern Before the Math")
    explore_col = st.selectbox("Pick a feature to plot against MPG:", features_list, index=features_list.index('weight') if 'weight' in features_list else 0)
    
    col_plot, col_corr = st.columns([3, 1])
    with col_plot:
        fig_explore = px.scatter(df, x=explore_col, y=target_col, title=f"{explore_col} vs {target_col}")
        fig_explore.update_traces(marker=dict(color='rgba(59, 130, 246, 0.6)', line=dict(color='white', width=1.5)))
        fig_explore.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), plot_bgcolor='rgba(15, 23, 42, 0.6)', paper_bgcolor='rgba(15, 23, 42, 0.6)')
        with st.container(border=True):
            st.plotly_chart(fig_explore, use_container_width=True)
    
    with col_corr:
        corr = df[explore_col].corr(df[target_col])
        st.metric(f"Correlation with {target_col}", f"{corr:.3f}", help="Correlation measures the strength and direction of the linear relationship between two variables. 1.0 is perfect positive, -1.0 is perfect negative, and 0 is no linear relationship.")
        
        if abs(corr) > 0.7:
            st.success(f"Strong relationship! `{explore_col}` looks like a powerful predictor.")
        elif abs(corr) > 0.4:
            st.info(f"Moderate relationship. `{explore_col}` has some predictive power.")
        else:
            st.warning(f"Weak relationship. `{explore_col}` alone may not predict `{target_col}` well.")
            
    st.markdown(f"> *Does the relationship between `{explore_col}` and `{target_col}` look like a straight line? Or is there a curve? Keep this in mind — it'll matter when we choose between linear and polynomial regression.*")

    st.divider()

    # 1.4 The Baseline
    st.subheader("The Baseline: How Bad is 'Dumb'?")

    st.markdown(f"""
    The simplest "model" is no model at all — just guess the average {target_col} 
    (**{mean_mpg:.1f}**) for every car. But how do we measure how wrong that guess is?
    """)

    with st.expander(" What is RMSE?", expanded=True):
        st.markdown("""
    **RMSE (Root Mean Squared Error)** measures the average size of your prediction mistakes, in the same units as the target.

    Here's how it works, step by step:
    1. For each car, calculate the **error**: `actual mpg − predicted mpg`
    2. **Square** each error (so negatives don't cancel out positives)
    3. Take the **mean** (average) of all squared errors
    4. Take the **square root** to get back to mpg units

    The result is a single number that says: *"On average, the prediction is off by this many mpg."*
        """)
        st.latex(r"RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (actual_i - predicted_i)^2}")
        st.markdown("**Lower RMSE = better model.** An RMSE of 0 would mean perfect prediction (unrealistic).")

    st.markdown(f"""
    Using the "always guess the average" strategy, our RMSE is **{baseline_rmse:.1f} {target_col}**.

    That's our bar to beat. Every model we build needs a LOWER RMSE than {baseline_rmse:.1f}. 
    If it can't, it's literally worse than guessing the average.
    """)

    fig_base = go.Figure()
    fig_base.add_trace(go.Scatter(x=df[explore_col], y=df[target_col], mode='markers', name='Data', marker=dict(color='rgba(59, 130, 246, 0.6)', line=dict(color='white', width=1.5))))
    fig_base.add_hline(y=mean_mpg, line_dash='dash', line_color='red', annotation_text=f"Mean Guess ({mean_mpg:.1f})")
    
    # Draw residuals for the first 50 points to save memory and DOM size
    for _, row in df.sample(60, random_state=42).iterrows():
        fig_base.add_trace(go.Scatter(
            x=[row[explore_col], row[explore_col]],
            y=[row[target_col], mean_mpg],
            mode='lines', line=dict(color='#fbbf24', width=2, dash='dot'), showlegend=False, hoverinfo='skip'
        ))
    
    fig_base.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
    st.plotly_chart(fig_base, use_container_width=True)
    
    st.caption("The vertical lines show how wrong the 'always guess the average' strategy is for each car. Our goal: make those lines shorter.")

    st.divider()

    # 1.5 Column Reference
    st.subheader("Data Dictionary")
    desc_map = {
        'mpg': "Miles per gallon — the target variable we predict",
        'cylinders': "Number of engine cylinders — more cylinders usually means lower mpg",
        'displacement': "Engine displacement in cubic inches — larger engines burn more gas",
        'horsepower': "Engine horsepower — more power usually means more fuel burned",
        'weight': "Vehicle weight in lbs — heavier cars tend to get worse mileage",
        'acceleration': "0-60 mph time in seconds — slower acceleration often correlates with better mpg",
        'model_year': "Model year (e.g. 70 = 1970) — newer cars tend to be more efficient",
        'origin': "Origin of the car (1: USA, 2: Europe, 3: Japan)"
    }
    
    col_details = []
    for col in df.columns:
        desc = desc_map.get(col, "Feature column")
        col_details.append({
            "Column": col,
            "Type": str(df[col].dtype),
            "Range": f"{df[col].min():.1f} - {df[col].max():.1f}" if pd.api.types.is_numeric_dtype(df[col]) else "N/A",
            "Description": desc
        })
    st.dataframe(pd.DataFrame(col_details), use_container_width=True, hide_index=True)
    
    st.markdown("**Sample Data (First 5 Rows)**")
    st.dataframe(df.head(5), use_container_width=True)
    
    st.info("💡 **Ready?** We've seen the data and the pattern. Now let's formalize it into math → **How Regression Works**")

with tab_foundations:
    # 2.1 The Simplest Model: One Feature, One Line
    st.subheader("2.1 The Simplest Model: One Feature, One Line")
    st.markdown("In **Simple Linear Regression**, we use only **one feature** to predict the target. The model assumes the relationship is a straight line.")
    
    col_21A, col_21B = st.columns([1, 1])
    with col_21A:
        with st.container(border=True):
            st.markdown(f"**{tooltip('The Equation', 'The foundational formula that calculates the mathematical prediction mapping inputs to outputs.')}**", unsafe_allow_html=True)
            st.latex(r"\text{predicted} = b + w \times \text{feature}")
            st.markdown(f"- $b$ is the **bias** (y-intercept). \n- $w$ is the **weight** (slope).")
    
    with col_21B:
        with st.container(border=True):
            row1 = df.iloc[0]
            st.markdown(f"**{tooltip('Example Calculation', 'Running actual data from our dataset through the equation above.')}**", unsafe_allow_html=True)
            st.markdown(f"For earliest {explore_col} (**{row1[explore_col]}**), if $b=30$ and $w=-0.01$:")
            st.code(f"predicted = 30 + (-0.01 * {row1[explore_col]})\n          = {30 - 0.01*row1[explore_col]:.1f} mpg", language="python")
            st.markdown(f"*Actual was **{row1[target_col]}** mpg. Off by **{abs(row1[target_col] - (30 - 0.01*row1[explore_col])):.1f}**!*")
    
    st.divider()

    # 2.2 Types of Regression
    st.subheader("2.2 Types of Regression")
    col1, col2, col3 = st.columns(3)
    np.random.seed(42)

    with col1:
        x_lin = np.linspace(0, 10, 30)
        y_lin = 2 * x_lin + 3 + np.random.randn(30) * 2
        fig_lin = go.Figure()
        fig_lin.add_trace(go.Scatter(x=x_lin, y=y_lin, mode='markers', name='Data', marker=dict(color='rgba(59, 130, 246, 0.6)', line=dict(color='white', width=1.5))))
        fig_lin.add_trace(go.Scatter(x=x_lin, y=2*x_lin+3, mode='lines', name='Fit', line=dict(color='red', width=3)))
        fig_lin.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), showlegend=False, title="Simple Linear Regression", plot_bgcolor='rgba(15, 23, 42, 0.6)', paper_bgcolor='rgba(15, 23, 42, 0.6)')
        with st.container(border=True):
            st.plotly_chart(fig_lin, use_container_width=True)
        st.latex(r"y = w \cdot x + b")
        st.caption("One feature, one straight line. The simplest form.")

    with col2:
        x_poly = np.linspace(0, 10, 30)
        y_poly = 0.5 * x_poly**2 - 3 * x_poly + 10 + np.random.randn(30) * 2
        fig_poly = go.Figure()
        fig_poly.add_trace(go.Scatter(x=x_poly, y=y_poly, mode='markers', name='Data', marker=dict(color='rgba(59, 130, 246, 0.6)', line=dict(color='white', width=1.5))))
        fig_poly.add_trace(go.Scatter(x=x_poly, y=0.5*x_poly**2 - 3*x_poly + 10, mode='lines', name='Fit', line=dict(color='red', width=3)))
        fig_poly.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), showlegend=False, title="Polynomial Regression", plot_bgcolor='rgba(15, 23, 42, 0.6)', paper_bgcolor='rgba(15, 23, 42, 0.6)')
        with st.container(border=True):
            st.plotly_chart(fig_poly, use_container_width=True)
        st.latex(r"y = w_2 x^2 + w_1 x + b")
        st.caption("Still one feature, but the relationship is curved. We add polynomial terms (x², x³...) to capture this.")

    with col3:
        x1_mult = np.random.rand(30) * 10
        x2_mult = np.random.rand(30) * 10
        y_mult = 3 * x1_mult + 2 * x2_mult + 5 + np.random.randn(30) * 2
        fig_mult = go.Figure()
        fig_mult.add_trace(go.Scatter3d(x=x1_mult, y=x2_mult, z=y_mult, mode='markers', marker=dict(size=4, color='rgba(59, 130, 246, 0.6)', line=dict(color='white', width=1.5))))
        
        xx_mult, yy_mult = np.meshgrid(np.linspace(0, 10, 10), np.linspace(0, 10, 10))
        zz_mult = 3 * xx_mult + 2 * yy_mult + 5
        fig_mult.add_trace(go.Surface(x=xx_mult, y=yy_mult, z=zz_mult, colorscale='Reds', opacity=0.5, showscale=False))
        fig_mult.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), showlegend=False, title="Multiple Regression", plot_bgcolor='rgba(15, 23, 42, 0.6)', paper_bgcolor='rgba(15, 23, 42, 0.6)')
        with st.container(border=True):
            st.plotly_chart(fig_mult, use_container_width=True)
        st.latex(r"y = w_1 x_1 + w_2 x_2 + b")
        st.caption("Two or more features. The line becomes a plane (or hyperplane in higher dimensions).")

    st.divider()

    # 2.3 Understanding Residuals
    st.subheader("2.3 Understanding Residuals")
    np.random.seed(42)
    x_res = np.linspace(1, 9, 8)
    y_true_res = 2 * x_res + 3
    y_noisy_res = y_true_res + np.random.randn(8) * 3
    
    fig_res = go.Figure()
    fig_res.add_trace(go.Scatter(x=x_res, y=y_true_res, mode='lines', name='Regression Line', line=dict(color='red', width=3)))
    fig_res.add_trace(go.Scatter(x=x_res, y=y_noisy_res, mode='markers', name='Actual Data', marker=dict(color='rgba(59, 130, 246, 0.8)', size=8, line=dict(color='white', width=1.5))))
    
    # Draw residual lines
    for i in range(len(x_res)):
        fig_res.add_trace(go.Scatter(
            x=[x_res[i], x_res[i]], y=[y_noisy_res[i], y_true_res[i]],
            mode='lines', line=dict(color='#fbbf24', dash='dot', width=2), showlegend=False
        ))
        
    # Formula annotation placed explicitly in the bottom left away from plot overlaps
    fig_res.add_annotation(
        xref="paper", yref="paper",
        x=0.98, y=0.05,
        xanchor="right",
        text="<b>Residual = Actual − Predicted</b>",
        showarrow=False,
        font=dict(color='#fbbf24', size=13),
        bgcolor="rgba(15, 23, 42, 0.85)",
        bordercolor="#fbbf24",
        borderwidth=1,
        borderpad=6
    )
    fig_res.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0), plot_bgcolor='rgba(15, 23, 42, 0.6)', paper_bgcolor='rgba(15, 23, 42, 0.6)')
    with st.container(border=True):
        st.plotly_chart(fig_res, use_container_width=True)
    
    st.latex(r"e_i = y_i - \hat{y}_i")
    st.markdown("A residual is the vertical distance between an actual data point and the regression line. If the dot is ABOVE the line, the model guessed too low (positive residual). If the dot is BELOW the line, the model guessed too high (negative residual). The model tries to make these as small as possible.")

    st.divider()

    # 2.4 What "Best" Means: The Loss Function
    st.subheader("2.4 What 'Best' Means: The Loss Function")
    st.markdown("We just saw the residuals (errors). A **Loss Function** takes all of those individual errors across the entire dataset and mathematically squashes them down into a **single score**. The algorithm's only goal is to find a line that gives the lowest score possible.\n\nThe most common loss function for regression is **Mean Squared Error (MSE)**.")
    
    col_24A, col_24B = st.columns([1, 1.2])
    with col_24A:
        with st.container(border=True):
            st.markdown(f"**{tooltip('Mean Squared Error (MSE)', 'A metric that computes the average of the squared mathematical differences between actual and algorithmic predicted values.')}**", unsafe_allow_html=True)
            st.latex(r"\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2")
    
    with col_24B:
        with st.container(border=True):
            st.markdown("**What is this math actually doing?**\n1. **Find the residue**: Take the actual value ($y_i$) minus the predicted value ($\hat{y}_i$).\n2. **Square it**: Multiply it by itself. This makes all negative numbers positive, AND it heavily punishes *large* errors (an error of 4 is 16x worse than an error of 1).\n3. **Average everything**: Add them all up ($\sum$) and divide by the number of cars ($n$) so the score is fair regardless of dataset size.")
   
    with st.expander("Why Square it instead of just using Absolute Math? (MSE vs MAE)"):
        st.write("We *could* just look at the absolute distance (Mean Absolute Error). But squaring the errors explicitly teaches the model: *'I would rather be slightly wrong on ten points than massively wrong on one exact point.'* It forces the regression line to be a smooth compromise that hugs the center of the entire dataset avoiding extreme outliers.", unsafe_allow_html=True)

    # 2.5 Can YOU Beat the Algorithm?
    st.subheader("2.5 Can YOU Beat the Algorithm?")
    st.markdown("Adjust the weight ($w$) and bias ($b$) to minimize your MSE. A perfect model will have an MSE of 0. Sklearn's optimal algorithm runs invisibly in the background. See how close you can get!")
    
    col_plot2, col_ctrl2 = st.columns([2, 1])
    
    # Need standardized limits based on selected_features[0]
    std_w = (df[target_col].max() - df[target_col].min()) / (df[selected_features[0]].max() - df[selected_features[0]].min() + 1e-5)
    std_b = df[target_col].mean()
    
    with col_ctrl2:
        # Calculate isolated single-feature optimal for comparison early so callbacks can access it
        opt_w, opt_b = np.polyfit(df[selected_features[0]], df[target_col], 1)
        opt_mse = np.mean((df[target_col] - (opt_w * df[selected_features[0]] + opt_b))**2)
        
        use_optimal = st.toggle("Auto-solve (Snap to Optimal)", help="Automatically snap your Weight and Bias to their exact mathematically perfect values.")
        
        if 'man_reg_w' not in st.session_state:
            st.session_state.man_reg_w = 0.0
        if 'man_reg_b' not in st.session_state:
            st.session_state.man_reg_b = 0.0
            
        # Dynamically secure bounding boxes so sliders never crash when attempting to adopt optimal shapes
        min_w_bound = min(-abs(std_w)*10, float(opt_w) - abs(opt_w)*1.0 - 0.01)
        max_w_bound = max(abs(std_w)*10, float(opt_w) + abs(opt_w)*1.0 + 0.01)
        
        min_b_bound = min(std_b - 100.0, float(opt_b) - abs(opt_b)*0.5 - 20)
        max_b_bound = max(std_b + 100.0, float(opt_b) + abs(opt_b)*0.5 + 20)
            
        if use_optimal:
            tab_manual_w = st.slider("Weight (w) Slope", min_w_bound, max_w_bound, value=float(opt_w), format="%.4f", disabled=True, key="opt_w_locked", help="The weight determines the slope (angle) of the line.")
            tab_manual_b = st.slider("Bias (b) Y-Intercept", min_b_bound, max_b_bound, value=float(opt_b), format="%.1f", disabled=True, key="opt_b_locked", help="The bias determines where the line crosses the y-axis.")
        else:
            tab_manual_w = st.slider("Weight (w) Slope", min_w_bound, max_w_bound, key="man_reg_w", format="%.4f", disabled=False, help="The weight determines the slope (angle) of the line.")
            tab_manual_b = st.slider("Bias (b) Y-Intercept", min_b_bound, max_b_bound, key="man_reg_b", format="%.1f", disabled=False, help="The bias determines where the line crosses the y-axis.")
        
        tab_train_pred = tab_manual_w * df[selected_features[0]] + tab_manual_b
        target_mse = np.mean((df[target_col] - tab_train_pred)**2)
        
        st.metric("Your MSE", f"{target_mse:.2f}", help="Mean Squared Error (MSE) measures the average squared distance from your line to the actual data points. Try to bring this number as close to 0 as possible.")
        
        show_opt = st.toggle("Show algorithm's optimal line", help="Reveal the mathematically perfect line calculated by the machine learning algorithm.")
        if show_opt:
            st.metric("Algorithm MSE", f"{opt_mse:.2f}", help="The absolute lowest possible MSE achievable for a straight line on this data.")
            gap = target_mse - opt_mse
            st.metric("The Gap", f"{gap:.2f}", help="The difference between your MSE and the mathematically perfect baseline. A gap of 0 means you beat the algorithm!")
            if gap < opt_mse * 0.1:
                st.success("Outstanding! You nearly matched the algorithm perfectly!")
            elif gap < opt_mse * 0.5:
                st.info("Close! Try tweaking the slope slightly.")
            else:
                st.warning("Keep going — the gap is still large.")
    
    with col_plot2:
        fig_man = go.Figure()
        fig_man.add_trace(go.Scatter(x=df[selected_features[0]], y=df[target_col], mode='markers', name='Data', marker=dict(color='rgba(56, 189, 248, 0.6)', line=dict(color='white', width=1.5))))
        
        line_x = np.array([df[selected_features[0]].min(), df[selected_features[0]].max()])
        fig_man.add_trace(go.Scatter(x=line_x, y=tab_manual_w * line_x + tab_manual_b, mode='lines', name='Your Line', line=dict(color='#ef4444', width=3)))
        
        if show_opt:
            fig_man.add_trace(go.Scatter(x=line_x, y=opt_w * line_x + opt_b, mode='lines', name='Sklearn Optimal', line=dict(color='#22c55e', width=3, dash='dash')))
            
        fig_man.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), showlegend=True, plot_bgcolor='rgba(15, 23, 42, 0.6)', paper_bgcolor='rgba(15, 23, 42, 0.6)', xaxis_title=selected_features[0], yaxis_title=target_col)
        with st.container(border=True):
            st.plotly_chart(fig_man, use_container_width=True)

    st.divider()

    # 2.6 When Lines Aren't Enough: Polynomial Regression
    st.subheader("2.6 When Lines Aren't Enough: Polynomial Regression")
    st.markdown("Sometimes, data curves. A straight line will never fit a curve perfectly. By adding polynomial terms ($x^2$, $x^3$), the model can bend to follow the data.")
    
    # 2.7 Overfitting Danger
    st.subheader("2.7 The Danger of Too Much Flexibility: Overfitting")
    st.markdown("As you increase polynomial degree, the model becomes more flexible. But more flexibility isn\'t always better — it can start memorizing noise instead of learning the true pattern.")
    
    col_u, col_g, col_o = st.columns(3)
    np.random.seed(42)
    x_cmplx = np.linspace(0, 10, 20)
    y_cmplx = 0.4 * x_cmplx**2 - 2 * x_cmplx + 5 + np.random.randn(20) * 1.5
    x_smooth = np.linspace(0, 10, 100)
    
    with col_u:
        st.markdown("**Too Simple (Underfitting)**")
        p1 = np.polyfit(x_cmplx, y_cmplx, 1)
        fig_u = go.Figure()
        fig_u.add_trace(go.Scatter(x=x_cmplx, y=y_cmplx, mode='markers', marker=dict(color='rgba(59, 130, 246, 0.6)', size=7, line=dict(color='white', width=1.5))))
        fig_u.add_trace(go.Scatter(x=x_smooth, y=np.polyval(p1, x_smooth), mode='lines', line=dict(color='#f59e0b', width=3)))
        fig_u.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0), showlegend=False, plot_bgcolor='rgba(15, 23, 42, 0.6)', paper_bgcolor='rgba(15, 23, 42, 0.6)')
        with st.container(border=True):
            st.plotly_chart(fig_u, use_container_width=True)
        st.error("Misses the pattern (**High Bias**)")
        
    with col_g:
        st.markdown("**Just Right**")
        p2 = np.polyfit(x_cmplx, y_cmplx, 2)
        fig_g = go.Figure()
        fig_g.add_trace(go.Scatter(x=x_cmplx, y=y_cmplx, mode='markers', marker=dict(color='rgba(59, 130, 246, 0.6)', size=7, line=dict(color='white', width=1.5))))
        fig_g.add_trace(go.Scatter(x=x_smooth, y=np.polyval(p2, x_smooth), mode='lines', line=dict(color='#22c55e', width=3)))
        fig_g.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0), showlegend=False, plot_bgcolor='rgba(15, 23, 42, 0.6)', paper_bgcolor='rgba(15, 23, 42, 0.6)')
        with st.container(border=True):
            st.plotly_chart(fig_g, use_container_width=True)
        st.success("Captures true trend")
        
    with col_o:
        st.markdown("**Too Complex (Overfitting)**")
        p10 = np.polyfit(x_cmplx, y_cmplx, 10)
        fig_o = go.Figure()
        fig_o.add_trace(go.Scatter(x=x_cmplx, y=y_cmplx, mode='markers', marker=dict(color='rgba(59, 130, 246, 0.6)', size=7, line=dict(color='white', width=1.5))))
        fig_o.add_trace(go.Scatter(x=x_smooth, y=np.polyval(p10, x_smooth), mode='lines', line=dict(color='#ef4444', width=3)))
        fig_o.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0), showlegend=False, plot_bgcolor='rgba(15, 23, 42, 0.6)', paper_bgcolor='rgba(15, 23, 42, 0.6)')
        with st.container(border=True):
            st.plotly_chart(fig_o, use_container_width=True)
        st.error("Memorizes noise (**High Variance**)")

    st.divider()

    # 2.8 Why Split the Data?
    st.subheader("2.8 Why Split the Data?")
    st.markdown("To prevent overfitting, we hide **20%** of the data during training. \n\nThink of it like an exam:\n- **Training Set (80%)**: The homework the model studies. We use it to calculate $b$ and $w$.\n- **Test Set (20%)**: The final exam. If the model memorized the homework (overfitting), it will fail the exam. If it genuinely learned the true pattern, it will pass.")

    # 2.9 Multiple Dimensions
    st.subheader("2.9 Adding More Features: Multiple Regression")
    st.markdown(f"We've only used a single feature so far. What if we use `weight` AND `horsepower` AND `model_year`? The math expands exactly the same way, but instead of 1 weight, we have $N$ weights. \n\nThe 2D line becomes a 3D plane, and eventually a multidimensional hyperplane.")

    st.info("💡 We understand the theory. Let's apply it to the actual model and measure performance → **Build & Evaluate**")



with tab_model:
    if len(selected_features) == 0:
        st.warning("Please select at least one feature.")
    else:
        # -----------------
        # Visual Rendering
        # Layouts run sequentially full-width instead of restricted side-by-side columns
        if True:
            with st.container(border=True):
                st.subheader("Visualization Space")
                
                show_residuals = st.checkbox("Show residual lines", value=False)
                
                if len(selected_features) == 1:
                    # Simple Linear Regression -> 2D Scatter + Curve
                    fig = go.Figure()
                    
                    # Scatter Train
                    fig.add_trace(go.Scatter(x=X_train[:, 0], y=y_train, mode='markers', name='Train Data', marker=dict(color='rgba(59, 130, 246, 0.6)', line=dict(color='white', width=1.5))))
                    # Scatter Test
                    fig.add_trace(go.Scatter(x=X_test[:, 0], y=y_test, mode='markers', name='Test Data', marker=dict(color='rgba(245, 158, 11, 0.8)', line=dict(color='white', width=1.5))))

                    if show_residuals:
                        for xi, yi_true, yi_pred in zip(X_test[:, 0], y_test, y_test_pred):
                            fig.add_trace(go.Scatter(
                                x=[xi, xi], y=[yi_true, yi_pred],
                                mode='lines',
                                line=dict(color='#fbbf24', width=2, dash='dot'),
                                showlegend=False, hoverinfo='skip'
                            ))

                    # Regression Line/Curve mapping
                    x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 200).reshape(-1, 1)
                    y_range = model.predict(poly.transform(x_range))
                    
                    fig.add_trace(go.Scatter(x=x_range[:, 0], y=y_range, mode='lines', name='Regression Fit', line=dict(color='#22c55e', width=3)))
                    
                    fig.update_layout(title="Regression Fit", xaxis_title=selected_features[0], yaxis_title=target_col, height=500, plot_bgcolor='rgba(15, 23, 42, 0.6)', paper_bgcolor='rgba(15, 23, 42, 0.6)')
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif len(selected_features) == 2 and poly_degree == 1:
                    # Multiple Linear Regression (2 vars) -> 3D Scatter + Plane
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter3d(
                        x=X_test[:, 0], y=X_test[:, 1], z=y_test,
                        mode='markers', name='Test Data',
                        marker=dict(size=4, color='rgba(245, 158, 11, 0.8)', line=dict(color='white', width=1.5))
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
                                line=dict(color='#fbbf24', width=2, dash='dot'),
                                showlegend=False, hoverinfo='skip'
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
                    ), height=500, plot_bgcolor='rgba(15, 23, 42, 0.6)', paper_bgcolor='rgba(15, 23, 42, 0.6)')
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.info("Visualizing Actual vs Predicted for higher dimensions or multi-feature polynomials.")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=y_test, y=y_test_pred, mode='markers', name='Predictions', marker=dict(color='rgba(168, 85, 247, 0.6)', line=dict(color='white', width=1.5))))
                    
                    min_val, max_val = min(y_test.min(), y_test_pred.min()), max(y_test.max(), y_test_pred.max())
                    fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Perfect Predict Line', line=dict(color='#ef4444', dash='dash')))
                    
                    fig.update_layout(title="Actual vs Predicted (Test Set)", xaxis_title="True Values", yaxis_title="Predicted Values", height=500, plot_bgcolor='rgba(15, 23, 42, 0.6)', paper_bgcolor='rgba(15, 23, 42, 0.6)')
                    st.plotly_chart(fig, use_container_width=True)

        # Render dynamic full-width formula box below the visualization
        if True:
            with st.container(border=True):
                st.subheader("The Exact Model Formula")
                feature_names = poly.get_feature_names_out(selected_features)
                weights = model.coef_
                bias = model.intercept_
                
                st.markdown("This equation represents the mathematically optimal behavior mapping drawn in the visualization space above.")
                
                col_gen, col_leg = st.columns([1, 1])
                with col_gen:
                    st.markdown("**Generic Template Form:**")
                    if len(selected_features) == 1 and poly_degree == 1:
                        st.latex(r"\hat{y} = b + w \cdot x")
                    elif len(selected_features) > 1 and poly_degree == 1:
                        st.latex(r"\hat{y} = b + w_1 x_1 + w_2 x_2 + \dots")
                    elif len(selected_features) == 1 and poly_degree > 1:
                        st.latex(r"\hat{y} = b + w_1 x + w_2 x^2 + \dots")
                    else:
                        st.latex(r"\hat{y} = b + \sum w_i x_i")
                
                with col_leg:
                    st.markdown("**Mathematical Legend:**")
                    st.markdown("- **$\hat{y}$** = Predicted Target (`" + target_col + "`)")
                    st.markdown("- **$b$** = Bias (y-intercept)")
                    st.markdown("- **$w$** = Weight (Feature Significance)")
                
                st.divider()
                st.markdown("**Your Active Model Equation:**")

                # Build precise mathematical string iteratively correcting signage natively
                eq_parts_tex = []
                for w, name in zip(weights, feature_names):
                    if w == 0: continue
                    sign = "+" if w > 0 else "-"
                    
                    # Handle polynomial powers and interaction terms generated by sklearn
                    formatted_parts = []
                    for part in name.split(' '):
                        if '^' in part:
                            base, exp = part.split('^', 1)
                            clean_base = base.replace('_', r'\ ')
                            formatted_parts.append(f"\\text{{{clean_base}}}^{{{exp}}}")
                        else:
                            clean_part = part.replace('_', r'\ ')
                            formatted_parts.append(f"\\text{{{clean_part}}}")
                            
                    clean_name = " \\cdot ".join(formatted_parts)
                    eq_parts_tex.append(f"{sign} {abs(w):.4f} \\cdot {clean_name}")

                clean_target = target_col.replace('_', r'\ ')
                equation_str = f"\\widehat{{\\text{{{clean_target}}}}} = {bias:.4f} "
                
                if len(eq_parts_tex) > 0:
                    # Append terms with clean signage
                    if eq_parts_tex[0].startswith('+ '):
                        equation_str += "+ " + eq_parts_tex[0][2:] + " "
                    elif eq_parts_tex[0].startswith('- '):
                        equation_str += "- " + eq_parts_tex[0][2:] + " "
                    
                    equation_str += " ".join(eq_parts_tex[1:])
                
                if len(weights) > 6:
                    st.latex(f"\\widehat{{\\text{{{clean_target}}}}} = {bias:.2f} + \\sum_{{i=1}}^{{{len(weights)}}} w_i x_i")
                    st.caption(f"({len(weights)} polynomial terms—displaying simplified summation for readability)")
                else:
                    st.latex(equation_str)
                
            st.divider()

            col_report, col_hist = st.columns([1, 1])
            with col_report:    
                with st.container(border=True):
                    st.markdown('### Model Report Card')
                    
                    improvement = ((baseline_rmse - test_rmse) / baseline_rmse) * 100
                    
                    st.markdown(f"**Baseline RMSE (Always guessing {mean_mpg:.1f}):** {baseline_rmse:.2f} mpg")
                    st.markdown(f"**Your Test RMSE:** {test_rmse:.2f} mpg")
                    st.markdown(f"**Improvement:** :green[-{improvement:.1f}% error reduction]")
                    
                    table_html = f"""
                    <style>
                    .metric-table {{ width: 100%; border-collapse: collapse; margin-top: 10px; margin-bottom: 20px; }}
                    .metric-table th {{ text-align: left; padding: 10px; border-bottom: 2px solid #334155; color: #94a3b8; font-weight: 600; text-transform: uppercase; font-size: 0.85rem; letter-spacing: 1px; }}
                    .metric-table td {{ padding: 10px; border-bottom: 1px solid #1e293b; color: #f1f5f9; }}
                    </style>
                    <table class="metric-table">
                      <tr><th>Metric</th><th>Train</th><th>Test</th></tr>
                      <tr>
                        <td>{tooltip('MSE', 'Mean Squared Error: Average of squared differences between predicted and actual values. Lower is better.')}</td>
                        <td>{train_mse:.2f}</td><td>{test_mse:.2f}</td>
                      </tr>
                      <tr>
                        <td>{tooltip('RMSE', 'Root Mean Squared Error: Square root of MSE. Same units as the target variable, making it easier to interpret.')}</td>
                        <td>{train_rmse:.2f}</td><td>{test_rmse:.2f}</td>
                      </tr>
                      <tr>
                        <td>{tooltip('R² Score', 'Coefficient of Determination: Proportion of variance explained by the model. 1.0 = perfect, 0.0 = no better than mean.')}</td>
                        <td>{train_r2:.4f}</td><td>{test_r2:.4f}</td>
                      </tr>
                    </table>
                    """
                    st.markdown(table_html, unsafe_allow_html=True)
                    
                    r2_gap = train_r2 - test_r2
                    if r2_gap < 0.05:
                        st.success('🟢 Good Fit — train and test scores match.')
                    elif r2_gap < 0.15:
                        st.warning('🟡 Slight Overfit — model struggles slightly on new data.')
                    else:
                        st.error('🔴 Overfitting Detected — model is memorizing training data.')
            
            with col_hist:
                with st.container(border=True):
                    st.markdown(f"### Residual Histogram")
                    residuals = y_test - y_test_pred
                    fig_res = px.histogram(x=residuals, nbins=20, title="Prediction Errors (Test Set)", labels={'x': 'Error', 'count': 'Frequency'})
                    fig_res.add_vline(x=0, line_color="red", line_dash="dash")
                    fig_res.update_layout(height=410, margin=dict(l=20, r=20, t=30, b=20), plot_bgcolor='rgba(15, 23, 42, 0.6)', paper_bgcolor='rgba(15, 23, 42, 0.6)')
                    st.plotly_chart(fig_res, use_container_width=True)
                    st.caption("Distribution of errors. Ideally, this should look like a bell curve centered at 0.")

        # -----------------
        # Live Code Viewer
        # -----------------
        st.divider()
        with st.expander("View Python Code", expanded=False):
            code_str = f'''import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# 1. Load Data
df = pd.read_csv('auto-mpg.csv')
X = df[{selected_features}].values
y = df['{target_col}'].values

# 2. Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_size}, random_state=42)

# 3. Polynomial mapping (if degree > 1)
poly = PolynomialFeatures(degree={poly_degree}, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# 4. Train Model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# 5. Evaluate
predictions = model.predict(X_test_poly)
print(f"MSE: {{mean_squared_error(y_test, predictions):.2f}}")
'''
            st.code(code_str, language="python")
            st.caption("This code matches the current sidebar settings exactly.")
            
        st.info("💡 The model works — but how does it actually find those weights? → **How the Algorithm Learns**")

with tab_hood:
    if len(selected_features) == 0:
        st.warning("Please select at least one feature from the sidebar to begin.")
    elif len(selected_features) != 1:
        st.info("Gradient descent visualization is available for single-feature regression only. Please select exactly one feature.")
    else:
        st.markdown("""
        There are two ways to find the best weight and bias:
        
        **Method 1 — OLS (Normal Equation):** A direct formula that computes 
        the exact answer in one step. Like using GPS to jump straight to 
        the destination.
        
        **Method 2 — Gradient Descent:** An iterative algorithm that starts 
        with random guesses and improves step by step. Like walking downhill 
        in fog — you can't see the bottom, but you know which way is down.
        
        For our dataset (392 rows, a few features), both arrive at the 
        same answer. OLS is faster. But for millions of rows, OLS runs 
        out of memory — that's when gradient descent becomes essential.
        """)
        st.divider()
        
        st.markdown("### Method 1: OLS — The Exact Solution")
        with st.container(border=True):
            st.info("Ordinary Least Squares (OLS) is how sklearn actually solves linear regression. Instead of iterating, it uses a direct formula from linear algebra called the Normal Equation to compute the exact optimal weights in a single step.")
            st.latex(r"\mathbf{w} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}")
            
            X_with_bias = np.column_stack([X_train[:, 0], np.ones(len(X_train))])
            XtX = X_with_bias.T @ X_with_bias
            XtX_inv = np.linalg.inv(XtX)
            Xty = X_with_bias.T @ y_train
            w_ols = XtX_inv @ Xty

            st.markdown("**Step 1: Compute X^T X** — How the features relate to themselves")
            st.caption("This matrix summarizes the internal structure of the input data. "
                       "The diagonal values measure how spread out each feature is. "
                       "The off-diagonal values measure how features correlate with each other.")
            st.code(f"X^T X =\n{XtX}", language="plaintext")

            st.markdown("**Step 2: Invert it → (X^T X)^-1** — Undo the feature structure")
            st.caption("Matrix inversion is the expensive step. For our small dataset "
                       "this is instant, but with millions of rows this becomes the bottleneck "
                       "that makes OLS impractical.")
            st.code(f"(X^T X)^-1 =\n{XtX_inv}", language="plaintext")

            st.markdown("**Step 3: Compute X^T y** — How features relate to the target")
            st.caption("This vector captures the direct relationship between each feature "
                       "and the target variable (mpg). Larger values mean stronger influence.")
            st.code(f"X^T y =\n{Xty}", language="plaintext")

            st.markdown("**Step 4: Multiply to get the answer**")
            st.latex(r"\mathbf{w} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}")
            
            st.markdown(f"**Result:** w = {w_ols[0]:.6f}, b = {w_ols[1]:.6f}")
            st.markdown("**Computed in:** 1 step, 0 iterations")
            st.caption(f"sklearn result: w = {model.coef_[0]:.6f}, b = {model.intercept_:.6f} — identical, because sklearn uses OLS internally.")
        
            with st.expander("Why not always use OLS?"):
                st.markdown("OLS requires computing $(\mathbf{X}^T \mathbf{X})^{-1}$, which is a matrix inversion. For small datasets like ours, this is instant. But with millions of rows or thousands of features, matrix inversion becomes very slow and memory-intensive. That is when gradient descent becomes useful — it trades exactness for scalability.")
            
        st.divider()
        
        st.markdown(f"### Method 2: {tooltip('Gradient Descent', 'An optimization algorithm used to minimize the loss function by iteratively moving in the direction of steepest descent.')} — The Iterative Alternative", unsafe_allow_html=True)
        
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
        if 'last_lr' not in st.session_state:
            st.session_state.last_lr = lr

        if init_weights != st.session_state.last_init_weights or lr != st.session_state.last_lr:
            st.session_state.last_init_weights = init_weights
            st.session_state.last_lr = lr
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
            fig_anim.add_trace(go.Scatter(x=X_train[:, 0], y=y_train, mode='markers', name='Train Data', marker=dict(color='rgba(59, 130, 246, 0.6)', line=dict(color='white', width=1.5))))
            
            t_x = np.array([X_train[:, 0].min(), X_train[:, 0].max()])
            t_y_gd = display_w * t_x + display_b
            fig_anim.add_trace(go.Scatter(x=t_x, y=t_y_gd, mode='lines', name='GD Current', line=dict(color='red', width=3)))
            
            t_y_sk = model.coef_[0] * t_x + model.intercept_
            fig_anim.add_trace(go.Scatter(x=t_x, y=t_y_sk, mode='lines', name='Optimal (sklearn)', line=dict(color='green', width=1, dash='dash')))
            
            fig_anim.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), plot_bgcolor='rgba(15, 23, 42, 0.6)', paper_bgcolor='rgba(15, 23, 42, 0.6)')
            st.plotly_chart(fig_anim, use_container_width=True)

        with col_plot2:
            st.markdown(f"**{tooltip('Loss Curve', 'Shows how the error (MSE) decreases with each iteration of gradient descent. The dashed red line is the sklearn optimal.')}**", unsafe_allow_html=True)
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(y=st.session_state.loss_history, mode='lines', name='Loss', line=dict(color='blue')))
            fig_loss.add_hline(y=sklearn_train_mse, line_dash='dash', line_color='red', annotation_text='sklearn MSE')
            fig_loss.add_trace(go.Scatter(x=[len(st.session_state.loss_history)-1], y=[st.session_state.loss_history[-1]], mode='markers', marker=dict(color='red', size=8), name='Current'))
            fig_loss.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="Iteration", yaxis_title="MSE Loss", plot_bgcolor='rgba(15, 23, 42, 0.6)', paper_bgcolor='rgba(15, 23, 42, 0.6)')
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
            fig_surf.update_layout(scene=dict(xaxis_title='Weight (w)', yaxis_title='Bias (b)', zaxis_title='MSE Loss', camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))), height=500, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor='rgba(15, 23, 42, 0.6)', paper_bgcolor='rgba(15, 23, 42, 0.6)')
            st.plotly_chart(fig_surf, use_container_width=True)
            
        with col_surf2:
            st.markdown(f"**{tooltip('Contour Map (Top-Down View)', 'A top-down view of the loss surface, like a topographic map. Each ring represents a constant error level. The red path shows the route gradient descent took.')}**", unsafe_allow_html=True)
            fig_cont = go.Figure(data=[go.Contour(x=w_range, y=b_range, z=Loss_grid, colorscale='Viridis', ncontours=25)])
            fig_cont.add_trace(go.Scatter(x=[opt_w], y=[opt_b], mode='markers', marker=dict(size=12, color='red', symbol='star'), name='Optimal'))
            if len(st.session_state.loss_history) > 1:
                fig_cont.add_trace(go.Scatter(x=st.session_state.w_history, y=st.session_state.b_history, mode='lines+markers', line=dict(color='red', width=2), marker=dict(size=6, symbol='arrow-right'), name='Path'))
                fig_cont.add_trace(go.Scatter(x=[st.session_state.w_history[0]], y=[st.session_state.b_history[0]], mode='markers', marker=dict(size=10, color='yellow'), name='Start'))
                fig_cont.add_trace(go.Scatter(x=[st.session_state.w_history[-1]], y=[st.session_state.b_history[-1]], mode='markers', marker=dict(size=10, color='green'), name='End'))
            fig_cont.update_layout(xaxis_title='Weight (w)', yaxis_title='Bias (b)', height=500, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor='rgba(15, 23, 42, 0.6)', paper_bgcolor='rgba(15, 23, 42, 0.6)')
            st.plotly_chart(fig_cont, use_container_width=True)
            
        st.markdown(f"The surface above shows MSE loss for {len(w_range) * len(b_range):,} different (w, b) combinations. The red diamond marks the minimum loss of **{sklearn_train_mse:.4f}** at `w={opt_w:.6f}`, `b={opt_b:.6f}`.")
        if len(st.session_state.loss_history) > 1:
            try: gap = abs(st.session_state.loss_history[-1] - sklearn_train_mse) / sklearn_train_mse 
            except: gap = 0
            st.success(f"**Path Tracking:** The red path shows how gradient descent navigated from its starting point to reach within **{gap:.2%}** of the optimal solution in **{st.session_state.gd_i}** steps.")
