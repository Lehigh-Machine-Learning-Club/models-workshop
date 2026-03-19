import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Classification", layout="wide")
st.title("Section 2: Classification Mechanisms")
st.markdown("Visualizing the mathematical boundaries that separate discrete classes.")

@st.cache_data
def load_iris_data():
    data = load_iris()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    target_names = data.target_names
    return X, y, feature_names, target_names

X, y, feature_names, target_names = load_iris_data()

tab1, tab2 = st.tabs(["K-Nearest Neighbors (KNN)", "Logistic Regression"])

with tab1:
    st.header("K-Nearest Neighbors")
    st.markdown("KNN classifies regions discretely based on majority votes of geographic neighbors.")
    
    st.sidebar.markdown("### KNN Controls")
    k_val = st.sidebar.slider("Number of Neighbors (K)", 1, 50, 5, key='knn_k')
    
    col_feat1, col_feat2 = st.columns(2)
    feat_x_idx = col_feat1.selectbox("Select Feature X", range(4), format_func=lambda x: feature_names[x], index=0, key='knn_fx')
    feat_y_idx = col_feat2.selectbox("Select Feature Y", range(4), format_func=lambda x: feature_names[x], index=1, key='knn_fy')
    
    if feat_x_idx == feat_y_idx:
        st.warning("Please select two distinct features for visualization.")
    else:
        # Jitter data
        np.random.seed(42)
        X_vis = X[:, [feat_x_idx, feat_y_idx]]
        # Calculate exactly overlapping points logic by adding standard normal noise scaled
        X_jitter = X_vis + np.random.normal(0, 0.02, X_vis.shape)
        
        clf = KNeighborsClassifier(n_neighbors=k_val)
        clf.fit(X_vis, y)
        
        c1, c2 = st.columns([1, 1.2])
        
        with c1:
            st.markdown("### Overfitting vs Underfitting")
            st.markdown("When **K=1**, the model memorizes the training data completely. Look at how fractured the decision boundary becomes to accommodate isolated dots.")
            if k_val == 1:
                st.info("K=1 Notice: At 1 neighbor, the model achieves exactly 100% training accuracy. The decision boundary snaps directly to every single point! Identical points have been purely jittered via np.random so you can see overlapping outliers.")
                
            accuracy = clf.score(X_vis, y)
            st.metric("Training Accuracy (2D)", f"{accuracy*100:.2f}%")
            
        with c2:
            # Visualization
            x_min, x_max = X_vis[:, 0].min() - 0.5, X_vis[:, 0].max() + 0.5
            y_min, y_max = X_vis[:, 1].min() - 0.5, X_vis[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                                 np.arange(y_min, y_max, 0.05))
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            fig = go.Figure()
            # Boundary Contour
            fig.add_trace(go.Contour(x=np.arange(x_min, x_max, 0.05), y=np.arange(y_min, y_max, 0.05), z=Z, showscale=False, opacity=0.3, colorscale='Viridis'))
            
            # Scatter Points
            fig.add_trace(go.Scatter(
                x=X_jitter[:, 0], y=X_jitter[:, 1],
                mode='markers',
                marker=dict(color=y, colorscale='Viridis', line=dict(color='black', width=1), size=8),
                hoverinfo='text',
                text=[f"Class: {target_names[val]}" for val in y]
            ))
            fig.update_layout(title="KNN Decision Boundary (2D Space)", xaxis_title=feature_names[feat_x_idx], yaxis_title=feature_names[feat_y_idx], height=450)
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Logistic Regression")
    st.markdown("Despite its name, Logistic Regression is fundamentally a **linear** model that maps into a probability space using the Sigmoid (or Softmax) function.")
    
    col_feat1_lr, col_feat2_lr = st.columns(2)
    lx = col_feat1_lr.selectbox("Feature X", range(4), format_func=lambda x: feature_names[x], index=2, key='lr_fx')
    ly = col_feat2_lr.selectbox("Feature Y", range(4), format_func=lambda x: feature_names[x], index=3, key='lr_fy')

    if lx == ly:
        st.warning("Please select distinct features.")
    else:
        # Logistic boundaries
        X_lr = X[:, [lx, ly]]
        lr_model = LogisticRegression(multi_class='multinomial', max_iter=200)
        lr_model.fit(X_lr, y)
        
        c_math, c_vis = st.columns([1, 1.2])
        
        with c_math:
            st.markdown("### The Sigmoid Bridge")
            st.markdown("The model computes a linear combination of features:")
            st.latex("z = W_1 x_1 + W_2 x_2 + b")
            st.markdown("And forces it between 0 and 1 for strict classification:")
            st.latex("P(y=C) = \\frac{1}{1 + e^{-z}}")
            st.info("Because the underlying combination $z$ is linear, the boundary where probabilities balance out is ALWAYS a perfectly straight line/hyperplane.")
            st.metric("Model Classification Accuracy", f"{lr_model.score(X_lr, y)*100:.2f}%")
            
        with c_vis:
            x_min, x_max = X_lr[:, 0].min() - 0.5, X_lr[:, 0].max() + 0.5
            y_min, y_max = X_lr[:, 1].min() - 0.5, X_lr[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                                 np.arange(y_min, y_max, 0.05))
            Z = lr_model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            fig_lr = go.Figure()
            # Boundary
            fig_lr.add_trace(go.Contour(x=np.arange(x_min, x_max, 0.05), y=np.arange(y_min, y_max, 0.05), z=Z, showscale=False, opacity=0.3, colorscale='Viridis'))
            
            # Scatter Points
            fig_lr.add_trace(go.Scatter(
                x=X_lr[:, 0], y=X_lr[:, 1],
                mode='markers',
                marker=dict(color=y, colorscale='Viridis', line=dict(color='white', width=1), size=8),
                hoverinfo='text',
                text=[f"Class: {target_names[val]}" for val in y]
            ))
            fig_lr.update_layout(title="Linear Boundaries in Logistic Regression", xaxis_title=feature_names[lx], yaxis_title=feature_names[ly], height=450)
            st.plotly_chart(fig_lr, use_container_width=True)
