import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Classification", layout="wide")

st.markdown("""
<style>
.tooltip-custom {
  position: relative;
  display: inline-block;
  border-bottom: 1px dotted #ccc;
  cursor: help;
}

.tooltip-custom .tooltiptext {
  visibility: hidden;
  width: 250px;
  background-color: #333;
  color: #fff;
  text-align: center;
  border-radius: 6px;
  padding: 5px;
  position: absolute;
  z-index: 10;
  bottom: 125%;
  left: 50%;
  margin-left: -125px;
  opacity: 0;
  transition: opacity 0.1s;
  font-size: 14px;
}

.tooltip-custom:hover .tooltiptext {
  visibility: visible;
  opacity: 1;
}
</style>
""", unsafe_allow_html=True)

st.title("Section 2: Classification Mechanisms", help="Classification mechanisms: The underlying mathematical or logical processes by which a model assigns a class label to an input.")
st.markdown("Visualizing the mathematical <span class='tooltip-custom'>boundaries<span class='tooltiptext'>A hypersurface that partitions the underlying vector space into sets, one for each class.</span></span> that separate discrete classes.", unsafe_allow_html=True)

@st.cache_data
def load_iris_data():
    data = load_iris()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    target_names = data.target_names
    return X, y, feature_names, target_names

X, y, feature_names, target_names = load_iris_data()

tab_data, tab_hyper, tab_knn, tab_lr, tab_choice = st.tabs([
    "Data Understanding", 
    "Hyperplanes and Decision Boundaries", 
    "K-Nearest Neighbors (KNN)", 
    "Logistic Regression",
    "Classification Model Choice"
])

with tab_data:
    st.header("Data Understanding: The Iris Dataset")
    st.markdown("""
    The **Iris dataset** is arguably the most famous dataset in the world of data science and pattern recognition. Introduced by the British statistician and biologist Ronald Fisher in his 1936 paper, it is widely used as a beginner-friendly benchmark for classification algorithms.
    """)

    st.markdown("""
    ### What are the datapoints?
    The dataset consists of **150 datapoints** (samples). Each datapoint represents a single Iris flower that was measured and recorded. These 150 flowers are evenly divided into three different species (50 flowers each):
    """)
    
    col_setosa, col_versicolor, col_virginica = st.columns(3)
    with col_setosa:
        st.image("assets/Iris_Setosa.jpg", caption="Iris setosa", use_container_width=True)
    with col_versicolor:
        st.image("assets/Iris_Versicolor.jpg", caption="Iris versicolor", use_container_width=True)
    with col_virginica:
        st.image("assets/Iris_Virginica.jpg", caption="Iris virginica", use_container_width=True)

    st.markdown("""
    ### What are the features?
    For each flower, four attributes (features) were measured in centimeters:
    1. **Sepal Length**: The length of the outer leaves of the flower.
    2. **Sepal Width**: The width of the outer leaves.
    3. **Petal Length**: The length of the inner petals.
    4. **Petal Width**: The width of the inner petals.
    """)
    
    st.image("assets/Iris_Features.jpeg", caption="Iris Features Diagram", use_container_width=False, width=600)

    st.markdown("""
    ### Significance in Data Science
    Because it is small, clean (no missing values), and easy to understand, the Iris dataset serves as the standard "Hello World" of machine learning. It perfectly demonstrates how algorithms can learn to draw a <span class='tooltip-custom'>decision boundary<span class='tooltiptext'>A hypersurface that partitions the underlying vector space into sets, one for each class.</span></span> to distinguish between different classes based on continuous numerical features. The dataset is linearly separable for one biological class (*Iris setosa* is isolated from the other two), while the other two classes (*versicolor* and *virginica*) have some overlap, providing a great test for model <span class='tooltip-custom'>classification accuracy<span class='tooltiptext'>The ratio of correct predictions to the total number of predictions made.</span></span>.
    """, unsafe_allow_html=True)

    st.markdown("### Code Snippet: Loading the Data")
    st.code("""
from sklearn.datasets import load_iris

# 1. Load the dataset
iris = load_iris()
X = iris.data
y = iris.target

print(f"Features: {iris.feature_names}")
print(f"Target Classes: {iris.target_names}")
""", language="python")

with tab_hyper:
    st.header("Hyperplanes and Decision Boundaries")
    st.markdown("""
    ### What is Classification?
    Classification is a type of supervised machine learning problem where the goal is to predict a discrete class label or category for a given input based on its features. Unlike regression (which predicts a continuous number), classification answers questions like "Is this email spam or not?" or "Which species of Iris is this?".

    ### What are Hyperplanes?
    A **hyperplane** is a flat affine subspace whose dimension is one less than that of its ambient space. 
    - In **2D space** (2 features), a hyperplane is a **1D line**.
    - In **3D space** (3 features), a hyperplane is a **2D flat plane**.
    - In higher dimensions, it is simply referred to as a hyperplane.
    
    Linear models (like Logistic Regression) use hyperplanes to split the space into distinct regions.

    ### What are Decision Boundaries?
    A <span class='tooltip-custom'>decision boundary<span class='tooltiptext'>A hypersurface that partitions the underlying vector space into sets, one for each class.</span></span> is the region of the problem space where the output label of a classifier is ambiguous. For a binary classification problem, it's the exact surface where the model evaluates the probability of either class to be equal. 
    
    While a linear model always produces a straight hyperplane as its decision boundary, more complex or non-linear models (like K-Nearest Neighbors) can create highly irregular, curved, or jagged boundaries.
    """, unsafe_allow_html=True)

    st.image("assets/2D_and_3D_DecisionBoundaries.png", caption="Examples of Hyperplane Decision Boundaries", use_container_width=True)

    st.markdown("""
    ### How we'll talk about them
    In the remainder of this section, we will visualize how different <span class='tooltip-custom'>classification mechanisms<span class='tooltiptext'>The underlying mathematical or logical processes by which a model assigns a class label to an input.</span></span> construct these boundaries. You will interact with 2D and 3D plots showing the training data points and the colored regions representing the model's decision zones.
    """, unsafe_allow_html=True)

    st.markdown("### Code Snippet: Importing Classification Mechanisms")
    st.code("""
# 2. Import the classification models we'll be visualizing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Import visualization libraries for the app
import numpy as np
import plotly.graph_objects as go
""", language="python")

with tab_knn:
    st.header("K-Nearest Neighbors")
    st.markdown("""
    **K-Nearest Neighbors (k-NN)** is a simple, intuitive, non-parametric algorithm used for classification. 
    Instead of learning mathematical weights or a strict formula during training, the model simply "memorizes" the entire training dataset. When asked to classify a new, unseen datapoint, it looks at the **k** closest training examples in the feature space and assigns the most common class among those neighbors (a majority vote).
    
    Because it relies on local distance rather than global equations, k-NN is capable of drawing highly complex, non-linear <span class='tooltip-custom'>decision boundaries<span class='tooltiptext'>A hypersurface that partitions the underlying vector space into sets, one for each class.</span></span>.
    """, unsafe_allow_html=True)
    st.markdown("### KNN Controls")
    
    col_k, col_train = st.columns([1, 2])
    with col_k:
        k_val = st.slider("Number of Neighbors (K)", 1, 50, 5, key='knn_k')
    with col_train:
        train_features_knn = st.multiselect("Select Feature(s) to Train On", feature_names, default=feature_names[:2], key='knn_train')
    
    dim_knn = st.radio("Visualization Dimension", ["2D", "3D"], key='knn_dim')
    
    if dim_knn == "2D":
        col_feat1, col_feat2 = st.columns(2)
        feat_x_idx = col_feat1.selectbox("Visualization X Axis", range(4), format_func=lambda x: feature_names[x], index=0, key='knn_fx')
        feat_y_idx = col_feat2.selectbox("Visualization Y Axis", range(4), format_func=lambda x: feature_names[x], index=1, key='knn_fy')
        feat_z_idx = None
        vis_features = [feat_x_idx, feat_y_idx]
        is_valid = feat_x_idx != feat_y_idx
    else:
        col_feat1, col_feat2, col_feat3 = st.columns(3)
        feat_x_idx = col_feat1.selectbox("Visualization X Axis", range(4), format_func=lambda x: feature_names[x], index=0, key='knn_fx_3d')
        feat_y_idx = col_feat2.selectbox("Visualization Y Axis", range(4), format_func=lambda x: feature_names[x], index=1, key='knn_fy_3d')
        feat_z_idx = col_feat3.selectbox("Visualization Z Axis", range(4), format_func=lambda x: feature_names[x], index=2, key='knn_fz_3d')
        vis_features = [feat_x_idx, feat_y_idx, feat_z_idx]
        is_valid = len(set(vis_features)) == 3
        
    if not is_valid:
        st.warning("Please select distinct features for visualization.")
    elif len(train_features_knn) == 0:
        st.warning("Please select at least 1 feature to train the model.")
    else:
        # Get indices for training
        train_indices = [list(feature_names).index(f) for f in train_features_knn]
        X_train = X[:, train_indices]
        
        # Jitter data for plotting only
        np.random.seed(42)
        X_vis = X[:, vis_features]
        X_jitter = X_vis + np.random.normal(0, 0.02, X_vis.shape)
        
        clf = KNeighborsClassifier(n_neighbors=k_val)
        clf.fit(X_train, y)
        
        c1, c2 = st.columns([1, 1.2])
        
        with c1:
            st.markdown("### <span class='tooltip-custom'>Overfitting<span class='tooltiptext'>When a model learns the training data too well, capturing noise and failing to generalize to new data.</span></span> vs <span class='tooltip-custom'>Underfitting<span class='tooltiptext'>When a model is too simple to capture the underlying patterns in the data, performing poorly on both training and new data.</span></span>", unsafe_allow_html=True)
            st.markdown("When **K=1**, the model memorizes the training data completely. Look at how fractured the decision boundary becomes to accommodate isolated dots (severe over-fitting). As K increases, the boundary smooths out, but if K becomes too large, the model might under-fit by ignoring distinct local patterns.")
            st.markdown("""
            **How to pick the best K:**
            Finding the best K is a balance between <span class='tooltip-custom'>overfitting<span class='tooltiptext'>When a model learns the training data too well, capturing noise and failing to generalize to new data.</span></span> and <span class='tooltip-custom'>underfitting<span class='tooltiptext'>When a model is too simple to capture the underlying patterns in the data, performing poorly on both training and new data.</span></span>. There is no statistical formula to find the exact best choice of K. It is usually found experimentally by analyzing validation errors.
            """, unsafe_allow_html=True)
            if k_val == 1:
                st.info("K=1 Notice: At 1 neighbor, the model achieves exactly 100% training accuracy. The decision boundary snaps directly to every single point! Identical points have been purely jittered via np.random so you can see overlapping outliers.")
                
            accuracy = clf.score(X_train, y)
            st.metric(f"Training Accuracy ({len(train_indices)} Features)", f"{accuracy*100:.2f}%", help="Classification accuracy: The ratio of correct predictions to the total number of predictions made.")
            if len(set(train_indices) - set(vis_features)) > 0:
                st.caption("*Note: Some training features are not visualized. They are held at their mean values to plot the decision boundary.*")
            
        with c2:
            fig = go.Figure()
            if dim_knn == "2D":
                # 2D Visualization limits based on original data axes
                x_min, x_max = X[:, feat_x_idx].min() - 0.5, X[:, feat_x_idx].max() + 0.5
                y_min, y_max = X[:, feat_y_idx].min() - 0.5, X[:, feat_y_idx].max() + 0.5
                xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                                     np.arange(y_min, y_max, 0.05))
                                     
                # Build input matrix
                grid_input = np.zeros((xx.size, len(train_indices)))
                for i, t_idx in enumerate(train_indices):
                    if t_idx == feat_x_idx:
                        grid_input[:, i] = xx.ravel()
                    elif t_idx == feat_y_idx:
                        grid_input[:, i] = yy.ravel()
                    else:
                        grid_input[:, i] = X[:, t_idx].mean()

                Z = clf.predict(grid_input)
                Z = Z.reshape(xx.shape)
                
                # Boundary Contour
                fig.add_trace(go.Contour(x=np.arange(x_min, x_max, 0.05), y=np.arange(y_min, y_max, 0.05), z=Z, showscale=False, opacity=0.3, colorscale='Viridis'))
                
                # Scatter Points
                fig.add_trace(go.Scatter(
                    x=X_jitter[:, 0], y=X_jitter[:, 1],
                    mode='markers',
                    marker=dict(color=y, colorscale='Viridis', line=dict(color='white', width=1), size=8),
                    hoverinfo='text',
                    text=[f"Class: {target_names[val]}" for val in y]
                ))
                fig.update_layout(title="KNN Decision Boundary (2D Space)", xaxis_title=feature_names[feat_x_idx], yaxis_title=feature_names[feat_y_idx], height=450)
            else:
                # 3D Visualization
                x_min, x_max = X[:, feat_x_idx].min() - 0.5, X[:, feat_x_idx].max() + 0.5
                y_min, y_max = X[:, feat_y_idx].min() - 0.5, X[:, feat_y_idx].max() + 0.5
                z_min, z_max = X[:, feat_z_idx].min() - 0.5, X[:, feat_z_idx].max() + 0.5
                
                xx, yy, zz = np.meshgrid(np.linspace(x_min, x_max, 15),
                                         np.linspace(y_min, y_max, 15),
                                         np.linspace(z_min, z_max, 15))
                                         
                grid_input = np.zeros((xx.size, len(train_indices)))
                for i, t_idx in enumerate(train_indices):
                    if t_idx == feat_x_idx:
                        grid_input[:, i] = xx.ravel()
                    elif t_idx == feat_y_idx:
                        grid_input[:, i] = yy.ravel()
                    elif t_idx == feat_z_idx:
                        grid_input[:, i] = zz.ravel()
                    else:
                        grid_input[:, i] = X[:, t_idx].mean()
                        
                Z = clf.predict(grid_input)
                
                # 3D Grid points for boundary
                fig.add_trace(go.Scatter3d(
                    x=xx.ravel(), y=yy.ravel(), z=zz.ravel(),
                    mode='markers',
                    marker=dict(color=Z, colorscale='Viridis', size=3, opacity=0.1),
                    hoverinfo='skip'
                ))
                
                # 3D Scatter Points
                fig.add_trace(go.Scatter3d(
                    x=X_jitter[:, 0], y=X_jitter[:, 1], z=X_jitter[:, 2],
                    mode='markers',
                    marker=dict(color=y, colorscale='Viridis', line=dict(color='black', width=1), size=6),
                    hoverinfo='text',
                    text=[f"Class: {target_names[val]}" for val in y]
                ))
                fig.update_layout(title="KNN Decision Regions (3D Space)", 
                                  scene=dict(
                                      xaxis_title=feature_names[feat_x_idx],
                                      yaxis_title=feature_names[feat_y_idx],
                                      zaxis_title=feature_names[feat_z_idx]
                                  ),
                                  height=450, margin=dict(l=0, r=0, b=0, t=40))
                
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Code Snippet: Creating This K-NN Model")
        dim_shape = len(train_indices)
        feature_names_str = f"{[feature_names[i] for i in train_indices]}"
        code_str = f"""# 3. Filter X down to {dim_shape} customized training features: {feature_names_str}
X_train_knn = X[:, {train_indices}]

# 4. Initialize the k-Nearest Neighbors model
clf = KNeighborsClassifier(n_neighbors={k_val})

# 5. Train the model
clf.fit(X_train_knn, y)

# Boundary prediction plotting code omitted for brevity.
"""
        st.code(code_str, language="python")

with tab_lr:
    st.header("Logistic Regression")
    st.markdown("""
    **Logistic Regression** is a fundamental statistical algorithm used for classification. Despite its name, it is a **linear** model used to predict categorical outcomes.
    
    It works by combining the features linearly (multiplying them by learned weights) and then passing that result through a mathematical function called the **Sigmoid** (for binary classification) or **Softmax** (for multi-class classification) function. This squashes the arbitrary linear output into a valid probability range between 0 and 1.
    """, unsafe_allow_html=True)
    st.markdown("### Logistic Regression Controls")
    
    col_c, col_int, col_lr_train = st.columns([1, 1, 2])
    with col_c:
        c_val = st.slider("Inverse Regularization Strength", 0.01, 10.0, 1.0, step=0.01, key='lr_c', help="Regularization: A technique used to reduce overfitting by adding a penalty term to the loss function.")
    with col_int:
        fit_intercept_val = st.toggle("Fit Intercept (Bias)", value=True, key='lr_int')
    with col_lr_train:
        train_features_lr = st.multiselect("Select Feature(s) to Train On", feature_names, default=feature_names[2:], key='lr_train')

    dim_lr = st.radio("Visualization Dimension", ["2D", "3D"], key='lr_dim')
    
    if dim_lr == "2D":
        col_feat1_lr, col_feat2_lr = st.columns(2)
        lx = col_feat1_lr.selectbox("Visualization X Axis", range(4), format_func=lambda x: feature_names[x], index=2, key='lr_fx')
        ly = col_feat2_lr.selectbox("Visualization Y Axis", range(4), format_func=lambda x: feature_names[x], index=3, key='lr_fy')
        lz = None
        vis_features_lr = [lx, ly]
        is_valid_lr = lx != ly
    else:
        col_feat1_lr, col_feat2_lr, col_feat3_lr = st.columns(3)
        lx = col_feat1_lr.selectbox("Visualization X Axis", range(4), format_func=lambda x: feature_names[x], index=0, key='lr_fx_3d')
        ly = col_feat2_lr.selectbox("Visualization Y Axis", range(4), format_func=lambda x: feature_names[x], index=1, key='lr_fy_3d')
        lz = col_feat3_lr.selectbox("Visualization Z Axis", range(4), format_func=lambda x: feature_names[x], index=2, key='lr_fz_3d')
        vis_features_lr = [lx, ly, lz]
        is_valid_lr = len(set(vis_features_lr)) == 3

    if not is_valid_lr:
        st.warning("Please select distinct features for visualization.")
    elif len(train_features_lr) == 0:
        st.warning("Please select at least 1 feature to train the model.")
    else:
        # Logistic boundaries
        train_indices_lr = [list(feature_names).index(f) for f in train_features_lr]
        X_train_lr = X[:, train_indices_lr]
        
        lr_model = LogisticRegression(multi_class='multinomial', max_iter=200, C=c_val, fit_intercept=fit_intercept_val)
        lr_model.fit(X_train_lr, y)
        
        X_vis_lr = X[:, vis_features_lr]
        
        st.markdown("### Linear Boundaries in Logistic Regression")
        fig_lr = go.Figure()
        if dim_lr == "2D":
            x_min, x_max = X[:, lx].min() - 0.5, X[:, lx].max() + 0.5
            y_min, y_max = X[:, ly].min() - 0.5, X[:, ly].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                                 np.arange(y_min, y_max, 0.05))
            
            grid_input = np.zeros((xx.size, len(train_indices_lr)))
            for i, t_idx in enumerate(train_indices_lr):
                if t_idx == lx:
                    grid_input[:, i] = xx.ravel()
                elif t_idx == ly:
                    grid_input[:, i] = yy.ravel()
                else:
                    grid_input[:, i] = X[:, t_idx].mean()

            Z = lr_model.predict(grid_input)
            Z = Z.reshape(xx.shape)
            
            # Boundary
            fig_lr.add_trace(go.Contour(x=np.arange(x_min, x_max, 0.05), y=np.arange(y_min, y_max, 0.05), z=Z, showscale=False, opacity=0.3, colorscale='Viridis'))
            
            # Scatter Points
            fig_lr.add_trace(go.Scatter(
                x=X_vis_lr[:, 0], y=X_vis_lr[:, 1],
                mode='markers',
                marker=dict(color=y, colorscale='Viridis', line=dict(color='white', width=1), size=8),
                hoverinfo='text',
                text=[f"Class: {target_names[val]}" for val in y]
            ))
            fig_lr.update_layout(xaxis_title=feature_names[lx], yaxis_title=feature_names[ly], height=600)
        else:
            # 3D
            x_min, x_max = X[:, lx].min() - 0.5, X[:, lx].max() + 0.5
            y_min, y_max = X[:, ly].min() - 0.5, X[:, ly].max() + 0.5
            z_min, z_max = X[:, lz].min() - 0.5, X[:, lz].max() + 0.5
            
            xx, yy, zz = np.meshgrid(np.linspace(x_min, x_max, 15),
                                     np.linspace(y_min, y_max, 15),
                                     np.linspace(z_min, z_max, 15))
            
            grid_input = np.zeros((xx.size, len(train_indices_lr)))
            for i, t_idx in enumerate(train_indices_lr):
                if t_idx == lx:
                    grid_input[:, i] = xx.ravel()
                elif t_idx == ly:
                    grid_input[:, i] = yy.ravel()
                elif t_idx == lz:
                    grid_input[:, i] = zz.ravel()
                else:
                    grid_input[:, i] = X[:, t_idx].mean()

            Z = lr_model.predict(grid_input)
            
            # Boundary Grid
            fig_lr.add_trace(go.Scatter3d(
                x=xx.ravel(), y=yy.ravel(), z=zz.ravel(),
                mode='markers',
                marker=dict(color=Z, colorscale='Viridis', size=3, opacity=0.1),
                hoverinfo='skip'
            ))
            
            # Scatter Points
            fig_lr.add_trace(go.Scatter3d(
                x=X_vis_lr[:, 0], y=X_vis_lr[:, 1], z=X_vis_lr[:, 2],
                mode='markers',
                marker=dict(color=y, colorscale='Viridis', line=dict(color='white', width=1), size=6),
                hoverinfo='text',
                text=[f"Class: {target_names[val]}" for val in y]
            ))
            fig_lr.update_layout(scene=dict(
                                     xaxis_title=feature_names[lx],
                                     yaxis_title=feature_names[ly],
                                     zaxis_title=feature_names[lz]
                                 ),
                                 height=600, margin=dict(l=0, r=0, b=0, t=10))
            
        st.plotly_chart(fig_lr, use_container_width=True)

        acc_lr = lr_model.score(X_train_lr, y)
        st.metric(f"Model Classification Accuracy ({len(train_indices_lr)} Features)", f"{acc_lr*100:.2f}%", help="Classification accuracy: The ratio of correct predictions to the total number of predictions made.")
        if len(set(train_indices_lr) - set(vis_features_lr)) > 0:
            st.caption("*Note: Some training features are not visualized. They are held at their mean values to plot the decision boundary.*")

        st.markdown("---")
        st.markdown("### The Sigmoid Bridge")
        sig_col1, sig_col2 = st.columns([1, 1])
        with sig_col1:
            st.markdown("The model computes a linear combination of features:")
            equation_terms = [f"W_{i+1} x_{i+1}" for i in range(len(train_indices_lr))]
            equation = " + ".join(equation_terms)
            equation += " + b" if fit_intercept_val else ""
            st.latex(f"z = {equation}")
            
            st.markdown("And forces it between 0 and 1 for strict classification:")
            st.latex("P(y=C) = \\frac{1}{1 + e^{-z}}")
        with sig_col2:
            st.image("assets/sigmoid_function.png", use_container_width=True)

        st.info("Because the underlying combination $z$ is linear, the boundary where probabilities balance out is ALWAYS a perfectly straight line/hyperplane.")

        st.markdown("---")
        st.markdown("### Binary Cross-Entropy Loss")
        st.markdown("""
        **What does it do?**
        In Logistic Regression, the algorithm learns the optimal weights ($W$) using a cost function called **Log Loss** or **Binary Cross-Entropy Loss** (generalized to Categorical Cross-Entropy for multi-class). This function measures the performance of a classification model where the prediction input is a probability value between 0 and 1. 

        **How is it used in our model?**
        The loss function heavily penalizes confidently wrong predictions. During training, the algorithm iteratively adjusts the feature weights in a way that minimizes this cross-entropy loss, descending the gradient towards the most optimal straight-line boundaries possible.
        """)

        st.markdown("---")
        st.markdown("### Binary vs. Multi-Class Logistic Regression")
        st.markdown("""
        - **Binary Logistic Regression:** Predicts between exactly two outcomes (e.g., spam vs. not spam). It uses a single **Sigmoid** function to output a probability $P$. The other class's probability is simply $1 - P$.
        - **Multi-Class (Multinomial) Logistic Regression:** Predicts between three or more outcomes (like our 3 Iris species here). Instead of a single Sigmoid, it uses the **Softmax** function, which takes a vector of linear scores and normalizes them into a probability distribution across all classes that sums to 1.
        """)

        st.markdown("### Code Snippet: Creating This Logistic Regression Model")
        dim_shape_lr = len(train_indices_lr)
        feature_names_str_lr = f"{[feature_names[i] for i in train_indices_lr]}"
        code_str_lr = f"""# 3. Filter X down to {dim_shape_lr} custom features: {feature_names_str_lr}
X_train_lr = X[:, {train_indices_lr}]

# 4. Initialize the Logistic Regression model
# Note: we use multinomial since there are 3 classes
lr_model = LogisticRegression(
    multi_class='multinomial', 
    max_iter=200, 
    C={c_val}, 
    fit_intercept={fit_intercept_val}
)

# 5. Train the model
lr_model.fit(X_train_lr, y)

# Boundary prediction plotting code omitted for brevity.
"""
        st.code(code_str_lr, language="python")

with tab_choice:
    st.header("Classification Model Choice")
    
    col_choice1, col_choice2 = st.columns(2)
    with col_choice1:
        st.markdown("""
        ### When to use Logistic Regression?
        - **Linear Separability**: When your data classes can be separated by a straight line or flat hyperplane.
        - **Interpretability**: When you explicitly need to know the influence of each feature. The learned weights directly tell you how much a feature impacts the odds of a specific class.
        - **Fast Inference**: Making predictions simply involves calculating a linear combination, making it extremely fast for deployment.
        - **Probabilistic Output**: When you care about the *confidence* of a prediction rather than just the final discrete label.
        """)
        st.image("assets/Log_Reg_Dataset.png", caption="Favoring Logistic Regression", use_container_width=True)

    with col_choice2:
        st.markdown("""
        ### When to use K-Nearest Neighbors (k-NN)?
        - **Non-Linear Boundaries**: When the relationship between features and classes is highly complex or irregular and cannot be captured by simple straight lines.
        - **Instance-Based Systems**: When you expect your model to continuously adapt to new data without retraining; adding a new datapoint to a k-NN dataset instantly influences future predictions locally.
        - **Few Features**: k-NN can struggle with the "Curse of Dimensionality" because calculating distance becomes less meaningful in very high dimensions. It usually works best with smaller, highly relevant feature sets.
        - **No Assumption of Data Distribution**: k-NN makes no mathematical assumptions about the underlying distribution of the data, whereas Logistic Regression assumes a linear relationship in the log-odds.
        """)
        st.image("assets/K_NN_Dataset2D.png", caption="Favoring k-NN", use_container_width=True)

    st.markdown("---")
    st.markdown("### What about other classification models?")
    st.markdown("""
    While Logistic Regression and k-NN are fundamental, modern machine learning provides many other classification algorithms:
    - **<span class='tooltip-custom'>Support Vector Machines (SVM)<span class='tooltiptext'>Finds the optimal hyperplane that maximizes the margin between classes, often mapping data to higher dimensions using kernels.</span></span>**: Great for high-dimensional spaces and rigorous mathematical margins.
    - **<span class='tooltip-custom'>Convolutional Neural Networks (CNNs)<span class='tooltiptext'>Deep learning networks specialized in processing grid-like data, such as images, using spatial convolutions to extract features.</span></span>**: State-of-the-art for image and video classification due to spatial feature extraction.
    - **<span class='tooltip-custom'>Random Forests & Gradient Boosting<span class='tooltiptext'>Ensemble methods that combine multiple decision trees into a single, more robust prediction model.</span></span>**: Powerful ensemble methods that combine many decision trees, currently state-of-the-art for tabular data.
    """, unsafe_allow_html=True)
