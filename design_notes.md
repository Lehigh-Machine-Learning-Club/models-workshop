# Design Notes: Mechanistic Interpretability Dashboard

## Application Architecture
- **Framework:** Streamlit (Provides rapid interactive UI updates and slider bindings)
- **Visualization Libraries:** Plotly (for interactive 2D scatter plots and contour decision boundaries), NetworkX combined with Plotly or Matplotlib (for rendering the 2-3-1 network nodes and topological weights). Math expressions rendered natively via Streamlit's LaTeX (`st.latex`).

## Section 3, Phase 1: 2-3-1 MLP (Poisonous Fruit Detector)

### Mathematical Assumptions
- **Input:** $X \in \mathbb{R}^{N \times 2}$ (Spikes, Spots)
- **Latent:** $H \in \mathbb{R}^{N \times 3}$
- **Output:** $\hat{Y} \in \mathbb{R}^{N \times 1}$ (Probability of being Poisonous)
- **Weights:**
  - $W_1 \in \mathbb{R}^{2 \times 3}$, $b_1 \in \mathbb{R}^{1 \times 3}$
  - $W_2 \in \mathbb{R}^{3 \times 1}$, $b_2 \in \mathbb{R}^{1 \times 1}$

### Forward Pass
1. $Z_1 = X W_1 + b_1$
2. $A_1 = \text{Activation}(Z_1)$ (where Activation can be ReLU, Sigmoid, Tanh, Linear, Step)
3. $Z_2 = A_1 W_2 + b_2$
4. $A_2 = \text{Sigmoid}(Z_2)$ (Output probability for binary classification)

### Loss Function
Binary Cross Entropy (BCE):
$L = -\frac{1}{N} \sum [Y \log(A_2) + (1-Y) \log(1-A_2)]$

### Backward Pass (Gradients)
*Assuming Sigmoid Output:*
1. $dZ_2 = A_2 - Y$
2. $dW_2 = \frac{1}{N} A_1^T dZ_2$
3. $db_2 = \frac{1}{N} \sum dZ_2$
4. $dA_1 = dZ_2 W_2^T$
5. $dZ_1 = dA_1 \odot \text{Activation}'(Z_1)$
6. $dW_1 = \frac{1}{N} X^T dZ_1$
7. $db_1 = \frac{1}{N} \sum dZ_1$

### UI/UX Layout Mapping
* **Left Column:** 2D Scatter plot containing the raw training data. A dynamically updating filled contour map serves as the background, representing the model's decision surface. 
* **Middle Column:** A static graph structure of the 2-3-1 network.
  - Nodes will display real-time activation values ($A_1[i]$) for a chosen data point.
  - Edges vary in thickness and hue (e.g. green for positive, red for negative weights).
* **Right Column:** 
  - Sliders mapped linearly to $W_1, b_1, W_2, b_2$.
  - Dropdown for selecting activation functions in the hidden layer.
  - Frame control buttons (Play, Pause, Step) and state machine parameters (Learning Rate).
