# **ROLE AND OBJECTIVE**

You are an expert AI coding agent and data scientist. Your objective is to build an interactive educational web dashboard using Streamlit.

This dashboard will demonstrate machine learning methodologies, progressing from fundamental Regression and Classification up to Multi-Layer Perceptron (MLP) Neural Networks, with a strong focus on **mechanistic interpretability** and the mathematical foundations of ML.

We will dissect and visualize what goes on under the hood in each methodology. The main page will have different "sections," each showing core components of the models (equations, decision boundaries, errors, animations, etc.).

## **SECTION 1: Linear Regression**

The core idea is to dissect and understand the core components, parameters, and biases of the linear regression model in detail and help visualize it.

* Take a standard dataset like the auto-mpg dataset.  
* Build dynamic selection of features (widget / dashboard / toggle buttons, Multiselect options).  
* Display real-time mathematical notations as features are selected (3blue1brown-style) \- both the core regression equation and the full equation based on the features selected.  
* Visualize the process of model evaluation (train-test split, loss estimation, error metrics, etc.).  
* Include dynamic visualizations to see the errors change in real-time.  
* Demonstrate Simple/multiple linear regression, and nonlinear/polynomial by allowing the user to play around with model features and params.

## **SECTION 2: Classification**

* Take a standard dataset like the iris dataset.  
* Add features to allow manual changes to the parameters of the current model.

### **KNN**

* Quickly demonstrate "classification" using a simple K-Nearest-Neighbor classification algorithm.  
* Allow the user to select either 2D or 3D visualization (3D must be rotatable).  
* Allow the user to change the value of "k" and see boundaries change in real-time.  
* Show what k=1 means (100% training accuracy). If data points are exactly equal in the iris dataset, add a randomized decimal place to ensure points aren’t exactly overlapping.

### **Logistic Regression**

* Build a dynamic dashboard similar to the linear regression section.  
* Explain visually how logistic regression is analogous to linear regression and how it's still fundamentally a "linear" model.  
* Highlight binary vs. multi-class classification boundaries.

## **SECTION 3: Neural Networks and MLP (Mechanistic Interpretability)**

Create an interactive dashboard about the basics of Neural Networks/MLP, heavily inspired by 3Blue1Brown and Sebastian Lague's visual style. The dashboard is designed to visualize the internal mechanics of a simple neural network in real-time. The goal is to demystify the "black box" by showing exactly how weights, biases, and hidden activations map to decision boundaries during training.

**The Universal Approximation Context:** While we are building a miniature model, the dashboard's core objective is to illustrate the foundational concept of neural networks as **general function approximators**. By watching the network warp its decision boundary in real-time, the user will see firsthand how these simple mathematical operations (weights, biases, and non-linearities) can be scaled and combined to approximate any arbitrary, highly complex function or problem.

### **Section 3, Phase 1: Simplest Example - Poisonous Fruit Detector**

#### **1\. Dataset & Architecture**

* **Dataset:** A synthetic, non-linear "toy" dataset (e.g., predicting fruit types based on features like "spikes" and "spots" \- blue vs. red fruits). Although this is a small-scale problem, it acts as a perfect proxy for understanding how networks handle non-linear function approximation in n-dimensional space.  
* **Network Architecture:** \* Input Layer: 2 nodes (features).  
  * Hidden Layer: Exactly 1 layer with **3 nodes**. (Chosen to keep the parameter count manageable for manual human tweaking, while still providing enough degrees of freedom to solve a non-linear problem).  
  * Output Layer: 1 node (binary classification).  
* **Implementation Strategy (From Scratch):** To maximize educational value and fully expose the underlying mechanics, this network will be built entirely from scratch using raw math and NumPy. We are intentionally avoiding high-level frameworks like PyTorch for this phase. Every forward pass, loss calculation, and backpropagation step (calculating derivatives and updating weights) will be explicitly coded using linear algebra. This ensures nothing is hidden behind auto-differentiation black boxes.  
* **Scalability Note:** This specific architecture is tiny, but it contains all the fundamental building blocks. The mechanics visualized here (gradient descent, activation thresholds, feature abstraction) are the exact same mechanics used when scaling up to arbitrary problem domains, whether that's image recognition or time-series forecasting.

#### **2\. UI/UX Layout (Desktop-Optimized)**
The dashboard will be divided into three primary vertical columns/blocks to cleanly separate the mathematical input from the visual output.

**1\. Left Column: The Output (Data & Decision Boundary)**

* **2D Scatter Plot:** The core visualization of the dataset (e.g., 'spikes' on the x-axis, 'spots' on the y-axis). Data points are color-coded by their true class (e.g., Blue vs. Red fruits).  
* **Dynamic Decision Surface:** A background contour map that updates in real-time, showing the network's continuous prediction landscape. The boundary where the contour shifts colors represents the exact decision threshold.  
* **Live Metrics Header:** A clean, bold display at the top of the plot showing the current global Loss and Accuracy.

**2\. Middle Column: The Mechanism (Network Architecture)**

* **Node & Edge Graph:** A visual topology of the 2-3-1 network.  
* **Live Activations:** Hidden and output nodes are drawn as large circles. The exact numerical activation value for a specific forward pass is printed directly inside the circle.  
* **Activation Heatmap:** The node circles dynamically shift color (e.g., pale yellow for near 0, deep red for near 1\) to visually represent their firing intensity.  
* **Activation Function Visualizer:** A small thumbnail or icon next to the hidden layer showing the mathematical curve of the currently selected activation function (e.g., a "S" curve for Sigmoid or a "V" shape for ReLU).  
* **Dynamic Connections:** The lines connecting nodes represent weights. Their thickness corresponds to the absolute magnitude of the weight, and their color indicates the sign (e.g., green for positive/excitatory, red for negative/inhibitory). During backpropagation, these lines will briefly pulse to indicate gradient flow.  
* **Latent Feature Labels:** The 3 hidden nodes will have brief, playful labels attached to them (e.g., "The Spiky Spotter") to constantly remind the user that they act as distinct feature detectors.

**3\. Right Column: The Controls (Sidebar & Parameters)**

* **Global Dashboard:** At the very top, a prominent counter showing the current Training Step/Epoch.  
* **Activation Function Selector:** A dropdown menu allowing users to choose the mathematical transformation used in the hidden layer:  
* **None (Linear):** To demonstrate why hidden layers are useless without non-linearity.  
* **Step Function:** The classic perceptron approach.  
* **Sigmoid:** For smooth, probabilistic transitions.  
* **ReLU (Rectified Linear Unit):** The modern standard for deep learning.  
* **Tanh:** For centered activations.  
* **Playback Suite:** Play, Pause, and Frame-by-Frame 'Step' buttons, alongside a slider to control the animation Speed (FPS).  
* **Parameter Sliders (Grouped):**  
  * Organized strictly by layer (Hidden Layer, Output Layer).  
  * Within each layer, separated logically: **Weights first**, followed by **Biases**.  
  * Each slider features its precise numerical value displayed directly above the handle for instant readability.  
* **Reset/Initialize Button:** A quick way to randomize weights or set them to zero to start the manual exploration process over.

#### **3\. Visualizing the Mechanics (The Core Loop)**

The animation will operate on a frame-by-frame basis.

* **Global Counter:** A metric display showing the total number of training steps/epochs completed.  
* **Forward Pass Visualization:**  
  * The decision boundary on the scatter plot updates.  
  * **Node Activations:** The hidden nodes are drawn as circles. The *actual calculated activation value* for a given data point is printed directly inside the circle.  
  * **Node Coloring:** The circles change color dynamically based on their activation level (e.g., yellow for low activation, red for high activation).  
* **Backward Pass Visualization:**  
  * Focuses on the parameter updates.  
  * Network connections (lines) briefly pulse in color or change thickness to signify gradient flow and weight adjustments.  
  * (Optional) Small indicators next to the sliders showing the delta (![][image1]) of the change.

#### **4\. User Controls & Interactions**

The right-hand sidebar will house all interactive elements, giving the user complete control over both the automated training process and manual parameter tuning.

* **Activation Selection Logic:** Changing the activation function in the sidebar instantly resets the "forward pass" view. If the network is currently training, it continues using the new math. This allows users to "hot-swap" functions to see how a complex, curvy boundary (Sigmoid) suddenly collapses into a rigid linear one (None).

* **Playback & Animation Controls:**  
  * **The "Frame" Concept:** The animation logic strictly defines **1 Frame \= 1 Iteration (1 Forward Pass \+ 1 Backward Pass)**. This ensures the user sees the boundary calculation (forward) and the subsequent parameter update (backward) as a single cohesive unit of learning.  
  * **Play/Pause Toggle:** Starts or stops the automated training loop, allowing users to freeze the network at any point to inspect activations and boundaries.  
  * **Frame-by-Frame Stepping:** A button to manually advance the network by exactly one iteration, perfect for slowly observing the gradient updates.  
  * **Animation Speed (FPS):** A slider or dropdown to control playback speed. At the maximum speed (e.g., 60 frames per second), the network will execute 60 full backpropagations per second. Given the simple dataset, this allows the network to fully train in under a minute while still providing a smooth visual representation of the learning curve.  
* **Manual Parameter Sliders:**  
  * **Initialization & Manual Play:** Sliders can be set to start at zero (or another default baseline). Before ever pressing "Play," the user can manually drag the sliders to see how shifting a specific weight or bias instantly recalculates the forward pass and alters the decision boundary in the left column.  
  * **Visual Feedback:** Designed as horizontal sliders with the exact numerical value of the parameter displayed directly on top of the slider handle for instant, precise reading.  
  * **Grouping & Hierarchy:** Parameters are grouped strictly by layer to maintain mechanistic clarity. Within the hidden layer, they are sorted by type: **All Weights are listed first, followed immediately by All Biases directly below them.**

#### **5\. Interpretability & Latent Features**

* **Feature Detectors:** A dedicated UI section, tooltip, or pop-up explaining what each of the 3 hidden neurons has actually "learned."  
* Since hidden nodes represent latent feature combinations, this section will assign intuitive, playful names (e.g., "The Spiky Spotter") to help the user map the mathematical abstraction back to the physical dataset.


### **Section 3, Phase 2: Scaling to MNIST (Multi-Layer Perceptron)**
* **Objective:** Once the foundation with the toy dataset is built and tested, the core design principles (activation visualization, parameter tracking, feature abstraction) will be scaled up to a Multi-Layer Perceptron (MLP) trained on the MNIST handwritten digit dataset.  
* **Implementation Shift:** While Phase 1 is built entirely from scratch in NumPy to understand the raw mechanics, Phase 2 may transition to PyTorch to handle the significantly larger parameter space (e.g., 784 input nodes) efficiently while retaining the same interpretability dashboard principles.  
* **Visualizing Larger Hidden Layers:** Even with hundreds of hidden neurons, the underlying principle remains the same. The dashboard will demonstrate that each neuron acts as a specific feature detector—identifying particular visual patterns like curves, loops, or specific stroke angles within the 28x28 pixel grid.  
* **Feature Visualization Maps:** We will adapt the "Feature Detectors" UI from Phase 1 to show visual representations (e.g., activation maximization or heatmaps) of what specific features those deeper neurons are activating on, bridging the gap between abstract numbers and recognizable image strokes.  
* **Stepping Stone to Advanced Interpretability:** This phase will act as a crucial stepping stone to study layer-level activation dynamics and representations, preparing the groundwork for eventually analyzing more complex architectures like small language models and sequence models.
* **Interactive Input:** Include a canvas page to allow the user to draw their own digit doodle and watch the network evaluate it in real-time.


====================


# **NEXT STEPS & AGENT DIRECTIVES**

1. Read and analyze these instructions thoroughly. Think step-by-step about the technical architecture required to achieve this in python, and Streamlit.  
2. Acknowledge this plan and wait for the user's explicit approval to begin coding.  
3. **CRITICAL:** You must actively create and maintain a design\_notes.md file in the root workspace. Log the architecture, data schemas, mathematical assumptions, and any architectural blueprints discussed here.  
4. Treat design\_notes.md as your external working memory. Update it as you progress, keeping it concise but comprehensive. The developer uses this file for tracking blueprints and drafting presentation slides. Let's go.

5. Since Section 3 is the most complex and the most  important part of the project, let's get started with Section 3 first. We will later work on Section 1 and Section 2 after we are done with Section 3.