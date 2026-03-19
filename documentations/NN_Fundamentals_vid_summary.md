# **Neural Network Fundamentals: A Visual and Code Breakdown**

**Source Material:** "How to Implement a Simple MLP from Scratch Using C\#" (Uploaded Video)

## **1\. The Problem Setup and Data Visualization (0:00 \- 1:11)**

The video introduces a basic classification task: determining if a fictitious purple, spikey fruit is "safe" or "poisonous."

**The Dataset (Visualized as a 2D Scatter Plot):**

* **X-Axis (Feature 1):** Spot Size.  
* **Y-Axis (Feature 2):** Spike Length.  
* **Data Points:** Fruits are plotted on this 2D plane. Safe fruits are colored **Blue**, and poisonous fruits are colored **Red**.  
* **The Initial Distribution:** The data is "linearly separable," meaning a single straight diagonal line can cleanly divide the blue dots from the red dots.

## **2\. The Simplest Network: A Linear Classifier (1:12 \- 2:25)**

To draw this dividing line (the decision boundary), the video starts with the absolute simplest neural network architecture.

### **Network Architecture: 2 Inputs ![][image1] 2 Outputs**

\[Input Layer\]                        \[Output Layer\]  
(Node: Spot Size) ────── W\_{1,1} ──────\> (Node: Output 1 \- Safe Score)  
                  \\                    /  
                         \\      /  
(Node: Spike Len) ────── W\_{2,1} ──────\> (Node: Output 2 \- Poison Score)  
                        \[+ W\_{1,2}, W\_{2,2} omitted for clarity\]

* **Inputs:** 2 nodes representing the ![][image2] and ![][image3] coordinates of a fruit.  
* **Outputs:** 2 nodes representing the raw "score" for Safe vs. Poisonous. (The video notes that while a single output node ![][image4] or ![][image5] works for binary classification, using two outputs future-proofs the network for multi-class problems).  
* **Weights (![][image6]):** Each connection between an input and output has a weight. The output is calculated as the sum of inputs multiplied by their respective weights.  
* **Classification Rule:** If Output\_1 \> Output\_2, the network predicts **Safe**. Otherwise, **Poisonous**.

## **3\. Implementation & The "Origin" Problem (2:26 \- 4:05)**

### **Initial C\# Implementation (Hardcoded)**

The video shows a raw Classify function implementing the linear math:

// Weights connecting each input to the first output  
double weight\_1\_1, weight\_2\_1;  
// Weights connecting each input to the second output  
double weight\_1\_2, weight\_2\_2;

public int Classify(double input\_1, double input\_2) {  
    double output\_1 \= (input\_1 \* weight\_1\_1) \+ (input\_2 \* weight\_2\_1);  
    double output\_2 \= (input\_1 \* weight\_1\_2) \+ (input\_2 \* weight\_2\_2);  
      
    // Return 0 (Safe) if output\_1 is greater, otherwise 1 (Poisonous)  
    return (output\_1 \> output\_2) ? 0 : 1;  
}

### **The Visualization**

The video demonstrates a Visualize() loop that runs the Classify() function on *every single pixel* in the graph display, coloring the pixel blue or red depending on the prediction. This reveals the **Decision Boundary**.

**The Limitation:** By manually tweaking sliders mapped to weight\_1\_1, weight\_2\_1, etc., the creator shows that changing weights only *rotates* the decision boundary around the graph's origin (0,0). It cannot be shifted up, down, left, or right.

### **The Solution: Adding Biases**

To allow the boundary to translate (shift) away from the origin, **biases** are introduced. A bias is a constant value added to the weighted sum of a node.

// Added to the top of the script  
double bias\_1, bias\_2;

// Updated Classify function  
double output\_1 \= (input\_1 \* weight\_1\_1) \+ (input\_2 \* weight\_2\_1) \+ bias\_1;  
double output\_2 \= (input\_1 \* weight\_1\_2) \+ (input\_2 \* weight\_2\_2) \+ bias\_2;

*Visual Result:* Tweaking the bias sliders now translates the decision boundary across the Cartesian plane, allowing the creator to easily draw a line separating the safe and poisonous data points.

## **4\. The Non-Linear Challenge & Hidden Layers (4:06 \- 7:18)**

**The New Dataset:** The video introduces a more complex, non-linear dataset. Safe (blue) dots are grouped in the bottom-left and top-right corners, while poisonous (red) dots occupy a band in the middle. **A single straight line can no longer separate the classes.**

To solve this, the network must be made larger by adding **Hidden Layers**.

### **Refactoring to Object-Oriented C\#**

Because hardcoding weight\_1\_1 becomes impossible for larger networks, the code is refactored into a scalable Object-Oriented structure.

**1\. The Layer Class:**

public class Layer {  
    int numNodesIn, numNodesOut;  
    double\[,\] weights; // 2D array for matrix representation  
    double\[\] biases;

    public Layer(int numNodesIn, int numNodesOut) {  
        this.numNodesIn \= numNodesIn;  
        this.numNodesOut \= numNodesOut;  
        weights \= new double\[numNodesIn, numNodesOut\];  
        biases \= new double\[numNodesOut\];  
    }

    public double\[\] CalculateOutputs(double\[\] inputs) {  
        double\[\] weightedInputs \= new double\[numNodesOut\];  
          
        // Loop over every output node  
        for (int nodeOut \= 0; nodeOut \< numNodesOut; nodeOut++) {  
            double weightedInput \= biases\[nodeOut\];  
              
            // Sum (input \* weight) for all incoming connections  
            for (int nodeIn \= 0; nodeIn \< numNodesIn; nodeIn++) {  
                weightedInput \+= inputs\[nodeIn\] \* weights\[nodeIn, nodeOut\];  
            }  
            weightedInputs\[nodeOut\] \= weightedInput;  
        }  
        return weightedInputs;   
    }  
}

**2\. The NeuralNetwork Class:**

public class NeuralNetwork {  
    Layer\[\] layers;

    public NeuralNetwork(params int\[\] layerSizes) {  
        layers \= new Layer\[layerSizes.Length \- 1\];  
        // If layerSizes is \[2, 3, 2\], it creates a (2-\>3) layer and a (3-\>2) layer.  
        for (int i \= 0; i \< layers.Length; i++) {  
            layers\[i\] \= new Layer(layerSizes\[i\], layerSizes\[i \+ 1\]);  
        }  
    }

    public double\[\] CalculateOutputs(double\[\] inputs) {  
        // Feed forward pass  
        foreach (Layer layer in layers) {  
            inputs \= layer.CalculateOutputs(inputs);  
        }  
        return inputs;  
    }  
      
    // Classify function omitted for brevity (uses argmax on outputs)  
}

### **The "Linear Trap" Visualization**

The creator sets up a \[2, 3, 2\] network (2 inputs, 1 hidden layer with 3 nodes, 2 outputs).

\[Input (2)\]  \====\>  \[Hidden (3)\]  \====\>  \[Output (2)\]

*Visual Result:* The creator tweaks the new sliders, but **the decision boundary remains completely straight**.

*Mechanistic Explanation:* Stacking multiple linear transformations (matrix multiplications) mathematically collapses into a single linear transformation. The network is essentially still a single-layer perceptron.

## **5\. Activation Functions: The Secret Sauce (7:19 \- 10:15)**

To bend the boundary, the network requires a **non-linear activation function**. The video uses a biological analogy: a neuron (node) receives a stimulus (weighted input) and only "fires" (outputs a value) if a certain threshold is met.

### **The Step Function (Binary Activation)**

First, a simple Step Function is tested.

double ActivationFunction(double weightedInput) {  
    // If input \> 0, return 1\. Else, return 0\.  
    return (weightedInput \> 0\) ? 1 : 0;  
}

*Visual Result:* The decision boundaries become sharp, blocky, and jagged. The creator successfully encloses the blue dots, but notes that the sliders are highly sensitive and "jerky." The output is discontinuous.

### **The Sigmoid Function (Smooth Activation)**

To fix the discontinuous nature of the Step Function, smooth functions are introduced. The video briefly flashes diagrams for Hyperbolic Tangent (![][image7]), SiLU, and ReLU, before settling on the **Sigmoid** function: ![][image8].

double ActivationFunction(double weightedInput) {  
    return 1 / (1 \+ Math.Exp(-weightedInput));  
}

**Integrating it into the Layer class:**

The CalculateOutputs function is updated. Instead of returning raw weightedInputs, it passes them through the activation function first:

// Inside Layer.CalculateOutputs() loop:  
activations\[nodeOut\] \= ActivationFunction(weightedInput);  
// ...  
return activations;

### **The Final Visualization**

With the Sigmoid function active, the creator manipulates the sliders for the weights and biases of the \[2, 3, 2\] network.

* **Result:** The decision boundaries are now beautifully smooth curves.  
* **Bias Context:** The video notes that in this setup, adjusting a bias effectively shifts the input value left or right along the X-axis of the Sigmoid curve, changing how easily that specific node "fires."  
* **Conclusion:** The creator successfully manipulates the network parameters by hand to draw a curved, triangular boundary that perfectly encapsulates the "safe" blue dots, achieving 100% accuracy on the non-linear dataset.
