import streamlit as st

st.set_page_config(
    page_title="Mechanistic Interpretability Dashboard",
    page_icon="🧠",
    layout="wide"
)

st.title("Mechanistic Interpretability Dashboard 🧠")
st.markdown("""
Welcome to the interactive educational dashboard! This application is designed to demystify machine learning models by letting you visualize and tweak their internal mechanics in real-time.

### Navigation
Please select a module from the **Sidebar** to start exploring:

1. **Linear Regression:** Understand the foundational building blocks of predictive lines and planes.
2. **Classification:** Explore KNN and Logistic decision boundaries.
3. **Phase 1: Poisonous Fruit Detector:** Build an intuition for Neural Networks using a tiny 2-3-1 MLP and non-linear boundaries.
4. **Phase 2: MNIST Scale-Up:** See how the principles from Phase 1 scale up to image recognition and 784-dimensional space.
""")
