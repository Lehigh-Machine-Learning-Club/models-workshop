import streamlit as st

st.set_page_config(
    page_title="Mechanistic Interpretability Dashboard",
        layout="wide"
)

from src.ui_components import inject_tooltip_css, section_divider, tip

inject_tooltip_css()

st.title("Mechanistic Interpretability Dashboard")
st.caption("Lehigh Machine Learning Club • April 15, 2025 Workshop")

section_divider()

col_intro, col_nav = st.columns([1.2, 1])

with col_intro:
    st.markdown(f"""
    Welcome to the **interactive educational dashboard** for the Lehigh Machine Learning Club!
    
    This application is designed to **demystify machine learning models** by letting you visualize 
    and tweak their internal mechanics in real-time. Every weight, gradient, and activation is visible — 
    no black boxes here.
    
    Built with using {tip("NumPy", "The fundamental package for scientific computing in Python. We use it to implement neural networks from scratch — no frameworks, no shortcuts.")} 
    and {tip("Streamlit", "An open-source Python framework for building interactive data applications. All the UI you see here is pure Python code.")}.
    
    **Event:** Creating Your First Model — An Introductory Workshop  
    **Focus:** Regression, Classification, and Neural Networks  
    **Philosophy:** If you can't see it, you can't understand it.
    """, unsafe_allow_html=True)

with col_nav:
    st.markdown("### Dashboard Sections")
    st.markdown("""
    | Section | What You'll Learn |
    |---|---|
    | **1. Linear Regression** | The foundational building blocks of predictive lines and planes |
    | **2. Classification** | KNN and Logistic Regression decision boundaries |
    | **3. Neural Networks: Toy MLP** | Build intuition for neural networks with a tiny 2-3-1 MLP — from the dataset problem, through NN concepts, to an interactive playground |
    | **4. Neural Networks: MNIST** | Scale up to 784 pixels × 109K parameters — explore how a trained network sees handwritten digits |
    
    Use the **sidebar** to navigate between sections →
    """)

section_divider()

st.markdown("""
### How to Use This Dashboard

1. **Start with the Toy MLP** if this is your first time — it builds up from "Why can't a straight line classify these fruits?"
2. **Use the sidebar controls** to change activation functions and watch the decision boundary morph
3. **Use the playback slider** to scrub through 10,000 pre-computed training epochs — or click Play for animation
4. **Hover over underlined terms** for instant explanations of technical concepts
5. **Move to MNIST** once you've built intuition — same mechanics, just 784 dimensions instead of 2

> **Tip:** The Toy MLP playground uses pre-computed checkpoints for buttery-smooth animation. The MNIST section loads a pretrained PyTorch model so you can focus on exploring its internals.
""")
