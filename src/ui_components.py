"""
Shared UI components for the Mechanistic Interpretability Dashboard.
Provides reusable tooltip, glossary, metric, and formatting utilities.
"""
import streamlit as st

# ---------------------
# Tooltip CSS & Helpers
# ---------------------

TOOLTIP_CSS = """
<style>
/* --- Custom Hover Tooltip --- */
.ct {
    position: relative;
    display: inline;
    cursor: help;
    border-bottom: 1.5px dotted #6C63FF;
    color: #6C63FF;
    font-weight: 500;
}
.ct .ctt {
    visibility: hidden;
    width: 280px;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    color: #e0e0e0;
    text-align: left;
    border-radius: 10px;
    padding: 12px 16px;
    position: absolute;
    z-index: 9999;
    bottom: 130%;
    left: 50%;
    margin-left: -140px;
    opacity: 0;
    transition: opacity 0.25s ease-in-out;
    font-size: 0.82rem;
    line-height: 1.45;
    font-weight: 400;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    border: 1px solid rgba(108, 99, 255, 0.3);
    pointer-events: none;
}
.ct .ctt::after {
    content: "";
    position: absolute;
    top: 100%;
    left: 50%;
    margin-left: -6px;
    border-width: 6px;
    border-style: solid;
    border-color: #16213e transparent transparent transparent;
}
.ct:hover .ctt {
    visibility: visible;
    opacity: 1;
}

/* --- Metric Card Styling --- */
.metric-card {
    background: linear-gradient(135deg, rgba(108,99,255,0.08) 0%, rgba(72,52,212,0.05) 100%);
    border: 1px solid rgba(108,99,255,0.2);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
    margin: 4px;
}
.metric-card .metric-label {
    font-size: 0.78rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}
.metric-card .metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #6C63FF;
}

/* --- Section Divider --- */
.section-divider {
    border: none;
    height: 2px;
    background: linear-gradient(90deg, transparent 0%, rgba(108,99,255,0.3) 50%, transparent 100%);
    margin: 2rem 0;
}
</style>
"""


def inject_tooltip_css():
    """Inject the custom CSS for hover tooltips and metric cards. Call once per page."""
    st.markdown(TOOLTIP_CSS, unsafe_allow_html=True)


def tip(term: str, definition: str) -> str:
    """
    Returns an HTML string for an inline hover tooltip.
    Use inside st.markdown(..., unsafe_allow_html=True).
    
    Example:
        st.markdown(f"The {tip('activation function', 'A mathematical ...')} transforms the input.", 
                     unsafe_allow_html=True)
    """
    # Escape quotes in definition to prevent HTML breakage
    safe_def = definition.replace('"', '&quot;').replace("'", "&#39;")
    return f'<span class="ct">{term}<span class="ctt">{safe_def}</span></span>'


def glossary_popover(term: str, content: str, icon: str = ""):
    """
    Creates a st.popover with rich content for longer explanations.
    Can contain LaTeX, markdown, etc.
    
    Args:
        term: The button label text
        content: Markdown content to display inside the popover
        icon: Emoji icon prefix
    """
    with st.popover(f"{icon} {term}"):
        st.markdown(content)


def metric_row(metrics: dict):
    """
    Renders a horizontal row of styled metric cards.
    
    Args:
        metrics: dict of {label: value} pairs, e.g. {"Loss": "0.342", "Accuracy": "91.2%"}
    """
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items()):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)


def section_header(title: str, subtitle: str = "", icon: str = ""):
    """Renders a consistent section header with optional subtitle and icon."""
    header = f"## {icon} {title}" if icon else f"## {title}"
    st.markdown(header)
    if subtitle:
        st.caption(subtitle)


def section_divider():
    """Renders a styled gradient divider between sections."""
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)


def math_block(latex: str, explanation: str = ""):
    """
    Renders a LaTeX equation block with an optional plain-English annotation below.
    
    Args:
        latex: LaTeX string (without $$ delimiters)
        explanation: Plain-English explanation shown as a caption below
    """
    st.latex(latex)
    if explanation:
        st.caption(f"↳ {explanation}")
