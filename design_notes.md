# Design Notes: Mechanistic Interpretability Dashboard

*Last updated: April 2025 — Post Phase 1 & Phase 2 overhaul*

## Application Architecture
- **Framework:** Streamlit 1.54+
- **Visualization:** Plotly (all interactive charts, decision boundaries, network graphs, heatmaps)
- **ML Backend:** NumPy (Phase 1 from-scratch MLP), PyTorch (Phase 2 MNIST)
- **UI Components:** Custom CSS tooltips + `st.popover()` for glossary, `@st.fragment` for animation isolation
- **Math Rendering:** Streamlit's native `st.latex`

## Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/train_mnist.py     # Generate pretrained model & artifacts
streamlit run app.py
```

---

## Phase 1: 2-3-1 MLP (Poisonous Fruit Detector)

### Architecture
- **Input:** X ∈ ℝ^{N × 2} (Spikiness, Spottiness)
- **Hidden:** H ∈ ℝ^{N × 3} (3 hidden neurons = feature detectors)
- **Output:** ŷ ∈ ℝ^{N × 1} (P(Poisonous))
- **Total Parameters:** 13 (W1: 2×3, b1: 1×3, W2: 3×1, b2: 1×1)

### Data
- `make_moons(n=200, noise=0.15)` normalized to [0,1] range
- Labels: 0 = Safe 🍏, 1 = Poisonous ☠️
- Non-linearly separable → demonstrates need for hidden layers

### Forward Pass
1. Z₁ = X·W₁ + b₁
2. A₁ = f(Z₁) — Activation can be: Linear, Step, Sigmoid, ReLU, Tanh
3. Z₂ = A₁·W₂ + b₂
4. A₂ = σ(Z₂) — Output always Sigmoid for binary classification probability

### Loss & Backprop
- Binary Cross-Entropy: L = -1/N Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]
- Standard backprop with chain rule; all gradients exposed via `mlp.get_gradients()`

### Streamlit Animation Strategy
- `@st.fragment(run_every="250ms")` isolates the playground visualization zone
- Only decision boundary + network graph rerun during animation; rest of page is static
- Grid resolution drops from 50 to 30 during playback for performance
- Target: 4-5 effective FPS (sufficient for live workshop presentation)

### Feature: Hover Tooltips
- CSS-injected hover tooltips via `unsafe_allow_html=True`
- `tip(term, definition)` helper returns inline HTML spans
- Styled with glassmorphism dark gradient background
- Zero JavaScript required; pure CSS `:hover` + `visibility` transition

### Page Structure (4 educational sections)
1. **"The Problem"** — Narrative intro, linear model failure demo (LogReg ~85%)
2. **"The Architecture"** — Forward pass math, activation function gallery (5 functions + derivatives)
3. **"The Playground"** — Interactive training dashboard (Play/Step/Reset, weight sliders, sample inspector)
4. **"What Did the Neurons Learn?"** — Neuron labels + individual activation maps

---

## Phase 2: MNIST Scale-Up (784 → 128 → 64 → 10)

### Architecture
- **Small model:** 784 → 64 → 10 (50,890 params, ~97.4% test accuracy)
- **Large model (primary):** 784 → 128 → 64 → 10 (109,386 params, ~97.8% test accuracy)
- ReLU activations, Softmax output, Cross-Entropy loss
- Trained with Adam optimizer, lr=0.001, 15 epochs on MPS/CPU

### Pre-saved Artifacts (in `models/`)
| File | Description |
|---|---|
| `mnist_mlp.pt` | Large model state dict |
| `training_history.json` | Loss/accuracy curves for both architectures |
| `architecture_comparison.json` | Side-by-side metrics |
| `test_predictions.npz` | All 10K test predictions + probabilities |
| `sample_digits.npz` | 5 samples per digit class |
| `misclassified.npz` | First 20 wrong predictions with images |

### Introspection API (`MNIST_MLP`)
- `get_layer_activations(x)` → dict of all layer outputs for one sample
- `get_feature_maps(layer_idx)` → weight matrices reshaped as 28×28 heatmaps
- `get_top_activating_neurons(x, layer, k)` → indices + values of top-k neurons
- `load_pretrained(path)` → classmethod for instant model loading
- `count_parameters()` → total trainable params

### Page Structure (7 sections)
1. **"From Fruits to Pixels"** — Bridge from Phase 1, architecture comparison
2. **"Exploring MNIST"** — Digit grid, individual pixel inspector
3. **"Inside the Trained Network"** — Feature detector grid (top 16 neurons by weight magnitude)
4. **"Watch a Digit Flow Through"** — Layer-by-layer activation inspector
5. **"Draw Your Own Digit"** — Canvas + file upload fallback
6. **"Where the Model Struggles"** — Confusion matrix, confidence distribution, misclassified gallery
7. **Bridge to Advanced Topics** — CNNs, Transformers, mechanistic interpretability

---

## Shared UI Components (`src/ui_components.py`)

| Component | Purpose |
|---|---|
| `tip(term, def)` | Inline CSS hover tooltip |
| `glossary_popover(term, content)` | Rich `st.popover` for equations/diagrams |
| `metric_row(metrics)` | Styled metric card row |
| `section_header(title, subtitle, icon)` | Consistent section titles |
| `section_divider()` | Gradient horizontal rule |
| `math_block(latex, explanation)` | LaTeX + plain-English annotation |
| `TOOLTIP_CSS` | Injected once per page via `inject_tooltip_css()` |

---

## Known Limitations
- Streamlit caps at ~10 FPS for complex Plotly charts during `@st.fragment` animation
- `streamlit-drawable-canvas` may not install on all platforms; file upload fallback provided
- `use_container_width` is deprecated in Streamlit 1.54+; should migrate to `width='stretch'`
- Phase 2 model loading requires `models/mnist_mlp.pt` — run training script first
