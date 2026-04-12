In `pages/1_Linear_Regression.py`, make the following changes:

STEP 1 — SIDEBAR CONTROLS
In the sidebar, after the existing `selected_features` multiselect widget,
add a conditional block: if len(selected_features) == 1, render the 
following widgets and store their values in variables:

- fit_mode = st.sidebar.radio("Optimization Mode", 
  ["Auto-Fit (Algorithm)", "Manual Fit (Human)"])
  
- show_residuals = st.sidebar.checkbox("Show Residuals", value=False)

- If fit_mode == "Manual Fit (Human)", show:
    manual_w = st.sidebar.slider("Weight (w)", -5.0, 5.0, 0.0, step=0.001)
    manual_b = st.sidebar.slider("Bias (b)", -50.0, 100.0, 0.0, step=0.1)
  Else, set manual_w = None, manual_b = None.

- If len(selected_features) != 1, force fit_mode = "Auto-Fit (Algorithm)" 
  and show_residuals = False.

STEP 2 — PREDICTION LOGIC BYPASS
After the existing model.fit() and model.predict() block, add:

  if fit_mode == "Manual Fit (Human)" and len(selected_features) == 1:
      y_train_pred = manual_w * X_train[:, 0] + manual_b
      y_test_pred  = manual_w * X_test[:, 0]  + manual_b
      mse_train = mean_squared_error(y_train, y_train_pred)
      mse_test  = mean_squared_error(y_test,  y_test_pred)
      r2_test   = r2_score(y_test, y_test_pred)
      r2_train  = r2_score(y_train, y_train_pred)

Do not change anything inside the sklearn training block itself.

STEP 3 — RESIDUAL LINES ON 2D PLOT
Inside the existing single-feature 2D Plotly figure block 
(where len(selected_features) == 1), before the regression 
fit line trace is added, insert:

  if show_residuals:
      for xi, yi_true, yi_pred in zip(X_test[:, 0], y_test, y_test_pred):
          fig.add_trace(go.Scatter(
              x=[xi, xi],
              y=[yi_true, yi_pred],
              mode='lines',
              line=dict(color='rgba(100,100,100,0.4)', width=1, dash='dash'),
              showlegend=False,
              hoverinfo='skip'
          ))

STEP 3.5 — FIX THE REGRESSION LINE DRAW LOGIC
In the single-feature 2D Plotly figure block, find where x_range 
and y_range are computed for drawing the red regression line:

    x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 200).reshape(-1, 1)
    y_range  = model.predict(poly.transform(x_range))

Replace this with a conditional:

    x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 200).reshape(-1, 1)
    
    if fit_mode == "Manual Fit (Human)":
        y_range = manual_w * x_range[:, 0] + manual_b
    else:
        y_range = model.predict(poly.transform(x_range))

This ensures the red line, the residual lines, and the metrics all 
respond to the same source of truth — the manual sliders in Manual 
mode, and the sklearn model in Auto-Fit mode.

STEP 4 — SIDEBAR LABEL FEEDBACK
After the fit_mode radio, if fit_mode == "Manual Fit (Human)", 
add a st.sidebar.caption that says:
"Drag w and b to fit the line manually. Can you beat the algorithm?"

Do not modify any other part of the file.

Prompt for 3D Residual Drop-lines
In `pages/1_Linear_Regression.py`, locate the block handling the 3D multiple linear regression visualization where `len(selected_features) == 2` and `poly_degree == 1`.

Make the following two visual upgrades to the Plotly figure. Do NOT change any training logic or other plotting blocks.

---

**1. Add 3D Residual Drop-lines**

Before the `go.Surface` trace is added, insert this loop to draw dotted vertical lines connecting each actual test point down to the predicted plane:

```python
if show_residuals:
    for i in range(len(X_test)):
        z_pred_point = model.predict(X_test[i].reshape(1, -1))[0]
        fig.add_trace(go.Scatter3d(
            x=[X_test[i, 0], X_test[i, 0]],
            y=[X_test[i, 1], X_test[i, 1]],
            z=[y_test[i], z_pred_point],
            mode='lines',
            line=dict(color='rgba(255,255,255,0.4)', width=2, dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ))
```

---

**2. Upgrade the Prediction Plane Surface**

Find the existing `go.Surface` trace and replace `colorscale='Blues'` with `colorscale='Viridis'`, set `opacity=0.7`, and ensure `showscale=False`:

```python
fig.add_trace(go.Surface(
    x=xx, y=yy, z=zz,
    colorscale='Viridis',
    opacity=0.7,
    name='Prediction Plane',
    showscale=False
))
```

---

Expected outcome:
- Dotted white drop-lines visually connect each test point to the regression plane, making residual magnitude clear in 3D
- The Viridis gradient (purple → green → yellow) maps depth across the plane, making it far more readable than a flat blue surface
