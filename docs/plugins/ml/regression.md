# Regression

### Bayesian Ridge

Trains a Bayesian Ridge regression model (uncertainty-aware linear).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `train` | table |
| **Output** | `result` | table |

**Properties:** `Max Iterations`

---

### Decision Tree Regressor

Trains a single Decision Tree regressor.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `train` | table |
| **Output** | `result` | table |

**Properties:** `Max Depth (0=auto)`, `Min Samples Split`, `Random Seed`

---

### Elastic Net

Trains an Elastic Net (mixed L1/L2) regression model.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `train` | table |
| **Output** | `result` | table |

**Properties:** `Alpha`, `L1 Ratio`, `Max Iterations`, `Random Seed`

---

### Extra Trees Regressor

Trains an Extra-Trees regressor.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `train` | table |
| **Output** | `result` | table |

**Properties:** `Trees`, `Max Depth (0=auto)`, `Random Seed`

---

### GB Regressor

Trains a Gradient Boosting regressor.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `train` | table |
| **Output** | `result` | table |

**Properties:** `Estimators`, `Max Depth`, `Learning Rate`, `Random Seed`

---

### Hist Gradient Boosting Regressor

Trains a histogram-based gradient boosting regressor (very fast on large tables).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `train` | table |
| **Output** | `result` | table |

**Properties:** `Iterations`, `Learning Rate`, `Max Depth (0=auto)`, `Random Seed`

---

### KNN Regressor

Trains a K-Nearest-Neighbours regressor.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `train` | table |
| **Output** | `result` | table |

**Properties:** `Neighbours`, `Weights`

---

### Lasso

Trains a Lasso (L1-regularized) regression model.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `train` | table |
| **Output** | `result` | table |

**Properties:** `Alpha (L1)`, `Max Iterations`, `Random Seed`

---

### Linear Regression

Trains an ordinary least-squares Linear Regression model.

??? note "Details"
    Options:

    - **target_column** -- column to predict
    - **feature_columns** -- feature columns (blank → all numeric)
    - **fit_intercept** -- whether to calculate the intercept

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `train` | table |
| **Output** | `result` | table |

**Properties:** `Fit Intercept`

---

### MLP Regressor

Trains a multi-layer perceptron regressor.

??? note "Details"
    Hidden layer sizes are entered as a comma-separated list of ints, e.g.
    "100" for one 100-neuron layer or "100, 50" for two layers.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `train` | table |
| **Output** | `result` | table |

**Properties:** `Activation`, `L2 Penalty`, `Max Iterations`, `Random Seed`

---

### RF Regressor

Trains a Random Forest regressor.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `train` | table |
| **Output** | `result` | table |

**Properties:** `Trees`, `Max Depth (0=auto)`, `Random Seed`

---

### Ridge

Trains a Ridge (L2-regularized) regression model.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `train` | table |
| **Output** | `result` | table |

**Properties:** `Alpha (L2)`, `Random Seed`

---

### SVR

Trains a Support Vector Regression model.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `train` | table |
| **Output** | `result` | table |

**Properties:** `Kernel`, `C (Regularization)`, `Epsilon`

---

### XGBoost Regressor

Trains an XGBoost regressor (gradient-boosted trees, optimized).

??? note "Details"
    Requires the ``xgboost`` package -- install with ``pip install xgboost``.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `train` | table |
| **Output** | `result` | table |

**Properties:** `Trees`, `Max Depth`, `Learning Rate`, `Subsample`, `Col Subsample`, `L1 (alpha)`, `L2 (lambda)`, `Random Seed`

---
