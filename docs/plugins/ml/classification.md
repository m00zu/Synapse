# Classification

### AdaBoost

Trains an AdaBoost classifier.

??? note "Details"
    Options:
    
    - **target_column** -- column to predict
    - **n_estimators** -- number of weak learners (default 50)
    - **learning_rate** -- weight applied to each classifier (default 1.0)
    - **random_seed** -- for reproducibility

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `train` | table |
| **Output** | `result` | table |

**Properties:** `Estimators`, `Learning Rate`, `Random Seed`

---

### Decision Tree

Trains a Decision Tree classifier.

??? note "Details"
    Options:
    
    - **target_column** -- column to predict
    - **max_depth** -- max tree depth (0 = unlimited)
    - **min_samples_split** -- minimum samples to split a node (default 2)
    - **random_seed** -- for reproducibility

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `train` | table |
| **Output** | `result` | table |

**Properties:** `Max Depth (0=auto)`, `Min Samples Split`, `Random Seed`

---

### Extra Trees

Trains an Extra-Trees classifier (randomized RF variant -- often faster).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `train` | table |
| **Output** | `result` | table |

**Properties:** `Trees`, `Max Depth (0=auto)`, `Random Seed`

---

### Gradient Boosting

Trains a Gradient Boosting classifier.

??? note "Details"
    Options:
    
    - **target_column** -- column to predict
    - **n_estimators** -- number of boosting stages (default 100)
    - **max_depth** -- max depth of individual trees (default 3)
    - **learning_rate** -- shrinkage factor (default 0.1)
    - **random_seed** -- for reproducibility

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `train` | table |
| **Output** | `result` | table |

**Properties:** `Estimators`, `Max Depth`, `Learning Rate`, `Random Seed`

---

### Hist Gradient Boosting

Trains a histogram-based gradient boosting classifier (sklearn's LightGBM-equivalent; very fast on large tables).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `train` | table |
| **Output** | `result` | table |

**Properties:** `Iterations`, `Learning Rate`, `Max Depth (0=auto)`, `Random Seed`

---

### KNN Classifier

Trains a K-Nearest Neighbors classifier.

??? note "Details"
    Options:
    
    - **target_column** -- column to predict
    - **n_neighbors** -- number of neighbors (default 5)
    - **weights** -- weight function (uniform or distance)

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `train` | table |
| **Output** | `result` | table |

**Properties:** `K Neighbors`, `Weights`

---

### LDA

Trains a Linear Discriminant Analysis classifier.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `train` | table |
| **Output** | `result` | table |

**Properties:** `Solver`

---

### Logistic Regression

Trains a Logistic Regression classifier.

??? note "Details"
    Options:
    
    - **target_column** -- column to predict
    - **C** -- inverse regularization strength
    - **max_iter** -- maximum iterations
    - **solver** -- optimization algorithm

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `train` | table |
| **Output** | `result` | table |

**Properties:** `C (Regularization)`, `Max Iterations`, `Solver`

---

### MLP Classifier

Trains a multi-layer perceptron classifier.

??? note "Details"
    Hidden layer sizes are entered as a comma-separated list of ints, e.g.
    "100" for one 100-neuron layer or "100, 50" for two layers.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `train` | table |
| **Output** | `result` | table |

**Properties:** `Activation`, `L2 Penalty`, `Max Iterations`, `Random Seed`

---

### Naive Bayes

Trains a Gaussian Naive Bayes classifier.

??? note "Details"
    Options:
    
    - **target_column** -- column to predict

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `train` | table |
| **Output** | `result` | table |

---

### QDA

Trains a Quadratic Discriminant Analysis classifier.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `train` | table |
| **Output** | `result` | table |

**Properties:** `Regularization`

---

### Random Forest

Trains a Random Forest classifier.

??? note "Details"
    Options:
    
    - **target_column** -- column to predict
    - **n_estimators** -- number of trees (default 100)
    - **max_depth** -- max tree depth (0 = unlimited)
    - **random_seed** -- for reproducibility

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `train` | table |
| **Output** | `result` | table |

**Properties:** `Trees`, `Max Depth (0=auto)`, `Random Seed`

---

### Ridge Classifier

Trains a Ridge-regularized linear classifier (fast linear baseline).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `train` | table |
| **Output** | `result` | table |

**Properties:** `Alpha (L2)`, `Random Seed`

---

### SVM Classifier

Trains a Support Vector Machine classifier.

??? note "Details"
    Options:
    
    - **target_column** -- column to predict
    - **kernel** -- kernel type (rbf, linear, poly, sigmoid)
    - **C** -- regularization parameter
    - **gamma** -- kernel coefficient (scale or auto)

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `train` | table |
| **Output** | `result` | table |

**Properties:** `Kernel`, `C (Regularization)`, `Gamma`

---

### XGBoost Classifier

Trains an XGBoost classifier (gradient-boosted trees, optimized).

??? note "Details"
    Requires the ``xgboost`` package -- install with ``pip install xgboost``.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `train` | table |
| **Output** | `result` | table |

**Properties:** `Trees`, `Max Depth`, `Learning Rate`, `Subsample`, `Col Subsample`, `L1 (alpha)`, `L2 (lambda)`, `Random Seed`

---
