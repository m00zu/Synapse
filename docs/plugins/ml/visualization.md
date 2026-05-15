# Visualization

### Cluster Visualization

2D scatter plot, optionally colored by cluster or class labels.

??? note "Details"
    Pick the X / Y axes from existing columns (e.g. ``umap_0`` / ``umap_1``
    after running the UMAP node).  If left blank, PCA is run on all numeric
    columns to produce 2D coordinates automatically.
    
    Coloring is optional:

    - ``class_column``: a true class / category column (e.g. ``activity``,
        ``label``).  Takes precedence when set.  Legend shows the raw value;
        title reads "Class Visualization".

    - ``cluster_column``: integer cluster IDs from K-Means / DBSCAN /
        Agglomerative.  Used when ``class_column`` is blank.  Legend shows
        ``Cluster N`` (and ``-1`` is rendered as ``Noise`` for DBSCAN).

    - If neither selector is set, falls back to a column named ``cluster``
        if one exists.  Otherwise a plain (uncolored) scatter is drawn.
    
    Options:
    
    - **x_column** -- column for X axis (blank = auto PCA on numeric columns)
    - **y_column** -- column for Y axis (blank = auto PCA on numeric columns)
    - **cluster_column** -- column with cluster labels (optional)
    - **class_column** -- column with true class labels (overrides cluster_column)

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `table` | table |
| **Output** | `figure` | figure |

---

### Feature Importance

Plots feature importance from a trained model as a horizontal bar chart.

??? note "Details"
    Works with tree-based models (Random Forest, Gradient Boosting, etc.)
    that expose a `feature_importances_` attribute, or linear models with
    `coef_`.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `sklearn_model` | sklearn_model |
| **Input** | `model` | model |
| **Output** | `importance` | table |

---

### Learning Curve

Plots training vs validation score as a function of training set size.

??? note "Details"
    Helps diagnose overfitting or underfitting.
    
    Options:
    
    - **target_column** -- column to predict
    - **cv_folds** -- number of cross-validation folds (default 5)
    - **scoring** -- scoring metric (accuracy, f1_macro, r2, etc.)

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `sklearn_model` | sklearn_model |
| **Input** | `table` | table |
| **Input** | `model` | model |
| **Output** | `figure` | figure |

**Properties:** `CV Folds`, `Scoring`

---

### Precision-Recall Curve

Plots a Precision-Recall curve with AUPRC.

??? note "Details"
    For binary classification, plots a single curve. For multi-class,
    plots one-vs-rest curves for each class.
    
    Options:
    
    - **true_column** -- column with true labels
    - **pred_column** -- column with prediction probabilities or scores

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `table` | table |
| **Output** | `figure` | figure |

---

### Regression Scatter

Scatter plot of true vs predicted values with an identity line.

??? note "Details"
    Useful for evaluating regression model performance visually.
    
    Options:
    
    - **true_column** -- column with true values
    - **pred_column** -- column with predicted values

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `table` | table |
| **Output** | `figure` | figure |

---

### ROC Curve

Plots a Receiver Operating Characteristic (ROC) curve with AUC.

??? note "Details"
    For binary classification, plots a single ROC curve. For multi-class,
    plots one-vs-rest curves for each class.
    
    Options:
    
    - **true_column** -- column with true labels
    - **pred_column** -- column with prediction probabilities or scores

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `table` | table |
| **Output** | `figure` | figure |

---
