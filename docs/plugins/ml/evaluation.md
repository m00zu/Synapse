# Evaluation

### Classification Report

Generates a classification report (precision, recall, F1, accuracy).

??? note "Details"
    Options:
    
    - **true_column** -- column with true labels
    - **pred_column** -- column with predicted labels

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `table` | table |
| **Output** | `report` | table |

---

### Confusion Matrix

Generates a confusion matrix from predictions.

??? note "Details"
    Expects a table with true labels and predicted labels columns.
    Outputs a matrix table and a heatmap figure.
    
    Options:
    
    - **true_column** -- column with true labels
    - **pred_column** -- column with predicted labels
    - **normalize** -- normalize matrix values (row-wise)

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `table` | table |
| **Output** | `matrix` | table |

**Properties:** `Normalize`

---

### Cross Validation

Runs K-fold cross-validation on a model and dataset.

??? note "Details"
    Options:
    
    - **target_column** -- the column to predict
    - **cv_folds** -- number of folds (default 5)
    - **scoring** -- metric to evaluate (accuracy, f1_macro, r2, etc.)

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `sklearn_model` | sklearn_model |
| **Input** | `table` | table |
| **Input** | `model` | model |
| **Output** | `scores` | table |

**Properties:** `Folds`, `Scoring`

---

### Grid Search Classifier

K-fold grid search across hyperparameters for a classifier.

??? note "Details"
    Pick a model, fill in values to sweep (one value = fixed; multiple =
    swept), run K-fold cross-validation.  Outputs the refit best estimator
    and a per-combo results table sorted by rank.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `table` | table |
| **Output** | `sklearn_model` | sklearn_model |
| **Output** | `table` | table |

---

### Grid Search Regressor

K-fold grid search across hyperparameters for a regressor.

??? note "Details"
    Pick a model, fill in values to sweep (one value = fixed; multiple =
    swept), run K-fold cross-validation.  Outputs the refit best estimator
    and a per-combo results table sorted by rank.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `table` | table |
| **Output** | `sklearn_model` | sklearn_model |
| **Output** | `table` | table |

---

### Model Evaluation

Comprehensive evaluation of a fitted model on a held-out table.

??? note "Details"
    Auto-detects task from the model's ``SklearnModelData.task`` field and
    computes the appropriate metric set.
    
    Classification metrics:

    - accuracy, balanced accuracy
    - precision / recall / f1 (macro)
    - matthews correlation, cohen kappa
    - roc_auc (binary or one-vs-rest), log_loss (when predict_proba available)
    
    Regression metrics:

    - r2, explained variance
    - rmse, mse, mae, median absolute error
    - mape (mean absolute percentage error)
    
    Output is a 2-column TableData (``metric``, ``value``).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `sklearn_model` | sklearn_model |
| **Input** | `table` | table |
| **Input** | `model` | model |
| **Output** | `metrics` | table |

---

### Predict

Applies a trained model to new data and outputs predictions.

??? note "Details"
    Connects a trained model and a table. The node uses the model's stored
    feature_names to select the right columns automatically.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `sklearn_model` | sklearn_model |
| **Input** | `table` | table |
| **Input** | `model` | model |
| **Output** | `result` | table |

---

### SHAP Dependence

SHAP dependence plot -- how a single feature drives the prediction, coloured by an interacting feature.

??? note "Details"
    Inputs:

    - ``feature`` -- the feature to plot on the x-axis.  Use the expanded
                      name from a fingerprint / vector column (e.g. ``fp[42]``)
                      or just the column name for scalar features.  An integer
                      is interpreted as the column index.

    - ``interaction_feature`` -- feature used for colouring (or ``auto`` for
                                  the strongest interaction).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `sklearn_model` | sklearn_model |
| **Input** | `table` | table |
| **Input** | `model` | model |
| **Output** | `figure` | figure |

**Properties:** `Max Samples`, `Background Samples`

---

### SHAP Sample

Per-sample SHAP waterfall -- why did the model predict X for THIS row?

??? note "Details"
    Outputs:

    - ``figure`` -- waterfall plot showing each feature's push from the
                     base value to the final prediction for the chosen row.

    - ``contributions`` -- table of (feature, value, shap_contribution)
                             rows, sorted by |contribution| descending.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `sklearn_model` | sklearn_model |
| **Input** | `table` | table |
| **Input** | `model` | model |
| **Output** | `contributions` | table |

**Properties:** `Sample Index`, `Max Samples`, `Background Samples`, `Top N Features`

---

### SHAP Summary

Global SHAP summary -- which features matter most, with sign and spread.

??? note "Details"
    Outputs:

    - ``figure`` -- matplotlib bar plot of mean |SHAP| per feature.
    - ``table``  -- one row per feature (mean_abs_shap / mean_shap / std_shap),
                     sorted by mean |SHAP| descending.
    
    For multi-class classification, SHAP values for the predicted class of
    each sample are used (matches what the model actually decided).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `sklearn_model` | sklearn_model |
| **Input** | `table` | table |
| **Input** | `model` | model |
| **Output** | `summary` | table |

**Properties:** `Max Samples`, `Background Samples`, `Top N Features`, `Plot Type`

---

### SHAP Values

Raw SHAP value matrix as a table -- one row per sample, one column per feature, plus ``base_value`` and ``prediction`` columns.

??? note "Details"
    Useful as input to downstream nodes (Heatmap, dimensionality reduction
    on SHAP, custom analysis).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `sklearn_model` | sklearn_model |
| **Input** | `table` | table |
| **Input** | `model` | model |
| **Output** | `shap_values` | table |

**Properties:** `Max Samples`, `Background Samples`

---
