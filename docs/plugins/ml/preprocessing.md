# Preprocessing

### Feature Selection

Selects the top K features based on statistical tests.

??? note "Details"
    Options:
    
    - **target_column** -- the column to predict
    - **k** -- number of top features to keep
    - **method** -- scoring function (f_classif, mutual_info_classif, f_regression)

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `table` | table |
| **Output** | `selected` | table |
| **Output** | `scores` | table |

**Properties:** `Top K Features`, `Method`

---

### Label Encoder

Encodes categorical columns to integer labels.

??? note "Details"
    Each unique value in the selected columns is mapped to an integer
    (0, 1, 2, ...). Useful for converting string labels before training.
    
    Options:
    
    - **columns** -- columns to encode (comma-separated)

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `table` | table |
| **Output** | `encoded` | table |

---

### MinMax Scaler

Scales numeric columns to a given range (default 0-1).

??? note "Details"
    Outputs the scaled table and the fitted scaler model (for applying
    the same transform to test data).
    
    Options:
    
    - **columns** -- columns to scale (blank = all numeric)

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `table` | table |
| **Input** | `sklearn_model` | sklearn_model |
| **Input** | `fitted_scaler` | fitted_scaler |
| **Output** | `scaled` | table |

---

### Standard Scaler

Scales numeric columns to zero mean and unit variance.

??? note "Details"
    Outputs the scaled table and the fitted scaler model (for applying
    the same transform to test data).
    
    Options:
    
    - **columns** -- columns to scale (blank = all numeric)

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `table` | table |
| **Input** | `sklearn_model` | sklearn_model |
| **Input** | `fitted_scaler` | fitted_scaler |
| **Output** | `scaled` | table |

---

### Train/Test Split

Splits a table into training and testing sets.

??? note "Details"
    Options:
    
    - **target_column** -- the column to predict
    - **test_size** -- fraction of data for testing (0.0-1.0)
    - **random_seed** -- for reproducibility (0 = random)
    - **stratify** -- preserve class proportions in the split

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `table` | table |
| **Output** | `train` | table |
| **Output** | `test` | table |

**Properties:** `Test Size`, `Random Seed`, `Stratify`

---
