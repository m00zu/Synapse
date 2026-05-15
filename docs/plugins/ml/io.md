# IO

### Model Load

Loads a trained sklearn model from a joblib file.

??? note "Details"
    Outputs a SklearnModelData that can be connected to Predict or
    Cross Validation nodes.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `file_path` | path |
| **Output** | `sklearn_model` | sklearn_model |
| **Output** | `model` | model |

---

### Model Save

Saves a trained sklearn model to disk using joblib.

??? note "Details"
    Supports `.joblib` and `.pkl` file extensions.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `file_path_in` | path |

---
