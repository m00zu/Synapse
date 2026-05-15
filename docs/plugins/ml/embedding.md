# Embedding

### UMAP

UMAP dimensionality reduction (Rust backend).

??? note "Details"
    Takes a table, picks the chosen feature columns (scalar or 1-D ndarray
    columns both supported via ``build_xy``), computes KNN + PCA-or-random
    init, runs UMAP, and appends ``umap_0``, ``umap_1`` (... up to
    ``n_components - 1``) columns to the output table.
    
    Backed by the vendored ``umap_rs_py`` Rust crate -- fast brute-force KNN
    with rayon parallelism and the patched UMAP optimizer.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `table` | table |
| **Output** | `result` | table |

**Properties:** `Components`, `Neighbours`, `Min Dist`, `Spread`, `Metric`, `p (Minkowski only)`, `Init`, `Epochs (0=auto)`, `Random Seed`

---
