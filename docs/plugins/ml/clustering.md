# Clustering

### Agglomerative

Clusters data using Agglomerative (hierarchical) clustering.

??? note "Details"
    Adds a 'cluster' column to the output table.
    
    Options:
    
    - **columns** -- columns to cluster on (blank = all numeric)
    - **n_clusters** -- number of clusters (default 3)
    - **linkage** -- linkage criterion (ward, complete, average, single)

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `table` | table |
| **Output** | `result` | table |

**Properties:** `Clusters`, `Linkage`

---

### DBSCAN

Clusters data using the DBSCAN density-based algorithm.

??? note "Details"
    Adds a 'cluster' column to the output table. Noise points are
    labelled -1.
    
    Options:
    
    - **columns** -- columns to cluster on (blank = all numeric)
    - **eps** -- maximum distance between neighbours (default 0.5)
    - **min_samples** -- minimum points to form a cluster (default 5)

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `table` | table |
| **Output** | `result` | table |

**Properties:** `Epsilon`, `Min Samples`

---

### K-Means

Clusters data using K-Means algorithm.

??? note "Details"
    Adds a 'cluster' column to the output table with the assigned cluster
    label for each row. Also outputs the fitted model.
    
    Options:
    
    - **columns** -- columns to cluster on (blank = all numeric)
    - **n_clusters** -- number of clusters (default 3)
    - **random_seed** -- for reproducibility

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `table` | table |
| **Output** | `result` | table |

**Properties:** `Clusters`, `Random Seed`

---
