# Descriptive & Comparison

### Data Summary

Computes pixel intensity histograms for images or descriptive statistics for DataFrames.

??? note "Details"
    Inputs:

    - **any** — an image (grayscale or RGB) or a pandas DataFrame
    - **mask** — optional mask to restrict image histograms to the masked region
    
    Outputs:

    - **table** — histogram bin counts (images) or `describe()` summary (DataFrames)
    - **figure** — distribution plot of the input data

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | in |
| **Input** | `mask` | mask |
| **Output** | `table` | table |
| **Output** | `fig` | fig |

---

### Outlier Detection

Detects and removes outliers in numerical data using statistical tests.

??? note "Details"
    Methods:

    - *ROUT (Prism Regression)* — robust nonlinear regression-based detection
    - *ROUT (Fast Math)* — faster variant of the ROUT method
    - *Grubbs* — classical single-outlier test applied iteratively
    
    - **Threshold** — Q value (ROUT) or alpha significance level (Grubbs).
    
    Outputs two tables: rows kept and rows removed.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `kept` | table |
| **Output** | `removed` | table |

**Properties:** `Method`

---

### Grouped Comparison

Tests whether there are significant differences among two or more groups.

??? note "Details"
    Tests:

    - *One-Way ANOVA* — parametric, assumes normal distribution and equal variances
    - *Kruskal-Wallis* — non-parametric rank-based alternative to ANOVA
    
    Outputs a summary table with test statistic, p-value, and significance flag.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | in |
| **Output** | `stats_table` | stats_table |

**Properties:** `Statistical Method`

---

### Pairwise Comparison

Performs pairwise comparisons between groups using parametric or non-parametric tests.

??? note "Details"
    Tests:

    - *Student's T-test* — parametric, assumes equal variance and normal distribution
    - *Welch's T-test* — parametric, does not assume equal variance
    - *Mann-Whitney U* — non-parametric rank-based test
    - *Kolmogorov-Smirnov* — tests whether two groups come from the same distribution
    - *Tukey HSD* — post-hoc test after ANOVA
    - *Dunn* — non-parametric post-hoc test (requires scikit-posthocs)
    - *Fisher's Z* — compare correlation coefficients between groups (target column = r values)
    
    - **Alternative** — two-sided (default), greater (group1 > group2), or less (group1 < group2). Tukey HSD and Dunn are always two-sided.
    
    - **P-Adj Method** — multiple comparison correction (Bonferroni, Holm, BH).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `stats_table` | stat |

**Properties:** `Statistical Method`, `Alternative`, `P-Adj Method`

---

### Normality Test

Tests whether each numerical column in a DataFrame follows a normal distribution.

??? note "Details"
    Tests:

    - *Shapiro-Wilk* — recommended for small to moderate samples
    - *Kolmogorov-Smirnov* — compares against a theoretical normal CDF
    - *Anderson-Darling* — weighted variant sensitive to distribution tails
    
    Outputs a summary table with test statistic, p-value (where applicable), and pass/fail result.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `results` | table |

**Properties:** `Test(s)`

---

### Pairwise Matrix

Computes a pairwise correlation or distance matrix for all numeric columns and visualises it as a heatmap.

??? note "Details"
    Correlation methods:

    - *Pearson* — linear correlation coefficient, assumes normality
    - *Spearman* — rank-based, robust to outliers and non-normal distributions
    - *Kendall* — rank-based, slower but more exact for small sample sizes
    
    Outputs a matrix table (for further analysis) and an annotated heatmap figure.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `table` | table |
| **Output** | `figure` | figure |

**Properties:** `Metric`, `Colormap`, ``, ``

---
