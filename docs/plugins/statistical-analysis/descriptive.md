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
| **Input** | `in` | table |
| **Output** | `stats_table` | stat |

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
    - *Two-sample Z-test* — compares means when variance is known or n is large
    - *Permutation test* — non-parametric, no distributional assumptions, resampling-based
    - *Tukey HSD* — post-hoc test after ANOVA
    - *Dunn* — non-parametric post-hoc test (requires scikit-posthocs)
    - *Fisher's Z* — compare correlation coefficients between groups (target column = r values)
    
    - **Alternative** — two-sided (default), greater (group1 > group2), or less (group1 < group2). Tukey HSD and Dunn are always two-sided.
    
    - **P-Adj Method** — multiple comparison correction (Bonferroni, Holm, BH).
    
    - **N Permutations** — number of resampling iterations for the permutation test (default 10,000).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `stats_table` | stat |

**Properties:** `Statistical Method`, `Alternative`, `N Permutations`, `P-Adj Method`

---

### Normality Test

Tests whether each numerical column in a DataFrame follows a normal distribution.

??? note "Details"
    Tests:

    - *Shapiro-Wilk* — recommended for small to moderate samples
    - *Kolmogorov-Smirnov* — compares against a theoretical normal CDF
    - *Anderson-Darling* — weighted variant sensitive to distribution tails
    
    Outputs:

    - **results** — summary table with test statistic, p-value, and pass/fail per column.
    - **qq_plot** — Q-Q (quantile-quantile) plots for each column. Points following the red dashed reference line indicate normality; systematic curvature suggests non-normal distribution.
    
    Use the **Group Column** option to test normality per group (e.g. per treatment condition before running a t-test or ANOVA).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `results` | table |
| **Output** | `qq_plot` | figure |

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

### Variance Test

Tests whether two or more groups have equal variance (homoscedasticity).

??? note "Details"
    Use this to decide between Student's t-test (equal variance) and Welch's t-test
    (unequal variance), or to check ANOVA assumptions.
    
    Tests:

    - *Levene's test* — robust, works for non-normal data (recommended default)
    - *Bartlett's test* — more powerful but assumes normality
    - *F-test* — classical variance ratio test for exactly 2 groups (sensitive to non-normality)
    
    Outputs a table with test statistic and p-value per group pair (F-test) or
    for all groups at once (Levene, Bartlett).
    
    A significant p-value (< 0.05) means variances are NOT equal — use Welch's t-test.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `result` | table |

**Properties:** `Test`

---

### Effect Size

Calculates effect sizes for pairwise group comparisons.

??? note "Details"
    Measures how large the difference between groups is, complementing
    p-values from statistical tests. Journals increasingly require effect
    sizes alongside significance testing.
    
    Methods:

    - *Auto* — Cohen's d for 2 groups, Eta-squared for 3+ groups
    - *Cohen's d* — standardised mean difference (pooled SD)
    - *Hedges' g* — Cohen's d with small-sample bias correction
    - *Glass's delta* — mean difference divided by the control group SD
    - *Rank-biserial r* — effect size for Mann-Whitney U (non-parametric)
    - *Eta-squared* — proportion of variance explained (ANOVA-style)
    - *Omega-squared* — bias-corrected eta-squared
    
    Output columns: group1, group2, n1, n2, effect_size, ci_lower,
    ci_upper, magnitude, method.
    
    **magnitude** uses conventional thresholds:

    - Cohen's d / Hedges' g / Glass's delta: negligible < 0.2, small < 0.5, medium < 0.8, large >= 0.8
    - Eta-squared / Omega-squared: negligible < 0.01, small < 0.06, medium < 0.14, large >= 0.14

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `results` | table |

**Properties:** `Method`, `CI Level`, `Bootstrap Iterations`

---

### Descriptive Stats

Computes comprehensive descriptive statistics for numeric columns.

??? note "Details"
    Calculates per-group (or overall) statistics including central tendency,
    dispersion, shape, and confidence intervals — everything needed for a
    publication-ready summary table.
    
    Output columns: group, column, n, mean, median, std, sem, ci_lower, ci_upper, min, q1, q3, max, iqr, skewness, kurtosis, cv.
    
    - **group_col** — optional grouping column. If set, statistics are computed per group. Leave blank for overall stats.
    - **value_cols** — columns to summarise. Leave blank for all numeric.
    - **ci_level** — confidence interval level (default 0.95).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `results` | table |

**Properties:** `CI Level`

---

### Distribution Fit

Fits data to candidate probability distributions and ranks them by goodness-of-fit (AIC / BIC / Kolmogorov-Smirnov).

??? note "Details"
    Select which distributions to test, or use **All** to try every candidate.
    The node outputs a ranking table with fitted parameters and a figure
    overlaying the best-fit PDFs on the empirical histogram.
    
    Candidate distributions: Normal, Log-Normal, Exponential, Gamma, Weibull,
    Beta, Rayleigh, Uniform, Cauchy, Logistic, Pareto, Student-t, Inverse Gaussian.
    
    Outputs:

    - **results** — one row per tested distribution with shape/loc/scale params,
      log-likelihood, AIC, BIC, KS statistic, and KS p-value, sorted by AIC.

    - **figure** — histogram of the data with top-N best-fit PDF curves overlaid.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `results` | table |

**Properties:** `Distributions`, `Overlay Top-N`, `Histogram Bins`, `Fig Width`, `Fig Height`

---
