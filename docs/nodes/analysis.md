# Analysis

### Contingency Analysis

Tests categorical association using chi-square and Fisher's exact tests.

??? note "Details"
    Input types:

    - *Raw Data (two columns)* -- a crosstab is built automatically from two categorical columns
    - *Contingency Matrix* -- a pre-built matrix of observed counts
    
    Outputs:

    - **test_results** -- Pearson chi-square, Yates-corrected chi-square, and Fisher's exact (2x2)
    - **observed_counts** -- the observed contingency table
    - **expected_counts** -- expected counts under the null hypothesis

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `test_results` | stat |
| **Output** | `observed_counts` | table |
| **Output** | `expected_counts` | table |

**Properties:** `Input Type`

---

### Data Summary

Computes pixel intensity histograms for images or descriptive statistics for DataFrames.

??? note "Details"
    Inputs:

    - **any** -- an image (grayscale or RGB) or a pandas DataFrame
    - **mask** -- optional mask to restrict image histograms to the masked region
    
    Outputs:

    - **table** -- histogram bin counts (images) or `describe()` summary (DataFrames)
    - **figure** -- distribution plot of the input data

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | in |
| **Input** | `mask` | mask |
| **Output** | `table` | table |
| **Output** | `fig` | fig |

---

### Descriptive Stats

Computes comprehensive descriptive statistics for numeric columns.

??? note "Details"
    Calculates per-group (or overall) statistics including central tendency,
    dispersion, shape, and confidence intervals -- everything needed for a
    publication-ready summary table.
    
    Output columns: group, column, n, mean, median, std, sem, ci_lower, ci_upper, min, q1, q3, max, iqr, skewness, kurtosis, cv.
    
    - **group_col** -- optional grouping column. If set, statistics are computed per group. Leave blank for overall stats.
    - **value_cols** -- columns to summarise. Leave blank for all numeric.
    - **ci_level** -- confidence interval level (default 0.95).

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

    - **results** -- one row per tested distribution with shape/loc/scale params,
      log-likelihood, AIC, BIC, KS statistic, and KS p-value, sorted by AIC.

    - **figure** -- histogram of the data with top-N best-fit PDF curves overlaid.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `results` | table |

**Properties:** `Distributions`, `Overlay Top-N`, `Histogram Bins`, `Fig Width`, `Fig Height`

---

### Effect Size

Calculates effect sizes for pairwise group comparisons.

??? note "Details"
    Measures how large the difference between groups is, complementing
    p-values from statistical tests. Journals increasingly require effect
    sizes alongside significance testing.
    
    Methods:

    - *Auto* -- Cohen's d for 2 groups, Eta-squared for 3+ groups
    - *Cohen's d* -- standardised mean difference (pooled SD)
    - *Hedges' g* -- Cohen's d with small-sample bias correction
    - *Glass's delta* -- mean difference divided by the control group SD
    - *Rank-biserial r* -- effect size for Mann-Whitney U (non-parametric)
    - *Eta-squared* -- proportion of variance explained (ANOVA-style)
    - *Omega-squared* -- bias-corrected eta-squared
    
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

### Grouped Comparison

Tests whether there are significant differences among two or more groups.

??? note "Details"
    Tests:

    - *One-Way ANOVA* -- parametric, assumes normal distribution and equal variances
    - *Kruskal-Wallis* -- non-parametric rank-based alternative to ANOVA
    
    Outputs a summary table with test statistic, p-value, and significance flag.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `stats_table` | stat |

**Properties:** `Statistical Method`

---

### Linear Regression

Performs ordinary least-squares (OLS) linear or polynomial regression.

??? note "Details"
    Set **Degree** > 1 for polynomial regression (e.g. 2 = quadratic, 3 = cubic).
    With degree 1 (default), this is standard linear regression.
    
    Outputs:

    - **coefficients** -- slope, intercept, standard error, 95% CI, and p-values per parameter
    - **residuals** -- fitted values, residuals, and standardized residuals for downstream plotting
    
    Summary statistics: R², adjusted R², F-statistic, and F p-value.
    
              R-squared, coefficient, residuals, predict, fitted values,
              multiple regression, quadratic, cubic, standard curve, Bradford,
              線性回歸, 多項式迴歸, 迴歸分析, 最小二乘法, 斜率, 截距, 決定係數

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `coefficients` | stat |
| **Output** | `residuals` | table |
| **Output** | `curve` | table |

**Properties:** `Polynomial Degree`, ``

---

### Mixed Effects Model

Fits a linear mixed-effects model (LMM) for hierarchical / nested data.

??? note "Details"
    Mixed-effects models are essential when observations are grouped (e.g.
    cells within wells, animals within treatment groups, repeated measures
    per subject). They estimate **fixed effects** (population-level trends)
    and **random effects** (group-level deviations) simultaneously.
    
    Configuration:

    - **y_col** -- dependent (response) variable.
    - **fixed_cols** -- fixed-effect predictor(s), comma-separated.
    - **group_col** -- grouping variable for random intercepts (required).
    - **random_slope_col** -- optional predictor for random slopes.
    - **REML** -- use Restricted ML (default) or Full ML estimation.
    
    Outputs:

    - **fixed_effects** -- coefficient table with SE, z-value, p-value, 95% CI.
    - **random_effects** -- per-group random intercept (and slope) estimates.
    - **summary** -- model-level statistics: log-likelihood, AIC, BIC,
      number of groups, ICC.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `fixed_effects` | stat |
| **Output** | `random_effects` | table |
| **Output** | `summary` | stat |

**Properties:** ``

---

### Model Predict

Predicts Y values from a fitted model and a new data table.

??? note "Details"
    Connect the **model** output from Linear Regression or Nonlinear Regression,
    then provide a table with the X column to predict on.
    
    The node auto-detects the X column name from the model metadata.
    Override with the **X Column** field if the new table uses a different name.
    
    Outputs the input table with an added **Predicted** column.
    
              Bradford, ELISA, 預測, 插值, 標準曲線

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `data` | table |
| **Output** | `out` | table |

**Properties:** ``, `Inverse X Min (0=auto)`, `Inverse X Max (0=auto)`

---

### Nonlinear Regression

Fits nonlinear curves to XY data using `scipy.optimize.curve_fit`.

??? note "Details"
    Built-in models:

    - *4PL (EC50 / Dose-Response)* -- four-parameter logistic for IC50/EC50
    - *Hill Equation* -- sigmoidal binding/dose-response
    - *One-Phase Exponential Decay* -- single-rate decay to plateau
    - *Two-Phase Exponential Decay* -- fast + slow decay components
    - *Exponential Growth* -- unbounded exponential increase
    - *Michaelis-Menten* -- enzyme kinetics saturation curve
    - *Gompertz Growth* -- asymmetric sigmoidal growth
    - *Sigmoidal (Logistic)* -- symmetric S-curve
    
    Outputs best-fit parameters with 95% CI and a smooth predicted curve table.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `parameters` | stat |
| **Output** | `curve` | table |

**Properties:** `Model`, `X Min (0=auto)`, `X Max (0=auto)`

---

### Normality Test

Tests whether each numerical column in a DataFrame follows a normal distribution.

??? note "Details"
    Tests:

    - *Shapiro-Wilk* -- recommended for small to moderate samples
    - *Kolmogorov-Smirnov* -- compares against a theoretical normal CDF
    - *Anderson-Darling* -- weighted variant sensitive to distribution tails
    
    Outputs:

    - **results** -- summary table with test statistic, p-value, and pass/fail per column.
    - **qq_plot** -- Q-Q (quantile-quantile) plots for each column. Points following the red dashed reference line indicate normality; systematic curvature suggests non-normal distribution.
    
    Use the **Group Column** option to test normality per group (e.g. per treatment condition before running a t-test or ANOVA).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `results` | table |
| **Output** | `qq_plot` | figure |

**Properties:** `Test(s)`

---

### Outlier Detection

Detects and removes outliers in numerical data using statistical tests.

??? note "Details"
    Methods:

    - *ROUT (Prism Regression)* -- robust nonlinear regression-based detection
    - *ROUT (Fast Math)* -- faster variant of the ROUT method
    - *Grubbs* -- classical single-outlier test applied iteratively
    
    - **Threshold** -- Q value (ROUT) or alpha significance level (Grubbs).
    
    Outputs two tables: rows kept and rows removed.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `kept` | table |
| **Output** | `removed` | table |

**Properties:** `Method`

---

### Pairwise Comparison

Performs pairwise comparisons between groups using parametric or non-parametric tests.

??? note "Details"
    Tests:

    - *Student's T-test* -- parametric, assumes equal variance and normal distribution
    - *Welch's T-test* -- parametric, does not assume equal variance
    - *Mann-Whitney U* -- non-parametric rank-based test
    - *Kolmogorov-Smirnov* -- tests whether two groups come from the same distribution
    - *Two-sample Z-test* -- compares means when variance is known or n is large
    - *Permutation test* -- non-parametric, no distributional assumptions, resampling-based
    - *Tukey HSD* -- post-hoc test after ANOVA
    - *Dunn* -- non-parametric post-hoc test (requires scikit-posthocs)
    - *Fisher's Z* -- compare correlation coefficients between groups (target column = r values)
    
    - **Alternative** -- two-sided (default), greater (group1 > group2), or less (group1 < group2). Tukey HSD and Dunn are always two-sided.
    
    - **P-Adj Method** -- multiple comparison correction (Bonferroni, Holm, BH).
    
    - **N Permutations** -- number of resampling iterations for the permutation test (default 10,000).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `stats_table` | stat |

**Properties:** `Statistical Method`, `Alternative`, `N Permutations`, `P-Adj Method`

---

### Pairwise Matrix

Computes a pairwise correlation or distance matrix for all numeric columns and visualises it as a heatmap.

??? note "Details"
    Correlation methods:

    - *Pearson* -- linear correlation coefficient, assumes normality
    - *Spearman* -- rank-based, robust to outliers and non-normal distributions
    - *Kendall* -- rank-based, slower but more exact for small sample sizes
    
    Outputs a matrix table (for further analysis) and an annotated heatmap figure.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `table` | table |
| **Output** | `figure` | figure |

**Properties:** `Metric`, `Colormap`, ``, ``

---

### PCA

Performs principal component analysis (PCA) for multivariate data exploration.

??? note "Details"
    Outputs:

    - **transformed** -- PC coordinates per sample (connect to ScatterPlotNode for PC1 vs PC2)
    - **loadings** -- feature contributions per principal component
    - **variance** -- eigenvalues and cumulative variance explained per component
    
    - **Standardize** -- when enabled, applies Z-score normalization before decomposition.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `transformed` | table |
| **Output** | `loadings` | table |
| **Output** | `variance` | stat |

**Properties:** ``

---

### Survival Analysis

Performs Kaplan-Meier survival analysis with log-rank test.

??? note "Details"
    Input columns:

    - **Time Column** -- duration or follow-up time
    - **Event Column** -- `1` = event occurred, `0` = censored
    - **Group Column** (optional) -- categorical grouping for multi-group comparison
    
    Outputs:

    - **km_table** -- survival function with 95% CI (feed into SurvivalPlotNode)
    - **log_rank** -- omnibus log-rank test statistic and p-value
    - **pairwise_stat** -- pairwise log-rank results with optional p-value adjustment
    
    - **P-Adj Method** -- multiple comparison correction for pairwise tests.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `km_table` | table |
| **Output** | `log_rank` | stat |
| **Output** | `pairwise_stat` | table |

**Properties:** `P-Adj Method (Pairwise)`

---

### Two-Way ANOVA

Performs two-way analysis of variance with interaction term (Type II SS).

??? note "Details"
    Input must be in long format with two factor columns and one numeric value column.
    
    Outputs:

    - **anova_table** -- sum of squares, df, F-statistic, and p-value per source
    - **group_means** -- mean, SD, SEM, and N for every factor combination

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `anova_table` | stat |
| **Output** | `group_means` | table |

---

### Variance Test

Tests whether two or more groups have equal variance (homoscedasticity).

??? note "Details"
    Use this to decide between Student's t-test (equal variance) and Welch's t-test
    (unequal variance), or to check ANOVA assumptions.
    
    Tests:

    - *Levene's test* -- robust, works for non-normal data (recommended default)
    - *Bartlett's test* -- more powerful but assumes normality
    - *F-test* -- classical variance ratio test for exactly 2 groups (sensitive to non-normality)
    
    Outputs a table with test statistic and p-value per group pair (F-test) or
    for all groups at once (Levene, Bartlett).
    
    A significant p-value (< 0.05) means variances are NOT equal -- use Welch's t-test.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `result` | table |

**Properties:** `Test`

---
