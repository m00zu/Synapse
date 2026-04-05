# Regression & Advanced

### Linear Regression

Performs ordinary least-squares (OLS) linear or polynomial regression.

??? note "Details"
    Set **Degree** > 1 for polynomial regression (e.g. 2 = quadratic, 3 = cubic).
    With degree 1 (default), this is standard linear regression.
    
    Outputs:

    - **coefficients** — slope, intercept, standard error, 95% CI, and p-values per parameter
    - **residuals** — fitted values, residuals, and standardized residuals for downstream plotting
    
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

### Nonlinear Regression

Fits nonlinear curves to XY data using `scipy.optimize.curve_fit`.

??? note "Details"
    Built-in models:

    - *4PL (EC50 / Dose-Response)* — four-parameter logistic for IC50/EC50
    - *Hill Equation* — sigmoidal binding/dose-response
    - *One-Phase Exponential Decay* — single-rate decay to plateau
    - *Two-Phase Exponential Decay* — fast + slow decay components
    - *Exponential Growth* — unbounded exponential increase
    - *Michaelis-Menten* — enzyme kinetics saturation curve
    - *Gompertz Growth* — asymmetric sigmoidal growth
    - *Sigmoidal (Logistic)* — symmetric S-curve
    
    Outputs best-fit parameters with 95% CI and a smooth predicted curve table.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `parameters` | stat |
| **Output** | `curve` | table |

**Properties:** `Model`, `X Min (0=auto)`, `X Max (0=auto)`

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

### Two-Way ANOVA

Performs two-way analysis of variance with interaction term (Type II SS).

??? note "Details"
    Input must be in long format with two factor columns and one numeric value column.
    
    Outputs:

    - **anova_table** — sum of squares, df, F-statistic, and p-value per source
    - **group_means** — mean, SD, SEM, and N for every factor combination

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `anova_table` | stat |
| **Output** | `group_means` | table |

---

### Contingency Analysis

Tests categorical association using chi-square and Fisher's exact tests.

??? note "Details"
    Input types:

    - *Raw Data (two columns)* — a crosstab is built automatically from two categorical columns
    - *Contingency Matrix* — a pre-built matrix of observed counts
    
    Outputs:

    - **test_results** — Pearson chi-square, Yates-corrected chi-square, and Fisher's exact (2x2)
    - **observed_counts** — the observed contingency table
    - **expected_counts** — expected counts under the null hypothesis

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `test_results` | stat |
| **Output** | `observed_counts` | table |
| **Output** | `expected_counts` | table |

**Properties:** `Input Type`

---

### Survival Analysis

Performs Kaplan-Meier survival analysis with log-rank test.

??? note "Details"
    Input columns:

    - **Time Column** — duration or follow-up time
    - **Event Column** — `1` = event occurred, `0` = censored
    - **Group Column** (optional) — categorical grouping for multi-group comparison
    
    Outputs:

    - **km_table** — survival function with 95% CI (feed into SurvivalPlotNode)
    - **log_rank** — omnibus log-rank test statistic and p-value
    - **pairwise_stat** — pairwise log-rank results with optional p-value adjustment
    
    - **P-Adj Method** — multiple comparison correction for pairwise tests.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `km_table` | table |
| **Output** | `log_rank` | stat |
| **Output** | `pairwise_stat` | table |

**Properties:** `P-Adj Method (Pairwise)`

---

### PCA

Performs principal component analysis (PCA) for multivariate data exploration.

??? note "Details"
    Outputs:

    - **transformed** — PC coordinates per sample (connect to ScatterPlotNode for PC1 vs PC2)
    - **loadings** — feature contributions per principal component
    - **variance** — eigenvalues and cumulative variance explained per component
    
    - **Standardize** — when enabled, applies Z-score normalization before decomposition.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `transformed` | table |
| **Output** | `loadings` | table |
| **Output** | `variance` | stat |

**Properties:** ``

---

### Mixed Effects Model

Fits a linear mixed-effects model (LMM) for hierarchical / nested data.

??? note "Details"
    Mixed-effects models are essential when observations are grouped (e.g.
    cells within wells, animals within treatment groups, repeated measures
    per subject). They estimate **fixed effects** (population-level trends)
    and **random effects** (group-level deviations) simultaneously.
    
    Configuration:

    - **y_col** — dependent (response) variable.
    - **fixed_cols** — fixed-effect predictor(s), comma-separated.
    - **group_col** — grouping variable for random intercepts (required).
    - **random_slope_col** — optional predictor for random slopes.
    - **REML** — use Restricted ML (default) or Full ML estimation.
    
    Outputs:

    - **fixed_effects** — coefficient table with SE, z-value, p-value, 95% CI.
    - **random_effects** — per-group random intercept (and slope) estimates.
    - **summary** — model-level statistics: log-likelihood, AIC, BIC,
      number of groups, ICC.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `fixed_effects` | stat |
| **Output** | `random_effects` | table |
| **Output** | `summary` | stat |

**Properties:** ``

---
