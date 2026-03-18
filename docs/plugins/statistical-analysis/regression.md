# Regression & Advanced

### Linear Regression

Performs ordinary least-squares (OLS) linear regression, simple or multiple.

??? note "Details"
    Outputs:

    - **coefficients** — slope, intercept, standard error, 95% CI, and p-values per parameter
    - **residuals** — fitted values, residuals, and standardized residuals for downstream plotting
    
    Summary statistics: R², adjusted R², F-statistic, and F p-value.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `coefficients` | stat |
| **Output** | `residuals` | table |

**Properties:** ``

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

**Properties:** `Model`

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
