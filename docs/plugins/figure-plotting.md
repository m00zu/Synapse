# Plotting

### Double-Variable Plot

Generates a 2D seaborn plot from a data table.

??? note "Details"
    Plot types:

    - *scatter* — X vs Y scatter plot
    - *box* — box-and-whisker plot
    - *violin* — violin density plot
    - *pairplot* — all-pairs scatter matrix
    
    Columns:

    - **x_col** — column for the X axis
    - **y_col** — column for the Y axis
    - **hue** — optional column for colour grouping

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `data` | table |
| **Output** | `plot` | figure |

**Properties:** `Plot Type`

---

### Swarm Plot + Stats

Creates a swarm plot with optional statistical annotation overlay.

??? note "Details"
    Accepts a data table and an optional stats table (from
    PairwiseComparisonNode) for significance-bracket overlays.
    
    Columns:

    - **target_column** — numeric column for the Y axis
    - **group_col** — categorical column that defines groups on the X axis
    - **x_axis_order** — comma-separated group order
    - **control_group** — reference group for fold-change ratios
    
    Options:

    - *use_stripplot* — switch from beeswarm to jittered strip layout
    - *show_error_bars* — overlay mean with SE/SD/CI/PI error bars
    - *enable_subgroups* — split group labels by delimiter for sub-bracket display

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `data` | table |
| **Input** | `stats` | stat |
| **Output** | `plot` | figure |

---

### Figure Editor

Interactively edits the aesthetics of any FigureData input via a popup dialog.

??? note "Details"
    Takes any FigureData, lets the user adjust titles, axes, colours,
    spines, lines, and annotations, then outputs the modified figure.
    Stored settings are persisted with the node and re-applied on every
    run.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `figure` | figure |
| **Output** | `plot` | figure |

---

### Violin Plot + Stats

Creates a violin plot with optional significance-bracket overlay.

??? note "Details"
    Connects to the same StatData output as SwarmPlotNode. Use
    **order** to fix the x-axis group order (comma-separated).
    
    Columns:

    - **x_col** — categorical group column
    - **y_col** — numeric value column
    - **order** — comma-separated group order for the X axis
    
    Options:

    - *inner_box* — draw a mini box plot inside each violin
    - *palette* — colour palette for groups

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `data` | table |
| **Input** | `stats` | stat |
| **Output** | `plot` | figure |

---

### Box Plot + Stats

Creates a box-and-whisker plot with optional significance-bracket overlay.

??? note "Details"
    Columns:

    - **x_col** — categorical group column
    - **y_col** — numeric value column
    - **order** — comma-separated group order for the X axis
    
    Options:

    - *show_points* — overlay individual data points on the boxes
    - *palette* — colour palette for groups

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `data` | table |
| **Input** | `stats` | stat |
| **Output** | `plot` | figure |

---

### Bar Plot + Stats

Creates a bar plot showing group means with error bars and optional significance-bracket overlay.

??? note "Details"
    Columns:

    - **x_col** — categorical group column
    - **y_col** — numeric value column
    - **order** — comma-separated group order for the X axis
    
    Options:

    - *error_type* — error bar measure: `se`, `sd`, `ci`, or `pi`
    - *show_bar_values* — annotate each bar with its numeric value
    - *palette* — colour palette for groups

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `data` | table |
| **Input** | `stats` | stat |
| **Output** | `plot` | figure |

---

### Scatter Plot

Creates a scatter plot (X vs Y) with optional regression line and hue grouping.

??? note "Details"
    Columns:

    - **x_col** — numeric column for the X axis
    - **y_col** — numeric column for the Y axis
    - **hue_col** — optional column for colour-coding by group
    
    Options:

    - *regression* — overlay a linear regression line
    - *palette* — colour palette for hue groups

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `data` | table |
| **Output** | `plot` | figure |

**Properties:** `Palette`, ``

---

### Histogram

Creates a histogram with optional grouping and KDE overlay.

??? note "Details"
    Columns:

    - **value_col** — numeric column to bin
    - **group_col** — optional categorical column for grouped histograms
    
    Options:

    - *bins* — number of bins (integer or `"auto"`)
    - *binwidth* — explicit bin width (overrides bins when set)
    - *kde* — overlay a kernel density estimate curve
    - *palette* — colour palette for groups

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `data` | table |
| **Output** | `plot` | figure |

**Properties:** `Palette`, ``

---

### Joint Plot

Creates a joint plot — scatter with marginal distributions on each axis.

??? note "Details"
    Columns:

    - **x_col** — numeric column for the X axis
    - **y_col** — numeric column for the Y axis
    - **hue_col** — optional column for colour-coding by group
    
    Options:

    - *kind* — scatter, kde, hex, hist, or reg (scatter + regression)
    - *marginal* — histogram, kde, or both for the marginal distributions
    - *palette* — colour palette for hue groups

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `data` | table |
| **Output** | `plot` | figure |

**Properties:** `Kind`, `Marginal`, `Palette`

---

### KDE Plot

Creates a kernel density estimate plot for smooth distribution visualisation.

??? note "Details"
    Supports optional grouping for comparing multiple distributions
    on the same axes.
    
    Columns:

    - **value_col** — numeric column to estimate density for
    - **group_col** — optional categorical column for overlaid group curves
    
    Options:

    - *fill* — fill the area under the density curve
    - *palette* — colour palette for groups

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `data` | table |
| **Output** | `plot` | figure |

**Properties:** `Palette`, ``

---

### XY Line Plot

Creates an XY line plot with error bars in the classic Prism graph style.

??? note "Details"
    Groups the data by an optional group column, computes mean +/- error
    per unique X value, and connects the means with lines. Optionally
    overlays individual data points and accepts a stats table from
    PairwiseComparisonNode for significance-bracket overlays.
    
    Columns:

    - **x_col** — numeric or categorical column for the X axis
    - **y_col** — numeric column for the Y axis
    - **group_col** — optional column to split data into separate lines
    
    Options:

    - *error_type* — error bar measure: `SEM`, `SD`, `95% CI`, or `None`
    - *show_points* — overlay individual data points
    - *x_order* — comma-separated order for X axis categories
    - *palette* — colour palette for groups

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `data` | table |
| **Input** | `stats` | stat |
| **Output** | `plot` | figure |

**Properties:** `Error Type`, ``, `Color Palette`

---

### Heatmap

Creates a heatmap with optional hierarchical clustering of rows and/or columns.

??? note "Details"
    Supports value annotations inside cells and a wide range of colour
    maps. Input can be a correlation matrix, gene-expression matrix, or
    any numeric table.
    
    Columns:

    - **row_label_col** — optional column to use as row labels
    - **value_cols** — comma-separated numeric columns (blank = all numeric)
    
    Options:

    - *cluster_rows* — apply hierarchical clustering to rows
    - *cluster_cols* — apply hierarchical clustering to columns
    - *annotate* — show numeric values inside cells
    - *cmap* — colour map (e.g. `viridis`, `coolwarm`, `RdYlGn`)

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `data` | table |
| **Output** | `plot` | figure |

**Properties:** `Colormap`, ``, ``, ``

---

### Volcano Plot

Creates a volcano plot showing log2(fold change) vs -log10(p-value).

??? note "Details"
    Colours up-regulated, down-regulated, and non-significant points
    separately, draws fold-change and significance threshold lines, and
    optionally labels the top N most significant features. Also outputs
    the significant-hit rows as a table for downstream filtering.
    
    Columns:

    - **fc_col** — column containing log2 fold-change values
    - **p_col** — column containing p-values
    - **label_col** — optional column for feature labels
    
    Parameters:

    - **fc_thresh** — fold-change threshold (`|log2FC|`)
    - **p_thresh** — p-value significance cutoff
    - **n_labels** — number of top significant features to label (0 = none)
    - **point_size** — scatter point size

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `data` | table |
| **Output** | `plot` | figure |
| **Output** | `significant` | table |

**Properties:** `FC Threshold (|log2FC|)`, `p-value Threshold`, `Top N Labels (0=none)`, `Label Font Size`, `Point Size`

---

### Regression Plot

Creates a scatter plot with a fitted regression line and optional 95% confidence band.

??? note "Details"
    Optionally accepts a pre-computed curve table from
    NonlinearRegressionNode to overlay a custom fit. For simple linear
    fits the equation and R-squared are annotated on the plot
    automatically.
    
    Columns:

    - **x_col** — numeric column for the X axis
    - **y_col** — numeric column for the Y axis
    - **group_col** — optional column for per-group fits
    
    Options:

    - *fit_type* — auto-fit when no curve input: `Linear`, `Polynomial deg 2`, `Polynomial deg 3`, or `None`
    - *show_ci* — show 95% confidence band around the fit
    - *show_equation* — annotate with equation and R-squared
    - *palette* — colour palette for groups

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `data` | table |
| **Input** | `curve` | table |
| **Output** | `plot` | figure |

**Properties:** `Auto-Fit (no curve input)`, ``, ``, `Color Palette`

---

### Survival Plot

Draws Kaplan-Meier survival curves from SurvivalAnalysisNode output.

??? note "Details"
    Accepts the `km_table` output and draws survival step-function curves
    with optional 95% CI shading, censoring tick marks, and an automatic
    log-rank p-value annotation from the `log_rank` port.
    
    Inputs:

    - `km_table` — Kaplan-Meier table with time, survival, and group columns
    - `log_rank` — StatData with overall log-rank test result
    - `pairwise_stat` — optional pairwise comparison table
    
    Options:

    - *show_ci* — shade the 95% confidence interval around each curve
    - *show_censored* — draw tick marks at censoring events
    - *show_pairwise* — display pairwise log-rank comparisons on the plot
    - *palette* — colour palette for groups

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `km_table` | table |
| **Input** | `log_rank` | stat |
| **Input** | `pairwise_stat` | table |
| **Output** | `plot` | figure |

**Properties:** ``, ``, ``, `Pairwise Stats Location`, `Pairwise X Offset`, `Pairwise Y Offset`, `Pairwise Font Size`, `Color Palette`

---

### Angle Distribution Plot

Creates a polar angle distribution plot for angular data.

??? note "Details"
    Display modes:

    - *Bin Arrows* — each angular bin is drawn as a proportional arrow from the origin (length = normalised bin frequency)
    - *KDE* — smooth kernel-density fill across the defined angular range
    - *Both* — overlay KDE on top of bin arrows
    
    The angular range is fully user-defined via **theta_min** / **theta_max**
    (degrees). Common presets: 0--90 (fibre orientation), 0--180, 0--360
    (full circle).
    
    Columns:

    - **angle_col** — column containing angle values
    - **group_col** — optional column for per-group curves in distinct colours
    
    Input angles may be in *Degrees* or *Radians* (set via **input_unit**).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `data` | table |
| **Output** | `plot` | figure |

---
