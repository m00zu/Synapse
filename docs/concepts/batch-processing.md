# Batch Processing

Batch processing lets you apply the same pipeline to hundreds of files automatically.

## How It Works

1. **Folder Iterator** provides one file path per iteration
2. A reader node loads each file (Image Reader, Table Reader, etc.)
3. Your processing pipeline runs on each file
4. **Batch Accumulator** collects results across all iterations
5. Downstream nodes (tables, plots) run once after all files are processed

## Setting Up a Batch Pipeline

### Step 1: Add a Folder Iterator

1. Add a **Folder Iterator** node
2. Set the **Folder** to your data directory
3. Set the **Pattern** to match your files (e.g., `*.tif`, `*.csv`, `*.png`)

### Step 2: Build Your Pipeline

Connect the Folder Iterator to a reader node, then to your processing nodes as usual.

### Step 3: Collect Results

Add a **Batch Accumulator** node between your pipeline output and any display/plot nodes.

The Batch Accumulator:

- Collects one table per iteration during the batch
- Merges all tables into one combined table after the batch completes
- Adds a `file` column with the source filename

### Step 4: Run

Click **Batch Run** (++ctrl+b++). Progress is shown in the status bar.

## Example Pipelines

### Image measurement

Measure particle properties across a folder of microscopy images.

```
Folder Iterator → Image Reader → Gaussian Filter → Threshold → Particle Props → Batch Accumulator → Data Table
```

### CSV aggregation

Load multiple CSV files, filter rows, and combine into a single summary table.

```
Folder Iterator → Table Reader → Filter Table → Aggregate Table → Batch Accumulator → Data Table
```

### Multi-file statistics

Run the same statistical test on each dataset and collect all p-values.

```
Folder Iterator → Table Reader → Group Comparison → Batch Accumulator → Sort Table → Data Table
```

### Batch figure export

Generate and save a plot for each input file.

```
Folder Iterator → Table Reader → Scatter Plot → Data Saver
```

## Batch Gate

The **Batch Gate** node pauses the batch pipeline at each iteration so you can review results and adjust node settings before continuing. Wire it anywhere in your pipeline as a checkpoint.

- **Next** — let the current iteration continue past the gate
- **Refresh** — re-evaluate upstream nodes (useful after changing a parameter)
- **Pass All** — stop pausing and let the remaining iterations run automatically

## Tips

!!! tip "Skipping the batch loop"
    If you've already run a batch and want to re-process only the downstream analysis (e.g., change a plot), just click **Run Graph** (++ctrl+w++). Synapse detects that the accumulators already have data and skips the per-file loop.

!!! tip "Stopping a batch"
    Click the **Stop** button to cancel. The batch stops after the current file finishes processing.

!!! warning "Memory"
    For very large batches (1000+ files), ensure you have sufficient RAM. Each iteration's results are held in memory until the batch completes.
