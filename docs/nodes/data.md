# Data

### Universal Node

Executes arbitrary Python code to process multiple inputs and push results to outputs.

??? note "Details"
    Available variables in user code:

    - `inputs` -- list of upstream data values
    - `output` -- assign the result here (auto-wrapped into `TableData`, `ImageData`, or `FigureData`)
    - `pd`, `np`, `plt`, `sns` -- pre-imported libraries

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | any |
| **Output** | `out` | any |

---
