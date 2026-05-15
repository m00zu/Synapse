# Collection

### Collect

Pack multiple data items into a named collection.

??? note "Details"
    Connect any number of items to the multi-input port. Each connection
    gets a name (auto-populated from the upstream port name, editable).
    The output is a single CollectionData that flows as one wire.
    
    Downstream nodes that expect a single item will automatically loop
    over all items in the collection and repack the results.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | any |
| **Output** | `collection` | collection |

---

### Collection Info

Outputs a table listing item names, types, shapes, and metadata.

??? note "Details"
    All number and string valued metadata fields are included as extra columns.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `collection` | collection |
| **Output** | `info` | table |

---

### Filter Collection

Keep or remove items by pattern matching on names.

??? note "Details"
    Supports simple wildcards (* and ?) or exact names.
    Multiple patterns separated by | (pipe).
    
    Mode:

    - *Keep* -- only matching items pass through
    - *Remove* -- matching items are excluded

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `collection` | collection |
| **Output** | `matched` | collection |
| **Output** | `rest` | collection |

**Properties:** `Mode`

---

### Map Names

Batch rename collection items using find/replace, prefix, or suffix.

??? note "Details"
    Operations (applied in order):
    1. Find/Replace -- replace substring in all names
    2. Prefix -- add text before each name
    3. Suffix -- add text after each name

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `collection` | collection |
| **Output** | `collection` | collection |

---

### Pop Collection

Extract one item from a collection and output the rest separately.

??? note "Details"
    Two outputs: the extracted item on **item**, and a new collection
    without that item on **rest**.  Type a name or pick from the dropdown.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `collection` | collection |
| **Output** | `item` | any |
| **Output** | `rest` | collection |

---

### Rename Collection

Rename items in a collection using a visual mapping table.

??? note "Details"
    When a collection is connected, the table auto-populates with original
    names. Edit the 'New Name' column to rename items. Leave blank to keep
    the original name.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `collection` | collection |
| **Output** | `collection` | collection |

---

### Save Collection

Saves all items in a collection to disk.

??? note "Details"
    Each item is saved as a separate file using the item name as a suffix.
    Supports images (TIFF, PNG), tables (CSV, TSV), and figures.
    
    If a path is connected, it is used as the base -- the item name is inserted
    before the extension.  Otherwise the folder + extension fields are used.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `file_path` | path |
| **Output** | `status` | table |

---

### Select Collection

Extract a single item from a collection by name.

??? note "Details"
    Type a name or pick from the dropdown. The dropdown auto-populates
    with available item names when the collection is connected.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `collection` | collection |
| **Output** | `out` | any |

---

### Split Collection

Split a collection into two groups by selecting which items go to each output.

??? note "Details"
    Type item names separated by ' | ' or pick from the dropdown to add.
    Selected items go to **selected**, the rest go to **rest**.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `collection` | collection |
| **Output** | `selected` | collection |
| **Output** | `rest` | collection |

---
