# Filters

### Set Scale

Manually set the pixel scale (µm/pixel) for an image.

??? note "Details"
    Use this when the image has no embedded calibration data (e.g. plain TIFF or PNG).
    
    Options:
    
    - **um_per_pixel** — micrometers per pixel
    - **known_distance** — a known real-world distance in µm
    - **distance_pixels** — the same distance measured in pixels
    
    If both known_distance and distance_pixels are set, um_per_pixel is calculated automatically.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `image` | image |
| **Output** | `image` | image |

**Properties:** `µm / pixel`, `Known Distance (µm)`, `Distance (pixels)`

---

### Gaussian Blur

Applies a Gaussian blur to smooth the image.

??? note "Details"
    - **sigma** — standard deviation of the Gaussian kernel; larger values produce stronger blurring (default: 10.0).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `image` | image |
| **Output** | `image` | image |

**Properties:** `Sigma (Blur Amount)`

---

### Threshold Local

Applies adaptive local thresholding to produce a binary mask.

??? note "Details"
    Computes a threshold for each pixel based on its local neighbourhood, making it robust to uneven illumination.
    
    - **block_size** — size of the local neighbourhood (must be odd; default: 25).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `image` | image |
| **Output** | `mask` | mask |

**Properties:** `Block Size (odd)`

---

### Binary Threshold

Applies interactive global thresholding using a histogram with a draggable threshold line.

??? note "Details"
    Drag the yellow threshold line on the histogram to select pixels. The green-shaded region shows which pixels will be included in the output mask. Works with 8-bit and 16-bit input images; the threshold value is in the original pixel-value space.
    
    Direction modes:

    - *Above (pixel > T)* — selected pixels are brighter than the threshold
    - *Below (pixel <= T)* — selected pixels are darker than the threshold
    - *Auto (Otsu)* — automatically finds the optimal threshold using Otsu's method
    - *Auto Otsu per image* — re-computes Otsu for every new input image (useful for batch workflows with varying brightness)
    
    Output is binary MaskData (255 = selected, 0 = not selected).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `image` | image |
| **Output** | `mask` | mask |

**Properties:** ``

---

### Rolling Ball

Subtracts slowly-varying background illumination using rolling-ball estimation.

??? note "Details"
    Models the image surface as a landscape and rolls a sphere of the given radius underneath it. The sphere's path estimates the background, which is then subtracted to leave only local foreground features (cells, fibres, etc.). Works on grayscale or RGB images.
    
    Rule of thumb: set **radius** to slightly larger than the largest object of interest. Larger radius removes broader background gradients (default: 100.0).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `image` | image |
| **Output** | `image` | image |

**Properties:** `Ball Radius (px)`

---

### Multi-Otsu Threshold

Splits image intensity into N classes using multi-Otsu thresholding.

??? note "Details"
    Uses `skimage.filters.threshold_multiotsu` to find optimal inter-class thresholds. Output is a LabelData integer array where each pixel is labelled 0 to N-1 (background = 0, brightest class = N-1).
    
    - **n_classes** — number of intensity classes to separate (default: 3).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `image` | image |
| **Output** | `label_image` | label_image |

**Properties:** `Classes`

---

### Bandpass Filter

Applies an FFT-based bandpass filter to a grayscale image.

??? note "Details"
    Keeps spatial frequencies corresponding to object sizes between the two cutoffs,
    analogous to ImageJ's Process > FFT > Bandpass Filter.
    
    - **Remove < (px)** — suppress structures smaller than this value (high-pass cutoff).
    - **Remove > (px)** — suppress structures larger than this value (low-pass cutoff). Set to `0` to disable low-pass (keep all large features).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `image` | image |
| **Output** | `image` | image |

**Properties:** `Remove < (px)`, `Remove > (px)`

---

### Frangi Tubeness

Enhances tubular structures using the Frangi multi-scale vesselness filter.

??? note "Details"
    Detects curvilinear features (blood vessels, filopodia, collagen fibres) across a
    range of scales using `skimage.filters.frangi`. Output is a response map normalised
    to 0-255 uint8 for downstream thresholding.
    
    - **Sigma Min** — smallest vessel width scale to detect.
    - **Sigma Max** — largest vessel width scale to detect.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `image` | image |
| **Output** | `image` | image |

**Properties:** `Sigma Min`, `Sigma Max`

---

### Canny Edge

Detects edges using the Canny algorithm, producing a thin binary edge mask.

??? note "Details"
    Converts input to grayscale, applies optional Gaussian blur, then runs the Canny
    algorithm with hysteresis thresholding. Leave both thresholds at `0` to use
    automatic Otsu-based values.
    
    - **Sigma** — scale of detected edges; larger values produce coarser edges.
    - **Low Threshold** — lower bound for hysteresis thresholding.
    - **High Threshold** — upper bound for hysteresis thresholding.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `image` | image |
| **Output** | `mask` | mask |

**Properties:** `Sigma`, `Low Threshold`, `High Threshold`

---

### Sobel Edge

Computes edge strength using the Sobel gradient-magnitude filter.

??? note "Details"
    Calculates the Sobel gradient in X and Y, combines them as `sqrt(Gx^2 + Gy^2)`,
    and scales to 0-255. Good for visualising edge strength. Connect to
    BinaryThresholdNode to convert the gradient into a mask.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `image` | image |
| **Output** | `image` | image |

---

### Prewitt Edge

Computes edge strength using the Prewitt gradient-magnitude filter.

??? note "Details"
    Similar to Sobel but uses equal-weight kernels. Slightly more sensitive to noise,
    but sometimes picks up finer detail at diagonal edges. Connect to
    BinaryThresholdNode to convert the gradient into a mask.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `image` | image |
| **Output** | `image` | image |

---

### Laplacian Edge

Highlights regions of rapid intensity change using the Laplacian of Gaussian (LoG) filter.

??? note "Details"
    Responds to blob-like features as well as sharp edges at the scale set by sigma.
    Output is a signed response normalised to 0-255 where `128` represents the
    zero-crossing. Connect to BinaryThresholdNode for a binary mask.
    
    - **Sigma** — spatial scale of the Gaussian smoothing before the Laplacian.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `image` | image |
| **Output** | `image` | image |

**Properties:** `Sigma`

---
