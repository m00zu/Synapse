use ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray2, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};

// ═══════════════════════════════════════════════════════════════════════════
//  Skeleton analysis (existing)
// ═══════════════════════════════════════════════════════════════════════════

fn compute_skeleton_stats_full_impl(
    pixels: &[u8],
    height: usize,
    width: usize,
) -> Result<Vec<(f64, f64, f64, usize)>, String> {
    if height == 0 || width == 0 {
        return Ok(Vec::new());
    }
    let expected = height
        .checked_mul(width)
        .ok_or_else(|| "height * width overflows usize".to_string())?;
    if pixels.len() != expected {
        return Err(format!(
            "pixels length {} does not match height*width {}",
            pixels.len(),
            expected
        ));
    }

    let fg: Vec<bool> = pixels.iter().map(|&v| v != 0).collect();
    let mut labels = vec![usize::MAX; expected];
    let mut components: Vec<Vec<usize>> = Vec::new();
    let mut nb_count = vec![0u8; expected];
    let mut q = VecDeque::<usize>::new();

    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            if !fg[i] {
                continue;
            }
            let y0 = y.saturating_sub(1);
            let y1 = (y + 1).min(height - 1);
            let x0 = x.saturating_sub(1);
            let x1 = (x + 1).min(width - 1);
            let mut count = 0u8;
            for ny in y0..=y1 {
                for nx in x0..=x1 {
                    if ny == y && nx == x {
                        continue;
                    }
                    if fg[ny * width + nx] {
                        count += 1;
                    }
                }
            }
            nb_count[i] = count;
        }
    }

    for i in 0..expected {
        if !fg[i] || labels[i] != usize::MAX {
            continue;
        }
        let cid = components.len();
        labels[i] = cid;
        q.push_back(i);
        let mut comp = Vec::new();
        while let Some(cur) = q.pop_front() {
            comp.push(cur);
            let y = cur / width;
            let x = cur % width;
            let y0 = y.saturating_sub(1);
            let y1 = (y + 1).min(height - 1);
            let x0 = x.saturating_sub(1);
            let x1 = (x + 1).min(width - 1);
            for ny in y0..=y1 {
                for nx in x0..=x1 {
                    if ny == y && nx == x {
                        continue;
                    }
                    let ni = ny * width + nx;
                    if fg[ni] && labels[ni] == usize::MAX {
                        labels[ni] = cid;
                        q.push_back(ni);
                    }
                }
            }
        }
        components.push(comp);
    }

    let sqrt2 = std::f64::consts::SQRT_2;
    let results: Vec<(f64, f64, f64, usize)> = components
        .par_iter()
        .enumerate()
        .map(|(cid, comp)| {
            let n = comp.len() as f64;
            let mut sum_y = 0.0f64;
            let mut sum_x = 0.0f64;
            let mut ortho = 0usize;
            let mut diag = 0usize;
            let mut junctions = 0usize;

            for &i in comp {
                let y = i / width;
                let x = i % width;
                sum_y += y as f64;
                sum_x += x as f64;
                if nb_count[i] >= 3 {
                    junctions += 1;
                }
                if x + 1 < width && labels[i + 1] == cid {
                    ortho += 1;
                }
                if y + 1 < height && labels[i + width] == cid {
                    ortho += 1;
                }
                if y + 1 < height && x + 1 < width && labels[i + width + 1] == cid {
                    diag += 1;
                }
                if y + 1 < height && x > 0 && labels[i + width - 1] == cid {
                    diag += 1;
                }
            }

            let total_edges = ortho as f64 + diag as f64 * sqrt2;
            let length = if total_edges == 0.0 { 1.0 } else { total_edges };
            let centroid_y = sum_y / n;
            let centroid_x = sum_x / n;
            (length, centroid_y, centroid_x, junctions)
        })
        .collect();

    Ok(results)
}

fn compute_skeleton_branch_stats_impl(
    pixels: &[u8],
    height: usize,
    width: usize,
    include_singletons: bool,
) -> Result<Vec<(f64, usize, usize)>, String> {
    if height == 0 || width == 0 {
        return Ok(Vec::new());
    }
    let expected = height
        .checked_mul(width)
        .ok_or_else(|| "height * width overflows usize".to_string())?;
    if pixels.len() != expected {
        return Err(format!(
            "pixels length {} does not match height*width {}",
            pixels.len(),
            expected
        ));
    }

    let fg: Vec<bool> = pixels.iter().map(|&v| v != 0).collect();
    let mut labels = vec![usize::MAX; expected];
    let mut components: Vec<Vec<usize>> = Vec::new();
    let mut nb_count = vec![0u8; expected];
    let mut q = VecDeque::<usize>::new();

    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            if !fg[i] {
                continue;
            }
            let y0 = y.saturating_sub(1);
            let y1 = (y + 1).min(height - 1);
            let x0 = x.saturating_sub(1);
            let x1 = (x + 1).min(width - 1);
            let mut count = 0u8;
            for ny in y0..=y1 {
                for nx in x0..=x1 {
                    if ny == y && nx == x {
                        continue;
                    }
                    if fg[ny * width + nx] {
                        count += 1;
                    }
                }
            }
            nb_count[i] = count;
        }
    }

    for i in 0..expected {
        if !fg[i] || labels[i] != usize::MAX {
            continue;
        }
        let cid = components.len();
        labels[i] = cid;
        q.push_back(i);
        let mut comp = Vec::new();
        while let Some(cur) = q.pop_front() {
            comp.push(cur);
            let y = cur / width;
            let x = cur % width;
            let y0 = y.saturating_sub(1);
            let y1 = (y + 1).min(height - 1);
            let x0 = x.saturating_sub(1);
            let x1 = (x + 1).min(width - 1);
            for ny in y0..=y1 {
                for nx in x0..=x1 {
                    if ny == y && nx == x {
                        continue;
                    }
                    let ni = ny * width + nx;
                    if fg[ni] && labels[ni] == usize::MAX {
                        labels[ni] = cid;
                        q.push_back(ni);
                    }
                }
            }
        }
        components.push(comp);
    }

    let sqrt2 = std::f64::consts::SQRT_2;
    let mut out: Vec<(f64, usize, usize)> = components
        .par_iter()
        .enumerate()
        .filter_map(|(cid, comp)| {
            let mut ortho = 0usize;
            let mut diag = 0usize;
            for &i in comp {
                let y = i / width;
                let x = i % width;
                if x + 1 < width && labels[i + 1] == cid {
                    ortho += 1;
                }
                if y + 1 < height && labels[i + width] == cid {
                    ortho += 1;
                }
                if y + 1 < height && x + 1 < width && labels[i + width + 1] == cid {
                    diag += 1;
                }
                if y + 1 < height && x > 0 && labels[i + width - 1] == cid {
                    diag += 1;
                }
            }

            let total_len = ortho as f64 + diag as f64 * sqrt2;
            if total_len == 0.0 && !include_singletons {
                return None;
            }

            let endpoint_idx = comp
                .iter()
                .copied()
                .find(|&idx| nb_count[idx] == 1)
                .unwrap_or(comp[0]);
            let y = endpoint_idx / width;
            let x = endpoint_idx % width;
            let length = if total_len == 0.0 { 1.0 } else { total_len };
            Some((length, y, x))
        })
        .collect();

    out.sort_by(|a, b| a.1.cmp(&b.1).then(a.2.cmp(&b.2)));
    Ok(out)
}

// ═══════════════════════════════════════════════════════════════════════════
//  Rolling-ball background subtraction
//  (not available in scirs2-vision — implemented from scratch)
//
//  Algorithm: the "rolling ball" algorithm models the image as a 3D surface
//  and slides a ball of given radius underneath it. The ball's envelope is
//  the estimated background. We use the Sternberg approach:
//    1. Build a 1-D ball profile of radius R
//    2. Erode (min-filter) along rows with the profile
//    3. Erode (min-filter) along cols with the profile
//    4. Dilate (max-filter) along rows with the profile
//    5. Dilate (max-filter) along cols with the profile
//  This separable approach is O(n) per pixel (independent of radius) when
//  using a sliding-window min/max (van Herk / Gil-Werman algorithm).
// ═══════════════════════════════════════════════════════════════════════════

/// Build the 1-D ball profile: for each offset dx in [-r, r],
/// height = sqrt(r^2 - dx^2).  Returned as f32 values.
fn ball_profile(radius: f32) -> Vec<f32> {
    let r = radius.ceil() as isize;
    let r2 = radius * radius;
    ((-r)..=r)
        .map(|dx| {
            let d2 = (dx as f32) * (dx as f32);
            if d2 <= r2 {
                (r2 - d2).sqrt()
            } else {
                0.0
            }
        })
        .collect()
}

/// Erode a row with the ball profile (subtract profile, take min).
fn erode_row(row: &[f32], profile: &[f32], out: &mut [f32]) {
    let n = row.len();
    let half = profile.len() / 2;
    for i in 0..n {
        let mut min_val = f32::MAX;
        for (k, &p) in profile.iter().enumerate() {
            let j = i as isize + k as isize - half as isize;
            if j >= 0 && (j as usize) < n {
                let v = row[j as usize] - p;
                if v < min_val {
                    min_val = v;
                }
            }
        }
        out[i] = min_val;
    }
}

/// Dilate a row with the ball profile (add profile, take max).
fn dilate_row(row: &[f32], profile: &[f32], out: &mut [f32]) {
    let n = row.len();
    let half = profile.len() / 2;
    for i in 0..n {
        let mut max_val = f32::MIN;
        for (k, &p) in profile.iter().enumerate() {
            let j = i as isize + k as isize - half as isize;
            if j >= 0 && (j as usize) < n {
                let v = row[j as usize] + p;
                if v > max_val {
                    max_val = v;
                }
            }
        }
        out[i] = max_val;
    }
}

/// Rolling-ball background estimation on a 2-D f32 image [0..1].
/// Returns the estimated background (same size).
fn rolling_ball_impl(
    data: &[f32],
    height: usize,
    width: usize,
    radius: f32,
) -> Vec<f32> {
    let profile = ball_profile(radius);
    let npix = height * width;
    let mut buf = data.to_vec();
    let mut tmp = vec![0.0f32; npix];

    // --- Erode rows (parallel) ---
    buf.par_chunks_mut(width)
        .zip(tmp.par_chunks_mut(width))
        .for_each(|(row, out)| {
            erode_row(row, &profile, out);
        });
    std::mem::swap(&mut buf, &mut tmp);

    // --- Erode columns (parallel over columns) ---
    // Transpose → erode rows → transpose back
    let mut col_buf = vec![0.0f32; npix];
    // Transpose buf into col_buf (row-of-columns)
    for y in 0..height {
        for x in 0..width {
            col_buf[x * height + y] = buf[y * width + x];
        }
    }
    let mut col_tmp = vec![0.0f32; npix];
    col_buf
        .par_chunks_mut(height)
        .zip(col_tmp.par_chunks_mut(height))
        .for_each(|(col, out)| {
            erode_row(col, &profile, out);
        });
    // Transpose back
    for x in 0..width {
        for y in 0..height {
            buf[y * width + x] = col_tmp[x * height + y];
        }
    }

    // --- Dilate rows (parallel) ---
    buf.par_chunks_mut(width)
        .zip(tmp.par_chunks_mut(width))
        .for_each(|(row, out)| {
            dilate_row(row, &profile, out);
        });
    std::mem::swap(&mut buf, &mut tmp);

    // --- Dilate columns (parallel) ---
    for y in 0..height {
        for x in 0..width {
            col_buf[x * height + y] = buf[y * width + x];
        }
    }
    col_buf
        .par_chunks_mut(height)
        .zip(col_tmp.par_chunks_mut(height))
        .for_each(|(col, out)| {
            dilate_row(col, &profile, out);
        });
    for x in 0..width {
        for y in 0..height {
            buf[y * width + x] = col_tmp[x * height + y];
        }
    }

    buf
}

// ═══════════════════════════════════════════════════════════════════════════
//  Euclidean Distance Transform (EDT)
//  (not available in scirs2-vision — implemented from scratch)
//
//  Uses the Meijster/Roerdink/Hesselink linear-time algorithm:
//    Phase 1: 1-D transform along columns
//    Phase 2: extend to 2-D along rows using parabola envelopes
//  Reference: "A General Algorithm for Computing Distance Transforms
//  in Linear Time" (2000).
// ═══════════════════════════════════════════════════════════════════════════

fn distance_transform_edt_impl(
    binary: &[u8], // nonzero = foreground; compute distance to nearest background (==0)
    height: usize,
    width: usize,
) -> Vec<f32> {
    let npix = height * width;
    if npix == 0 {
        return Vec::new();
    }

    let inf = (height + width) as f32;

    // Meijster/Roerdink/Hesselink two-phase linear-time EDT.
    //
    // Phase 1: For each column x, compute g(x,y) = min vertical distance
    //          to nearest background pixel (binary == 0).
    //          Background pixels get g = 0; foreground pixels get g > 0.
    let mut g = vec![0.0f32; npix];

    let col_results: Vec<Vec<f32>> = (0..width)
        .into_par_iter()
        .map(|x| {
            let mut col = vec![0.0f32; height];
            // Forward pass: distance to nearest bg above
            col[0] = if binary[x] != 0 { inf } else { 0.0 };
            for y in 1..height {
                if binary[y * width + x] == 0 {
                    col[y] = 0.0;
                } else {
                    col[y] = col[y - 1] + 1.0;
                }
            }
            // Backward pass: distance to nearest bg below
            for y in (0..height - 1).rev() {
                let d = col[y + 1] + 1.0;
                if d < col[y] {
                    col[y] = d;
                }
            }
            col
        })
        .collect();

    for x in 0..width {
        for y in 0..height {
            g[y * width + x] = col_results[x][y];
        }
    }

    // Phase 2: row transform using parabola envelope
    // For each row y, compute: dt[y][x] = min_{x'} { g[y][x']^2 + (x - x')^2 }
    let mut dt = vec![0.0f32; npix];
    // This uses the lower envelope of parabolas.
    let row_results: Vec<Vec<f32>> = (0..height)
        .into_par_iter()
        .map(|y| {
            let row_start = y * width;
            let f: Vec<f32> = (0..width).map(|x| {
                let gv = g[row_start + x];
                gv * gv // squared vertical distance
            }).collect();

            // Lower envelope of parabolas
            let mut v = vec![0usize; width]; // locations of parabolas
            let mut z = vec![0.0f32; width + 1]; // boundaries between parabolas
            let mut k = 0usize;
            v[0] = 0;
            z[0] = f32::NEG_INFINITY;
            z[1] = f32::INFINITY;

            for q in 1..width {
                loop {
                    let vk = v[k];
                    // intersection of parabola at vk and q
                    let s = ((f[q] - f[vk]) + (q * q) as f32 - (vk * vk) as f32)
                        / (2.0 * (q as f32 - vk as f32));
                    if s > z[k] {
                        k += 1;
                        v[k] = q;
                        z[k] = s;
                        z[k + 1] = f32::INFINITY;
                        break;
                    }
                    if k == 0 {
                        v[0] = q;
                        z[1] = f32::INFINITY;
                        break;
                    }
                    k -= 1;
                }
            }

            let mut result = vec![0.0f32; width];
            k = 0;
            for q in 0..width {
                while z[k + 1] < q as f32 {
                    k += 1;
                }
                let dx = q as f32 - v[k] as f32;
                result[q] = (dx * dx + f[v[k]]).sqrt();
            }
            result
        })
        .collect();

    // Write results
    for y in 0..height {
        for x in 0..width {
            dt[y * width + x] = row_results[y][x];
        }
    }

    dt
}

// ═══════════════════════════════════════════════════════════════════════════
//  Connected Component Labeling (two-pass union-find)
//  Adapted from scirs2-vision semantic_segmentation.rs
// ═══════════════════════════════════════════════════════════════════════════

fn find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        parent[x as usize] = parent[parent[x as usize] as usize]; // path compression
        x = parent[x as usize];
    }
    x
}

fn union(parent: &mut [u32], a: u32, b: u32) {
    let ra = find(parent, a);
    let rb = find(parent, b);
    if ra != rb {
        parent[ra as usize] = rb;
    }
}

/// Connected component labeling with 4-connectivity (matching scipy.ndimage.label default).
/// Input: binary mask (nonzero = foreground).
/// Returns: (label_array, num_labels).
fn label_2d_impl(
    binary: &[u8],
    height: usize,
    width: usize,
    connectivity: u8,
) -> (Vec<u32>, u32) {
    let npix = height * width;
    let mut labels = vec![0u32; npix];
    let mut parent = vec![0u32; npix + 1]; // 1-indexed
    let mut next_label = 1u32;

    // First pass
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            if binary[i] == 0 {
                continue;
            }

            let mut neighbors = Vec::with_capacity(4);

            // Left
            if x > 0 && labels[i - 1] > 0 {
                neighbors.push(labels[i - 1]);
            }
            // Up
            if y > 0 && labels[i - width] > 0 {
                neighbors.push(labels[i - width]);
            }
            if connectivity == 8 {
                // Up-left
                if y > 0 && x > 0 && labels[i - width - 1] > 0 {
                    neighbors.push(labels[i - width - 1]);
                }
                // Up-right
                if y > 0 && x + 1 < width && labels[i - width + 1] > 0 {
                    neighbors.push(labels[i - width + 1]);
                }
            }

            if neighbors.is_empty() {
                labels[i] = next_label;
                parent[next_label as usize] = next_label;
                next_label += 1;
            } else {
                // SAFETY: `neighbors` is guaranteed non-empty (checked above)
                let min_label = *neighbors.iter().min().unwrap();
                labels[i] = min_label;
                for &n in &neighbors {
                    union(&mut parent, n, min_label);
                }
            }
        }
    }

    // Second pass: resolve to consecutive labels
    let mut label_map = vec![0u32; next_label as usize];
    let mut component_id = 0u32;
    for i in 1..next_label as usize {
        let root = find(&mut parent, i as u32) as usize;
        if label_map[root] == 0 {
            component_id += 1;
            label_map[root] = component_id;
        }
        label_map[i] = label_map[root];
    }

    // Relabel
    for i in 0..npix {
        if labels[i] > 0 {
            labels[i] = label_map[labels[i] as usize];
        }
    }

    (labels, component_id)
}

// ═══════════════════════════════════════════════════════════════════════════
//  Watershed Segmentation (marker-controlled, priority-queue flooding)
//  Adapted from scirs2-vision segmentation/watershed.rs
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy)]
struct WsPixel {
    y: usize,
    x: usize,
    priority: u8,
}

impl Eq for WsPixel {}
impl PartialEq for WsPixel {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}
impl Ord for WsPixel {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: lower priority value = higher queue priority
        other.priority.cmp(&self.priority)
    }
}
impl PartialOrd for WsPixel {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

const NEIGHBORS_8: [(i32, i32); 8] = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
];

/// Marker-based watershed on a u8 grayscale image.
/// `image`: grayscale pixel values (typically the negated distance transform mapped to u8).
/// `markers`: integer label for each pixel (0 = unlabeled, >0 = seed region).
/// `mask`: optional binary mask (nonzero = region to segment).
/// Returns: label array.
fn watershed_impl(
    image: &[u8],
    markers: &[u32],
    mask: Option<&[u8]>,
    height: usize,
    width: usize,
) -> Vec<u32> {
    let npix = height * width;
    let mut labels = markers.to_vec();
    let mut in_queue = vec![false; npix];
    let mut queue = BinaryHeap::new();

    // Seed the queue with boundary pixels of markers
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            if let Some(m) = mask {
                if m[i] == 0 {
                    continue;
                }
            }
            if labels[i] == 0 {
                continue;
            }
            // Check if any neighbor is unlabeled
            for &(dy, dx) in &NEIGHBORS_8 {
                let ny = y as i32 + dy;
                let nx = x as i32 + dx;
                if ny >= 0 && ny < height as i32 && nx >= 0 && nx < width as i32 {
                    let ni = ny as usize * width + nx as usize;
                    let mask_ok = mask.map_or(true, |m| m[ni] != 0);
                    if mask_ok && labels[ni] == 0 && !in_queue[ni] {
                        queue.push(WsPixel {
                            y: ny as usize,
                            x: nx as usize,
                            priority: image[ni],
                        });
                        in_queue[ni] = true;
                    }
                }
            }
        }
    }

    // Flood
    while let Some(px) = queue.pop() {
        let i = px.y * width + px.x;
        let mut neighbor_label = 0u32;
        let mut conflict = false;

        for &(dy, dx) in &NEIGHBORS_8 {
            let ny = px.y as i32 + dy;
            let nx = px.x as i32 + dx;
            if ny >= 0 && ny < height as i32 && nx >= 0 && nx < width as i32 {
                let ni = ny as usize * width + nx as usize;
                let lbl = labels[ni];
                if lbl > 0 {
                    if neighbor_label == 0 {
                        neighbor_label = lbl;
                    } else if lbl != neighbor_label {
                        conflict = true;
                    }
                }
            }
        }

        if conflict {
            // Watershed line: leave as 0
            continue;
        }
        if neighbor_label == 0 {
            continue;
        }

        labels[i] = neighbor_label;

        // Enqueue unlabeled neighbors
        for &(dy, dx) in &NEIGHBORS_8 {
            let ny = px.y as i32 + dy;
            let nx = px.x as i32 + dx;
            if ny >= 0 && ny < height as i32 && nx >= 0 && nx < width as i32 {
                let ni = ny as usize * width + nx as usize;
                let mask_ok = mask.map_or(true, |m| m[ni] != 0);
                if mask_ok && labels[ni] == 0 && !in_queue[ni] {
                    queue.push(WsPixel {
                        y: ny as usize,
                        x: nx as usize,
                        priority: image[ni],
                    });
                    in_queue[ni] = true;
                }
            }
        }
    }

    labels
}

// ═══════════════════════════════════════════════════════════════════════════
//  Morphological Operations (erode, dilate, open, close, tophat, blackhat)
//  Adapted from scirs2-vision morphology.rs — rewritten for raw buffers
// ═══════════════════════════════════════════════════════════════════════════

/// Build a filled-disk structuring element of radius r.
/// Returns flat array of (2r+1)^2, row-major, with 1 inside disk.
fn disk_kernel_flat(radius: usize) -> (Vec<u8>, usize) {
    let size = 2 * radius + 1;
    let mut k = vec![0u8; size * size];
    let c = radius as isize;
    let r2 = (radius * radius) as isize;
    for y in 0..size {
        for x in 0..size {
            let dy = y as isize - c;
            let dx = x as isize - c;
            if dy * dy + dx * dx <= r2 {
                k[y * size + x] = 1;
            }
        }
    }
    (k, size)
}

/// Grayscale erosion: local min under structuring element.
fn erode_impl(
    image: &[u8],
    height: usize,
    width: usize,
    kernel: &[u8],
    ksize: usize,
) -> Vec<u8> {
    let npix = height * width;
    let mut out = vec![0u8; npix];
    let half = ksize / 2;

    // Parallelize over rows
    out.par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row_out)| {
            for x in 0..width {
                let mut min_val = 255u8;
                for ky in 0..ksize {
                    for kx in 0..ksize {
                        if kernel[ky * ksize + kx] == 0 {
                            continue;
                        }
                        let iy = y as isize + ky as isize - half as isize;
                        let ix = x as isize + kx as isize - half as isize;
                        let pix = if iy >= 0
                            && iy < height as isize
                            && ix >= 0
                            && ix < width as isize
                        {
                            image[iy as usize * width + ix as usize]
                        } else {
                            255 // border = max for erosion
                        };
                        if pix < min_val {
                            min_val = pix;
                        }
                    }
                }
                row_out[x] = min_val;
            }
        });

    out
}

/// Grayscale dilation: local max under structuring element.
fn dilate_impl(
    image: &[u8],
    height: usize,
    width: usize,
    kernel: &[u8],
    ksize: usize,
) -> Vec<u8> {
    let npix = height * width;
    let mut out = vec![0u8; npix];
    let half = ksize / 2;

    out.par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row_out)| {
            for x in 0..width {
                let mut max_val = 0u8;
                for ky in 0..ksize {
                    for kx in 0..ksize {
                        if kernel[ky * ksize + kx] == 0 {
                            continue;
                        }
                        let iy = y as isize + ky as isize - half as isize;
                        let ix = x as isize + kx as isize - half as isize;
                        let pix = if iy >= 0
                            && iy < height as isize
                            && ix >= 0
                            && ix < width as isize
                        {
                            image[iy as usize * width + ix as usize]
                        } else {
                            0 // border = 0 for dilation
                        };
                        if pix > max_val {
                            max_val = pix;
                        }
                    }
                }
                row_out[x] = max_val;
            }
        });

    out
}

// ═══════════════════════════════════════════════════════════════════════════
//  CLAHE (Contrast Limited Adaptive Histogram Equalization)
//  Faithfully ported from skimage.exposure._adapthist
// ═══════════════════════════════════════════════════════════════════════════

const NUM_BINS: usize = 256;

/// Reflect index into [0, n-1], matching numpy 'reflect' padding mode.
fn reflect_index(mut i: isize, n: usize) -> usize {
    let n = n as isize;
    if n <= 1 {
        return 0;
    }
    loop {
        if i < 0 {
            i = -i;
        } else if i >= n {
            i = 2 * (n - 1) - i;
        } else {
            return i as usize;
        }
    }
}

/// Clip histogram and redistribute excess (matching skimage's clip_histogram).
fn clip_histogram_redistribute(hist: &mut [u32], clip_limit: u32) {
    let nbins = hist.len();

    // Phase 1: Clip bins above limit and count excess
    let mut n_excess: i64 = 0;
    for bin in hist.iter_mut() {
        if *bin > clip_limit {
            n_excess += (*bin - clip_limit) as i64;
            *bin = clip_limit;
        }
    }
    if n_excess <= 0 {
        return;
    }

    // Phase 2: Bulk redistribution
    let bin_incr = (n_excess / nbins as i64) as u32;
    if bin_incr > 0 {
        let upper = clip_limit - bin_incr;

        // Low bins: safely add bin_incr (won't exceed clip_limit)
        for bin in hist.iter_mut() {
            if *bin < upper {
                n_excess -= bin_incr as i64;
                *bin += bin_incr;
            }
        }

        // Mid bins [upper, clip_limit): set to clip_limit, account for used excess
        for bin in hist.iter_mut() {
            if *bin >= upper && *bin < clip_limit {
                n_excess -= (clip_limit - *bin) as i64;
                *bin = clip_limit;
            }
        }
    }

    // Phase 3: Scatter remaining excess one-by-one (matching skimage's while loop)
    while n_excess > 0 {
        let prev = n_excess;
        for index in 0..nbins {
            let under_count = hist.iter().filter(|&&b| b < clip_limit).count() as i64;
            if under_count == 0 {
                return;
            }
            let step_size = (under_count / n_excess).max(1) as usize;

            let mut idx = index;
            while idx < nbins {
                if hist[idx] < clip_limit {
                    hist[idx] += 1;
                    n_excess -= 1;
                    if n_excess <= 0 {
                        return;
                    }
                }
                idx += step_size;
            }
        }
        if n_excess == prev {
            break;
        }
    }
}

fn clahe_impl(
    image: &[u8],
    height: usize,
    width: usize,
    tile_size: usize,
    clip_limit: f32,
) -> Vec<u8> {
    let k = tile_size;

    // Step 1: Compute padding to match skimage exactly.
    // skimage pads k//2 before and (k - s%k)%k + ceil(k/2) after.
    let pad_start_y = k / 2;
    let pad_end_y = (k - height % k) % k + (k + 1) / 2;
    let pad_start_x = k / 2;
    let pad_end_x = (k - width % k) % k + (k + 1) / 2;
    let ph = height + pad_start_y + pad_end_y;
    let pw = width + pad_start_x + pad_end_x;

    // Step 2: Reflect-pad the image
    let mut padded = vec![0u8; ph * pw];
    for y in 0..ph {
        let sy = reflect_index(y as isize - pad_start_y as isize, height);
        for x in 0..pw {
            let sx = reflect_index(x as isize - pad_start_x as isize, width);
            padded[y * pw + x] = image[sy * width + sx];
        }
    }

    // Step 3: Compute histogram tiles.
    // ns_hist = padded/k - 1 tiles, extracted from offset k/2.
    // ns_proc = padded/k processing blocks covering the entire padded image.
    let ns_hist_y = ph / k - 1;
    let ns_hist_x = pw / k - 1;
    let ns_proc_y = ph / k;
    let ns_proc_x = pw / k;
    let tile_area = (k * k) as u64;

    let mut histograms = vec![[0u32; NUM_BINS]; ns_hist_y * ns_hist_x];
    for ty in 0..ns_hist_y {
        for tx in 0..ns_hist_x {
            let y0 = k / 2 + ty * k;
            let x0 = k / 2 + tx * k;
            for dy in 0..k {
                for dx in 0..k {
                    let val = padded[(y0 + dy) * pw + (x0 + dx)] as usize;
                    histograms[ty * ns_hist_x + tx][val] += 1;
                }
            }
        }
    }

    // Step 4: Clip and redistribute.
    // skimage: clim = int(clip(clip_limit * kernel_elements, 1, None))
    let clim = if clip_limit > 0.0 {
        (clip_limit * tile_area as f32).max(1.0) as u32
    } else {
        tile_area as u32
    };
    for hist in histograms.iter_mut() {
        clip_histogram_redistribute(hist, clim);
    }

    // Step 5: Compute CDF mappings.
    // map_histogram: cumsum * 255 / n_pixels, clipped to 255.
    let mut cdfs = vec![[0u32; NUM_BINS]; ns_hist_y * ns_hist_x];
    for (i, hist) in histograms.iter().enumerate() {
        let mut cum = 0u64;
        for b in 0..NUM_BINS {
            cum += hist[b] as u64;
            cdfs[i][b] = ((cum * 255) / tile_area).min(255) as u32;
        }
    }

    // Step 6: Build map_array with edge padding (ns_hist+2 in each dim).
    // np.pad(hist, [[1,1],[1,1],[0,0]], mode='edge')
    let may = ns_hist_y + 2;
    let max = ns_hist_x + 2;
    let mut map_array = vec![[0u32; NUM_BINS]; may * max];
    for ty in 0..may {
        for tx in 0..max {
            let sy = ty.saturating_sub(1).min(ns_hist_y - 1);
            let sx = tx.saturating_sub(1).min(ns_hist_x - 1);
            map_array[ty * max + tx] = cdfs[sy * ns_hist_x + sx];
        }
    }

    // Step 7: Bilinear interpolation per processing block.
    // For block (by, bx), pixel (dy, dx): interpolate between 4 surrounding CDFs.
    // Coefficients: fx = dx/k, fy = dy/k (matching skimage).
    let mut result = vec![0u8; ph * pw];
    for by in 0..ns_proc_y {
        for bx in 0..ns_proc_x {
            for dy in 0..k {
                for dx in 0..k {
                    let y = by * k + dy;
                    let x = bx * k + dx;
                    let val = padded[y * pw + x] as usize;

                    let fx = dx as f32 / k as f32;
                    let fy = dy as f32 / k as f32;

                    let tl = map_array[by * max + bx][val] as f32;
                    let tr = map_array[by * max + (bx + 1)][val] as f32;
                    let bl = map_array[(by + 1) * max + bx][val] as f32;
                    let br = map_array[(by + 1) * max + (bx + 1)][val] as f32;

                    let top = (1.0 - fx) * tl + fx * tr;
                    let bot = (1.0 - fx) * bl + fx * br;
                    let mapped = (1.0 - fy) * top + fy * bot;

                    result[y * pw + x] = mapped.clamp(0.0, 255.0) as u8;
                }
            }
        }
    }

    // Step 8: Unpad to original size
    let mut out = vec![0u8; height * width];
    for y in 0..height {
        for x in 0..width {
            out[y * width + x] = result[(y + pad_start_y) * pw + (x + pad_start_x)];
        }
    }

    out
}

// ═══════════════════════════════════════════════════════════════════════════
//  PyO3 Bindings
// ═══════════════════════════════════════════════════════════════════════════

#[pyclass(name = "image_process_rs")]
struct ImageProcessRs;

#[pymethods]
impl ImageProcessRs {
    #[new]
    fn new() -> Self {
        Self
    }

    #[pyo3(signature = (img_np, include_singletons=false))]
    fn skeleton_branch_stats(
        &self,
        py: Python<'_>,
        img_np: PyReadonlyArrayDyn<'_, u8>,
        include_singletons: bool,
    ) -> PyResult<Vec<(f64, usize, usize)>> {
        let shape = img_np.shape();
        if shape.len() != 2 {
            return Err(PyValueError::new_err("Expected a 2D uint8 numpy array"));
        }
        let height = shape[0];
        let width = shape[1];
        let pixels = img_np
            .as_slice()
            .map_err(|_| PyValueError::new_err("Array must be contiguous"))?
            .to_vec();

        py.detach(move || {
            compute_skeleton_branch_stats_impl(&pixels, height, width, include_singletons)
                .map_err(PyValueError::new_err)
        })
    }

    fn __repr__(&self) -> &'static str {
        "image_process_rs()"
    }
}

// --- Standalone pyfunction exports ---

#[pyfunction]
#[pyo3(signature = (img_np, include_singletons=false))]
fn skeleton_branch_stats(
    py: Python<'_>,
    img_np: PyReadonlyArrayDyn<'_, u8>,
    include_singletons: bool,
) -> PyResult<Vec<(f64, usize, usize)>> {
    let shape = img_np.shape();
    if shape.len() != 2 {
        return Err(PyValueError::new_err("Expected a 2D uint8 numpy array"));
    }
    let (height, width) = (shape[0], shape[1]);
    let pixels = img_np
        .as_slice()
        .map_err(|_| PyValueError::new_err("Array must be contiguous"))?
        .to_vec();

    py.detach(move || {
        compute_skeleton_branch_stats_impl(&pixels, height, width, include_singletons)
            .map_err(PyValueError::new_err)
    })
}

#[pyfunction]
fn skeleton_stats_full(
    py: Python<'_>,
    img_np: PyReadonlyArrayDyn<'_, u8>,
) -> PyResult<Vec<(f64, f64, f64, usize)>> {
    let shape = img_np.shape();
    if shape.len() != 2 {
        return Err(PyValueError::new_err("Expected a 2D uint8 numpy array"));
    }
    let (height, width) = (shape[0], shape[1]);
    let pixels = img_np
        .as_slice()
        .map_err(|_| PyValueError::new_err("Array must be contiguous"))?
        .to_vec();

    py.detach(move || {
        compute_skeleton_stats_full_impl(&pixels, height, width)
            .map_err(PyValueError::new_err)
    })
}

/// Helper: create a PyArray2 from a flat Vec and shape.
fn vec_to_pyarray2<T: numpy::Element>(
    py: Python<'_>,
    data: Vec<T>,
    height: usize,
    width: usize,
) -> PyResult<Bound<'_, PyArray2<T>>> {
    let arr = Array2::from_shape_vec((height, width), data)
        .map_err(|e| PyValueError::new_err(format!("shape error: {}", e)))?;
    Ok(PyArray2::from_owned_array(py, arr))
}

/// Rolling-ball background subtraction.
/// Input: 2D float32 array (grayscale, [0..1] range), radius in pixels.
/// Returns: 2D float32 background array (same shape).
#[pyfunction]
fn rolling_ball<'py>(
    py: Python<'py>,
    img_np: PyReadonlyArray2<'py, f32>,
    radius: f32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let shape = img_np.shape();
    let (height, width) = (shape[0], shape[1]);
    let data = img_np
        .as_slice()
        .map_err(|_| PyValueError::new_err("Array must be contiguous"))?
        .to_vec();
    if radius <= 0.0 {
        return Err(PyValueError::new_err("radius must be > 0"));
    }
    let bg = py.detach(move || rolling_ball_impl(&data, height, width, radius));
    vec_to_pyarray2(py, bg, height, width)
}

/// Rolling-ball for multi-channel (e.g. RGB) images.
/// Input: 3D float32 array [H, W, C], radius in pixels.
/// Returns: 2D float32 array [H*W, C] (reshape to [H,W,C] in Python).
#[pyfunction]
fn rolling_ball_rgb<'py>(
    py: Python<'py>,
    img_np: PyReadonlyArrayDyn<'py, f32>,
    radius: f32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let shape = img_np.shape();
    if shape.len() != 3 {
        return Err(PyValueError::new_err("Expected a 3D float32 array [H,W,C]"));
    }
    let (height, width, channels) = (shape[0], shape[1], shape[2]);
    let data = img_np
        .as_slice()
        .map_err(|_| PyValueError::new_err("Array must be contiguous"))?
        .to_vec();
    if radius <= 0.0 {
        return Err(PyValueError::new_err("radius must be > 0"));
    }
    let bg = py.detach(move || {
        let npix = height * width;
        let channel_data: Vec<Vec<f32>> = (0..channels)
            .map(|c| (0..npix).map(|i| data[i * channels + c]).collect())
            .collect();
        let channel_bgs: Vec<Vec<f32>> = channel_data
            .par_iter()
            .map(|ch| rolling_ball_impl(ch, height, width, radius))
            .collect();
        let mut result = vec![0.0f32; npix * channels];
        for c in 0..channels {
            for i in 0..npix {
                result[i * channels + c] = channel_bgs[c][i];
            }
        }
        result
    });
    vec_to_pyarray2(py, bg, height * width, channels)
}

/// Euclidean distance transform.
/// Input: 2D uint8 array (nonzero = foreground, distance from background).
/// Returns: 2D float32 array of Euclidean distances.
#[pyfunction]
fn distance_transform_edt<'py>(
    py: Python<'py>,
    binary_np: PyReadonlyArray2<'py, u8>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let shape = binary_np.shape();
    let (height, width) = (shape[0], shape[1]);
    let data = binary_np
        .as_slice()
        .map_err(|_| PyValueError::new_err("Array must be contiguous"))?
        .to_vec();
    let dt = py.detach(move || distance_transform_edt_impl(&data, height, width));
    vec_to_pyarray2(py, dt, height, width)
}

/// Connected component labeling.
/// Input: 2D uint8 array (nonzero = foreground), connectivity (4 or 8).
/// Returns: (2D uint32 label array, num_labels).
#[pyfunction]
#[pyo3(signature = (binary_np, connectivity=4))]
fn label_2d<'py>(
    py: Python<'py>,
    binary_np: PyReadonlyArray2<'py, u8>,
    connectivity: u8,
) -> PyResult<(Bound<'py, PyArray2<u32>>, u32)> {
    if connectivity != 4 && connectivity != 8 {
        return Err(PyValueError::new_err("connectivity must be 4 or 8"));
    }
    let shape = binary_np.shape();
    let (height, width) = (shape[0], shape[1]);
    let data = binary_np
        .as_slice()
        .map_err(|_| PyValueError::new_err("Array must be contiguous"))?
        .to_vec();
    let (labels, n) = py.detach(move || label_2d_impl(&data, height, width, connectivity));
    Ok((vec_to_pyarray2(py, labels, height, width)?, n))
}

/// Marker-based watershed segmentation.
/// Input: grayscale u8 image, u32 marker array, optional u8 mask.
/// Returns: u32 label array.
#[pyfunction]
#[pyo3(signature = (image_np, markers_np, mask_np=None))]
fn watershed<'py>(
    py: Python<'py>,
    image_np: PyReadonlyArray2<'py, u8>,
    markers_np: PyReadonlyArray2<'py, u32>,
    mask_np: Option<PyReadonlyArray2<'py, u8>>,
) -> PyResult<Bound<'py, PyArray2<u32>>> {
    let shape = image_np.shape();
    let (height, width) = (shape[0], shape[1]);
    let mshape = markers_np.shape();
    if mshape[0] != height || mshape[1] != width {
        return Err(PyValueError::new_err("markers shape must match image shape"));
    }
    let image = image_np
        .as_slice()
        .map_err(|_| PyValueError::new_err("image must be contiguous"))?
        .to_vec();
    let markers = markers_np
        .as_slice()
        .map_err(|_| PyValueError::new_err("markers must be contiguous"))?
        .to_vec();
    let mask = match mask_np {
        Some(m) => {
            let ms = m.shape();
            if ms[0] != height || ms[1] != width {
                return Err(PyValueError::new_err("mask shape must match image shape"));
            }
            Some(m.as_slice()
                .map_err(|_| PyValueError::new_err("mask must be contiguous"))?
                .to_vec())
        }
        None => None,
    };
    let labels = py.detach(move || {
        watershed_impl(&image, &markers, mask.as_deref(), height, width)
    });
    vec_to_pyarray2(py, labels, height, width)
}

/// Grayscale erosion with disk structuring element.
#[pyfunction]
fn erode<'py>(
    py: Python<'py>,
    img_np: PyReadonlyArray2<'py, u8>,
    radius: usize,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let shape = img_np.shape();
    let (height, width) = (shape[0], shape[1]);
    let data = img_np
        .as_slice()
        .map_err(|_| PyValueError::new_err("Array must be contiguous"))?
        .to_vec();
    let (kernel, ksize) = disk_kernel_flat(radius);
    let result = py.detach(move || erode_impl(&data, height, width, &kernel, ksize));
    vec_to_pyarray2(py, result, height, width)
}

/// Grayscale dilation with disk structuring element.
#[pyfunction]
fn dilate<'py>(
    py: Python<'py>,
    img_np: PyReadonlyArray2<'py, u8>,
    radius: usize,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let shape = img_np.shape();
    let (height, width) = (shape[0], shape[1]);
    let data = img_np
        .as_slice()
        .map_err(|_| PyValueError::new_err("Array must be contiguous"))?
        .to_vec();
    let (kernel, ksize) = disk_kernel_flat(radius);
    let result = py.detach(move || dilate_impl(&data, height, width, &kernel, ksize));
    vec_to_pyarray2(py, result, height, width)
}

/// Morphological opening (erode then dilate).
#[pyfunction]
fn opening<'py>(
    py: Python<'py>,
    img_np: PyReadonlyArray2<'py, u8>,
    radius: usize,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let shape = img_np.shape();
    let (height, width) = (shape[0], shape[1]);
    let data = img_np
        .as_slice()
        .map_err(|_| PyValueError::new_err("Array must be contiguous"))?
        .to_vec();
    let (kernel, ksize) = disk_kernel_flat(radius);
    let result = py.detach(move || {
        let eroded = erode_impl(&data, height, width, &kernel, ksize);
        dilate_impl(&eroded, height, width, &kernel, ksize)
    });
    vec_to_pyarray2(py, result, height, width)
}

/// Morphological closing (dilate then erode).
#[pyfunction]
fn closing<'py>(
    py: Python<'py>,
    img_np: PyReadonlyArray2<'py, u8>,
    radius: usize,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let shape = img_np.shape();
    let (height, width) = (shape[0], shape[1]);
    let data = img_np
        .as_slice()
        .map_err(|_| PyValueError::new_err("Array must be contiguous"))?
        .to_vec();
    let (kernel, ksize) = disk_kernel_flat(radius);
    let result = py.detach(move || {
        let dilated = dilate_impl(&data, height, width, &kernel, ksize);
        erode_impl(&dilated, height, width, &kernel, ksize)
    });
    vec_to_pyarray2(py, result, height, width)
}

/// White top-hat: original − opening. Extracts bright features smaller than the kernel.
#[pyfunction]
fn white_tophat<'py>(
    py: Python<'py>,
    img_np: PyReadonlyArray2<'py, u8>,
    radius: usize,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let shape = img_np.shape();
    let (height, width) = (shape[0], shape[1]);
    let data = img_np
        .as_slice()
        .map_err(|_| PyValueError::new_err("Array must be contiguous"))?
        .to_vec();
    let (kernel, ksize) = disk_kernel_flat(radius);
    let result = py.detach(move || {
        let eroded = erode_impl(&data, height, width, &kernel, ksize);
        let opened = dilate_impl(&eroded, height, width, &kernel, ksize);
        data.iter().zip(opened.iter()).map(|(&o, &op)| o.saturating_sub(op)).collect::<Vec<u8>>()
    });
    vec_to_pyarray2(py, result, height, width)
}

/// Black top-hat: closing − original. Extracts dark features smaller than the kernel.
#[pyfunction]
fn black_tophat<'py>(
    py: Python<'py>,
    img_np: PyReadonlyArray2<'py, u8>,
    radius: usize,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let shape = img_np.shape();
    let (height, width) = (shape[0], shape[1]);
    let data = img_np
        .as_slice()
        .map_err(|_| PyValueError::new_err("Array must be contiguous"))?
        .to_vec();
    let (kernel, ksize) = disk_kernel_flat(radius);
    let result = py.detach(move || {
        let dilated = dilate_impl(&data, height, width, &kernel, ksize);
        let closed = erode_impl(&dilated, height, width, &kernel, ksize);
        closed.iter().zip(data.iter()).map(|(&c, &o)| c.saturating_sub(o)).collect::<Vec<u8>>()
    });
    vec_to_pyarray2(py, result, height, width)
}

/// CLAHE — Contrast Limited Adaptive Histogram Equalization.
/// Input: 2D uint8 grayscale image, tile_size (e.g. 8), clip_limit (e.g. 2.0).
/// Returns: enhanced 2D uint8 image.
#[pyfunction]
#[pyo3(signature = (img_np, tile_size=8, clip_limit=2.0))]
fn clahe<'py>(
    py: Python<'py>,
    img_np: PyReadonlyArray2<'py, u8>,
    tile_size: usize,
    clip_limit: f32,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    if tile_size == 0 {
        return Err(PyValueError::new_err("tile_size must be > 0"));
    }
    if clip_limit <= 0.0 {
        return Err(PyValueError::new_err("clip_limit must be > 0"));
    }
    let shape = img_np.shape();
    let (height, width) = (shape[0], shape[1]);
    let data = img_np
        .as_slice()
        .map_err(|_| PyValueError::new_err("Array must be contiguous"))?
        .to_vec();
    let result = py.detach(move || clahe_impl(&data, height, width, tile_size, clip_limit));
    vec_to_pyarray2(py, result, height, width)
}

// ─── Module registration ─────────────────────────────────────────────────

#[pymodule]
fn image_process_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ImageProcessRs>()?;
    // Existing
    m.add_function(wrap_pyfunction!(skeleton_branch_stats, m)?)?;
    m.add_function(wrap_pyfunction!(skeleton_stats_full, m)?)?;
    // New functions
    m.add_function(wrap_pyfunction!(rolling_ball, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_ball_rgb, m)?)?;
    m.add_function(wrap_pyfunction!(distance_transform_edt, m)?)?;
    m.add_function(wrap_pyfunction!(label_2d, m)?)?;
    m.add_function(wrap_pyfunction!(watershed, m)?)?;
    m.add_function(wrap_pyfunction!(erode, m)?)?;
    m.add_function(wrap_pyfunction!(dilate, m)?)?;
    m.add_function(wrap_pyfunction!(opening, m)?)?;
    m.add_function(wrap_pyfunction!(closing, m)?)?;
    m.add_function(wrap_pyfunction!(white_tophat, m)?)?;
    m.add_function(wrap_pyfunction!(black_tophat, m)?)?;
    m.add_function(wrap_pyfunction!(clahe, m)?)?;
    Ok(())
}
