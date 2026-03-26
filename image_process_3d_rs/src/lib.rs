use ndarray::{Array3, Axis};
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::VecDeque;

// ══════════════════════════════════════════════════════════════════════
// 1.  Gaussian Filter 3D  (separable, parallel)
// ══════════════════════════════════════════════════════════════════════

/// 1D Gaussian kernel (normalized).
fn make_gaussian_kernel(sigma: f64) -> Vec<f64> {
    let radius = (sigma * 4.0).ceil() as usize;
    let size = 2 * radius + 1;
    let mut kernel = vec![0.0f64; size];
    let s2 = 2.0 * sigma * sigma;
    let mut sum = 0.0;
    for i in 0..size {
        let x = i as f64 - radius as f64;
        kernel[i] = (-x * x / s2).exp();
        sum += kernel[i];
    }
    for v in &mut kernel {
        *v /= sum;
    }
    kernel
}

/// Apply 1D convolution along axis 0 (Z), parallel over YX planes.
fn convolve_axis0(input: &Array3<f32>, kernel: &[f64]) -> Array3<f32> {
    let (nz, ny, nx) = input.dim();
    let r = kernel.len() / 2;
    let mut output = Array3::<f32>::zeros((nz, ny, nx));

    // Parallel over Y rows
    output
        .axis_iter_mut(Axis(1))
        .into_par_iter()
        .enumerate()
        .for_each(|(y, mut out_yx)| {
            for x in 0..nx {
                for z in 0..nz {
                    let mut sum = 0.0f64;
                    for (ki, &kv) in kernel.iter().enumerate() {
                        let zz = z as isize + ki as isize - r as isize;
                        let zz = zz.clamp(0, nz as isize - 1) as usize;
                        sum += input[(zz, y, x)] as f64 * kv;
                    }
                    out_yx[(z, x)] = sum as f32;
                }
            }
        });
    output
}

/// Apply 1D convolution along axis 1 (Y), parallel over Z slices.
fn convolve_axis1(input: &Array3<f32>, kernel: &[f64]) -> Array3<f32> {
    let (nz, ny, nx) = input.dim();
    let r = kernel.len() / 2;
    let mut output = Array3::<f32>::zeros((nz, ny, nx));

    output
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(z, mut out_z)| {
            for y in 0..ny {
                for x in 0..nx {
                    let mut sum = 0.0f64;
                    for (ki, &kv) in kernel.iter().enumerate() {
                        let yy = y as isize + ki as isize - r as isize;
                        let yy = yy.clamp(0, ny as isize - 1) as usize;
                        sum += input[(z, yy, x)] as f64 * kv;
                    }
                    out_z[(y, x)] = sum as f32;
                }
            }
        });
    output
}

/// Apply 1D convolution along axis 2 (X), parallel over Z slices.
fn convolve_axis2(input: &Array3<f32>, kernel: &[f64]) -> Array3<f32> {
    let (nz, ny, nx) = input.dim();
    let r = kernel.len() / 2;
    let mut output = Array3::<f32>::zeros((nz, ny, nx));

    output
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(z, mut out_z)| {
            for y in 0..ny {
                for x in 0..nx {
                    let mut sum = 0.0f64;
                    for (ki, &kv) in kernel.iter().enumerate() {
                        let xx = x as isize + ki as isize - r as isize;
                        let xx = xx.clamp(0, nx as isize - 1) as usize;
                        sum += input[(z, y, xx)] as f64 * kv;
                    }
                    out_z[(y, x)] = sum as f32;
                }
            }
        });
    output
}

/// 3D Gaussian filter with anisotropic sigma.
///
/// Args:
///     volume: float32 array (Z, Y, X)
///     sigma_z, sigma_y, sigma_x: Gaussian sigma per axis (in voxels)
///
/// Returns: float32 array (Z, Y, X)
#[pyfunction]
#[pyo3(signature = (volume, sigma_z, sigma_y, sigma_x))]
fn gaussian_filter_3d<'py>(
    py: Python<'py>,
    volume: PyReadonlyArray3<'py, f32>,
    sigma_z: f64,
    sigma_y: f64,
    sigma_x: f64,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let arr = volume.as_array().to_owned();

    let result = py
        .detach(move || {
            let kz = make_gaussian_kernel(sigma_z);
            let ky = make_gaussian_kernel(sigma_y);
            let kx = make_gaussian_kernel(sigma_x);

            let tmp = convolve_axis2(&arr, &kx);
            let tmp = convolve_axis1(&tmp, &ky);
            convolve_axis0(&tmp, &kz)
        });

    Ok(result.into_pyarray(py))
}

// ══════════════════════════════════════════════════════════════════════
// 2.  Distance Transform EDT 3D  (Meijster algorithm, anisotropic)
// ══════════════════════════════════════════════════════════════════════

const INF: f64 = 1e18;

/// 3D Euclidean Distance Transform with anisotropic spacing.
///
/// Args:
///     binary: uint8 array (Z, Y, X), nonzero = foreground
///     spacing_z, spacing_y, spacing_x: voxel spacing in µm
///
/// Returns: float32 distance map (Z, Y, X) in µm
#[pyfunction]
#[pyo3(signature = (binary, spacing_z=1.0, spacing_y=1.0, spacing_x=1.0))]
fn distance_transform_edt_3d<'py>(
    py: Python<'py>,
    binary: PyReadonlyArray3<'py, u8>,
    spacing_z: f64,
    spacing_y: f64,
    spacing_x: f64,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let arr = binary.as_array();
    let (nz, ny, nx) = arr.dim();

    let result = py.detach(move || {
        // Phase 1: squared distances along X (parallel over Z*Y lines)
        let mut dt = vec![0.0f64; nz * ny * nx];

        // Initialize: foreground = 0, background = INF
        // Note: EDT computes distance from background (0) pixels
        // scipy convention: distance_transform_edt(binary) = distance from 0-pixels
        // So foreground (nonzero) pixels get distance values, background = 0
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    if arr[(z, y, x)] == 0 {
                        dt[z * ny * nx + y * nx + x] = 0.0;
                    } else {
                        dt[z * ny * nx + y * nx + x] = INF;
                    }
                }
            }
        }

        // Meijster phase along X
        edt_phase(&mut dt, nz, ny, nx, 2, spacing_x); // axis 2 = X
        // Phase along Y
        edt_phase(&mut dt, nz, ny, nx, 1, spacing_y);
        // Phase along Z
        edt_phase(&mut dt, nz, ny, nx, 0, spacing_z);

        // Sqrt and convert to f32
        let mut result = Array3::<f32>::zeros((nz, ny, nx));
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    result[(z, y, x)] = dt[z * ny * nx + y * nx + x].sqrt() as f32;
                }
            }
        }
        result
    });

    Ok(result.into_pyarray(py))
}

/// Meijster EDT phase along one axis (operates on squared distances).
fn edt_phase(dt: &mut [f64], nz: usize, ny: usize, nx: usize, axis: usize, spacing: f64) {
    let sp2 = spacing * spacing;

    // Determine the iteration dimensions
    let (n_lines, line_len) = match axis {
        0 => (ny * nx, nz), // lines along Z
        1 => (nz * nx, ny), // lines along Y
        2 => (nz * ny, nx), // lines along X
        _ => unreachable!(),
    };

    // Process lines in parallel
    let dt_ptr = dt.as_mut_ptr() as usize;

    (0..n_lines).into_par_iter().for_each(|line_idx| {
        let dt_ptr = dt_ptr as *mut f64;

        // Compute indices for this line
        let indices: Vec<usize> = match axis {
            0 => {
                let y = line_idx / nx;
                let x = line_idx % nx;
                (0..line_len).map(|z| z * ny * nx + y * nx + x).collect()
            }
            1 => {
                let z = line_idx / nx;
                let x = line_idx % nx;
                (0..line_len).map(|y| z * ny * nx + y * nx + x).collect()
            }
            2 => {
                let z = line_idx / ny;
                let y = line_idx % ny;
                (0..line_len).map(|x| z * ny * nx + y * nx + x).collect()
            }
            _ => unreachable!(),
        };

        // Read current values along this line
        let mut f: Vec<f64> = indices.iter().map(|&i| unsafe { *dt_ptr.add(i) }).collect();

        // Forward pass: compute squared distance contribution
        let n = f.len();
        if n == 0 {
            return;
        }

        // Meijster lower envelope
        let mut v = vec![0usize; n]; // locations of parabolas
        let mut z_bounds = vec![0.0f64; n + 1]; // boundaries between parabolas
        z_bounds[0] = -INF;
        z_bounds[1] = INF;
        let mut k = 0usize;

        for q in 1..n {
            loop {
                let s = ((f[q] + sp2 * (q * q) as f64) - (f[v[k]] + sp2 * (v[k] * v[k]) as f64))
                    / (2.0 * sp2 * (q as f64 - v[k] as f64));
                if s > z_bounds[k] {
                    k += 1;
                    v[k] = q;
                    z_bounds[k] = s;
                    z_bounds[k + 1] = INF;
                    break;
                }
                if k == 0 {
                    v[0] = q;
                    break;
                }
                k -= 1;
            }
        }

        // Backward pass: evaluate lower envelope
        k = 0;
        for q in 0..n {
            while z_bounds[k + 1] < q as f64 {
                k += 1;
            }
            let dq = (q as f64 - v[k] as f64) * spacing;
            f[q] = dq * dq + unsafe { *dt_ptr.add(indices[v[k]]) };
        }

        // Write back (need to re-read dt values for the envelope evaluation)
        // Actually, we need f[v[k]] from the INPUT, not output.
        // The standard Meijster algorithm: we already have the input in f before the envelope.
        // Let me redo this properly:

        // Re-read input
        let input: Vec<f64> = indices.iter().map(|&i| unsafe { *dt_ptr.add(i) }).collect();

        // Recompute envelope with input values
        let mut v2 = vec![0usize; n];
        let mut z2 = vec![0.0f64; n + 1];
        z2[0] = -INF;
        z2[1] = INF;
        let mut k2 = 0usize;

        for q in 1..n {
            loop {
                let s = ((input[q] + sp2 * (q * q) as f64)
                    - (input[v2[k2]] + sp2 * (v2[k2] * v2[k2]) as f64))
                    / (2.0 * sp2 * (q as f64 - v2[k2] as f64));
                if s > z2[k2] {
                    k2 += 1;
                    v2[k2] = q;
                    z2[k2] = s;
                    z2[k2 + 1] = INF;
                    break;
                }
                if k2 == 0 {
                    v2[0] = q;
                    break;
                }
                k2 -= 1;
            }
        }

        k2 = 0;
        for q in 0..n {
            while z2[k2 + 1] < q as f64 {
                k2 += 1;
            }
            let dq = (q as f64 - v2[k2] as f64) * spacing;
            let val = dq * dq + input[v2[k2]];
            unsafe { *dt_ptr.add(indices[q]) = val; }
        }
    });
}

// ══════════════════════════════════════════════════════════════════════
// 3.  Label 3D  (BFS connected components)
// ══════════════════════════════════════════════════════════════════════

/// 3D connected component labeling (6-connectivity).
///
/// Args:
///     binary: uint8 array (Z, Y, X), nonzero = foreground
///
/// Returns: (labels, n_labels) — int32 label array + count
#[pyfunction]
fn label_3d<'py>(
    py: Python<'py>,
    binary: PyReadonlyArray3<'py, u8>,
) -> PyResult<(Bound<'py, PyArray3<i32>>, i32)> {
    let arr = binary.as_array();
    let (nz, ny, nx) = arr.dim();

    let mut labels = Array3::<i32>::zeros((nz, ny, nx));
    let mut current_label: i32 = 0;
    let mut queue = VecDeque::new();

    // 6-connectivity neighbors
    let neighbors: [(isize, isize, isize); 6] = [
        (-1, 0, 0), (1, 0, 0),
        (0, -1, 0), (0, 1, 0),
        (0, 0, -1), (0, 0, 1),
    ];

    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                if arr[(z, y, x)] != 0 && labels[(z, y, x)] == 0 {
                    current_label += 1;
                    labels[(z, y, x)] = current_label;
                    queue.push_back((z, y, x));

                    while let Some((cz, cy, cx)) = queue.pop_front() {
                        for &(dz, dy, dx) in &neighbors {
                            let nz2 = cz as isize + dz;
                            let ny2 = cy as isize + dy;
                            let nx2 = cx as isize + dx;
                            if nz2 >= 0
                                && nz2 < nz as isize
                                && ny2 >= 0
                                && ny2 < ny as isize
                                && nx2 >= 0
                                && nx2 < nx as isize
                            {
                                let (nz2, ny2, nx2) =
                                    (nz2 as usize, ny2 as usize, nx2 as usize);
                                if arr[(nz2, ny2, nx2)] != 0 && labels[(nz2, ny2, nx2)] == 0 {
                                    labels[(nz2, ny2, nx2)] = current_label;
                                    queue.push_back((nz2, ny2, nx2));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok((labels.into_pyarray(py), current_label))
}

// ══════════════════════════════════════════════════════════════════════
// 4.  Watershed 3D  (bucket queue for O(1) push/pop)
// ══════════════════════════════════════════════════════════════════════

const N_BUCKETS: usize = 65536;

/// Bucket queue: quantize float values into fixed buckets for O(1) priority queue.
struct BucketQueue {
    buckets: Vec<Vec<(usize, usize, usize)>>,
    current: usize,
    count: usize,
}

impl BucketQueue {
    fn new() -> Self {
        Self {
            buckets: (0..N_BUCKETS).map(|_| Vec::new()).collect(),
            current: 0,
            count: 0,
        }
    }

    #[inline]
    fn push(&mut self, val: f32, z: usize, y: usize, x: usize, vmin: f32, vmax: f32) {
        let range = vmax - vmin;
        let bucket = if range <= 0.0 {
            0
        } else {
            let normalized = ((val - vmin) / range).clamp(0.0, 0.99999);
            (normalized * N_BUCKETS as f32) as usize
        };
        self.buckets[bucket].push((z, y, x));
        if bucket < self.current {
            self.current = bucket;
        }
        self.count += 1;
    }

    #[inline]
    fn pop(&mut self) -> Option<(usize, usize, usize)> {
        if self.count == 0 {
            return None;
        }
        while self.current < N_BUCKETS {
            if let Some(v) = self.buckets[self.current].pop() {
                self.count -= 1;
                return Some(v);
            }
            self.current += 1;
        }
        None
    }
}

/// 3D marker-based watershed segmentation.
///
/// Args:
///     image: float32 (Z, Y, X) — intensity/distance image
///     markers: int32 (Z, Y, X) — seed labels (0 = unlabeled)
///     mask: optional uint8 (Z, Y, X) — restrict to nonzero region
///
/// Returns: int32 (Z, Y, X) labeled segmentation
#[pyfunction]
#[pyo3(signature = (image, markers, mask=None))]
fn watershed_3d<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<'py, f32>,
    markers: PyReadonlyArray3<'py, i32>,
    mask: Option<PyReadonlyArray3<'py, u8>>,
) -> PyResult<Bound<'py, PyArray3<i32>>> {
    let img = image.as_array();
    let mrk = markers.as_array();
    let (nz, ny, nx) = img.dim();

    if mrk.dim() != (nz, ny, nx) {
        return Err(PyValueError::new_err("markers shape mismatch"));
    }

    // Find value range for bucket quantization
    let mut vmin = f32::MAX;
    let mut vmax = f32::MIN;
    for &v in img.iter() {
        if v < vmin { vmin = v; }
        if v > vmax { vmax = v; }
    }

    let mask_arr = mask.as_ref().map(|m| m.as_array());
    let mut labels = mrk.to_owned();
    let mut bq = BucketQueue::new();

    let neighbors: [(isize, isize, isize); 6] = [
        (-1, 0, 0), (1, 0, 0),
        (0, -1, 0), (0, 1, 0),
        (0, 0, -1), (0, 0, 1),
    ];

    // Seed the queue with all labeled voxels bordering unlabeled ones
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                if labels[(z, y, x)] > 0 {
                    for &(dz, dy, dx) in &neighbors {
                        let nz2 = z as isize + dz;
                        let ny2 = y as isize + dy;
                        let nx2 = x as isize + dx;
                        if nz2 >= 0 && nz2 < nz as isize
                            && ny2 >= 0 && ny2 < ny as isize
                            && nx2 >= 0 && nx2 < nx as isize
                        {
                            if labels[(nz2 as usize, ny2 as usize, nx2 as usize)] == 0 {
                                bq.push(img[(z, y, x)], z, y, x, vmin, vmax);
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    // Flood
    while let Some((vz, vy, vx)) = bq.pop() {
        let lbl = labels[(vz, vy, vx)];
        if lbl == 0 { continue; }

        for &(dz, dy, dx) in &neighbors {
            let nz2 = vz as isize + dz;
            let ny2 = vy as isize + dy;
            let nx2 = vx as isize + dx;
            if nz2 >= 0 && nz2 < nz as isize
                && ny2 >= 0 && ny2 < ny as isize
                && nx2 >= 0 && nx2 < nx as isize
            {
                let (nz2, ny2, nx2) = (nz2 as usize, ny2 as usize, nx2 as usize);
                if labels[(nz2, ny2, nx2)] != 0 { continue; }
                if let Some(ref ma) = mask_arr {
                    if ma[(nz2, ny2, nx2)] == 0 { continue; }
                }
                labels[(nz2, ny2, nx2)] = lbl;
                bq.push(img[(nz2, ny2, nx2)], nz2, ny2, nx2, vmin, vmax);
            }
        }
    }

    Ok(labels.into_pyarray(py))
}

// ══════════════════════════════════════════════════════════════════════
// 5.  Binary Fill Holes 3D  (flood from border)
// ══════════════════════════════════════════════════════════════════════

/// Fill holes in a 3D binary volume.
/// A hole is a background region not connected to the volume border.
///
/// Args:
///     binary: uint8 (Z, Y, X), nonzero = foreground
///
/// Returns: uint8 (Z, Y, X) with holes filled
#[pyfunction]
fn binary_fill_holes_3d<'py>(
    py: Python<'py>,
    binary: PyReadonlyArray3<'py, u8>,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let arr = binary.as_array();
    let (nz, ny, nx) = arr.dim();

    // Flood fill background from all border voxels
    let mut visited = vec![false; nz * ny * nx];
    let mut queue = VecDeque::new();

    let idx = |z: usize, y: usize, x: usize| z * ny * nx + y * nx + x;

    // Seed all border background voxels
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let is_border = z == 0 || z == nz - 1 || y == 0 || y == ny - 1 || x == 0 || x == nx - 1;
                if is_border && arr[(z, y, x)] == 0 && !visited[idx(z, y, x)] {
                    visited[idx(z, y, x)] = true;
                    queue.push_back((z, y, x));
                }
            }
        }
    }

    let neighbors: [(isize, isize, isize); 6] = [
        (-1, 0, 0), (1, 0, 0),
        (0, -1, 0), (0, 1, 0),
        (0, 0, -1), (0, 0, 1),
    ];

    while let Some((cz, cy, cx)) = queue.pop_front() {
        for &(dz, dy, dx) in &neighbors {
            let nz2 = cz as isize + dz;
            let ny2 = cy as isize + dy;
            let nx2 = cx as isize + dx;
            if nz2 >= 0
                && nz2 < nz as isize
                && ny2 >= 0
                && ny2 < ny as isize
                && nx2 >= 0
                && nx2 < nx as isize
            {
                let (nz2, ny2, nx2) = (nz2 as usize, ny2 as usize, nx2 as usize);
                let i = idx(nz2, ny2, nx2);
                if !visited[i] && arr[(nz2, ny2, nx2)] == 0 {
                    visited[i] = true;
                    queue.push_back((nz2, ny2, nx2));
                }
            }
        }
    }

    // Result: foreground OR interior (unvisited background = holes)
    let mut result = Array3::<u8>::zeros((nz, ny, nx));
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                if arr[(z, y, x)] != 0 || !visited[idx(z, y, x)] {
                    result[(z, y, x)] = 1;
                }
            }
        }
    }

    Ok(result.into_pyarray(py))
}

// ══════════════════════════════════════════════════════════════════════
// 6.  Remove Small Objects 3D
// ══════════════════════════════════════════════════════════════════════

/// Remove connected components smaller than min_size from a 3D binary volume.
///
/// Args:
///     binary: uint8 (Z, Y, X), nonzero = foreground
///     min_size: minimum number of voxels to keep
///
/// Returns: uint8 (Z, Y, X) with small objects removed
#[pyfunction]
#[pyo3(signature = (binary, min_size))]
fn remove_small_objects_3d<'py>(
    py: Python<'py>,
    binary: PyReadonlyArray3<'py, u8>,
    min_size: usize,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let arr = binary.as_array();
    let (nz, ny, nx) = arr.dim();

    // Label first
    let mut labels = vec![0i32; nz * ny * nx];
    let mut current_label: i32 = 0;
    let mut queue = VecDeque::new();
    let mut sizes: Vec<usize> = vec![0]; // sizes[label] = count, 0-indexed label starts at 1

    let idx = |z: usize, y: usize, x: usize| z * ny * nx + y * nx + x;

    let neighbors: [(isize, isize, isize); 6] = [
        (-1, 0, 0), (1, 0, 0),
        (0, -1, 0), (0, 1, 0),
        (0, 0, -1), (0, 0, 1),
    ];

    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let i = idx(z, y, x);
                if arr[(z, y, x)] != 0 && labels[i] == 0 {
                    current_label += 1;
                    labels[i] = current_label;
                    queue.push_back((z, y, x));
                    let mut count = 1usize;

                    while let Some((cz, cy, cx)) = queue.pop_front() {
                        for &(dz, dy, dx) in &neighbors {
                            let nz2 = cz as isize + dz;
                            let ny2 = cy as isize + dy;
                            let nx2 = cx as isize + dx;
                            if nz2 >= 0 && nz2 < nz as isize
                                && ny2 >= 0 && ny2 < ny as isize
                                && nx2 >= 0 && nx2 < nx as isize
                            {
                                let (nz2, ny2, nx2) = (nz2 as usize, ny2 as usize, nx2 as usize);
                                let j = idx(nz2, ny2, nx2);
                                if arr[(nz2, ny2, nx2)] != 0 && labels[j] == 0 {
                                    labels[j] = current_label;
                                    queue.push_back((nz2, ny2, nx2));
                                    count += 1;
                                }
                            }
                        }
                    }
                    sizes.push(count);
                }
            }
        }
    }

    // Build output: keep only components >= min_size
    let mut result = Array3::<u8>::zeros((nz, ny, nx));
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let lbl = labels[idx(z, y, x)];
                if lbl > 0 && sizes[lbl as usize] >= min_size {
                    result[(z, y, x)] = 1;
                }
            }
        }
    }

    Ok(result.into_pyarray(py))
}

// ══════════════════════════════════════════════════════════════════════
// Module registration
// ══════════════════════════════════════════════════════════════════════

#[pymodule]
fn image_process_3d_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gaussian_filter_3d, m)?)?;
    m.add_function(wrap_pyfunction!(distance_transform_edt_3d, m)?)?;
    m.add_function(wrap_pyfunction!(label_3d, m)?)?;
    m.add_function(wrap_pyfunction!(watershed_3d, m)?)?;
    m.add_function(wrap_pyfunction!(binary_fill_holes_3d, m)?)?;
    m.add_function(wrap_pyfunction!(remove_small_objects_3d, m)?)?;
    Ok(())
}
