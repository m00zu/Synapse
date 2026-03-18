use memchr::memmem;
use memmap2::Mmap;
use numpy::ndarray::{Array2, Array3};
use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::IntoPyObjectExt;
use rayon::prelude::*;
use regex::Regex;
use std::fs::File;
use std::path::Path;
use std::sync::OnceLock;

// Internal error type — lets all core functions run inside
// py.detach() closures without needing a Python token.
type Res<T> = Result<T, String>;

// ──────────────────────────────────────────────────────────────────────
// 1.  get_meta_block  — locate the XML metadata inside the OIR binary
// ──────────────────────────────────────────────────────────────────────
fn get_meta_block(path: &Path) -> Res<String> {
    let file = File::open(path).map_err(|e| format!("Cannot open file: {e}"))?;
    let mmap = unsafe { Mmap::map(&file) }.map_err(|e| format!("Cannot mmap file: {e}"))?;

    // memmem uses SIMD (Aho-Corasick / two-way) for multi-byte patterns.
    let start = memmem::find(&mmap, b"<fileinfo").unwrap_or(0);
    let end = memmem::find(&mmap[start..], b"<annotation")
        .map(|p| start + p)
        .unwrap_or_else(|| (start + 40_000).min(mmap.len()));

    Ok(String::from_utf8_lossy(&mmap[start..end]).into_owned())
}

// ──────────────────────────────────────────────────────────────────────
// 2.  extract_xmldata  — pull a value out of <tag>…</tag>
// ──────────────────────────────────────────────────────────────────────
enum XmlValue {
    Str(String),
    Float(f64),
    Int(i64),
}

fn extract_xmldata(xml: &str, field: &str, as_float: bool, as_int: bool) -> Option<XmlValue> {
    let open = format!("<{field}>");
    let close = format!("</{field}>");
    let a = xml.find(&open)?;
    let after_open = a + open.len();
    let b = xml[after_open..].find(&close)?;
    let inner = xml[after_open..after_open + b].trim();

    if as_float {
        inner.parse::<f64>().ok().map(XmlValue::Float)
    } else if as_int {
        inner.parse::<i64>().ok().map(XmlValue::Int)
    } else {
        Some(XmlValue::Str(inner.to_string()))
    }
}

fn xml_int(xml: &str, field: &str) -> Option<i64> {
    match extract_xmldata(xml, field, false, true)? {
        XmlValue::Int(v) => Some(v),
        _ => None,
    }
}
fn xml_float(xml: &str, field: &str) -> Option<f64> {
    match extract_xmldata(xml, field, true, false)? {
        XmlValue::Float(v) => Some(v),
        _ => None,
    }
}
fn xml_str(xml: &str, field: &str) -> Option<String> {
    match extract_xmldata(xml, field, false, false)? {
        XmlValue::Str(v) => Some(v),
        _ => None,
    }
}

// ──────────────────────────────────────────────────────────────────────
// 3.  read_meta_block  — parse dimensions / channels / bit depth
// ──────────────────────────────────────────────────────────────────────
struct OirMeta {
    size_x: usize,
    size_y: usize,
    n_channels: usize,
    line_rate: f64,
    bit_depth: usize,
}

fn read_meta_block(meta: &str) -> Res<OirMeta> {
    let p_resonant = meta.find("<lsmimage:scannerSettings type=\"Resonant\">");
    let p_galvano  = meta.find("<lsmimage:scannerSettings type=\"Galvano\">");

    let meta_scan = match (p_resonant, p_galvano) {
        (Some(pr), Some(pg)) => {
            let scanner_type = xml_str(meta, "lsmimage:scannerType");
            if scanner_type.as_deref() == Some("Resonant") { &meta[pr..] } else { &meta[pg..pr] }
        }
        (Some(pr), None) => &meta[pr..],
        (None, Some(pg)) => &meta[pg..],
        (None, None)     => meta,
    };

    let bit_depth  = xml_int(meta, "commonphase:bitCounts").unwrap_or(12) as usize;
    let size_x     = xml_int(meta_scan, "commonparam:width")
        .ok_or_else(|| "Missing width in metadata".to_string())? as usize;
    let size_y     = xml_int(meta_scan, "commonparam:height")
        .ok_or_else(|| "Missing height in metadata".to_string())? as usize;
    let line_rate  = xml_float(meta_scan, "commonparam:lineSpeed")
        .ok_or_else(|| "Missing lineSpeed in metadata".to_string())?;

    // Compile the regex only once per process lifetime.
    static CH_RE: OnceLock<Regex> = OnceLock::new();
    let re = CH_RE.get_or_init(|| {
        Regex::new(r#"<commonphase:channel\s+id="[^"]+"\s+order="(\d+)""#).unwrap()
    });

    let unique_orders: std::collections::HashSet<&str> =
        re.captures_iter(meta).filter_map(|c| c.get(1).map(|m| m.as_str())).collect();
    let n_channels = if unique_orders.is_empty() { 4 } else { unique_orders.len() };

    Ok(OirMeta { size_x, size_y, n_channels, line_rate, bit_depth })
}

// ──────────────────────────────────────────────────────────────────────
// 4.  read_oir_frames_tiled  — decode tiled pixel data
//
//  Optimisations vs the original:
//   • mmap instead of fs::read  — avoids a full file copy into heap
//   • memmem::find_iter         — SIMD tile-marker search, replaces
//                                  pos_4 pre-scan + per-byte filter loop
//   • rayon starts_per_div      — tile offset search parallelised across divs
//   • rayon channel decode      — channels decoded in parallel
//   • integer (v*255)/max_val   — replaces float division, exact match
// ──────────────────────────────────────────────────────────────────────
fn read_oir_frames_tiled(
    path: &Path,
    size_x: usize,
    size_y: usize,
    n_ch: usize,
    line_rate: f64,
    flag: i32,
    bitsize: usize,
) -> Res<Array3<u16>> {
    let file = File::open(path).map_err(|e| format!("Cannot open file: {e}"))?;
    let raw = unsafe { Mmap::map(&file) }.map_err(|e| format!("Cannot mmap file: {e}"))?;

    let lines_per_tile = (30.0_f64 / line_rate).ceil() as usize;
    let n_div          = ((size_y as f64) / (lines_per_tile as f64)).ceil() as usize;

    // Parallel: for each division, scan for its tile marker with SIMD memmem.
    let starts_per_div: Vec<Vec<usize>> = (1..=n_div)
        .into_par_iter()
        .map(|i_div| {
            let lit = format!("_{}", i_div - 1);
            let l   = lit.len();
            let mut s: Vec<usize> = memmem::find_iter(&*raw, lit.as_bytes())
                .filter_map(|start| {
                    let marker = start + l + 4;
                    if raw.get(marker) == Some(&4u8) {
                        let data = marker + 3;
                        if data < raw.len() { Some(data) } else { None }
                    } else {
                        None
                    }
                })
                .collect();

            if flag == 0 && !s.is_empty() {
                let first      = s[0];
                let look_start = first.saturating_sub(99);
                let look_end   = (first + 1).min(raw.len());
                if raw[look_start..look_end].windows(3).any(|w| w == b"REF") {
                    if s.len() >= n_ch { s = s[n_ch..].to_vec(); } else { s.clear(); }
                }
            }
            s
        })
        .collect();

    // Keep raw values — no normalization to 8-bit.
    if bitsize > 16 {
        return Err(format!("Unsupported bit depth: {bitsize} (max 16)"));
    }

    // Parallel: each channel fills its own flat row-major buffer independently.
    let channel_data: Vec<Vec<u16>> = (0..n_ch)
        .into_par_iter()
        .map(|ch| {
            let mut data         = vec![0u16; size_y * size_x];
            let mut rows_written = 0usize;

            for j in 0..n_div {
                let lines_this = lines_per_tile.min(size_y - j * lines_per_tile);
                let starts     = &starts_per_div[j];

                if ch >= starts.len() {
                    rows_written += lines_this;
                    continue;
                }

                let p     = starts[ch];
                let b0    = p + 1;
                let b1    = (b0 + 2 * size_x * lines_this).min(raw.len());
                let block = &raw[b0..b1];
                let block = if block.len() % 2 == 1 { &block[..block.len() - 1] } else { block };
                let n_pairs = block.len() / 2;

                for row in 0..lines_this {
                    let dest_row = rows_written + row;
                    if dest_row >= size_y { break; }

                    let src_off = row * size_x;
                    let dst_off = dest_row * size_x;
                    let cols = size_x.min(n_pairs.saturating_sub(src_off));

                    for col in 0..cols {
                        let bi = (src_off + col) * 2;
                        let v  = u16::from_le_bytes([block[bi], block[bi + 1]]);
                        data[dst_off + col] = v;
                    }
                }
                rows_written += lines_this;
            }
            data
        })
        .collect();

    let result = Array3::from_shape_fn((size_y, size_x, n_ch), |(y, x, c)| {
        channel_data[c][y * size_x + x]
    });
    Ok(result)
}

// ──────────────────────────────────────────────────────────────────────
// 5.  read_single_oir
// ──────────────────────────────────────────────────────────────────────
fn read_single_oir(path: &Path) -> Res<(Array3<u16>, OirMeta)> {
    let meta_str = get_meta_block(path)?;
    let meta     = read_meta_block(&meta_str)?;
    let img      = read_oir_frames_tiled(
        path, meta.size_x, meta.size_y, meta.n_channels,
        meta.line_rate, 0, meta.bit_depth,
    )?;
    Ok((img, meta))
}

// ──────────────────────────────────────────────────────────────────────
// 6.  Channel helpers
// ──────────────────────────────────────────────────────────────────────

// Remap user-facing 1-indexed RGB channel to raw BGR storage index.
fn remap_channel(c: usize, n_ch: usize) -> usize {
    if n_ch >= 3 && c <= 3 {
        match c { 1 => 2, 2 => 1, 3 => 0, _ => c.saturating_sub(1) }
    } else {
        c.saturating_sub(1)
    }
}

// Parse a Python channel argument (int, list[int], or None) into Vec<usize>.
fn parse_channel_arg(channel: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<usize>> {
    match channel {
        None => Ok(vec![2]),
        Some(obj) => {
            if let Ok(val) = obj.extract::<i64>() {
                Ok(vec![val as usize])
            } else if let Ok(list) = obj.cast::<PyList>() {
                Ok(list.iter()
                    .filter_map(|item| item.extract::<i64>().ok())
                    .map(|v| v as usize)
                    .collect())
            } else if let Ok(val) = obj.extract::<Vec<i64>>() {
                Ok(val.into_iter().map(|v| v as usize).collect())
            } else {
                Ok(vec![2])
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────
// 7.  read_oir_file  — public Python API (single file)
// ──────────────────────────────────────────────────────────────────────
#[pyfunction]
#[pyo3(signature = (file, channel=None))]
fn read_oir_file<'py>(
    py: Python<'py>,
    file: &str,
    channel: Option<Bound<'py, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let path = Path::new(file);

    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string();
    let group = if let Some(pos) = name.rfind('_') {
        name[..pos].to_string()
    } else {
        name.clone()
    };

    let channels = parse_channel_arg(channel.as_ref())?;
    
    let (full_img, _meta) = py.detach(move || {
        read_single_oir(path)
    }).map_err(PyValueError::new_err)?;
    
    let (h, w, total_ch) = (full_img.shape()[0], full_img.shape()[1], full_img.shape()[2]);

    let ch_indices: Vec<usize> = channels
        .iter()
        .filter(|&&c| c > 0 && c <= total_ch)
        .map(|&c| remap_channel(c, total_ch))
        .collect();

    if ch_indices.is_empty() || ch_indices.len() == 1 {
        let ch_idx = if ch_indices.is_empty() { 1_usize.min(total_ch.saturating_sub(1)) } else { ch_indices[0] };
        let slice = Array2::from_shape_fn((h, w), |(y, x)| full_img[[y, x, ch_idx]]);
        let py_arr = slice.into_pyarray(py).into_any();
        (name, py_arr, group, (h, w)).into_py_any(py)
    } else {
        let n_out = ch_indices.len();
        let slice = Array3::from_shape_fn((h, w, n_out), |(y, x, c)| {
            full_img[[y, x, ch_indices[c]]]
        });
        let py_arr = slice.into_pyarray(py).into_any();
        (name, py_arr, group, (h, w, n_out)).into_py_any(py)
    }
}

// ──────────────────────────────────────────────────────────────────────
// 8.  read_oir_file_batch  — parallel multi-file read via rayon
//
//  All file I/O and pixel decoding runs inside py.detach() so
//  the GIL is fully released and rayon can use every available CPU core.
//  Results are collected as owned Vec<u8> buffers and converted to
//  numpy arrays only after re-acquiring the GIL.
// ──────────────────────────────────────────────────────────────────────
struct RawOirResult {
    name:  String,
    group: String,
    h:     usize,
    w:     usize,
    n_out: usize,   // 1 → 2-D output,  >1 → 3-D output
    data:  Vec<u16>, // row-major: [h*w] or [h*w*n_out]
}

#[pyfunction]
#[pyo3(signature = (files, channel=None))]
fn read_oir_file_batch<'py>(
    py: Python<'py>,
    files: Vec<String>,
    channel: Option<Bound<'py, PyAny>>,
) -> PyResult<Vec<Py<PyAny>>> {
    // Parse channel spec while holding the GIL.
    let channels = parse_channel_arg(channel.as_ref())?;

    // Process all files in parallel with rayon.
    // Rayon's threads are native OS threads, not Python interpreter threads,
    // so they run freely alongside the GIL-holding main thread without conflict.
    // Each file also uses rayon internally for channel decode — nested
    // parallelism is handled gracefully by rayon's work-stealing pool.
    let raw_results: Vec<Res<RawOirResult>> = py.detach(|| files.par_iter().map(|path_str| {
            let path  = Path::new(path_str.as_str());
            let name  = path.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
            let group = if let Some(pos) = name.rfind('_') {
                name[..pos].to_string()
            } else {
                name.clone()
            };

            let (full_img, _meta) = read_single_oir(path)?;
            let (h, w, total_ch) = (full_img.shape()[0], full_img.shape()[1], full_img.shape()[2]);

            let ch_indices: Vec<usize> = channels
                .iter()
                .filter(|&&c| c > 0 && c <= total_ch)
                .map(|&c| remap_channel(c, total_ch))
                .collect();

            if ch_indices.is_empty() || ch_indices.len() == 1 {
                let ch_idx = if ch_indices.is_empty() { 1_usize.min(total_ch.saturating_sub(1)) } else { ch_indices[0] };
                let mut data = Vec::with_capacity(h * w);
                for y in 0..h { for x in 0..w { data.push(full_img[[y, x, ch_idx]]); } }
                Ok(RawOirResult { name, group, h, w, n_out: 1, data })
            } else {
                let n_out = ch_indices.len();
                let mut data = Vec::with_capacity(h * w * n_out);
                for y in 0..h {
                    for x in 0..w {
                        for &ci in &ch_indices { data.push(full_img[[y, x, ci]]); }
                    }
                }
                Ok(RawOirResult { name, group, h, w, n_out, data })
            }
        }).collect());

    // Convert raw buffers → ndarray → numpy, build Python tuples.
    raw_results.into_iter().map(|r: Res<RawOirResult>| {
        let rr = r.map_err(PyValueError::new_err)?;
        if rr.n_out == 1 {
            let arr = Array2::from_shape_vec((rr.h, rr.w), rr.data)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let py_arr = arr.into_pyarray(py).into_any();
            (rr.name, py_arr, rr.group, (rr.h, rr.w)).into_py_any(py)
        } else {
            let arr = Array3::from_shape_vec((rr.h, rr.w, rr.n_out), rr.data)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let py_arr = arr.into_pyarray(py).into_any();
            (rr.name, py_arr, rr.group, (rr.h, rr.w, rr.n_out)).into_py_any(py)
        }
    }).collect()
}

// ──────────────────────────────────────────────────────────────────────
// 9.  calculate_histogram_and_thres  — bincount + reverse cumsum
// ──────────────────────────────────────────────────────────────────────

// Pure computation (no Python token) — shared by single and batch paths.
fn compute_histogram_pure(
    counts: Vec<i64>,
    max_val: usize,
    thres_step: usize,
) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
    let size = max_val + 1;
    let mut cum = vec![0i64; size];
    cum[size - 1] = counts[size - 1];
    for i in (0..size - 1).rev() {
        cum[i] = cum[i + 1] + counts[i];
    }
    let all_thres: Vec<i64> = (0..size).step_by(thres_step).map(|v| v as i64).collect();
    let thres_cnt: Vec<i64> = all_thres.iter().map(|&t| cum[t as usize]).collect();
    (counts, thres_cnt, all_thres)
}

fn finalize_histogram<'py>(
    py: Python<'py>,
    counts: Vec<i64>,
    max_val: usize,
    thres_step: usize,
) -> PyResult<Py<PyAny>> {
    compute_histogram_pure(counts, max_val, thres_step).into_py_any(py)
}

#[pyfunction]
#[pyo3(signature = (img_np, thres_step=5))]
fn calculate_histogram_and_thres<'py>(
    py: Python<'py>,
    img_np: Bound<'py, PyAny>,
    thres_step: usize,
) -> PyResult<Py<PyAny>> {
    // u8 path: max_val is always 255, skip the max scan entirely.
    if let Ok(arr) = img_np.extract::<numpy::PyReadonlyArrayDyn<'_, u8>>() {
        let sl = arr.as_slice()
            .map_err(|_| PyValueError::new_err("Array must be contiguous"))?;
        let mut counts = vec![0i64; 256];
        for &px in sl { counts[px as usize] += 1; }
        return finalize_histogram(py, counts, 255, thres_step);
    }

    // u16 path: determine range from actual max.
    if let Ok(arr) = img_np.extract::<numpy::PyReadonlyArrayDyn<'_, u16>>() {
        let sl = arr.as_slice()
            .map_err(|_| PyValueError::new_err("Array must be contiguous"))?;
        let max_pixel = sl.iter().copied().max().unwrap_or(0) as usize;
        let max_val   = if max_pixel <= 255 { 255 }
                        else if max_pixel <= 4095 { 4095 }
                        else { 65535 };
        let mut counts = vec![0i64; max_val + 1];
        for &px in sl { counts[px as usize] += 1; }
        return finalize_histogram(py, counts, max_val, thres_step);
    }

    Err(PyValueError::new_err("Expected uint8 or uint16 numpy array"))
}

// ──────────────────────────────────────────────────────────────────────
// 10.  calculate_histogram_batch  — parallel histograms via rayon
//
//  Images are copied into owned Vecs while holding the GIL (numpy array
//  views cannot cross the GIL boundary safely), then the GIL is released
//  and rayon processes all images simultaneously.
// ──────────────────────────────────────────────────────────────────────
enum RawImage {
    U8(Vec<u8>),
    U16(Vec<u16>),
}

#[pyfunction]
#[pyo3(signature = (imgs, thres_step=5))]
fn calculate_histogram_batch(
    py: Python<'_>,
    imgs: Vec<Bound<'_, PyAny>>,
    thres_step: usize,
) -> PyResult<Vec<(Vec<i64>, Vec<i64>, Vec<i64>)>> {
    // Extract pixel data into owned Vecs while holding the GIL.
    let raw_images: Vec<RawImage> = imgs.iter().map(|obj| -> PyResult<RawImage> {
        if let Ok(arr) = obj.extract::<numpy::PyReadonlyArrayDyn<u8>>() {
            Ok(RawImage::U8(
                arr.as_slice()
                    .map_err(|_| PyValueError::new_err("u8 array must be contiguous"))?
                    .to_vec(),
            ))
        } else if let Ok(arr) = obj.extract::<numpy::PyReadonlyArrayDyn<u16>>() {
            Ok(RawImage::U16(
                arr.as_slice()
                    .map_err(|_| PyValueError::new_err("u16 array must be contiguous"))?
                    .to_vec(),
            ))
        } else {
            Err(PyValueError::new_err("Expected uint8 or uint16 numpy array"))
        }
    }).collect::<PyResult<Vec<_>>>()?;

    // Compute all histograms in parallel with rayon.
    // Rayon's threads are native OS threads and don't need the GIL.
    Ok(py.detach(|| raw_images.into_par_iter().map(|img| match img {
        RawImage::U8(sl) => {
            let mut counts = vec![0i64; 256];
            for px in sl { counts[px as usize] += 1; }
            compute_histogram_pure(counts, 255, thres_step)
        }
        RawImage::U16(sl) => {
            let max_pixel = sl.iter().copied().max().unwrap_or(0) as usize;
            let max_val = if max_pixel <= 255 { 255 }
                          else if max_pixel <= 4095 { 4095 }
                          else { 65535 };
            let mut counts = vec![0i64; max_val + 1];
            for px in sl { counts[px as usize] += 1; }
            compute_histogram_pure(counts, max_val, thres_step)
        }
    }).collect()))
}

// ──────────────────────────────────────────────────────────────────────
// 11.  compute_group_statistics  — mean & std over dict-of-dict
// ──────────────────────────────────────────────────────────────────────
#[pyfunction]
fn compute_group_statistics<'py>(
    py: Python<'py>,
    cnt_result: Bound<'py, PyDict>,
) -> PyResult<Py<PyAny>> {
    let grouped_stats    = PyDict::new(py);
    let valid_cnt_result = PyDict::new(py);

    for (group_key, items_obj) in cnt_result.iter() {
        let items: &Bound<'_, PyDict> = items_obj.cast()
            .map_err(|_| PyValueError::new_err("Expected dict for group items"))?;

        let mut all_vals: Vec<Vec<f64>> = Vec::new();
        for (_name, value) in items.iter() {
            if value.is_none() { continue; }
            if let Ok(v) = value.extract::<Vec<f64>>() { all_vals.push(v); }
        }
        if all_vals.is_empty() { continue; }

        let n   = all_vals.len() as f64;
        let len = all_vals[0].len();

        let mut mean = vec![0.0f64; len];
        for row in &all_vals {
            for (i, &v) in row.iter().enumerate() { mean[i] += v; }
        }
        for m in &mut mean { *m /= n; }

        let mut variance = vec![0.0f64; len];
        for row in &all_vals {
            for (i, &v) in row.iter().enumerate() {
                let diff = v - mean[i];
                variance[i] += diff * diff;
            }
        }
        let std_dev: Vec<f64> = variance.iter().map(|&v| (v / n).sqrt()).collect();

        let stat_dict = PyDict::new(py);
        stat_dict.set_item("Average",   &mean)?;
        stat_dict.set_item("Std. Dev.", &std_dev)?;
        grouped_stats.set_item(&group_key, stat_dict)?;
        valid_cnt_result.set_item(&group_key, items_obj)?;
    }

    (valid_cnt_result, grouped_stats).into_py_any(py)
}

// ──────────────────────────────────────────────────────────────────────
// 12.  compute_blanked_ratios  — blank subtraction & ratio
// ──────────────────────────────────────────────────────────────────────
#[pyfunction]
fn compute_blanked_ratios<'py>(
    py: Python<'py>,
    valid_cnt_result: Bound<'py, PyDict>,
    grouped_stats: Bound<'py, PyDict>,
    blanked_group_pairs: Bound<'py, PyDict>,
    img_size: (usize, usize),
) -> PyResult<Py<PyAny>> {
    let grouped_blank_stats        = PyDict::new(py);
    let single_blanked_stats       = PyDict::new(py);
    let single_blanked_ratio_stats = PyDict::new(py);
    let total_pixels               = (img_size.0 * img_size.1) as f64;

    for (blank_g_key, target_groups_obj) in blanked_group_pairs.iter() {
        let stat_entry = match grouped_stats.get_item(&blank_g_key)? {
            Some(v) => v,
            None    => continue,
        };
        let stat_dict: &Bound<'_, PyDict> = stat_entry.cast()
            .map_err(|_| PyValueError::new_err("Expected dict for stats"))?;
        let blanking_arr: Vec<f64> = stat_dict
            .get_item("Average")?
            .ok_or_else(|| PyValueError::new_err("Missing 'Average'"))?
            .extract()?;

        let target_groups: Vec<String> = target_groups_obj.extract()?;

        for g in &target_groups {
            let g_stat = match grouped_stats.get_item(g)? {
                Some(v) => v,
                None    => continue,
            };
            let g_dict: &Bound<'_, PyDict> = g_stat.cast()
                .map_err(|_| PyValueError::new_err("Expected dict"))?;
            let g_avg: Vec<f64> = g_dict
                .get_item("Average")?
                .ok_or_else(|| PyValueError::new_err("Missing 'Average'"))?
                .extract()?;

            let blanked: Vec<f64> = g_avg.iter().zip(&blanking_arr).map(|(v, b)| v - b).collect();
            grouped_blank_stats.set_item(g, &blanked)?;

            let group_samples = match valid_cnt_result.get_item(g)? {
                Some(v) => v,
                None    => continue,
            };
            let samples_dict: &Bound<'_, PyDict> = group_samples.cast()
                .map_err(|_| PyValueError::new_err("Expected dict"))?;

            for (sample_key, s_value_obj) in samples_dict.iter() {
                if s_value_obj.is_none() { continue; }
                let s_value: Vec<f64> = s_value_obj.extract()?;
                let blanked_s: Vec<f64> = s_value.iter().zip(&blanking_arr).map(|(s, b)| s - b).collect();
                let ratio_s:   Vec<f64> = s_value.iter().zip(&blanking_arr).map(|(s, b)| (s - b) / total_pixels).collect();
                single_blanked_stats.set_item(&sample_key, &blanked_s)?;
                single_blanked_ratio_stats.set_item(&sample_key, &ratio_s)?;
            }
        }
    }

    (grouped_blank_stats, single_blanked_stats, single_blanked_ratio_stats).into_py_any(py)
}

// ──────────────────────────────────────────────────────────────────────
// Module
// ──────────────────────────────────────────────────────────────────────
#[pymodule]
fn oir_reader_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_oir_file, m)?)?;
    m.add_function(wrap_pyfunction!(read_oir_file_batch, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_histogram_and_thres, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_histogram_batch, m)?)?;
    m.add_function(wrap_pyfunction!(compute_group_statistics, m)?)?;
    m.add_function(wrap_pyfunction!(compute_blanked_ratios, m)?)?;
    Ok(())
}
