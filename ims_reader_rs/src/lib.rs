mod fast_read;

use hdf5_pure::{AttrValue, File as H5File};
use ndarray::{Array2, Array3, Array4, s};
use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::IntoPyObjectExt;
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;

type Res<T> = Result<T, String>;

// ──────────────────────────────────────────────────────────────────────
// 1.  Metadata
// ──────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct ImsMeta {
    x: usize,
    y: usize,
    z: usize,
    n_channels: usize,
    pixel_size_x: f64,
    pixel_size_y: f64,
    pixel_size_z: f64,
    ext_min: [f64; 3],
    ext_max: [f64; 3],
    recording_date: String,
}

fn attr_str(attrs: &HashMap<String, AttrValue>, name: &str) -> Res<String> {
    match attrs.get(name) {
        Some(AttrValue::String(s)) => Ok(s.clone()),
        Some(AttrValue::StringArray(v)) => Ok(v.join("")),
        Some(AttrValue::AsciiString(s)) => Ok(s.clone()),
        Some(AttrValue::AsciiStringArray(v)) => Ok(v.join("")),
        Some(AttrValue::I64(v)) => Ok(v.to_string()),
        Some(AttrValue::F64(v)) => Ok(v.to_string()),
        Some(other) => Ok(format!("{:?}", other)),
        None => Err(format!("Missing attribute '{name}'")),
    }
}

fn attr_f64(attrs: &HashMap<String, AttrValue>, name: &str) -> Res<f64> {
    let s = attr_str(attrs, name)?;
    s.trim().parse::<f64>()
        .map_err(|e| format!("Cannot parse '{name}' as f64: {e}"))
}

fn attr_usize(attrs: &HashMap<String, AttrValue>, name: &str) -> Res<usize> {
    let s = attr_str(attrs, name)?;
    s.trim().parse::<usize>()
        .map_err(|e| format!("Cannot parse '{name}' as usize: {e}"))
}

fn read_metadata(file: &H5File) -> Res<ImsMeta> {
    let img_group = file.group("DataSetInfo/Image")
        .map_err(|e| format!("Cannot open DataSetInfo/Image: {e}"))?;
    let attrs = img_group.attrs()
        .map_err(|e| format!("Cannot read attrs: {e}"))?;

    let x = attr_usize(&attrs, "X")?;
    let y = attr_usize(&attrs, "Y")?;
    let z = attr_usize(&attrs, "Z")?;

    let ext_min0 = attr_f64(&attrs, "ExtMin0")?;
    let ext_max0 = attr_f64(&attrs, "ExtMax0")?;
    let ext_min1 = attr_f64(&attrs, "ExtMin1")?;
    let ext_max1 = attr_f64(&attrs, "ExtMax1")?;
    let ext_min2 = attr_f64(&attrs, "ExtMin2")?;
    let ext_max2 = attr_f64(&attrs, "ExtMax2")?;

    let pixel_size_x = (ext_max0 - ext_min0) / x as f64;
    let pixel_size_y = (ext_max1 - ext_min1) / y as f64;
    let pixel_size_z = (ext_max2 - ext_min2) / z as f64;

    let recording_date = attr_str(&attrs, "RecordingDate").unwrap_or_default();

    // Count channels
    let tp = file.group("DataSet/ResolutionLevel 0/TimePoint 0")
        .map_err(|e| format!("Cannot open TimePoint 0: {e}"))?;
    let channel_names = tp.groups()
        .map_err(|e| format!("Cannot list channels: {e}"))?;
    let n_channels = channel_names.len();

    Ok(ImsMeta {
        x, y, z, n_channels,
        pixel_size_x, pixel_size_y, pixel_size_z,
        ext_min: [ext_min0, ext_min1, ext_min2],
        ext_max: [ext_max0, ext_max1, ext_max2],
        recording_date,
    })
}

// ──────────────────────────────────────────────────────────────────────
// 2.  Channel reading
// ──────────────────────────────────────────────────────────────────────

fn read_channel(file: &H5File, ch: usize, meta: &ImsMeta) -> Res<Array3<u16>> {
    let ds_path = format!("DataSet/ResolutionLevel 0/TimePoint 0/Channel {}/Data", ch);
    let ds = file.dataset(&ds_path)
        .map_err(|e| format!("Cannot open {ds_path}: {e}"))?;

    let shape = ds.shape()
        .map_err(|e| format!("Cannot get shape: {e}"))?;
    if shape.len() != 3 {
        return Err(format!("Expected 3D dataset, got {}D", shape.len()));
    }
    let (dz, dy, dx) = (shape[0] as usize, shape[1] as usize, shape[2] as usize);

    // Read raw bytes and reinterpret as u16 (zero-copy on LE)
    let raw_bytes = ds.read_raw_bytes()
        .map_err(|e| format!("Cannot read {ds_path}: {e}"))?;

    let n_elements = dz * dy * dx;
    // Safety: raw_bytes is LE uint16 data, we reinterpret the bytes directly
    let raw: Vec<u16> = if cfg!(target_endian = "little") {
        // On LE: direct pointer cast, no per-element conversion
        let mut v = std::mem::ManuallyDrop::new(raw_bytes);
        let ptr = v.as_mut_ptr() as *mut u16;
        let len = n_elements;
        let cap = v.capacity() / 2;
        unsafe { Vec::from_raw_parts(ptr, len, cap) }
    } else {
        let src = &raw_bytes[..n_elements * 2];
        (0..n_elements).map(|i| u16::from_le_bytes([src[i*2], src[i*2+1]])).collect()
    };

    let full = Array3::from_shape_vec((dz, dy, dx), raw)
        .map_err(|e| format!("Shape mismatch: {e}"))?;

    let z = meta.z.min(dz);
    let y = meta.y.min(dy);
    let x = meta.x.min(dx);

    Ok(full.slice(s![..z, ..y, ..x]).to_owned())
}

fn read_all_channels(path: &Path, channels: Option<&[usize]>) -> Res<(Array4<u16>, ImsMeta)> {
    // Read metadata
    let file = H5File::open(path)
        .map_err(|e| format!("Cannot open {}: {e}", path.display()))?;
    let meta = read_metadata(&file)?;
    drop(file);

    // Fast mmap + parallel chunk assembly (single file open, all channels)
    let combined = fast_read::read_all_channels_fast(path, &meta, channels)?;
    Ok((combined, meta))
}

// ──────────────────────────────────────────────────────────────────────
// 3.  MIP — parallel over rows
// ──────────────────────────────────────────────────────────────────────

fn mip_channel(vol: &ndarray::ArrayView3<u16>) -> Array2<u16> {
    let (nz, ny, nx) = vol.dim();
    let mut out = Array2::<u16>::zeros((ny, nx));

    out.outer_iter_mut()
        .into_par_iter()
        .enumerate()
        .for_each(|(yi, mut row)| {
            for xi in 0..nx {
                let mut mx = 0u16;
                for zi in 0..nz {
                    let v = vol[(zi, yi, xi)];
                    if v > mx { mx = v; }
                }
                row[xi] = mx;
            }
        });

    out
}

// ──────────────────────────────────────────────────────────────────────
// 4.  Python API
// ──────────────────────────────────────────────────────────────────────

#[pyfunction]
#[pyo3(signature = (path, channel=None))]
fn read_ims<'py>(py: Python<'py>, path: &str, channel: Option<usize>) -> PyResult<Py<PyAny>> {
    let p = Path::new(path);
    let channels_arg = channel.map(|c| vec![c]);
    let ch_slice = channels_arg.as_deref();

    let (data, meta) = py
        .detach(move || read_all_channels(p, ch_slice))
        .map_err(PyValueError::new_err)?;

    let meta_dict = meta_to_dict(py, &meta)?;

    if channel.is_some() {
        let squeezed = data.index_axis(ndarray::Axis(0), 0).to_owned();
        (squeezed.into_pyarray(py), meta_dict)
            .into_py_any(py).map_err(PyValueError::new_err)
    } else {
        (data.into_pyarray(py), meta_dict)
            .into_py_any(py).map_err(PyValueError::new_err)
    }
}

#[pyfunction]
#[pyo3(signature = (path, channel=None))]
fn read_ims_mip<'py>(py: Python<'py>, path: &str, channel: Option<usize>) -> PyResult<Py<PyAny>> {
    let p = Path::new(path);
    let channels_arg = channel.map(|c| vec![c]);
    let ch_slice = channels_arg.as_deref();

    let (data, meta) = py
        .detach(move || read_all_channels(p, ch_slice))
        .map_err(PyValueError::new_err)?;

    let meta_dict = meta_to_dict(py, &meta)?;
    let n_ch = data.dim().0;

    if n_ch == 1 || channel.is_some() {
        let vol = data.index_axis(ndarray::Axis(0), 0);
        let mip = mip_channel(&vol);
        (mip.into_pyarray(py), meta_dict)
            .into_py_any(py).map_err(PyValueError::new_err)
    } else {
        let mips: Vec<Array2<u16>> = (0..n_ch)
            .into_par_iter()
            .map(|c| mip_channel(&data.index_axis(ndarray::Axis(0), c)))
            .collect();
        let (y, x) = (meta.y, meta.x);
        let mut combined = Array3::<u16>::zeros((n_ch, y, x));
        for (i, m) in mips.into_iter().enumerate() {
            combined.slice_mut(s![i, .., ..]).assign(&m);
        }
        (combined.into_pyarray(py), meta_dict)
            .into_py_any(py).map_err(PyValueError::new_err)
    }
}

#[pyfunction]
#[pyo3(signature = (paths, channel=None, mip=false))]
fn read_ims_batch<'py>(
    py: Python<'py>,
    paths: Vec<String>,
    channel: Option<usize>,
    mip: bool,
) -> PyResult<Vec<Py<PyAny>>> {
    let channels_arg = channel.map(|c| vec![c]);

    let results: Vec<Res<(String, Array4<u16>, ImsMeta)>> = py.detach(|| {
        paths.iter().map(|path_str| {
            let p = Path::new(path_str);
            let name = p.file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();
            let ch_slice = channels_arg.as_deref();
            let (data, meta) = read_all_channels(p, ch_slice)?;
            Ok((name, data, meta))
        }).collect()
    });

    let mut out = Vec::with_capacity(results.len());
    for result in results {
        match result {
            Ok((name, data, meta)) => {
                let meta_dict = meta_to_dict(py, &meta)?;
                if mip {
                    let n_ch = data.dim().0;
                    if n_ch == 1 || channel.is_some() {
                        let vol = data.index_axis(ndarray::Axis(0), 0);
                        let m = mip_channel(&vol);
                        let item = (&name, m.into_pyarray(py), meta_dict)
                            .into_py_any(py).map_err(PyValueError::new_err)?;
                        out.push(item);
                    } else {
                        let mips: Vec<Array2<u16>> = (0..n_ch)
                            .into_par_iter()
                            .map(|c| mip_channel(&data.index_axis(ndarray::Axis(0), c)))
                            .collect();
                        let (y, x) = (meta.y, meta.x);
                        let mut combined = Array3::<u16>::zeros((n_ch, y, x));
                        for (i, m) in mips.into_iter().enumerate() {
                            combined.slice_mut(s![i, .., ..]).assign(&m);
                        }
                        let item = (&name, combined.into_pyarray(py), meta_dict)
                            .into_py_any(py).map_err(PyValueError::new_err)?;
                        out.push(item);
                    }
                } else if channel.is_some() {
                    let sq = data.index_axis(ndarray::Axis(0), 0).to_owned();
                    let item = (&name, sq.into_pyarray(py), meta_dict)
                        .into_py_any(py).map_err(PyValueError::new_err)?;
                    out.push(item);
                } else {
                    let item = (&name, data.into_pyarray(py), meta_dict)
                        .into_py_any(py).map_err(PyValueError::new_err)?;
                    out.push(item);
                }
            }
            Err(e) => eprintln!("Warning: {e}"),
        }
    }
    Ok(out)
}

#[pyfunction]
fn read_ims_meta(py: Python<'_>, path: &str) -> PyResult<Py<PyAny>> {
    let p = Path::new(path);
    let file = H5File::open(p)
        .map_err(|e| PyValueError::new_err(format!("Cannot open {path}: {e}")))?;
    let meta = read_metadata(&file).map_err(PyValueError::new_err)?;
    meta_to_dict(py, &meta)?
        .into_py_any(py).map_err(PyValueError::new_err)
}

// ──────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────

fn meta_to_dict(py: Python<'_>, meta: &ImsMeta) -> PyResult<HashMap<String, Py<PyAny>>> {
    let mut d: HashMap<String, Py<PyAny>> = HashMap::new();
    d.insert("x".into(), meta.x.into_py_any(py)?);
    d.insert("y".into(), meta.y.into_py_any(py)?);
    d.insert("z".into(), meta.z.into_py_any(py)?);
    d.insert("n_channels".into(), meta.n_channels.into_py_any(py)?);
    d.insert("pixel_size_x".into(), meta.pixel_size_x.into_py_any(py)?);
    d.insert("pixel_size_y".into(), meta.pixel_size_y.into_py_any(py)?);
    d.insert("pixel_size_z".into(), meta.pixel_size_z.into_py_any(py)?);
    d.insert("ext_min".into(), meta.ext_min.to_vec().into_py_any(py)?);
    d.insert("ext_max".into(), meta.ext_max.to_vec().into_py_any(py)?);
    d.insert("recording_date".into(), meta.recording_date.clone().into_py_any(py)?);
    Ok(d)
}

#[pymodule]
fn ims_reader_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_ims, m)?)?;
    m.add_function(wrap_pyfunction!(read_ims_mip, m)?)?;
    m.add_function(wrap_pyfunction!(read_ims_batch, m)?)?;
    m.add_function(wrap_pyfunction!(read_ims_meta, m)?)?;
    Ok(())
}
