//! Fast IMS reader: uses hdf5-pure for metadata/navigation,
//! reads chunk data directly with parallel assembly.

use hdf5_pure::File as H5File;
use hdf5_pure::data_layout::DataLayout;
use hdf5_pure::object_header::ObjectHeader;
use hdf5_pure::message_type::MessageType;
use hdf5_pure::filter_pipeline::FilterPipeline;
use ndarray::{Array4, s};
use rayon::prelude::*;
use std::path::Path;

use crate::ImsMeta;

type Res<T> = Result<T, String>;

#[derive(Debug, Clone)]
struct ChunkLoc {
    file_offset: usize,
    size: usize, // size on disk (may be compressed)
    cz: usize,
    cy: usize,
    cx: usize,
}

struct DatasetInfo {
    chunks: Vec<ChunkLoc>,
    chunk_dims: Vec<usize>, // (cdz, cdy, cdx)
    ds_dims: (usize, usize, usize),
    pipeline: Option<FilterPipeline>,
}

pub fn read_all_channels_fast(
    path: &Path,
    meta: &ImsMeta,
    channels: Option<&[usize]>,
) -> Res<Array4<u16>> {
    let h5 = H5File::open(path)
        .map_err(|e| format!("Cannot parse HDF5: {e}"))?;
    let data = h5.as_bytes();

    let ch_indices: Vec<usize> = match channels {
        Some(chs) => chs.to_vec(),
        None => (0..meta.n_channels).collect(),
    };

    let n_ch = ch_indices.len();
    let (z, y, x) = (meta.z, meta.y, meta.x);
    let mut combined = Array4::<u16>::zeros((n_ch, z, y, x));

    for (ci, &ch) in ch_indices.iter().enumerate() {
        let ds_path = format!("DataSet/ResolutionLevel 0/TimePoint 0/Channel {}/Data", ch);
        let info = extract_dataset_info(&h5, &ds_path)?;

        let (dz, dy, dx) = info.ds_dims;
        let cdz = info.chunk_dims[0];
        let cdy = info.chunk_dims[1];
        let cdx = info.chunk_dims[2];
        let elem_size = 2usize;
        let raw_chunk_bytes = cdz * cdy * cdx * elem_size;
        let has_filters = info.pipeline.is_some();

        let sz = z.min(dz);
        let sy = y.min(dy);
        let sx = x.min(dx);

        let ch_slice = combined.slice_mut(s![ci, .., .., ..]);
        let out_ptr = ch_slice.as_ptr() as usize;

        info.chunks.par_iter().for_each(|chunk| {
            let out_ptr = out_ptr as *mut u16;

            let z_start = chunk.cz * cdz;
            let y_start = chunk.cy * cdy;
            let x_start = chunk.cx * cdx;

            if z_start >= sz || y_start >= sy || x_start >= sx {
                return;
            }

            // Get raw chunk bytes — decompress if needed
            let raw_data: Vec<u8>;
            let chunk_bytes: &[u8] = if has_filters {
                let compressed = &data[chunk.file_offset..chunk.file_offset + chunk.size];
                // Use flate2 for gzip/deflate (most common in IMS)
                raw_data = decompress_chunk(compressed, raw_chunk_bytes);
                &raw_data
            } else {
                &data[chunk.file_offset..chunk.file_offset + chunk.size]
            };

            let z_end = (z_start + cdz).min(sz);
            let y_end = (y_start + cdy).min(sy);
            let x_end = (x_start + cdx).min(sx);

            for iz in 0..(z_end - z_start) {
                for iy in 0..(y_end - y_start) {
                    let src_row_start = (iz * cdy * cdx + iy * cdx) * elem_size;
                    let dst_row_start = (z_start + iz) * sy * sx + (y_start + iy) * sx + x_start;
                    let row_len = x_end - x_start;

                    for ix in 0..row_len {
                        let si = src_row_start + ix * 2;
                        if si + 1 < chunk_bytes.len() {
                            let val = u16::from_le_bytes([chunk_bytes[si], chunk_bytes[si + 1]]);
                            unsafe { *out_ptr.add(dst_row_start + ix) = val; }
                        }
                    }
                }
            }
        });
    }

    Ok(combined)
}

/// Decompress a gzip/deflate compressed chunk
fn decompress_chunk(compressed: &[u8], expected_size: usize) -> Vec<u8> {
    use std::io::Read;
    // Try raw deflate first, then gzip wrapper
    let mut out = vec![0u8; expected_size];

    // Try zlib (deflate with zlib header — most common in HDF5)
    if let Ok(mut decoder) = flate2::read::ZlibDecoder::new(compressed).read_to_end(&mut Vec::new()) {
        let _ = decoder;
    }
    let mut decoder = flate2::read::ZlibDecoder::new(compressed);
    let mut result = Vec::with_capacity(expected_size);
    if decoder.read_to_end(&mut result).is_ok() && result.len() == expected_size {
        return result;
    }

    // Try raw deflate
    let mut decoder = flate2::read::DeflateDecoder::new(compressed);
    let mut result = Vec::with_capacity(expected_size);
    if decoder.read_to_end(&mut result).is_ok() && result.len() == expected_size {
        return result;
    }

    // Try gzip
    let mut decoder = flate2::read::GzDecoder::new(compressed);
    let mut result = Vec::with_capacity(expected_size);
    if decoder.read_to_end(&mut result).is_ok() && result.len() == expected_size {
        return result;
    }

    // Fallback: return zeros
    eprintln!("Warning: failed to decompress chunk ({} -> {} bytes)", compressed.len(), expected_size);
    out
}

fn extract_dataset_info(file: &H5File, ds_path: &str) -> Res<DatasetInfo> {
    let data = file.as_bytes();
    let sb = file.superblock();
    let base = sb.base_address;

    let addr = hdf5_pure::group_v2::resolve_path_any(data, sb, ds_path)
        .map_err(|e| format!("Cannot resolve {ds_path}: {e}"))?;

    let hdr = ObjectHeader::parse_with_base(
        data, addr as usize, sb.offset_size, sb.length_size, base,
    ).map_err(|e| format!("Cannot parse header: {e}"))?;

    // Get filter pipeline if present
    let pipeline = hdr.messages.iter()
        .find(|m| m.msg_type == MessageType::FilterPipeline)
        .and_then(|msg| FilterPipeline::parse(&msg.data).ok());

    // Get dataspace for dimensions
    let ds_msg = hdr.messages.iter()
        .find(|m| m.msg_type == MessageType::Dataspace)
        .ok_or("No Dataspace message")?;
    let dataspace = hdf5_pure::dataspace::Dataspace::parse(&ds_msg.data, sb.length_size)
        .map_err(|e| format!("Cannot parse dataspace: {e}"))?;

    let layout_msg = hdr.messages.iter()
        .find(|m| m.msg_type == MessageType::DataLayout)
        .ok_or("No DataLayout message")?;

    let layout = DataLayout::parse(
        &layout_msg.data, sb.offset_size, sb.length_size,
    ).map_err(|e| format!("Cannot parse layout: {e}"))?;

    let ds_dims = (
        dataspace.dimensions[0] as usize,
        dataspace.dimensions[1] as usize,
        dataspace.dimensions[2] as usize,
    );

    match layout {
        DataLayout::Chunked {
            chunk_dimensions,
            btree_address,
            ..
        } => {
            let btree_addr = btree_address.ok_or("No btree address")?;
            let ndims = chunk_dimensions.len();
            let rank = ndims - 1;
            let chunk_dims: Vec<usize> = chunk_dimensions[..rank]
                .iter().map(|&d| d as usize).collect();

            let raw_chunks = hdf5_pure::chunked_read::collect_chunk_info(
                data, btree_addr + base, ndims, sb.offset_size, sb.length_size,
            ).map_err(|e| format!("Cannot collect chunks: {e}"))?;

            let mut locs = Vec::with_capacity(raw_chunks.len());
            for ci in &raw_chunks {
                let offset = (ci.address + base) as usize;
                let cz = ci.offsets[0] as usize / chunk_dims[0];
                let cy = ci.offsets[1] as usize / chunk_dims[1];
                let cx = ci.offsets[2] as usize / chunk_dims[2];

                locs.push(ChunkLoc {
                    file_offset: offset,
                    size: ci.chunk_size as usize,
                    cz, cy, cx,
                });
            }

            Ok(DatasetInfo {
                chunks: locs,
                chunk_dims,
                ds_dims,
                pipeline,
            })
        }
        DataLayout::Contiguous { address, size } => {
            let addr = address.ok_or("No contiguous address")? + base;
            Ok(DatasetInfo {
                chunks: vec![ChunkLoc {
                    file_offset: addr as usize,
                    size: size as usize,
                    cz: 0, cy: 0, cx: 0,
                }],
                chunk_dims: vec![ds_dims.0, ds_dims.1, ds_dims.2],
                ds_dims,
                pipeline: None,
            })
        }
        _ => Err("Unsupported data layout".into()),
    }
}
