//! TRELLIS.2 sparse tensor primitives: coordinate hashmap, submanifold 3x3x3
//! sparse convolution, C2S/S2C coordinate ops and exact sparse full attention
//! helpers, shared by the shape SLAT flow DiT (T4) and later sparse VAEs (T5+).
//!
//! Reference: `trellis2/modules/sparse/basic.py` (`SparseTensor`), and
//! `trellis2/modules/sparse/attention/full_attn.py`
//! (`sparse_scaled_dot_product_attention`).
//!
//! Sparse tensor model (single batch element, `num_samples = 1`):
//! - `SparseFeats { coords: Vec<[i32; 3]>, feats: Tensor }` with row `i` of
//!   `feats` (`[N, C]`, F32) corresponding to `coords[i] = (x, y, z)`.
//! - Coordinates are non-negative, below the grid resolution and unique.
//! - Row order is preserved by every op here except `spatial_to_channel`,
//!   which sorts parents row-major exactly like the reference
//!   `SparseSpatial2Channel` (`code.unique()` sorts ascending).
//!
//! Sparse convolution (locked decision 7): submanifold 3x3x3, stride 1,
//! padding 1, output coords == input coords:
//! ```text
//! out[i, o] = bias[o] + sum_{kd,kh,kw} sum_c W[o,kd,kh,kw,c] * x[coord_i + (kd-1,kh-1,kw-1), c]
//! ```
//! with missing neighbors contributing zero. The canonical weights store the
//! kernel in the flex_gemm layout `[out, kd, kh, kw, in]` (NOT the PyTorch
//! `[out, in, kd, kh, kw]`); `SparseConv3d::load` transposes it into
//! `[27, Cout, Cin]` (`k = kd*9 + kh*3 + kw`, kw fastest, matching the
//! in-memory flattening of `(kd, kh, kw)`) and the forward is 27
//! hashmap-gather + dense GEMM pairs accumulated in F32.

use candle_core::{DType, Device, Result as CandleResult, Tensor};
use rustc_hash::FxHashMap;

use super::attention_math::exact_query_chunked_sdpa;
use crate::weights::VarBuilder;

/// A single-batch sparse feature tensor: row `i` of `feats` belongs to
/// `coords[i]`.
#[derive(Debug, Clone)]
pub(crate) struct SparseFeats {
    /// `(x, y, z)` integer latent coordinates, batch dim dropped.
    pub coords: Vec<[i32; 3]>,
    /// `[N, C]` F32 features.
    pub feats: Tensor,
}

impl SparseFeats {
    /// Crate-internal constructor; consumed by the shape VAE decoder (T5).
    #[allow(dead_code)]
    pub fn new(coords: Vec<[i32; 3]>, feats: Tensor) -> CandleResult<Self> {
        let rows = feats.dim(0)?;
        if rows != coords.len() {
            candle_core::bail!(
                "SparseFeats coords/features mismatch: {} coords vs {} feature rows",
                coords.len(),
                rows
            );
        }
        if feats.dtype() != DType::F32 {
            candle_core::bail!("SparseFeats feats must be F32, got {:?}", feats.dtype());
        }
        Ok(Self { coords, feats })
    }
}

/// Coordinate hashmap `coord -> row index`, built once per grid and reused
/// across the 27 conv offsets and mesh extraction.
#[derive(Debug, Clone)]
pub struct CoordMap {
    map: FxHashMap<[i32; 3], u32>,
}

impl CoordMap {
    /// Build from unique coordinates; later duplicates overwrite earlier
    /// entries (inputs are required to be duplicate-free).
    pub fn new(coords: &[[i32; 3]]) -> Self {
        let mut map = FxHashMap::default();
        map.reserve(coords.len());
        for (index, coord) in coords.iter().enumerate() {
            map.insert(*coord, index as u32);
        }
        Self { map }
    }

    pub fn get(&self, coord: [i32; 3]) -> Option<u32> {
        self.map.get(&coord).copied()
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

/// Kernel offset `(dx, dy, dz)` for flat index `k = kd*9 + kh*3 + kw`
/// (kw fastest), i.e. `delta = (kd-1, kh-1, kw-1)` for a 3^3 kernel with
/// padding 1.
pub fn kernel_offset_delta(k: usize) -> [i32; 3] {
    debug_assert!(k < 27);
    [
        (k / 9) as i32 - 1,
        ((k % 9) / 3) as i32 - 1,
        (k % 3) as i32 - 1,
    ]
}

/// Submanifold 3x3x3 sparse convolution (`SparseConv3d`): output coords equal
/// input coords; the kernel is stored per offset as contiguous `[Cin, Cout]`
/// (the flex_gemm `[Cout, kd, kh, kw, Cin]` layout transposed by the loader).
#[derive(Debug, Clone)]
pub struct SparseConv3d {
    /// Per offset `[Cin, Cout]`, index `k = kd*9 + kh*3 + kw` (kw fastest),
    /// each a standalone contiguous buffer with start offset 0.
    kernels: Vec<Tensor>,
    bias: Option<Tensor>,
    in_channels: usize,
    out_channels: usize,
}

/// Precomputed gather/scatter rows for the 27 submanifold offsets of one
/// fixed coordinate set.
///
/// A submanifold convolution keeps the coordinate set unchanged, so every
/// `ConvNeXt` block in a VAE stage performs the same 27 hashmap sweeps and
/// uploads the same index arrays. The sparse VAE decoder previously rebuilt
/// them per block (36 convolutions, 9 distinct coordinate sets); caching one
/// instance per stage removes most of the host-side hashmap work that
/// dominated the shape-VAE stage on every backend.
/// Per-offset source rows for a submanifold sparse convolution. The output
/// row set equals `coords`, so every offset can be expressed as a dense
/// `[N]` gather table: row `i` points at the voxel holding the neighbor at
/// `coords[i] + delta`, or at a zero sentinel row when the neighbor is
/// absent. Dense tables let the convolution accumulate with plain adds;
/// scatter-add would issue `27 * N * Cout` atomics (17.5 s for the 2M-row
/// stage on CUDA) while gathers stream at memory bandwidth.
#[derive(Debug)]
enum OffsetSource {
    /// `delta == 0`: the source row is the output row.
    Identity,
    /// `[N]` U32 gather rows; `N` (the appended zero row) marks an absent
    /// neighbor.
    Dense(Tensor),
    /// No output row has a neighbor at this offset.
    Empty,
}

#[derive(Debug)]
pub struct SparseConvIndices {
    rows: usize,
    /// True when at least one output row lacks a neighbor at some offset, so
    /// gathers need the appended zero sentinel row.
    has_missing: bool,
    /// Indexed by kernel offset (`kernel_offset_delta`).
    offsets: Vec<OffsetSource>,
}

impl SparseConvIndices {
    pub fn new(coords: &[[i32; 3]], device: &Device) -> CandleResult<Self> {
        let profile = std::env::var_os("LUX3D_STAGE_TIME").is_some();
        let mark = std::time::Instant::now();
        let map = CoordMap::new(coords);
        let voxels = coords.len();
        let sentinel = voxels as u32;
        // The 26 non-identity offsets are independent read-only sweeps over
        // the same coordinate map. At the 2M-voxel stage the combined sweeps
        // cost seconds on a single thread, so build them on scoped threads.
        let rows: Vec<(Vec<u32>, bool, bool)> = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(27);
            for k in 0..27 {
                let map = &map;
                handles.push(scope.spawn(move || {
                    if k == 13 {
                        return (Vec::new(), true, false);
                    }
                    let delta = kernel_offset_delta(k);
                    let mut source: Vec<u32> = vec![sentinel; voxels];
                    let mut any = false;
                    let mut missing = false;
                    for (index, coord) in coords.iter().enumerate() {
                        if let Some(found) = map.get([
                            coord[0] + delta[0],
                            coord[1] + delta[1],
                            coord[2] + delta[2],
                        ]) {
                            source[index] = found;
                            any = true;
                        } else {
                            missing = true;
                        }
                    }
                    (source, any, missing)
                }));
            }
            handles
                .into_iter()
                .map(|handle| handle.join().expect("sparse index worker panicked"))
                .collect()
        });
        let mut offsets = Vec::with_capacity(27);
        let mut has_missing = false;
        for (source, any, missing) in rows {
            has_missing |= missing;
            offsets.push(if !any {
                OffsetSource::Empty
            } else if source.is_empty() {
                OffsetSource::Identity
            } else {
                OffsetSource::Dense(Tensor::from_vec(source, (voxels,), device)?)
            });
        }
        if profile {
            eprintln!(
                "sparse conv indices: {:.2}s ({} voxels)",
                mark.elapsed().as_secs_f64(),
                coords.len()
            );
        }
        Ok(Self {
            rows: coords.len(),
            has_missing,
            offsets,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
}

impl SparseConv3d {
    /// Test-only view of the per-offset kernels (`[Cin, Cout]`, index
    /// `k = kd*9 + kh*3 + kw`). Hidden from docs; the integration tests use it
    /// to pin the flex_gemm loader transpose.
    #[doc(hidden)]
    pub fn kernels_for_test(&self) -> &[Tensor] {
        &self.kernels
    }

    /// Load from a VarBuilder rooted at the conv tensors. The checkpoint
    /// stores `weight` as `[Cout, kd, kh, kw, Cin]` (flex_gemm layout); it is
    /// transposed into `[27, Cout, Cin]` in F32.
    pub fn load(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        bias: bool,
    ) -> CandleResult<Self> {
        // GGUF packages store the rank-5 kernel flattened (candle's GGUF
        // reader caps tensor rank at 4); the helper reshapes it back.
        let profile = std::env::var_os("LUX3D_STAGE_TIME").is_some();
        let device = vb.device().clone();
        // The checkpoint stores the decoder convolutions in FP16 and the
        // reference runs them in FP16. Keeping the kernels F16 halves their
        // VRAM (~0.9 GiB for the shape VAE) and lets the gathers/GEMMs halve
        // their bandwidth; CPU keeps F32 (candle's CPU f16 matmul is a
        // fallback path).
        let kernel_dtype = if matches!(&device, Device::Cpu) {
            DType::F32
        } else {
            DType::F16
        };
        // Build the 27 `[Cin, Cout]` offset kernels on the host. Doing the
        // transpose on the GPU was the dominant Vulkan load cost (10.2 s for
        // the 432 MB stage-0 conv: candle's generic strided-copy shader runs
        // at ~43 MB/s). A parallel CPU transpose plus one upload per kernel is
        // ~10x faster and the uploaded buffers are already contiguous, which
        // is also what the GEMM wants.
        let mark = std::time::Instant::now();
        let cpu_vb = vb.clone().set_device(Device::Cpu);
        let weight = super::nn_blocks::get_weight_shaped(
            &cpu_vb,
            (out_channels, 3, 3, 3, in_channels),
            "weight",
        )?;
        if profile {
            eprintln!(
                "[stage] sparse conv load get [{in_channels}->{out_channels}]: {:.2}s",
                mark.elapsed().as_secs_f64()
            );
        }
        let mark = std::time::Instant::now();
        let weight27 = weight
            .to_dtype(kernel_dtype)?
            .reshape((out_channels, 27, in_channels))?;
        let kernels_cpu: Vec<Tensor> = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(27);
            for k in 0..27 {
                let weight27 = &weight27;
                handles.push(scope.spawn(move || -> CandleResult<Tensor> {
                    weight27
                        .narrow(1, k, 1)?
                        .squeeze(1)?
                        .transpose(0, 1)?
                        .contiguous()
                }));
            }
            handles
                .into_iter()
                .map(|handle| handle.join().expect("sparse kernel split panicked"))
                .collect::<CandleResult<Vec<_>>>()
        })?;
        if profile {
            eprintln!(
                "[stage] sparse conv load split [{in_channels}->{out_channels}]: {:.2}s",
                mark.elapsed().as_secs_f64()
            );
        }
        let mark = std::time::Instant::now();
        let kernels = kernels_cpu
            .into_iter()
            .map(|kernel| kernel.to_device(&device))
            .collect::<CandleResult<Vec<_>>>()?;
        if profile {
            eprintln!(
                "[stage] sparse conv load upload [{in_channels}->{out_channels}]: {:.2}s",
                mark.elapsed().as_secs_f64()
            );
        }
        let bias = if bias {
            Some(vb.get(out_channels, "bias")?.to_dtype(DType::F32)?)
        } else {
            None
        };
        Ok(Self {
            kernels,
            bias,
            in_channels,
            out_channels,
        })
    }

    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    /// `coords: [N]`, `feats: [N, Cin]` -> `[N, Cout]` F32. 27 gather + GEMM
    /// pairs accumulated in F32; missing neighbors contribute zero.
    pub fn forward(&self, coords: &[[i32; 3]], feats: &Tensor) -> CandleResult<Tensor> {
        let indices = SparseConvIndices::new(coords, feats.device())?;
        self.forward_with_indices(feats, &indices)
    }

    /// Same as [`Self::forward`] with precomputed neighbor rows (reuse them
    /// across every convolution over the same coordinates).
    pub fn forward_with_indices(
        &self,
        feats: &Tensor,
        indices: &SparseConvIndices,
    ) -> CandleResult<Tensor> {
        let (rows, channels) = feats.dims2()?;
        if rows != indices.rows || channels != self.in_channels {
            candle_core::bail!(
                "SparseConv3d expects [N, {}] with N = {}, got [{rows}, {channels}]",
                self.in_channels,
                indices.rows
            );
        }
        let device = feats.device();
        let profile = std::env::var_os("LUX3D_SPARSE_CONV_PROFILE").is_some();
        if profile {
            device.synchronize()?;
        }
        let mark = std::time::Instant::now();
        if rows == 0 {
            return Tensor::zeros((0, self.out_channels), DType::F32, device);
        }
        let kernel_dtype = self.kernels[0].dtype();
        let feats_in = if feats.dtype() == kernel_dtype {
            feats.clone()
        } else {
            feats.to_dtype(kernel_dtype)?
        };
        // Row `rows` of the padded tensor is all-zero, so gather tables can
        // use it for missing neighbors without a per-row mask. Interior
        // stages have no missing neighbors, and then the zero row/copy is
        // skipped entirely.
        let padded = if indices.has_missing {
            Some(Tensor::cat(
                &[
                    feats_in.clone(),
                    Tensor::zeros((1, channels), kernel_dtype, device)?,
                ],
                0,
            )?)
        } else {
            None
        };
        // Gather + GEMM per offset, accumulated with plain adds. Scatter add
        // would issue `27 * N * Cout` atomics (17.5 s for the 2M-row stage on
        // CUDA); these gathers stream at memory bandwidth.
        let mut acc: Option<Tensor> = None;
        for (k, source) in indices.offsets.iter().enumerate() {
            let input = match source {
                OffsetSource::Empty => continue,
                OffsetSource::Identity => feats_in.narrow(0, 0, rows)?,
                OffsetSource::Dense(src) => padded
                    .as_ref()
                    .expect("dense gather with missing sentinel row")
                    .index_select(src, 0)?,
            };
            let product = input.matmul(&self.kernels[k])?;
            let product = if product.dtype() == DType::F32 {
                product
            } else {
                product.to_dtype(DType::F32)?
            };
            acc = Some(match acc {
                None => product,
                Some(total) => total.add(&product)?,
            });
        }
        let mut out = match acc {
            Some(out) => out,
            None => Tensor::zeros((rows, self.out_channels), DType::F32, device)?,
        };
        if let Some(bias) = &self.bias {
            out = out.broadcast_add(bias)?;
        }
        drop(padded);
        if profile {
            device.synchronize()?;
            eprintln!(
                "sparse conv [{rows}, {channels}] -> {}: {:.3}s",
                self.out_channels,
                mark.elapsed().as_secs_f64()
            );
        }
        Ok(out)
    }
}

/// Binarized subdivision: `leaf = sub > 0` per parent (`[N, 8]` logits ->
/// `[N, 8]` bools in row-major subidx order).
fn subdivision_leaves(subdiv_logits: &Tensor, parents: usize) -> CandleResult<Vec<[bool; 8]>> {
    let (rows, channels) = subdiv_logits.dims2()?;
    if rows != parents || channels != 8 {
        candle_core::bail!("subdivision logits must be [{parents}, 8], got [{rows}, {channels}]");
    }
    let values = subdiv_logits
        .to_device(&candle_core::Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    Ok(values
        .chunks_exact(8)
        .map(|chunk| {
            let mut leaf = [false; 8];
            for (i, &v) in chunk.iter().enumerate() {
                leaf[i] = v > 0.0;
            }
            leaf
        })
        .collect())
}

/// Output of `channel_to_spatial`.
#[derive(Debug, Clone)]
pub struct Channel2SpatialOutput {
    /// Child coords `parent * 2 + (dx, dy, dz)`, grouped by parent in parent
    /// order, children ascending in `subidx`.
    pub coords: Vec<[i32; 3]>,
    /// `[M, child_channels]` selected features.
    pub feats: Tensor,
    /// Binarized subdivision per parent, `[N, 8]` F32 (row-major subidx).
    pub sub_binary: Tensor,
}

/// C2S (`SparseChannel2Spatial`, factor 2): subdivision logits `[N, 8]`
/// (`leaf = logits > 0`) select, per parent, which children survive.
///
/// `feats: [N, C]` with `C` a multiple of 8, channel layout
/// `[child, channel]`: reshaped to `[N*8, C/8]`, output row `j` is
/// `feats[parent*8 + subidx]` with `subidx = dx + 2*dy + 4*dz`. Children are
/// emitted grouped by parent in parent order, ascending `subidx` (row-major
/// nonzero order, exactly `sub.nonzero()[:, -1]` in the reference).
pub fn channel_to_spatial(
    coords: &[[i32; 3]],
    feats: &Tensor,
    subdiv_logits: &Tensor,
) -> CandleResult<Channel2SpatialOutput> {
    let (rows, channels) = feats.dims2()?;
    if rows != coords.len() {
        candle_core::bail!(
            "channel_to_spatial coords/features mismatch: {} coords vs {rows} rows",
            coords.len()
        );
    }
    if channels % 8 != 0 {
        candle_core::bail!(
            "channel_to_spatial expects channels divisible by 8 (layout [child, channel]), got {channels}"
        );
    }
    let leaves = subdivision_leaves(subdiv_logits, rows)?;
    let (out_coords, src_rows) = child_layout(coords, &leaves);
    Ok(Channel2SpatialOutput {
        feats: select_children(feats, &src_rows, out_coords.len())?,
        sub_binary: sub_binary_from_leaves(&leaves, rows, feats.device())?,
        coords: out_coords,
    })
}

/// Apply the same C2S subdivision to two feature tensors on the same coords
/// (the `h` and residual `x` paths of `SparseResBlockC2S3d`). The reference
/// runs `channel_to_spatial` twice; both calls binarize identical logits and
/// rebuild identical child rows, so do it once here.
pub fn channel_to_spatial_pair(
    coords: &[[i32; 3]],
    first: &Tensor,
    second: &Tensor,
    subdiv_logits: &Tensor,
) -> CandleResult<(Channel2SpatialOutput, Channel2SpatialOutput)> {
    let (rows, channels) = first.dims2()?;
    if rows != coords.len() || second.dim(0)? != rows {
        candle_core::bail!(
            "channel_to_spatial_pair coords/features mismatch: {} coords vs {rows} rows",
            coords.len()
        );
    }
    if channels % 8 != 0 || second.dim(1)? % 8 != 0 {
        candle_core::bail!(
            "channel_to_spatial_pair expects channels divisible by 8, got {channels} and {}",
            second.dim(1)?
        );
    }
    let leaves = subdivision_leaves(subdiv_logits, rows)?;
    let (out_coords, src_rows) = child_layout(coords, &leaves);
    let sub_binary = sub_binary_from_leaves(&leaves, rows, first.device())?;
    let first_out = Channel2SpatialOutput {
        feats: select_children(first, &src_rows, out_coords.len())?,
        sub_binary: sub_binary.clone(),
        coords: out_coords.clone(),
    };
    let second_out = Channel2SpatialOutput {
        feats: select_children(second, &src_rows, out_coords.len())?,
        sub_binary,
        coords: out_coords,
    };
    Ok((first_out, second_out))
}

fn child_layout(coords: &[[i32; 3]], leaves: &[[bool; 8]]) -> (Vec<[i32; 3]>, Vec<u32>) {
    let mut out_coords: Vec<[i32; 3]> = Vec::new();
    let mut src_rows: Vec<u32> = Vec::new();
    for (parent, coord) in coords.iter().enumerate() {
        for (subidx, &leaf) in leaves[parent].iter().enumerate() {
            if leaf {
                let dx = (subidx % 2) as i32;
                let dy = ((subidx / 2) % 2) as i32;
                let dz = (subidx / 4) as i32;
                out_coords.push([coord[0] * 2 + dx, coord[1] * 2 + dy, coord[2] * 2 + dz]);
                src_rows.push((parent * 8 + subidx) as u32);
            }
        }
    }
    (out_coords, src_rows)
}

fn select_children(feats: &Tensor, src_rows: &[u32], children: usize) -> CandleResult<Tensor> {
    let (rows, channels) = feats.dims2()?;
    let flat = feats.reshape((rows * 8, channels / 8))?;
    let src = Tensor::from_vec(src_rows.to_vec(), (children,), feats.device())?;
    flat.index_select(&src, 0)
}

fn sub_binary_from_leaves(
    leaves: &[[bool; 8]],
    rows: usize,
    device: &Device,
) -> CandleResult<Tensor> {
    Tensor::from_vec(
        leaves
            .iter()
            .flat_map(|leaf| leaf.iter().map(|&l| l as u8 as f32))
            .collect(),
        (rows, 8),
        device,
    )
}

/// The C2S skip ("skin") connection: repeat every channel `repeats` times
/// (`x.feats.repeat_interleave(repeats, dim=1)`) so a `[N, Cin]` tensor
/// becomes `[N, 8 * out_channels]` before `channel_to_spatial`.
/// `repeats = out_channels // (Cin / 8)` at the call site.
pub fn repeat_channels(feats: &Tensor, repeats: usize) -> CandleResult<Tensor> {
    if repeats == 1 {
        return Ok(feats.clone());
    }
    let (rows, channels) = feats.dims2()?;
    feats
        .unsqueeze(2)? // [N, C, 1]
        .broadcast_as((rows, channels, repeats))?
        .contiguous()?
        .reshape((rows, channels * repeats))
}

/// Output of `spatial_to_channel`.
#[derive(Debug, Clone)]
pub struct Spatial2ChannelOutput {
    /// Parent coords `child // 2`, row-major sorted (ascending `code`, the
    /// reference `code.unique()` order).
    pub coords: Vec<[i32; 3]>,
    /// `[Np, 8 * C]` F32 with channel layout `[parent, child, channel]`
    /// (empty child slots are zero, matching the reference zero-init).
    pub feats: Tensor,
}

/// S2C (`SparseSpatial2Channel`, factor 2, DIM 3): each child contributes its
/// features to parent `child // 2` at slot `subidx = dx + 2*dy + 4*dz` where
/// `(dx, dy, dz) = child % 2`. Parents are sorted row-major (lexicographic
/// `(x, y, z)` — identical to the reference row-major code order for a single
/// batch). Encoder-side op; kept for tests and round-trip validation.
pub fn spatial_to_channel(
    coords: &[[i32; 3]],
    feats: &Tensor,
) -> CandleResult<Spatial2ChannelOutput> {
    let (rows, channels) = feats.dims2()?;
    if rows != coords.len() {
        candle_core::bail!(
            "spatial_to_channel coords/features mismatch: {} coords vs {rows} rows",
            coords.len()
        );
    }
    // Dedupe parents and sort row-major (lexicographic (x, y, z) == the
    // reference ascending row-major `code` for one batch element).
    let mut parents: Vec<[i32; 3]> = coords
        .iter()
        .map(|c| [c[0] / 2, c[1] / 2, c[2] / 2])
        .collect();
    parents.sort_unstable();
    parents.dedup();
    let parent_index: FxHashMap<[i32; 3], usize> =
        parents.iter().enumerate().map(|(i, p)| (*p, i)).collect();
    let mut out = vec![0.0_f32; parents.len() * 8 * channels];
    let device = feats.device().clone();
    let feats = feats.to_device(&candle_core::Device::Cpu)?;
    let values = feats.flatten_all()?.to_vec1::<f32>()?;
    for (child, coord) in coords.iter().enumerate() {
        let parent = [coord[0] / 2, coord[1] / 2, coord[2] / 2];
        let slot = parent_index[&parent];
        let dx = (coord[0] % 2) as usize;
        let dy = (coord[1] % 2) as usize;
        let dz = (coord[2] % 2) as usize;
        let subidx = dx + 2 * dy + 4 * dz;
        let dst = (slot * 8 + subidx) * channels;
        out[dst..dst + channels].copy_from_slice(&values[child * channels..(child + 1) * channels]);
    }
    let parent_count = parents.len();
    Ok(Spatial2ChannelOutput {
        coords: parents,
        feats: Tensor::from_vec(out, (parent_count, 8 * channels), &device)?,
    })
}

/// Exact block-diagonal full attention for a single batch element: self
/// attention over `qkv: [T, 3, H, head_dim]` -> `[T, H, head_dim]`, scale
/// `1/sqrt(head_dim)`, softmax over all T keys (identical to the dense path
/// with `B = 1`). Queries are chunked; keys are never chunked, so softmax
/// stays exact and no `[H, T, T]` buffer larger than `[H, chunk, T]` is
/// materialized.
pub fn sparse_self_attention(qkv: &Tensor, chunk_size: usize) -> CandleResult<Tensor> {
    let (tokens, fused, heads, head_dim) = qkv.dims4()?;
    if fused != 3 {
        candle_core::bail!("sparse_self_attention expects [T, 3, H, D], got {qkv:?}");
    }
    let qkv = qkv.unsqueeze(0)?; // [1, T, 3, H, D]
    let q = qkv.narrow(2, 0, 1)?.squeeze(2)?; // [1, T, H, D]
    let k = qkv.narrow(2, 1, 1)?.squeeze(2)?;
    let v = qkv.narrow(2, 2, 1)?.squeeze(2)?;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let out = exact_query_chunked_sdpa(
        &q.permute((0, 2, 1, 3))?, // [1, H, T, D]
        &k.permute((0, 2, 1, 3))?,
        &v.permute((0, 2, 1, 3))?,
        scale,
        chunk_size,
    )?;
    Ok(out
        .permute((0, 2, 1, 3))?
        .reshape((tokens, heads, head_dim))?)
}

/// Exact cross attention for a single batch element: `q: [T, H, head_dim]`
/// from the sparse tokens, `kv: [Tkv, 2, H, head_dim]` from the conditioning
/// context -> `[T, H, head_dim]`, scale `1/sqrt(head_dim)`, softmax over all
/// `Tkv` context keys.
pub fn sparse_cross_attention(q: &Tensor, kv: &Tensor, chunk_size: usize) -> CandleResult<Tensor> {
    let (tokens, heads, head_dim) = q.dims3()?;
    let (_kv_tokens, fused, kv_heads, kv_head_dim) = kv.dims4()?;
    if fused != 2 || kv_heads != heads || kv_head_dim != head_dim {
        candle_core::bail!(
            "sparse_cross_attention expects kv [Tkv, 2, H, D], got {kv:?} for q {q:?}"
        );
    }
    let scale = 1.0 / (head_dim as f32).sqrt();
    let out = exact_query_chunked_sdpa(
        &q.unsqueeze(0)?.permute((0, 2, 1, 3))?, // [1, H, T, D]
        &kv.narrow(1, 0, 1)?
            .squeeze(1)?
            .unsqueeze(0)?
            .permute((0, 2, 1, 3))?, // k
        &kv.narrow(1, 1, 1)?
            .squeeze(1)?
            .unsqueeze(0)?
            .permute((0, 2, 1, 3))?, // v
        scale,
        chunk_size,
    )?;
    Ok(out
        .permute((0, 2, 1, 3))?
        .reshape((tokens, heads, head_dim))?)
}
